// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "gda/proxy/route_device.hpp"
#include "gda/proxy/collective_device.hpp"
namespace rocshmem {
__device__ inline void GDAContext::internal_require_pxn_route(
    GDATeam *team, int pe_start, int stride, int pe_size,
    const char *operation) {
  const bool world = pe_start == 0 && stride == 1 &&
                     pe_size == constmem.num_pes;
  const bool cross_node = world ? constmem.world_proxy_node_count > 1
                                : proxy::HasInTeamProxyRoute(team);
  if (cross_node) {
    LOGD_ERROR_ABORT("PXN %s has no supported proxy algorithm for this team",
                     operation);
  }
}

template <typename TryProxy>
__device__ inline bool GDAContext::internal_try_pxn_collective(GDATeam *team, int pe_start, int stride, int pe_size,
    const char *operation, TryProxy try_proxy) {
  if (!constmem.pxn_enabled) {
    return false;
  }
  if (try_proxy()) {
    return true;
  }
  internal_require_pxn_route(team, pe_start, stride, pe_size, operation);
  return false;  // Only a node-local fallback is permitted in PXN mode.
}

template <typename T>
__device__ void GDAContext::internal_broadcast(T *dst, const T *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync, int64_t proxy_epoch_base,
    int proxy_qp_row, GDATeam *team_obj) {  // NOLINT(runtime/int)
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (internal_try_pxn_collective(team_obj, pe_start, stride, pe_size, "broadcast",
          [&] __device__ () {
            return internal_try_pxn_proxy_broadcast<T>(team_obj, dst, src, nelems, pe_root, pe_start, stride, pe_size,
                p_sync, wf_info, proxy_epoch_base, proxy_qp_row);
          })) {
    return;
  }
  if (constmem.num_pes < 4) {
    internal_put_broadcast(dst, src, nelems, pe_root, pe_start, stride, pe_size, wf_info);
    internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
    return;
  }

  // The same multi-proxy transfer uses the unrestricted route when PXN is off.
  if (!constmem.pxn_enabled && internal_aligned_multi_proxy_broadcast<T>(dst, src, nelems, pe_root, pe_start, stride, pe_size,
          p_sync, wf_info, proxy_epoch_base, proxy_qp_row, false)) {
    return;
  }

  // Hierarchical broadcast: choose one leader in each IPC island.  Only the
  // leader fetches the message from the root (striped across merged NICs for
  // an inter-node root); all other local PEs consume the leader's copy over
  // IPC.  This avoids sending the full message once per GPU across the fabric.
  int node_leader = constmem.my_pe;
  for (int rank = 0; rank < pe_size; rank++) {
    const int peer = pe_start + rank * stride;
    int local_pe{-1};
    if (peer == constmem.my_pe ||
        ipcImpl_.isIpcAvailable(constmem.my_pe, peer, &local_pe)) {
      node_leader = peer;
      break;
    }
  }

  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);

  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<T *>(src), nbytes);
  } else if (constmem.my_pe == node_leader) {
    internal_get_broadcast(dst, src, nelems, pe_root, wf_info);
  }

  // Node leaders must finish the inter-node phase before local IPC readers
  // consume their destination buffers.
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  if (constmem.my_pe != node_leader && constmem.my_pe != pe_root) {
    internal_getmem_wg(dst, dst, nbytes, node_leader, node_leader, wf_info);
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
}

template <typename T>
__device__ void GDAContext::internal_put_broadcast(T *dst, const T *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (constmem.my_pe != i)
        internal_putmem_nbi_wg(dst, src, nelems * sizeof(T), i, i, wf_info);
    }
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<T *>(src), nelems * sizeof(T));
  }
}

template <typename T>
__device__ void GDAContext::internal_get_broadcast(T *dst, const T *src,
    int nelems, int pe_root, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<T *>(src), nelems * sizeof(T));
    return;
  }

  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe_root, &local_pe)) {
    const uint64_t offset = reinterpret_cast<const char *>(src) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ipcImpl_.ipcCopy_wg<MemcpyKind::GetBlocking>(dst, ipcImpl_.ipc_bases[local_pe] + offset, nbytes, local_pe);
    return;
  }

  // QP rows are mapped round-robin across merged NICs.  Stripe a large
  // inter-node message over the useful QP rows, submit every chunk first,
  // and only then wait for completion.
  if (is_thread_zero_in_block()) {
    constexpr size_t MIN_BYTES_PER_RAIL = 64 * 1024;
    uint32_t desired_rails = static_cast<uint32_t>((nbytes + MIN_BYTES_PER_RAIL - 1) / MIN_BYTES_PER_RAIL);
    if (desired_rails == 0) desired_rails = 1;
    desired_rails = min(desired_rails, num_qps_per_pe);

    // Share the available rows among concurrently active workgroups to avoid
    // multiplying the outstanding reads by gridDim.x.
    const uint32_t active_wgs = static_cast<uint32_t>(gridDim.x);
    uint32_t rails = (desired_rails + active_wgs - 1) / active_wgs;
    if (rails == 0) rails = 1;
    rails = min(rails, num_qps_per_pe);

    const size_t chunk = (nbytes + rails - 1) / rails;
    const uint64_t src_offset = reinterpret_cast<const char *>(src) - base_heap[constmem.my_pe];
    char *dst_bytes = reinterpret_cast<char *>(dst);

    for (uint32_t rail = 0; rail < rails; rail++) {
      const size_t begin = static_cast<size_t>(rail) * chunk;
      if (begin >= nbytes) break;
      const size_t length = ((nbytes - begin) < chunk) ? (nbytes - begin) : chunk;
      const uint32_t qp_row = (static_cast<uint32_t>(blockIdx.x) + rail * active_wgs) % num_qps_per_pe;
      const uint32_t qp_index = qp_row * constmem.num_pes + pe_root;
      qps[qp_index].get_nbi(dst_bytes + begin,
                            base_heap[pe_root] + src_offset + begin,
                            length, wf_info);
    }

    for (uint32_t rail = 0; rail < rails; rail++) {
      const size_t begin = static_cast<size_t>(rail) * chunk;
      if (begin >= nbytes) break;
      const uint32_t qp_row = (static_cast<uint32_t>(blockIdx.x) + rail * active_wgs) % num_qps_per_pe;
      const uint32_t qp_index = qp_row * constmem.num_pes + pe_root;
      qps[qp_index].quiet(wf_info);
    }
  }
}
}
