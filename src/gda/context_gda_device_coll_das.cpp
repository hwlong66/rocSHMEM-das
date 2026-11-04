// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/******************************************************************************
 * 1. 较大的DAS自研改动在本文件中维护，尽量保留原文件中的社区代码，
 *    方便后续维护及同步上游更新。
 * 2. 修改已有函数时，在此实现对应的_das函数；启用USE_DAS后(默认开启)，原函数
 *    在入口调用_das函数并返回，未启用时执行原来的社区实现。
 * 3. 新增辅助函数也在此维护，公共逻辑尽量复用原文件中的函数；
 *****************************************************************************/

#include "rocshmem/rocshmem.hpp"
#include "context_incl.hpp"
#include "context_gda_tmpl_device.hpp"
#include "util.hpp"
#include "gda_team.hpp"


#if defined(USE_DAS)
#include "gda/proxy/sync_device.hpp"

namespace rocshmem {

// One signal protocol serves scalar, wave and workgroup callers. Only the
// participants in the requested scope synchronize their GPU execution.
__device__ bool GDAContext::internal_try_pxn_sync_das(GDATeam *team_obj, int64_t *p_sync,
    ThreadScope scope) {
  if (!constmem.pxn_enabled) {
    return false;
  }
  const auto *info = team_obj->tinfo_wrt_world;
  const bool world = info->pe_start == 0 && info->stride == 1 && team_obj->num_pes == constmem.num_pes;
  const bool eligible = world
      ? proxy::IsWorldProxySyncEligible(constmem.my_pe, info->pe_start, info->stride, team_obj->num_pes)
      : proxy::HasInTeamProxyRoute(team_obj);
  if (!eligible) {
    internal_require_pxn_route(team_obj, info->pe_start, info->stride, team_obj->num_pes, "sync");
    return false;
  }
  __threadfence_system();
  if (scope == ThreadScope::wg) {
    __syncthreads();
  } else if (scope == ThreadScope::wave) {
    __builtin_amdgcn_wave_barrier();
  }
  const bool leader = scope == ThreadScope::thread ||
      (scope == ThreadScope::wg ? is_thread_zero_in_block() : is_thread_zero_in_wave());
  ActiveWFInfo wf_info(ctx_id_, scope);
  if (leader) {
    const int64_t epoch = static_cast<int64_t>(team_obj->in_team_proxy_sync_epoch);
    if (world) {
      internal_world_proxy_sync_thread(constmem.my_pe, team_obj->num_pes, p_sync, wf_info, epoch,
          team_obj->pool_index_ % num_qps_per_pe);
    } else {
      internal_team_proxy_sync_thread(team_obj, p_sync, wf_info, epoch);
    }
    team_obj->in_team_proxy_sync_epoch = epoch + 1;
  }
  __threadfence_system();
  if (scope == ThreadScope::wg) {
    __syncthreads();
  } else if (scope == ThreadScope::wave) {
    __builtin_amdgcn_wave_barrier();
  }
  return true;
}

__device__ bool GDAContext::internal_try_pxn_sync_range_das(int pe_start, int stride, int pe_size,
    int64_t *p_sync, ThreadScope scope) {
  if (!constmem.pxn_enabled || pe_start != 0 || stride != 1 || pe_size != constmem.num_pes) {
    return false;
  }
  return internal_try_pxn_sync_das(reinterpret_cast<GDATeam *>(ROCSHMEM_TEAM_WORLD), p_sync, scope);
}

// A wave uses one selected same-NIC proxy per IPC island. Data is pushed
// between proxies; the remaining team members consume it through local IPC.
__device__ bool GDAContext::internal_proxy_broadcast_wave_das(GDATeam *team_obj, void *dst,
    const void *src, size_t bytes, int root, int64_t *p_sync, ActiveWFInfo &wf_info, int64_t &epoch) {
  const auto *info = team_obj->tinfo_wrt_world;
  if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
      proxy::ProxyCollectiveLayout::None) {
    return false;
  }
  const bool world = info->pe_start == 0 && info->stride == 1 && team_obj->num_pes == constmem.num_pes;
  const int nodes = world ? constmem.world_proxy_node_count : team_obj->in_team_proxy_node_count;
  int root_first = -1;
  int root_proxy = -1;
  int local_proxy = -1;
  for (int node = 0; node < nodes; node++) {
    const int first = world ? constmem.world_proxy_node_first_pes[node]
                            : team_obj->in_team_proxy_node_first_pes[node];
    const int end = world ? proxy::WorldProxyNodeEnd(first)
                         : node + 1 < nodes ? team_obj->in_team_proxy_node_first_pes[node + 1]
                                            : constmem.num_pes;
    const int selected = world ? first : team_obj->in_team_proxy_pes[node];
    if (first == constmem.ipc_first_pe) {
      local_proxy = selected;
    }
    if (root >= first && root < end) {
      root_first = first;
      root_proxy = selected;
    }
  }
  if (root_proxy < 0 || local_proxy < 0) {
    return false;
  }
  const int qp_row = world && !constmem.world_proxy_nic_layout_enabled ? 0
      : team_obj->pool_index_ % num_qps_per_pe;
  constexpr size_t tile_bytes = 64 * 1024;
  if (bytes == 0) {
    internal_dispatch_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
  }
  for (size_t offset = 0; offset < bytes; offset += tile_bytes) {
    const size_t count = min(tile_bytes, bytes - offset);
    char *target = static_cast<char *>(dst) + offset;
    const char *source = static_cast<const char *>(src) + offset;
    if (constmem.my_pe == root) {
      memcpy_wave<MemcpyKind::Get>(target, const_cast<char *>(source), count);
    }
    internal_dispatch_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
    if (constmem.my_pe == root_proxy && root_proxy != root) {
      internal_getmem_wave(target, target, count, root, root, wf_info);
    }
    internal_dispatch_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
    if (constmem.my_pe == root_proxy && is_thread_zero_in_wave()) {
      for (int node = 0; node < nodes; node++) {
        const int first = world ? constmem.world_proxy_node_first_pes[node]
                                : team_obj->in_team_proxy_node_first_pes[node];
        const int peer = world ? first : team_obj->in_team_proxy_pes[node];
        if (first != root_first) {
          internal_putmem(target, target, count, peer, qp_row * constmem.num_pes + peer, wf_info);
        }
      }
    }
    internal_dispatch_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
    if (constmem.my_pe != local_proxy && constmem.my_pe != root) {
      internal_getmem_wave(target, target, count, local_proxy, local_proxy, wf_info);
    }
    internal_dispatch_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
  }
  return true;
}

__device__ bool GDAContext::internal_try_pxn_broadcastmem_wave_das(rocshmem_team_t team, void *dst,
    const void *src, int bytes, int root) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const auto *info = team_obj->tinfo_wrt_world;
  return internal_try_pxn_collective(team_obj, info->pe_start, info->stride, team_obj->num_pes, "broadcast_wave",
      [&] __device__ () {
        ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
        int64_t epoch = static_cast<int64_t>(team_obj->proxy_broadcast_sync_epoch);
        if (!internal_proxy_broadcast_wave_das(team_obj, dst, src, bytes, team_obj->get_pe_in_world(root),
                team_obj->bcast_pSync, wf_info, epoch)) {
          return false;
        }
        if (is_thread_zero_in_wave()) {
          team_obj->proxy_broadcast_sync_epoch = epoch;
        }
        __threadfence_system();
        __builtin_amdgcn_wave_barrier();
        return true;
      });
}

__device__ bool GDAContext::internal_try_pxn_fcollectmem_wave_das(rocshmem_team_t team, void *dst,
    const void *src, int bytes) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const auto *info = team_obj->tinfo_wrt_world;
  return internal_try_pxn_collective(team_obj, info->pe_start, info->stride, team_obj->num_pes, "fcollect_wave",
      [&] __device__ () {
        if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
            proxy::ProxyCollectiveLayout::None) {
          return false;
        }
        constexpr size_t capacity = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double);
        char *stage = static_cast<char *>(team_obj->pWrk);
        ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
        int64_t epoch = static_cast<int64_t>(team_obj->proxy_fcollect_sync_epoch);
        long *p_sync = team_obj->alltoall_pSync;
        if (bytes == 0) {
          internal_dispatch_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
        }
        for (size_t offset = 0; offset < static_cast<size_t>(bytes); offset += capacity) {
          const size_t count = min(capacity, static_cast<size_t>(bytes) - offset);
          memcpy_wave<MemcpyKind::Get>(stage, const_cast<char *>(static_cast<const char *>(src) + offset), count);
          internal_dispatch_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
          for (int rank = 0; rank < team_obj->num_pes; rank++) {
            if (!internal_proxy_broadcast_wave_das(team_obj,
                    static_cast<char *>(dst) + static_cast<size_t>(rank) * bytes + offset,
                    stage, count, team_obj->get_pe_in_world(rank), p_sync, wf_info, epoch)) {
              return false;
            }
          }
        }
        if (is_thread_zero_in_wave()) {
          team_obj->proxy_fcollect_sync_epoch = epoch;
        }
        __threadfence_system();
        __builtin_amdgcn_wave_barrier();
        return true;
      });
}

__device__ bool GDAContext::internal_try_pxn_alltoallmem_wave_das(rocshmem_team_t team, void *dst,
    const void *src, int bytes) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const auto *info = team_obj->tinfo_wrt_world;
  return internal_try_pxn_collective(team_obj, info->pe_start, info->stride, team_obj->num_pes, "alltoall_wave",
      [&] __device__ () {
        if (proxy::SelectProxyCollectiveLayout(team_obj, info->pe_start, info->stride, team_obj->num_pes) ==
            proxy::ProxyCollectiveLayout::None) {
          return false;
        }
        constexpr size_t capacity = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / 2;
        char *stage = static_cast<char *>(team_obj->pWrk);
        char *received = stage + capacity;
        ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
        int64_t epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch);
        long *p_sync = team_obj->alltoall_pSync;
        if (bytes == 0) {
          internal_dispatch_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
        }
        for (int target = 0; target < team_obj->num_pes; target++) {
          for (size_t offset = 0; offset < static_cast<size_t>(bytes); offset += capacity) {
            const size_t count = min(capacity, static_cast<size_t>(bytes) - offset);
            memcpy_wave<MemcpyKind::Get>(stage,
                const_cast<char *>(static_cast<const char *>(src) + static_cast<size_t>(target) * bytes + offset), count);
            internal_dispatch_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
            for (int rank = 0; rank < team_obj->num_pes; rank++) {
              if (!internal_proxy_broadcast_wave_das(team_obj, received, stage, count,
                      team_obj->get_pe_in_world(rank), p_sync, wf_info, epoch)) {
                return false;
              }
              if (team_obj->my_pe == target) {
                char *output = static_cast<char *>(dst) + static_cast<size_t>(rank) * bytes + offset;
                for (size_t i = get_flat_block_id() % WF_SIZE; i < count; i += WF_SIZE) {
                  output[i] = uncached_load(received + i);
                }
              }
              internal_dispatch_proxy_sync_wave(team_obj, p_sync, wf_info, epoch++);
            }
          }
        }
        if (is_thread_zero_in_wave()) {
          team_obj->proxy_alltoall_sync_epoch = epoch;
        }
        __threadfence_system();
        __builtin_amdgcn_wave_barrier();
        return true;
      });
}

__device__ void GDAContext::internal_get_broadcastmem_wg_das(void *dst, const void *src, int nelems,
    int pe_root, ActiveWFInfo &wf_info) {
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<void *>(src), nelems);
    return;
  }

  const size_t nbytes = static_cast<size_t>(nelems);
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

__device__ void GDAContext::broadcastmem_wg_das(rocshmem_team_t team, void *dst,
    const void *src, int nelems, int pe_root) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(pe_root);
  if (internal_try_pxn_collective(team_obj, pe_start, stride, pe_size, "broadcast",
          [&] __device__ () {
            if (proxy::SelectProxyCollectiveLayout(team_obj, pe_start, stride, pe_size) ==
                proxy::ProxyCollectiveLayout::None) {
              return false;
            }
            // Publish GPU-written input before another local proxy reads it.
            internal_try_pxn_sync_das(team_obj, team_obj->barrier_pSync, ThreadScope::wg);
            ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
            const int64_t epoch = static_cast<int64_t>(team_obj->proxy_broadcast_sync_epoch);
            const int phases = nelems == 0 ? 1 : proxy::TeamBroadcastPhaseCount(team_obj);
            if (nelems == 0) {
              internal_dispatch_proxy_sync_wg(team_obj, p_sync, wf_info, epoch);
            } else {
              internal_broadcast(static_cast<char *>(dst), static_cast<const char *>(src), nelems,
                  pe_root_world, pe_start, stride, pe_size, p_sync, epoch,
                  team_obj->pool_index_ % num_qps_per_pe, team_obj);
            }
            if (is_thread_zero_in_block()) {
              team_obj->proxy_broadcast_sync_epoch = epoch + phases;
            }
            __syncthreads();
            return true;
          })) {
    return;
  }
  internal_broadcastmem_wg(dst, src, nelems, pe_root_world, pe_start, stride,
               pe_size, p_sync);
}

__device__ void GDAContext::internal_broadcastmem_wg_das(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync) {  // NOLINT(runtime/int)
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (constmem.num_pes < 4) {
    internal_put_broadcastmem_wg(dst, src, nelems, pe_root, pe_start, stride, pe_size, wf_info);
    internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
    return;
  }

  // Attempt multi-proxy broadcast for large messages with multiple nodes
  if (internal_proxy_broadcast(dst, src, nelems, pe_root, pe_start,
                                  stride, pe_size, p_sync, wf_info)) {
    return;  // Proxy broadcast handled the operation
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

  const size_t nbytes = static_cast<size_t>(nelems);
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<void *>(src), nbytes);
  } else if (constmem.my_pe == node_leader) {
    internal_get_broadcastmem_wg(dst, src, nelems, pe_root, wf_info);
  }

  // Node leaders must finish the inter-node phase before local IPC readers
  // consume their destination buffers.
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  if (constmem.my_pe != node_leader && constmem.my_pe != pe_root) {
    internal_getmem_wg(dst, dst, nbytes, node_leader, node_leader, wf_info);
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
}

__device__ bool GDAContext::internal_proxy_broadcast(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)

  constexpr size_t MIN_BYTES_PER_PROXY = 64 * 1024;
  const size_t nbytes = static_cast<size_t>(nelems);
  const int local_shm_size = constmem.ipc_shm_size;

  // Multi-proxy hierarchical broadcast.  One PE per local GPU/NIC transfers
  // one segment between nodes; IPC stages the root data and fans the segments
  // out locally.  Restrict this path to complete, contiguous node groups so
  // proxy slot N names the same local GPU/NIC on every node.
  const bool use_proxy_bcast =
      nbytes >= kHierarchicalMinBytes && stride == 1 && local_shm_size > 1 &&
      constmem.ipc_stride == 1 && pe_start % local_shm_size == 0 &&
      pe_size % local_shm_size == 0 && pe_size > local_shm_size;
  if (!use_proxy_bcast) {
    return false;
  }

  // Calculate optimal proxy count: balance parallelism vs overhead per proxy
  const size_t useful_proxies = (nbytes + MIN_BYTES_PER_PROXY - 1) / MIN_BYTES_PER_PROXY;
  const int proxies = min(local_shm_size, static_cast<int>(useful_proxies));
  const int local_slot = constmem.my_pe - constmem.ipc_first_pe;
  const int root_rank = pe_root - pe_start;
  const int root_node_first = pe_start + (root_rank / local_shm_size) * local_shm_size;
  const bool on_root_node = constmem.ipc_first_pe == root_node_first;
  const size_t chunk = (nbytes + proxies - 1) / proxies;
  char *dst_bytes = reinterpret_cast<char *>(dst);
  const char *src_bytes = reinterpret_cast<const char *>(src);

  // Root keeps its complete result.  The other root-node proxies stage only
  // their segment in dst, making a symmetric source address available to
  // the corresponding proxy on every remote node.
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst_bytes, const_cast<char *>(src_bytes), nbytes);
  } else if (on_root_node && local_slot < proxies) {
    const size_t begin = static_cast<size_t>(local_slot) * chunk;
    if (begin < nbytes) {
      const size_t length = min(chunk, nbytes - begin);
      internal_getmem_wg(dst_bytes + begin, src_bytes + begin, length,
                         pe_root, pe_root, wf_info);
    }
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  // Every remote-node proxy fetches one disjoint segment.  Since NIC
  // selection is unique per local PE, the node drives its available NICs in
  // parallel instead of funneling all traffic through one leader GPU.
  if (!on_root_node && local_slot < proxies) {
    const size_t begin = static_cast<size_t>(local_slot) * chunk;
    if (begin < nbytes) {
      const size_t length = min(chunk, nbytes - begin);
      const int source_proxy = root_node_first + local_slot;
      const uint32_t qp_row = static_cast<uint32_t>(blockIdx.x) % num_qps_per_pe;
      const int qp_index = qp_row * constmem.num_pes + source_proxy;
      internal_getmem_wg(dst_bytes + begin, dst_bytes + begin, length,
                         source_proxy, qp_index, wf_info);
    }
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  // Assemble the complete result from this node's local proxy buffers.
  for (int slot = 0; slot < proxies; slot++) {
    if (slot == local_slot) continue;
    const size_t begin = static_cast<size_t>(slot) * chunk;
    if (begin >= nbytes) break;
    const size_t length = min(chunk, nbytes - begin);
    const int local_proxy = constmem.ipc_first_pe + slot;
    internal_getmem_wg(dst_bytes + begin, dst_bytes + begin, length,
                       local_proxy, local_proxy, wf_info);
  }

  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);

  return true;  // Proxy broadcast completed successfully
}

}  // namespace rocshmem
#endif  // defined(USE_DAS)

