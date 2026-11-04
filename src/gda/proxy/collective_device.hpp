// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_PROXY_COLLECTIVE_DEVICE_HPP_
#define LIBRARY_SRC_GDA_PROXY_COLLECTIVE_DEVICE_HPP_

#if defined(USE_DAS)

namespace rocshmem {

namespace proxy {


// An epoch identifies one complete synchronization phase, not one IPC domain or PE.
constexpr int kWorldBroadcastSyncPhases = 3;
constexpr int kPartialTeamBroadcastSyncPhases = 4;

/**
 * @brief Return the broadcast synchronization phases shared by collectives.
 *
 * @param team Team whose route is being used.
 * @return The phase count reserved by the selected broadcast protocol.
 */
__device__ inline int TeamBroadcastPhaseCount(const GDATeam *team) {
  const auto *info = team->tinfo_wrt_world;
  const bool is_world_team = info->pe_start == 0 && info->stride == 1 &&
                             team->num_pes == constmem.num_pes;
  return HasInTeamProxyRoute(team) && !is_world_team
             ? kPartialTeamBroadcastSyncPhases
             : kWorldBroadcastSyncPhases;
}

// Preserve the original full-vector traversal; sliced reduce-scatter visits
// only this WG's interval within every destination block.
struct ProxyReductionRange {
  size_t block_elements;
  size_t total_elements;
  size_t element_offset;
  size_t element_count;
  bool is_partitioned_scatter;

  __device__ static ProxyReductionRange Make(int nelems, int peers, bool scatter,
      int elem_offset, int elem_count) {
    const size_t element_count = elem_count < 0 ? nelems : elem_count;
    return {static_cast<size_t>(nelems), scatter ? static_cast<size_t>(peers) * nelems : nelems,
        static_cast<size_t>(elem_offset), element_count,
        scatter && (elem_offset != 0 || element_count != static_cast<size_t>(nelems))};
  }
  __device__ int segment_count(int peers) const { return is_partitioned_scatter ? peers : 1; }
  __device__ size_t segment_begin(int segment) const { return is_partitioned_scatter ? segment * block_elements + element_offset : 0; }
  __device__ size_t segment_end(int segment) const { return is_partitioned_scatter ? segment_begin(segment) + element_count : total_elements; }
};

}  // namespace proxy


// ############################################################################
// BROADCAST AND FCOLLECT
// ############################################################################

/**
 * @brief Split a broadcast across proxies at equal local PE indices.
 *
 * Local PE i receives its segment from local PE i on the root IPC domain.
 * PXN validates matching NICs at these indices and uses WORLD proxy
 * synchronization. Without PXN, unrestricted slot mapping and
 * team synchronization are retained.
 */
template <typename T>
__device__ bool GDAContext::internal_aligned_multi_proxy_broadcast(T *dst, const T *src, int nelems,
    int pe_root, int pe_start, int stride, int pe_size, long *p_sync, ActiveWFInfo &wf_info,
    int64_t proxy_epoch_base, int proxy_qp_row, bool require_matching_nic) {  // NOLINT(runtime/int)
  constexpr size_t kMinBytesPerProxy = 64 * 1024;
  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);
  const int local_size = constmem.ipc_shm_size;
  const bool eligible = require_matching_nic
      ? nbytes > 0 &&
            proxy::IsAlignedWorldProxyDataEligible(pe_start, stride, pe_size)
      : nbytes >= kHierarchicalMinBytes && stride == 1 && local_size > 1 &&
            constmem.ipc_stride == 1 && pe_start % local_size == 0 &&
            pe_size % local_size == 0 && pe_size > local_size;
  if (!eligible) {
    return false;
  }

  const int local_slot = constmem.my_pe - constmem.ipc_first_pe;
  const int root_rank = pe_root - pe_start;
  const int root_ipc_domain_first_pe = pe_start + (root_rank / local_size) * local_size;
  const int root_slot = require_matching_nic ? pe_root - root_ipc_domain_first_pe : 0;
  const int proxy_rank = (local_slot - root_slot + local_size) % local_size;
  const size_t useful = (nbytes + kMinBytesPerProxy - 1) / kMinBytesPerProxy;
  const int proxies = min(local_size, static_cast<int>(useful));
  const bool on_root_ipc_domain = constmem.ipc_first_pe == root_ipc_domain_first_pe;
  // Keep proxy boundaries aligned for vector IPC copies. The final proxy
  // still transfers only the remaining bytes, never the alignment padding.
  const size_t chunk = ((nbytes + proxies - 1) / proxies + 15) & ~size_t{15};
  char *dst_bytes = reinterpret_cast<char *>(dst);
  const char *src_bytes = reinterpret_cast<const char *>(src);

  if (require_matching_nic && constmem.my_pe == pe_root && is_thread_zero_in_block()) {
    LOGD_INFO("PXN broadcast data route: root PE %d, first proxy slot %d, %d proxy lane(s), %llu byte(s)",
        pe_root, root_slot, proxies, static_cast<unsigned long long>(nbytes));
  }

  // Stage one segment on each selected proxy in the root IPC domain.
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst_bytes, const_cast<char *>(src_bytes), nbytes);
  } else if (on_root_ipc_domain && proxy_rank < proxies) {
    const size_t begin = static_cast<size_t>(proxy_rank) * chunk;
    if (begin < nbytes) {
      const size_t length = min(chunk, nbytes - begin);
      internal_proxy_getmem_wg(dst_bytes + begin, src_bytes + begin, length, pe_root, 0, wf_info);
    }
  }

  if (require_matching_nic) {
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, proxy_epoch_base, proxy_qp_row);
  } else {
    internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
  }

  // Both policies push the same staged segments. The policy only selects
  // the participating IPC domains and the QP row used by their proxies.
  if (on_root_ipc_domain && proxy_rank < proxies) {
    const size_t begin = static_cast<size_t>(proxy_rank) * chunk;
    if (begin < nbytes) {
      const size_t length = min(chunk, nbytes - begin);
      const uint32_t qp_row = require_matching_nic && proxy_qp_row >= 0
          ? static_cast<uint32_t>(proxy_qp_row) % num_qps_per_pe
          : static_cast<uint32_t>(blockIdx.x) % num_qps_per_pe;
      const int ipc_domain_count = require_matching_nic ? constmem.world_proxy_ipc_domain_count : pe_size / local_size;
      for (int ipc_domain = 0; ipc_domain < ipc_domain_count; ipc_domain++) {
        const int first = require_matching_nic
            ? constmem.world_proxy_ipc_domain_first_pes[ipc_domain] : pe_start + ipc_domain * local_size;
        if (first == root_ipc_domain_first_pe) {
          continue;
        }
        const int target_proxy = first + local_slot;
        internal_proxy_putmem_wg(dst_bytes + begin, dst_bytes + begin, length, target_proxy, qp_row, wf_info);
      }
    }
  }

  if (require_matching_nic) {
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, proxy_epoch_base + 1,
        proxy_qp_row);
  } else {
    internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
  }

  // Assemble the complete result from local proxy buffers.
  for (int rank = 0; rank < proxies; rank++) {
    const int slot = (root_slot + rank) % local_size;
    if (slot == local_slot) {
      continue;
    }
    const size_t begin = static_cast<size_t>(rank) * chunk;
    if (begin >= nbytes) {
      break;
    }
    const size_t length = min(chunk, nbytes - begin);
    const int local_proxy = constmem.ipc_first_pe + slot;
    internal_proxy_getmem_wg(dst_bytes + begin, dst_bytes + begin, length, local_proxy, 0, wf_info);
  }

  if (require_matching_nic) {
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, proxy_epoch_base + 2,
        proxy_qp_row);
  } else {
    internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
  }
  return true;
}

/**
 * @brief Try the eligible PXN broadcast path before the original GDA path.
 *
 * @param team_obj Team route metadata, or nullptr for an internal operation.
 * @param dst Destination buffer.
 * @param src Root source buffer.
 * @param nelems Number of elements.
 * @param pe_root Root world PE.
 * @param pe_start First world PE in the operation.
 * @param stride World-rank stride between operation members.
 * @param pe_size Number of operation members.
 * @param p_sync Synchronization storage.
 * @param wf_info Active workgroup metadata.
 * @param proxy_epoch_base First synchronization epoch for this operation.
 * @param proxy_qp_row Preferred QP row, or -1 for automatic selection.
 * @return true when a PXN path completed the broadcast.
 */
template <typename T>
__device__ bool GDAContext::internal_try_pxn_proxy_broadcast(GDATeam *team_obj, T *dst, const T *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size, long *p_sync, ActiveWFInfo &wf_info,
    int64_t proxy_epoch_base, int proxy_qp_row) {  // NOLINT(runtime/int)
  // These layouts describe PXN routes; the unrestricted path is selected
  // separately by internal_broadcast() when PXN is disabled.
  const auto layout = proxy::SelectProxyCollectiveLayout(team_obj, pe_start, stride, pe_size);
  using Layout = proxy::ProxyCollectiveLayout;

  switch (layout) {
    case Layout::TeamProxyRoute:
      return internal_routed_team_proxy_broadcast<T>(team_obj, dst, src, nelems, pe_root, p_sync, wf_info,
          proxy_epoch_base, proxy_qp_row);
    case Layout::WorldAlignedData:
      return internal_aligned_multi_proxy_broadcast<T>(dst, src, nelems, pe_root, pe_start, stride, pe_size,
          p_sync, wf_info, proxy_epoch_base, proxy_qp_row, /*require_matching_nic=*/true);
    case Layout::WorldSyncOnly:
      // Sync is available, but data lanes still need their own route check.
      if (internal_routed_multi_proxy_broadcast<T>(dst, src, nelems, pe_root, pe_start, stride, pe_size,
          p_sync, wf_info, proxy_epoch_base, proxy_qp_row)) {
        return true;
      }
      return internal_single_proxy_world_broadcast<T>(dst, src, nelems, pe_root, pe_start, stride, pe_size,
          p_sync, wf_info, proxy_epoch_base, proxy_qp_row);
    case Layout::None:
      return false;
  }
  return false;
}

/**
 * @brief Select packed WORLD fcollect or routed broadcasts per source PE.
 *
 * @param team_obj Team and proxy layout metadata.
 * @param dst Concatenated destination buffer.
 * @param src This PE's source segment.
 * @param nelems Elements in each source segment.
 * @return true when the hierarchical broadcast path completed fcollect.
 */
template <typename T>
__device__ bool GDAContext::internal_try_hierarchical_fcollect(GDATeam *team_obj, T *dst, const T *src,
    int nelems) {
  const auto *team_info = team_obj->tinfo_wrt_world;
  const int pe_size = team_obj->num_pes;
  const bool has_in_team_route = proxy::HasInTeamProxyRoute(team_obj);
  const auto layout = proxy::SelectProxyCollectiveLayout(team_obj, team_info->pe_start, team_info->stride,
      pe_size);
  if (constmem.pxn_enabled && nelems > 0 &&
      (layout == proxy::ProxyCollectiveLayout::WorldAlignedData ||
       layout == proxy::ProxyCollectiveLayout::WorldSyncOnly)) {
    int max_local_size = 0;
    for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
      const int first = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
      max_local_size = max(max_local_size, proxy::WorldProxyIpcDomainEnd(first) - first);
    }
    // Scratch capacity is a safety bound, not a performance threshold.
    // Larger packed messages were slower than per-PE broadcasts in testing.
    // Use the largest IPC domain so every PE makes the same protocol choice.
    constexpr size_t kMaxPackedBytes = 64 * 1024;
    constexpr size_t kPackedCapacity = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / 2;
    constexpr size_t kPackedLimit = kMaxPackedBytes < kPackedCapacity ? kMaxPackedBytes : kPackedCapacity;
    if (max_local_size > 0 && static_cast<size_t>(nelems) <=
        kPackedLimit / sizeof(T) / static_cast<size_t>(max_local_size)) {
      internal_ipc_domain_proxy_fcollect(team_obj, dst, src, nelems);
      return true;
    }
  }
  if ((pe_size < 4 && !has_in_team_route && !constmem.pxn_enabled) ||
      (static_cast<size_t>(nelems) * sizeof(T) < kHierarchicalMinBytes && !has_in_team_route &&
      layout == proxy::ProxyCollectiveLayout::None)) {
    return false;
  }

  // One network copy per source, then IPC fan-out at the destination.
  for (int source_rank = 0; source_rank < pe_size; source_rank++) {
    const int root = team_obj->get_pe_in_world(source_rank);
    internal_broadcast(dst + source_rank * nelems, src, nelems, root, team_info->pe_start, team_info->stride,
        pe_size, team_obj->alltoall_pSync,
        static_cast<int64_t>(team_obj->proxy_fcollect_sync_epoch +
                             proxy::TeamBroadcastPhaseCount(team_obj) * source_rank),
        team_obj->pool_index_ % num_qps_per_pe, team_obj);
  }
  if (is_thread_zero_in_block()) {
    team_obj->proxy_fcollect_sync_epoch += proxy::TeamBroadcastPhaseCount(team_obj) * pe_size;
  }
  __syncthreads();
  return true;
}

/** @brief Pack local contributions into one WORLD broadcast per IPC domain when scratch permits. */
template <typename T>
__device__ void GDAContext::internal_ipc_domain_proxy_fcollect(GDATeam *team_obj, T *dst, const T *src, int nelems) {
  constexpr size_t kHalfElements = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / (2 * sizeof(T));
  T *packed = reinterpret_cast<T *>(team_obj->pWrk);
  T *received = packed + kHalfElements;
  const int tid = get_flat_block_id();
  const int block_size = get_flat_block_size();
  const int qp_row = team_obj->pool_index_ % num_qps_per_pe;
  long *p_sync = team_obj->alltoall_pSync;
  int64_t epoch = static_cast<int64_t>(team_obj->proxy_fcollect_sync_epoch);
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);

  if (constmem.my_pe == constmem.ipc_first_pe) {
    for (int slot = 0; slot < constmem.ipc_shm_size; slot++) {
      const int peer = constmem.ipc_first_pe + slot;
      internal_proxy_getmem_wg(packed + slot * nelems, src, static_cast<size_t>(nelems) * sizeof(T),
          peer, 0, wf_info);
    }
  }
  // All source contributions must be packed before any in-place result is written.
  internal_world_proxy_sync_wg(constmem.my_pe, team_obj->num_pes, p_sync, wf_info, epoch++, qp_row);
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    const int root = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
    const int count = (proxy::WorldProxyIpcDomainEnd(root) - root) * nelems;
    internal_broadcast(received, packed, count, root, 0, 1, team_obj->num_pes, p_sync, epoch,
        qp_row, team_obj);
    T *ipc_domain_dest = dst + static_cast<size_t>(root) * nelems;
    // Keep bypass loads for NIC-written tiles, but move aligned data in
    // 16-byte chunks instead of issuing one uncached load per element.
    // Arbitrary user offsets may be unaligned; retain the scalar path there.
    const bool aligned = reinterpret_cast<uintptr_t>(ipc_domain_dest) % 16 == 0 &&
                         reinterpret_cast<uintptr_t>(received) % 16 == 0;
    if (aligned) {
      memcpy_wg<MemcpyKind::Get>(ipc_domain_dest, received, static_cast<size_t>(count) * sizeof(T));
    } else {
      for (int i = tid; i < count; i += block_size) {
        ipc_domain_dest[i] = uncached_load(received + i);
      }
    }
    __syncthreads();
    epoch += proxy::kWorldBroadcastSyncPhases;
    // Protect the packed input and receive tile until every consumer finishes.
    internal_world_proxy_sync_wg(constmem.my_pe, team_obj->num_pes, p_sync, wf_info, epoch++, qp_row);
  }
  if (is_thread_zero_in_block()) {
    team_obj->proxy_fcollect_sync_epoch = epoch;
  }
  __syncthreads();
}

/**
 * @brief Broadcast a WORLD message through one same-NIC proxy per IPC domain.
 *
 * @tparam T Element type.
 * @param dst Destination buffer.
 * @param src Root source buffer.
 * @param nelems Number of elements.
 * @param pe_root Root world PE.
 * @return true when this transport handled the operation.
 */
template <typename T>
__device__ bool GDAContext::internal_single_proxy_world_broadcast(T *dst, const T *src, int nelems,
    int pe_root, int pe_start, int stride, int pe_size, long *p_sync, ActiveWFInfo &wf_info,
    int64_t proxy_epoch_base, int proxy_qp_row) {
  if (!proxy::IsWorldProxySyncEligible(constmem.my_pe, pe_start, stride,
      pe_size) || proxy::IsAlignedWorldProxyDataEligible(pe_start, stride, pe_size)) {
    return false;
  }

  const int local_proxy = constmem.ipc_first_pe;
  int root_proxy = -1;
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    const int first = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
    const int end = ipc_domain + 1 < constmem.world_proxy_ipc_domain_count
                        ? constmem.world_proxy_ipc_domain_first_pes[ipc_domain + 1]
                        : constmem.num_pes;
    if (pe_root >= first && pe_root < end) {
      root_proxy = first;
    }
  }
  if (root_proxy < 0) {
    return false;
  }

  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<T *>(src), nbytes);
  } else if (local_proxy == root_proxy && constmem.my_pe == local_proxy) {
    internal_proxy_getmem_wg(dst, src, nbytes, pe_root, 0, wf_info);
  }
  internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, proxy_epoch_base, proxy_qp_row);

  if (constmem.my_pe == root_proxy) {
    const uint32_t qp_row = proxy_qp_row >= 0
        ? static_cast<uint32_t>(proxy_qp_row) % num_qps_per_pe
        : static_cast<uint32_t>(blockIdx.x) % num_qps_per_pe;
    for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
      const int peer = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
      if (peer != root_proxy) {
        internal_proxy_putmem_wg(dst, dst, nbytes, peer, qp_row, wf_info);
      }
    }
  }
  internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, proxy_epoch_base + 1, proxy_qp_row);

  if (constmem.my_pe != local_proxy && constmem.my_pe != pe_root) {
    internal_proxy_getmem_wg(dst, dst, nbytes, local_proxy, 0, wf_info);
  }
  internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, proxy_epoch_base + 2, proxy_qp_row);
  return true;
}

/**
 * @brief Broadcast through one in-team same-NIC proxy per IPC domain.
 *
 * @tparam T Element type.
 * @param team_obj Partial-team route state.
 * @param dst Destination and proxy staging buffer.
 * @param src Root source buffer.
 * @param nelems Number of elements.
 * @param pe_root Root world PE.
 * @return true when the team has a usable in-team route.
 */
template <typename T>
__device__ bool GDAContext::internal_routed_team_proxy_broadcast(GDATeam *team_obj, T *dst, const T *src,
    int nelems, int pe_root, long *p_sync, ActiveWFInfo &wf_info, int64_t proxy_epoch_base,
    int proxy_qp_row) {
  const auto *team_info = team_obj->tinfo_wrt_world;
  if (team_obj->in_team_proxy_ipc_domain_count <= 1 || (team_info->pe_start == 0 && team_info->stride == 1 &&
      team_obj->num_pes == constmem.num_pes)) {
    return false;
  }

  int local_proxy = -1;
  int root_proxy = -1;
  for (int ipc_domain = 0; ipc_domain < team_obj->in_team_proxy_ipc_domain_count; ipc_domain++) {
    const int first = team_obj->in_team_proxy_ipc_domain_first_pes[ipc_domain];
    const int end = ipc_domain + 1 < team_obj->in_team_proxy_ipc_domain_count
                        ? team_obj->in_team_proxy_ipc_domain_first_pes[ipc_domain + 1]
                        : constmem.num_pes;
    if (first == constmem.ipc_first_pe) {
      local_proxy = team_obj->in_team_proxy_pes[ipc_domain];
    }
    if (pe_root >= first && pe_root < end) {
      root_proxy = team_obj->in_team_proxy_pes[ipc_domain];
    }
  }
  assert(local_proxy >= 0 && root_proxy >= 0);

  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<T *>(src), nbytes);
  }
  // The root must publish its tile before its IPC-domain proxy can read it over IPC.
  // Combining these two steps used to race and forward the previous root's tile.
  __threadfence_system();
  internal_team_proxy_sync_wg(team_obj, p_sync, wf_info, proxy_epoch_base);

  if (constmem.my_pe == root_proxy && root_proxy != pe_root) {
    internal_proxy_getmem_wg(dst, src, nbytes, pe_root, 0, wf_info);
    __syncthreads();
    // IPC Get stores into local memory; republish the tile with the Put path
    // before the NIC reads this registered proxy buffer.
    memcpy_wg<MemcpyKind::Put>(dst, dst, nbytes);
  }
  __threadfence_system();
  internal_team_proxy_sync_wg(team_obj, p_sync, wf_info, proxy_epoch_base + 1);

  if (constmem.my_pe == root_proxy) {
    const int qp_row = proxy_qp_row % num_qps_per_pe;
    for (int ipc_domain = 0; ipc_domain < team_obj->in_team_proxy_ipc_domain_count; ipc_domain++) {
      const int peer = team_obj->in_team_proxy_pes[ipc_domain];
      if (peer != root_proxy) {
        internal_proxy_putmem_wg(dst, dst, nbytes, peer, qp_row, wf_info);
      }
    }
  }
  internal_team_proxy_sync_wg(team_obj, p_sync, wf_info, proxy_epoch_base + 2);

  if (constmem.my_pe != local_proxy && constmem.my_pe != pe_root) {
    internal_proxy_getmem_wg(dst, dst, nbytes, local_proxy, 0, wf_info);
  }
  internal_team_proxy_sync_wg(team_obj, p_sync, wf_info, proxy_epoch_base + 3);
  return true;
}

/**
 * @brief Split a WORLD broadcast across routes selected by matching NIC names.
 * Proxy PEs may occupy different local indices in the two IPC domains.
 *
 * @tparam T Element type.
 * @param dst Destination and proxy staging buffer.
 * @param src Root source buffer.
 * @param nelems Number of elements.
 * @param pe_root Root world PE.
 * @return true when striped WORLD routing is available.
 */
template <typename T>
__device__ bool GDAContext::internal_routed_multi_proxy_broadcast(T *dst, const T *src, int nelems,
    int pe_root, int pe_start, int stride, int pe_size, long *p_sync, ActiveWFInfo &wf_info,
    int64_t proxy_epoch_base, int proxy_qp_row) {
  if (!constmem.world_proxy_nic_layout_available || nelems <= 0 ||
      proxy::IsAlignedWorldProxyDataEligible(pe_start, stride,
      pe_size) || !proxy::IsWorldProxySyncEligible(constmem.my_pe, pe_start, stride, pe_size)) {
    return false;
  }

  int root_ipc_domain_first_pe = -1;
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    const int first = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
    if (pe_root >= first && pe_root < proxy::WorldProxyIpcDomainEnd(first)) {
      root_ipc_domain_first_pe = first;
    }
  }
  if (root_ipc_domain_first_pe < 0) {
    return false;
  }

  constexpr size_t kMinBytesPerLane = 64 * 1024;
  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);
  // Epochs start at one; each WORLD broadcast reserves one set of sync phases.
  const int64_t lane_rotation_index = (proxy_epoch_base - 1) / proxy::kWorldBroadcastSyncPhases;
  char *dst_bytes = reinterpret_cast<char *>(dst);
  const char *src_bytes = reinterpret_cast<const char *>(src);

  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst_bytes, const_cast<char *>(src_bytes), nbytes);
  }

  // Stage only the segments assigned to this source-side proxy. The source
  // address remains symmetric; root's entire destination is already staged.
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    const int target_ipc_domain_first_pe = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
    if (target_ipc_domain_first_pe == root_ipc_domain_first_pe) {
      continue;
    }
    const auto plan = proxy::PlanWorldProxyTransfer(root_ipc_domain_first_pe, target_ipc_domain_first_pe, nbytes, kMinBytesPerLane);
    for (int lane = 0; lane < plan.active_lanes; lane++) {
      const auto route = proxy::SelectWorldProxyLane(root_ipc_domain_first_pe, target_ipc_domain_first_pe, lane, lane_rotation_index);
      if (constmem.my_pe != route.source_pe || pe_root == route.source_pe) {
        continue;
      }
      const size_t begin = plan.begin(lane);
      const size_t bytes = plan.bytes(lane, nbytes);
      internal_proxy_getmem_wg(dst_bytes + begin, src_bytes + begin, bytes, pe_root, 0, wf_info);
    }
  }
  internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, proxy_epoch_base, proxy_qp_row);

  // Each source proxy pushes its assigned segment through the same-NIC
  // route selected above. Target proxies do not read from other IPC domains.
  if (constmem.ipc_first_pe == root_ipc_domain_first_pe) {
    for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
      const int target_ipc_domain_first_pe = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
      if (target_ipc_domain_first_pe == root_ipc_domain_first_pe) {
        continue;
      }
      const auto plan = proxy::PlanWorldProxyTransfer(root_ipc_domain_first_pe, target_ipc_domain_first_pe, nbytes, kMinBytesPerLane);
      for (int lane = 0; lane < plan.active_lanes; lane++) {
        const auto route = proxy::SelectWorldProxyLane(root_ipc_domain_first_pe, target_ipc_domain_first_pe, lane, lane_rotation_index);
        if (constmem.my_pe != route.source_pe) {
          continue;
        }
        const size_t begin = plan.begin(lane);
        const size_t bytes = plan.bytes(lane, nbytes);
        const uint32_t qp_row = proxy_qp_row >= 0
            ? static_cast<uint32_t>(proxy_qp_row) % num_qps_per_pe
            : static_cast<uint32_t>(blockIdx.x) % num_qps_per_pe;
        internal_proxy_putmem_wg(dst_bytes + begin, dst_bytes + begin, bytes, route.target_pe, qp_row, wf_info);
      }
    }
  }
  internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, proxy_epoch_base + 1, proxy_qp_row);

  // Local consumers assemble the complete result over IPC. No extra network
  // traffic is generated after the target proxies finish their RDMA reads.
  if (constmem.ipc_first_pe == root_ipc_domain_first_pe) {
    if (constmem.my_pe != pe_root) {
      internal_proxy_getmem_wg(dst_bytes, dst_bytes, nbytes, pe_root, 0, wf_info);
    }
  } else {
    const int target_ipc_domain_first_pe = constmem.ipc_first_pe;
    const auto plan = proxy::PlanWorldProxyTransfer(root_ipc_domain_first_pe, target_ipc_domain_first_pe, nbytes, kMinBytesPerLane);
    for (int lane = 0; lane < plan.active_lanes; lane++) {
      const auto route = proxy::SelectWorldProxyLane(root_ipc_domain_first_pe, target_ipc_domain_first_pe, lane, lane_rotation_index);
      if (constmem.my_pe == route.target_pe) {
        continue;
      }
      const size_t begin = plan.begin(lane);
      const size_t bytes = plan.bytes(lane, nbytes);
      internal_proxy_getmem_wg(dst_bytes + begin, dst_bytes + begin, bytes, route.target_pe, 0, wf_info);
    }
  }
  internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, proxy_epoch_base + 2, proxy_qp_row);
  return true;
}

// ############################################################################
// ALL-TO-ALL
// ############################################################################

namespace proxy {

/**
 * @brief Check whether every target has a proxy and the scratch fits.
 *
 * Direct means target-selected proxy routing, not bypassing the proxy.
 * @param elem_bytes Size of one element in bytes.
 * @return true when the direct WORLD proxy all-to-all path is usable.
 */
__device__ __forceinline__ bool CanUseDirectWorldProxyAlltoall(size_t elem_bytes) {
  if (!constmem.world_proxy_nic_layout_available) {
    return false;
  }
  int max_local_pe_count = 0;
  int max_batches = 0;
  for (int source = 0; source < constmem.world_proxy_ipc_domain_count; source++) {
    const int source_ipc_domain_first_pe = constmem.world_proxy_ipc_domain_first_pes[source];
    max_local_pe_count = max(max_local_pe_count, WorldProxyIpcDomainEnd(source_ipc_domain_first_pe) - source_ipc_domain_first_pe);
    for (int target = 0; target < constmem.world_proxy_ipc_domain_count; target++) {
      if (source == target) {
        continue;
      }
      const int target_ipc_domain_first_pe = constmem.world_proxy_ipc_domain_first_pes[target];
      const int target_end = WorldProxyIpcDomainEnd(target_ipc_domain_first_pe);
      for (int pe = target_ipc_domain_first_pe; pe < target_end; pe++) {
        const int batch = SelectWorldAlltoallTargetAssignment(source_ipc_domain_first_pe, target_ipc_domain_first_pe, pe).scratch_batch;
        if (batch < 0) {
          return false;
        }
        max_batches = max(max_batches, batch + 1);
      }
    }
  }
  const size_t scratch_elements = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / elem_bytes;
  return max_batches > 0 && max_local_pe_count <= scratch_elements / max_batches;
}

/** @brief Return scratch batches required by a target IPC domain. */
__device__ __forceinline__ int WorldAlltoallScratchBatchCount(int target_ipc_domain_first_pe) {
  int batches = 0;
  const int target_end = WorldProxyIpcDomainEnd(target_ipc_domain_first_pe);
  for (int source = 0; source < constmem.world_proxy_ipc_domain_count; source++) {
    const int source_ipc_domain_first_pe = constmem.world_proxy_ipc_domain_first_pes[source];
    if (source_ipc_domain_first_pe == target_ipc_domain_first_pe) {
      continue;
    }
    for (int pe = target_ipc_domain_first_pe; pe < target_end; pe++) {
      const auto assignment = SelectWorldAlltoallTargetAssignment(source_ipc_domain_first_pe, target_ipc_domain_first_pe, pe);
      batches = max(batches, assignment.scratch_batch + 1);
    }
  }
  return batches;
}

}  // namespace proxy

/**
 * @brief Try the eligible PXN all-to-all algorithm before the original path.
 *
 * @param team_obj Team and proxy layout metadata.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param nelems Elements per peer.
 * @param elem_offset First element in this tile.
 * @param elem_count Number of elements in this tile.
 * @param wf_info Active workgroup metadata.
 * @return true when a PXN path completed this tile.
 */
template <typename T>
__device__ bool GDAContext::internal_try_pxn_proxy_alltoall(GDATeam *team_obj, T *dst, const T *src,
    int nelems, int elem_offset, int elem_count, ActiveWFInfo &wf_info) {
  if (elem_count <= 0) {
    return false;
  }
  const auto *team_info = team_obj->tinfo_wrt_world;
  const auto layout = proxy::SelectProxyCollectiveLayout(team_obj, team_info->pe_start, team_info->stride,
      team_obj->num_pes);
  if (layout == proxy::ProxyCollectiveLayout::TeamProxyRoute) {
    internal_routed_team_proxy_alltoall(team_obj, dst, src, nelems, elem_offset, elem_count);
    return true;
  }
  if (layout == proxy::ProxyCollectiveLayout::WorldAlignedData) {
    internal_aligned_world_proxy_alltoall(team_obj, dst, src, nelems, elem_offset, elem_count, wf_info);
    return true;
  }
  if (layout == proxy::ProxyCollectiveLayout::WorldSyncOnly) {
    if (proxy::CanUseDirectWorldProxyAlltoall(sizeof(T))) {
      internal_routed_world_proxy_alltoall(team_obj, dst, src, nelems, elem_offset, elem_count, wf_info);
    } else {
      internal_broadcast_world_proxy_alltoall(team_obj, dst, src, nelems, elem_offset, elem_count, wf_info);
    }
    return true;
  }
  return false;
}

/**
 * @brief Run aligned WORLD all-to-all through corresponding same-NIC PEs.
 *
 * @tparam T Element type.
 * @param team_obj WORLD team state and scratch storage.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param nelems Elements per peer.
 * @param elem_offset First element processed by this invocation.
 * @param elem_count Number of elements processed by this invocation.
 * @param wf_info Active workgroup metadata.
 */
template <typename T>
__device__ void GDAContext::internal_aligned_world_proxy_alltoall(GDATeam *team_obj, T *dst, const T *src,
    int nelems, int elem_offset, int elem_count, ActiveWFInfo &wf_info) {
  const int local_pe_count = constmem.ipc_shm_size;
  const int ipc_domain_first_pe = constmem.ipc_first_pe;
  const int local_slot = ipcImpl_.shm_rank;
  const int my_pe = constmem.my_pe;
  if (is_thread_zero_in_block()) {
    // This PE owns the QP used for each send between matching slots in different IPC domains.
    // A PE with multiple NICs needs a QP-row map before we can name its NIC.
    if (constmem.world_proxy_nic_layout_available) {
      LOGD_INFO("PXN alltoall data path: aligned WORLD, proxy PE %d, NIC id %d", my_pe,
          constmem.world_proxy_pe_nic_name_ids[my_pe]);
    } else {
      LOGD_INFO("PXN alltoall data path: aligned WORLD, proxy PE %d, NIC depends on QP row", my_pe);
    }
  }
  T *scratch = reinterpret_cast<T *>(team_obj->pWrk);
  long *p_sync = team_obj->alltoall_pSync;
  const size_t scratch_bytes = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double);
  const size_t capacity = scratch_bytes / (local_pe_count * sizeof(T));
  const int tile_capacity = static_cast<int>(capacity < static_cast<size_t>(elem_count) ? capacity : elem_count);
  int64_t sync_epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch + 1);

  // The local destination writes never touch the network.
  for (int slot = 0; slot < local_pe_count; slot++) {
    const int target_pe = ipc_domain_first_pe + slot;
    T *target = dst + my_pe * nelems + elem_offset;
    const T *source = src + target_pe * nelems + elem_offset;
    const size_t bytes = static_cast<size_t>(elem_count) * sizeof(T);
    if (target_pe == my_pe) {
      memcpy_wg<MemcpyKind::Put>(target, const_cast<T *>(source), bytes);
    } else {
      internal_proxy_putmem_wg(target, source, bytes, target_pe, 0, wf_info);
    }
  }

  for (int target_ipc_domain_first_pe = 0; target_ipc_domain_first_pe < constmem.num_pes; target_ipc_domain_first_pe += local_pe_count) {
    if (target_ipc_domain_first_pe == ipc_domain_first_pe) {
      continue;
    }
    for (int offset = elem_offset; offset < elem_offset + elem_count; offset += tile_capacity) {
      const int count = min(tile_capacity, elem_offset + elem_count - offset);
      const size_t bytes = static_cast<size_t>(count) * sizeof(T);

      // Each source writes a distinct scratch slot on every local NIC PE.
      for (int target_slot = 0; target_slot < local_pe_count; target_slot++) {
        const int proxy_pe = ipc_domain_first_pe + target_slot;
        T *stage = scratch + local_slot * tile_capacity;
        const T *source = src + (target_ipc_domain_first_pe + target_slot) * nelems + offset;
        if (proxy_pe == my_pe) {
          memcpy_wg<MemcpyKind::Put>(stage, const_cast<T *>(source), bytes);
        } else {
          internal_proxy_putmem_wg(stage, source, bytes, proxy_pe, 0, wf_info);
        }
      }
      __threadfence_system();
      internal_world_proxy_sync_wg(my_pe, constmem.num_pes, p_sync, wf_info, sync_epoch++);

      // The matching-NIC PE sends all local sources to its remote counterpart.
      const int target_pe = target_ipc_domain_first_pe + local_slot;
      for (int source_slot = 0; source_slot < local_pe_count; source_slot++) {
        const int source_pe = ipc_domain_first_pe + source_slot;
        T *target = dst + source_pe * nelems + offset;
        const T *stage = scratch + source_slot * tile_capacity;
        internal_proxy_putmem_wg(target, stage, bytes, target_pe, 0, wf_info);
      }
      internal_world_proxy_sync_wg(my_pe, constmem.num_pes, p_sync, wf_info, sync_epoch++);
    }
  }

  if (is_thread_zero_in_block()) {
    team_obj->proxy_alltoall_sync_epoch = sync_epoch - 1;
  }
  __syncthreads();
}

/**
 * @brief Broadcast each partial-team source row, retaining the local column.
 *
 * @tparam T Element type.
 * @param team_obj Partial-team routing and scratch state.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param nelems Elements per peer.
 * @param elem_offset First element to transfer.
 * @param elem_count Number of elements to transfer.
 */
template <typename T>
__device__ void GDAContext::internal_routed_team_proxy_alltoall(GDATeam *team_obj, T *dst, const T *src,
    int nelems, int elem_offset, int elem_count) {
  T *scratch = reinterpret_cast<T *>(team_obj->pWrk);
  long *p_sync = team_obj->alltoall_pSync;
  const int team_size = team_obj->num_pes;
  const int tile_capacity = static_cast<int>(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T));
  const int tid = get_flat_block_id();
  const int block_size = get_flat_block_size();
  const int qp_row = team_obj->pool_index_ % num_qps_per_pe;
  const int broadcast_phases = proxy::TeamBroadcastPhaseCount(team_obj);
  int64_t epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch);
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  const size_t row_elements = static_cast<size_t>(team_size) * nelems;
  const size_t column_begin = static_cast<size_t>(team_obj->my_pe) * nelems;
  const size_t wanted_begin = column_begin + elem_offset;
  const size_t wanted_end = wanted_begin + elem_count;

  for (int source_rank = 0; source_rank < team_size; source_rank++) {
    const int source_pe = team_obj->get_pe_in_world(source_rank);
    // All members broadcast the same contiguous row tiles. Each receiver keeps
    // only its requested column interval, including workgroup subranges.
    // This replaces one tiny broadcast per target with one stream per source.
    for (size_t offset = 0; offset < row_elements; offset += tile_capacity) {
      const int count = static_cast<int>(min(static_cast<size_t>(tile_capacity), row_elements - offset));
      internal_broadcast(scratch, src + offset, count, source_pe,
          team_obj->tinfo_wrt_world->pe_start, team_obj->tinfo_wrt_world->stride, team_size, p_sync, epoch,
          qp_row, team_obj);
      const size_t begin = max(offset, wanted_begin);
      const size_t end = min(offset + count, wanted_end);
      for (size_t i = begin + tid; i < end; i += block_size) {
        dst[static_cast<size_t>(source_rank) * nelems + i - column_begin] = uncached_load(scratch + i - offset);
      }
      __syncthreads();
      // No PE may overwrite the receive tile before every consumer finishes.
      internal_team_proxy_sync_wg(team_obj, p_sync, wf_info, epoch + broadcast_phases);
      epoch += broadcast_phases + 1;
    }
  }
  if (is_thread_zero_in_block()) {
    team_obj->proxy_alltoall_sync_epoch = epoch;
  }
  __syncthreads();
}

// Every PE must agree before a locally regular layout can use alltoall.
__device__ inline bool GDAContext::internal_world_proxy_alltoallv_check_uniform_layout(GDATeam *team_obj,
    const size_t dest_nelems[], const size_t dest_displs[], const size_t source_nelems[],
    const size_t source_displs[], ActiveWFInfo &wf_info, int64_t &epoch, size_t &common_count) {
  const int pe_size = constmem.num_pes;
  const int proxy_qp_row = team_obj->pool_index_ % num_qps_per_pe;
  // Select the regular fast path collectively. A locally regular PE may
  // belong to a globally asymmetric exchange, so local checks alone are unsafe.
  size_t *control = reinterpret_cast<size_t *>(team_obj->pWrk);
  const size_t local_count = source_nelems[0];
  bool local_regular = local_count <= 0x7fffffff;
  for (int peer = 0; peer < pe_size && local_regular; peer++) {
    local_regular = source_nelems[peer] == local_count && dest_nelems[peer] == local_count &&
        source_displs[peer] == static_cast<size_t>(peer) * local_count &&
        dest_displs[peer] == static_cast<size_t>(peer) * local_count;
  }
  if (is_thread_zero_in_block()) {
    control[0] = local_regular;
    control[1] = local_count;
  }
  __threadfence_system();
  __syncthreads();
  internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
      wf_info, epoch++, proxy_qp_row);

  // Aggregate over local IPC first; only one descriptor per IPC domain crosses
  // the network. Slots 0/2/4/6 hold local, fetched, aggregate and received pairs.
  bool regular = true;
  common_count = 0;
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    const int first = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
    if (constmem.my_pe == first) {
      if (is_thread_zero_in_block()) {
        control[4] = control[0];
        control[5] = control[1];
      }
      __syncthreads();
      for (int peer = first + 1; peer < proxy::WorldProxyIpcDomainEnd(first); peer++) {
        internal_proxy_getmem_wg(control + 2, control, 2 * sizeof(size_t), peer,
            proxy_qp_row, wf_info);
        if (is_thread_zero_in_block()) {
          control[4] = control[4] && uncached_load(control + 2) &&
              control[5] == uncached_load(control + 3);
        }
        __syncthreads();
      }
    }
    __threadfence_system();
    __syncthreads();
    // Root-side proxies must not fetch the descriptor before aggregation ends.
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
        wf_info, epoch++, proxy_qp_row);
    internal_broadcast(control + 6, control + 4, 2, first, 0, 1, pe_size,
        team_obj->alltoall_pSync, epoch, proxy_qp_row, team_obj);
    epoch += proxy::kWorldBroadcastSyncPhases;
    const size_t ipc_domain_regular = __hip_atomic_load(control + 6, __ATOMIC_ACQUIRE,
        __HIP_MEMORY_SCOPE_SYSTEM);
    const size_t ipc_domain_element_count = __hip_atomic_load(control + 7, __ATOMIC_ACQUIRE,
        __HIP_MEMORY_SCOPE_SYSTEM);
    if (ipc_domain == 0) {
      common_count = ipc_domain_element_count;
    }
    regular = regular && ipc_domain_regular && ipc_domain_element_count == common_count;
    // Every lane must consume this descriptor before its receive slot is reused.
    __syncthreads();
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
        wf_info, epoch++, proxy_qp_row);
  }
  return regular;
}

// Broadcast each packed source row; keep its count table outside the payload tile.
template <typename T>
__device__ bool GDAContext::internal_world_proxy_alltoallv_packed_rows(GDATeam *team_obj, T *dest,
    const size_t dest_nelems[], const size_t dest_displs[], T *source, const size_t source_nelems[],
    const size_t source_displs[], int wg_id, int wg_count, ActiveWFInfo &wf_info, int64_t &epoch) {
  char *workspace = reinterpret_cast<char *>(team_obj->pWrk);
  constexpr size_t kWorkspaceBytes = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double);
  const size_t half_bytes = kWorkspaceBytes / 2;
  T *stage = reinterpret_cast<T *>(workspace);
  T *received = reinterpret_cast<T *>(workspace + half_bytes);
  size_t *stage_count = reinterpret_cast<size_t *>(stage);
  size_t *received_count = reinterpret_cast<size_t *>(received);
  const int pe_size = constmem.num_pes;
  const int proxy_qp_row = team_obj->pool_index_ % num_qps_per_pe;
  const size_t metadata_bytes = static_cast<size_t>(pe_size) * sizeof(size_t);
  if (metadata_bytes >= half_bytes || (half_bytes - metadata_bytes) / sizeof(T) == 0) {
    return false;
  }
  size_t *row_counts = reinterpret_cast<size_t *>(workspace + kWorkspaceBytes - metadata_bytes);
  const size_t row_capacity = (half_bytes - metadata_bytes) / sizeof(T);
  for (int source_pe = 0; source_pe < pe_size; source_pe++) {
    if (constmem.my_pe == source_pe) {
      for (int peer = get_flat_block_id(); peer < pe_size; peer += get_flat_block_size()) {
        const size_t total = source_nelems[peer];
        stage_count[peer] = total / wg_count + (static_cast<size_t>(wg_id) < total % wg_count);
      }
    }
    __threadfence_system();
    __syncthreads();
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
        wf_info, epoch++, proxy_qp_row);
    internal_broadcast(received_count, stage_count, pe_size, source_pe, 0, 1, pe_size,
        team_obj->alltoall_pSync, epoch, proxy_qp_row, team_obj);
    epoch += proxy::kWorldBroadcastSyncPhases;
    for (int peer = get_flat_block_id(); peer < pe_size; peer += get_flat_block_size()) {
      row_counts[peer] = __hip_atomic_load(received_count + peer, __ATOMIC_ACQUIRE,
          __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
        wf_info, epoch++, proxy_qp_row);
    size_t row_elements = 0;
    size_t wanted_begin = 0;
    for (int peer = 0; peer < pe_size; peer++) {
      if (peer < constmem.my_pe) {
        wanted_begin += row_counts[peer];
      }
      row_elements += row_counts[peer];
    }
    const size_t total = dest_nelems[source_pe];
    const size_t local_elements = total / wg_count + (static_cast<size_t>(wg_id) < total % wg_count);
    const size_t wanted_elements = min(row_counts[constmem.my_pe], local_elements);
    const size_t local_begin = (total / wg_count) * wg_id + min(static_cast<size_t>(wg_id), total % wg_count);
    for (size_t offset = 0; offset < row_elements; offset += row_capacity) {
      const size_t tile = min(row_capacity, row_elements - offset);
      if (constmem.my_pe == source_pe) {
        size_t peer_begin = 0;
        for (int peer = 0; peer < pe_size; peer++) {
          const size_t begin = max(offset, peer_begin);
          const size_t end = min(offset + tile, peer_begin + row_counts[peer]);
          if (begin < end) {
            const size_t send_total = source_nelems[peer];
            const size_t send_begin = (send_total / wg_count) * wg_id +
                min(static_cast<size_t>(wg_id), send_total % wg_count);
            memcpy_wg<MemcpyKind::Put>(stage + begin - offset,
                source + source_displs[peer] + send_begin + begin - peer_begin,
                (end - begin) * sizeof(T));
          }
          peer_begin += row_counts[peer];
        }
      }
      __threadfence_system();
      __syncthreads();
      internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
          wf_info, epoch++, proxy_qp_row);
      internal_broadcast(received, stage, static_cast<int>(tile), source_pe, 0, 1, pe_size,
          team_obj->alltoall_pSync, epoch, proxy_qp_row, team_obj);
      epoch += proxy::kWorldBroadcastSyncPhases;
      const size_t begin = max(offset, wanted_begin);
      const size_t end = min(offset + tile, wanted_begin + wanted_elements);
      if (begin < end) {
        for (size_t i = get_flat_block_id(); i < end - begin; i += get_flat_block_size()) {
          dest[dest_displs[source_pe] + local_begin + begin - wanted_begin + i] =
              uncached_load(received + begin - offset + i);
        }
      }
      __syncthreads();
      internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
          wf_info, epoch++, proxy_qp_row);
    }
  }
  return true;
}

// Fall back to one PE pair at a time only when the row count table cannot fit.
template <typename T>
__device__ void GDAContext::internal_world_proxy_alltoallv_pairwise(GDATeam *team_obj, T *dest,
    const size_t dest_nelems[], const size_t dest_displs[], T *source, const size_t source_nelems[],
    const size_t source_displs[], int wg_id, int wg_count, ActiveWFInfo &wf_info, int64_t &epoch) {
  char *workspace = reinterpret_cast<char *>(team_obj->pWrk);
  constexpr size_t kWorkspaceBytes = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double);
  const size_t half_bytes = kWorkspaceBytes / 2;
  T *stage = reinterpret_cast<T *>(workspace);
  T *received = reinterpret_cast<T *>(workspace + half_bytes);
  size_t *stage_count = reinterpret_cast<size_t *>(stage);
  size_t *received_count = reinterpret_cast<size_t *>(received);
  const int pe_size = constmem.num_pes;
  const int proxy_qp_row = team_obj->pool_index_ % num_qps_per_pe;
  const size_t capacity = half_bytes / sizeof(T);
  __shared__ volatile size_t transfer_count;
  for (int source_pe = 0; source_pe < pe_size; source_pe++) {
    for (int target_pe = 0; target_pe < pe_size; target_pe++) {
      if (constmem.my_pe == source_pe && is_thread_zero_in_block()) {
        // Each on-stream workgroup owns a disjoint part of every PE pair.
        const size_t total = source_nelems[target_pe];
        *stage_count = total / wg_count + (static_cast<size_t>(wg_id) < total % wg_count);
      }
      __threadfence_system();
      __syncthreads();
      internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
          wf_info, epoch++, proxy_qp_row);
      internal_broadcast(received_count, stage_count, 1, source_pe,
                         0, 1, pe_size, team_obj->alltoall_pSync, epoch,
                         proxy_qp_row, team_obj);
      epoch += proxy::kWorldBroadcastSyncPhases;
      // Read NIC-written control once with system acquire semantics, then
      // publish one immutable loop bound to the entire workgroup. The receive
      // tile is reused for payloads below, so it must not remain the loop bound.
      if (is_thread_zero_in_block()) {
        transfer_count = __hip_atomic_load(received_count, __ATOMIC_ACQUIRE,
            __HIP_MEMORY_SCOPE_SYSTEM);
      }
      __syncthreads();
      internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
          wf_info, epoch++, proxy_qp_row);
      for (size_t offset = 0; offset < transfer_count; offset += capacity) {
        const size_t tile = min(capacity, static_cast<size_t>(transfer_count) - offset);
        if (constmem.my_pe == source_pe) {
          memcpy_wg<MemcpyKind::Put>(
              stage, source + source_displs[target_pe] +
                  (source_nelems[target_pe] / wg_count) * wg_id +
                  min(static_cast<size_t>(wg_id), source_nelems[target_pe] % wg_count) + offset,
              tile * sizeof(T));
        }
        // Root-IPC-domain proxies may fetch stage before broadcast's first barrier.
        // Publish every lane's stores before allowing those IPC readers in.
        __threadfence_system();
        __syncthreads();
        internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
            wf_info, epoch++, proxy_qp_row);
        internal_broadcast(received, stage, static_cast<int>(tile), source_pe,
                           0, 1, pe_size, team_obj->alltoall_pSync, epoch,
                           proxy_qp_row, team_obj);
        epoch += proxy::kWorldBroadcastSyncPhases;
        if (constmem.my_pe == target_pe) {
          const size_t total = dest_nelems[source_pe];
          const size_t count = total / wg_count + (static_cast<size_t>(wg_id) < total % wg_count);
          const size_t begin = (total / wg_count) * wg_id + min(static_cast<size_t>(wg_id), total % wg_count);
          const size_t valid = min(tile, count > offset ? count - offset : size_t{0});
          for (size_t i = get_flat_block_id(); i < valid; i += get_flat_block_size()) {
            dest[dest_displs[source_pe] + begin + offset + i] = uncached_load(received + i);
          }
        }
        __syncthreads();
        // Do not reuse the receive tile until every destination has consumed it.
        internal_world_proxy_sync_wg(constmem.my_pe, pe_size, team_obj->alltoall_pSync,
            wf_info, epoch++, proxy_qp_row);
      }
    }
  }
}

/**
 * @brief Select regular, packed-row or pairwise WORLD all-to-allv proxy transfer.
 *
 * @tparam T Element type.
 * @param team_obj WORLD team routing and scratch state.
 * @param dest Destination buffer and per-source metadata.
 * @param dest_nelems Destination element counts.
 * @param dest_displs Destination element displacements.
 * @param source Source buffer and per-target metadata.
 * @param source_nelems Source element counts.
 * @param source_displs Source element displacements.
 */
template <typename T>
__device__ void GDAContext::internal_world_proxy_alltoallv(GDATeam *team_obj, T *dest,
    const size_t dest_nelems[], const size_t dest_displs[], T *source, const size_t source_nelems[],
    const size_t source_displs[], int wg_id, int wg_count) {
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  int64_t epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch);
  size_t common_count = 0;
  const bool regular = internal_world_proxy_alltoallv_check_uniform_layout(team_obj, dest_nelems, dest_displs,
      source_nelems, source_displs, wf_info, epoch, common_count);
  if (is_thread_zero_in_block()) {
    team_obj->proxy_alltoall_sync_epoch = epoch;
  }
  __syncthreads();
  if (regular) {
    size_t begin = 0;
    size_t count = 0;
    split_wg_range(common_count, wg_id, wg_count, &begin, &count);
    alltoall_wg_das(reinterpret_cast<rocshmem_team_t>(team_obj), dest, source,
        static_cast<int>(common_count), static_cast<int>(begin), static_cast<int>(count));
    return;
  }
  if (!internal_world_proxy_alltoallv_packed_rows(team_obj, dest, dest_nelems, dest_displs,
          source, source_nelems, source_displs, wg_id, wg_count, wf_info, epoch)) {
    internal_world_proxy_alltoallv_pairwise(team_obj, dest, dest_nelems, dest_displs,
        source, source_nelems, source_displs, wg_id, wg_count, wf_info, epoch);
  }
  if (is_thread_zero_in_block()) {
    team_obj->proxy_alltoall_sync_epoch = epoch;
  }
  __syncthreads();
}

/**
 * @brief Run routed WORLD all-to-all for unequal IPC domain sizes.
 *
 * @tparam T Element type.
 * @param team_obj WORLD team routing and scratch state.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param nelems Elements per peer.
 * @param elem_offset First element to transfer.
 * @param elem_count Number of elements to transfer.
 * @param wf_info Active workgroup metadata.
 */
template <typename T>
__device__ void GDAContext::internal_routed_world_proxy_alltoall(GDATeam *team_obj, T *dst, const T *src,
    int nelems, int elem_offset, int elem_count, ActiveWFInfo &wf_info) {
  const int my_pe = constmem.my_pe;
  const int source_ipc_domain_first_pe = constmem.ipc_first_pe;
  const int source_size = constmem.ipc_shm_size;
  T *scratch = reinterpret_cast<T *>(team_obj->pWrk);
  long *p_sync = team_obj->alltoall_pSync;
  int max_local_pe_count = 0;
  int max_batches = 0;
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    const int first = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
    max_local_pe_count = max(max_local_pe_count, proxy::WorldProxyIpcDomainEnd(first) - first);
    max_batches = max(max_batches, proxy::WorldAlltoallScratchBatchCount(first));
  }
  const size_t scratch_bytes = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double);
  const int tile_capacity = static_cast<int>(scratch_bytes / (max_local_pe_count * max_batches * sizeof(T)));
  int64_t sync_epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch + 1);

  // Within an IPC domain, the source writes directly to each local target.
  for (int slot = 0; slot < source_size; slot++) {
    const int target_pe = source_ipc_domain_first_pe + slot;
    T *target = dst + my_pe * nelems + elem_offset;
    const T *source = src + target_pe * nelems + elem_offset;
    const size_t bytes = static_cast<size_t>(elem_count) * sizeof(T);
    if (target_pe == my_pe) {
      memcpy_wg<MemcpyKind::Put>(target, const_cast<T *>(source), bytes);
    } else {
      internal_proxy_putmem_wg(target, source, bytes, target_pe, 0, wf_info);
    }
  }

  // Every PE follows the same target-domain, batch and tile order, even when
  // its own IPC domain is the target. This keeps world proxy sync collective.
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    const int target_ipc_domain_first_pe = constmem.world_proxy_ipc_domain_first_pes[ipc_domain];
    const int target_end = proxy::WorldProxyIpcDomainEnd(target_ipc_domain_first_pe);
    const int batches = proxy::WorldAlltoallScratchBatchCount(target_ipc_domain_first_pe);
    for (int batch = 0; batch < batches; batch++) {
      for (int offset = elem_offset; offset < elem_offset + elem_count; offset += tile_capacity) {
        const int count = min(tile_capacity, elem_offset + elem_count - offset);
        const size_t bytes = static_cast<size_t>(count) * sizeof(T);

        if (source_ipc_domain_first_pe != target_ipc_domain_first_pe) {
          for (int target_pe = target_ipc_domain_first_pe; target_pe < target_end; target_pe++) {
            const auto assignment = proxy::SelectWorldAlltoallTargetAssignment(source_ipc_domain_first_pe, target_ipc_domain_first_pe,
                target_pe);
            if (assignment.scratch_batch != batch || assignment.source_proxy_pe != my_pe) {
              continue;
            }
            for (int slot = 0; slot < source_size; slot++) {
              const int source_pe = source_ipc_domain_first_pe + slot;
              T *stage = scratch +
                  (batch * max_local_pe_count + slot) * tile_capacity;
              const T *source = src + target_pe * nelems + offset;
              if (source_pe == my_pe) {
                memcpy_wg<MemcpyKind::Put>(stage, const_cast<T *>(source), bytes);
              } else {
                internal_proxy_getmem_wg(stage, source, bytes, source_pe, 0, wf_info);
              }
            }
          }
        }
        __threadfence_system();
        internal_world_proxy_sync_wg(my_pe, constmem.num_pes, p_sync, wf_info, sync_epoch++);

        if (my_pe >= target_ipc_domain_first_pe && my_pe < target_end) {
          for (int source_ipc_domain_index = 0; source_ipc_domain_index < constmem.world_proxy_ipc_domain_count; source_ipc_domain_index++) {
            const int remote_first = constmem.world_proxy_ipc_domain_first_pes[source_ipc_domain_index];
            if (remote_first == target_ipc_domain_first_pe) {
              continue;
            }
            const auto assignment = proxy::SelectWorldAlltoallTargetAssignment(remote_first, target_ipc_domain_first_pe, my_pe);
            if (assignment.scratch_batch != batch) {
              continue;
            }
            const int remote_size = proxy::WorldProxyIpcDomainEnd(remote_first) - remote_first;
            for (int slot = 0; slot < remote_size; slot++) {
              const int source_pe = remote_first + slot;
              T *target = dst + source_pe * nelems + offset;
              const T *stage = scratch +
                  (batch * max_local_pe_count + slot) * tile_capacity;
              internal_proxy_getmem_wg(target, stage, bytes, assignment.source_proxy_pe, 0, wf_info);
            }
          }
        }
        internal_world_proxy_sync_wg(my_pe, constmem.num_pes, p_sync, wf_info, sync_epoch++);
      }
    }
  }

  if (is_thread_zero_in_block()) {
    team_obj->proxy_alltoall_sync_epoch = sync_epoch - 1;
  }
  __syncthreads();
}

/**
 * @brief Run the WORLD all-to-all broadcast fallback for missing direct lanes.
 *
 * @tparam T Element type.
 * @param team_obj WORLD team routing and scratch state.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param nelems Elements per peer.
 * @param elem_offset First element to transfer.
 * @param elem_count Number of elements to transfer.
 * @param wf_info Active workgroup metadata.
 */
template <typename T>
__device__ void GDAContext::internal_broadcast_world_proxy_alltoall(GDATeam *team_obj, T *dst, const T *src,
    int nelems, int elem_offset, int elem_count, ActiveWFInfo &wf_info) {
  T *scratch = reinterpret_cast<T *>(team_obj->pWrk);
  long *p_sync = team_obj->alltoall_pSync;
  const int pe_size = constmem.num_pes;
  const size_t scratch_bytes = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double);
  const size_t tile_capacity = scratch_bytes / sizeof(T);
  const size_t total = static_cast<size_t>(pe_size) * nelems;
  const size_t wanted_begin = static_cast<size_t>(constmem.my_pe) * nelems + elem_offset;
  const size_t wanted_end = wanted_begin + elem_count;
  int64_t sync_epoch = static_cast<int64_t>(team_obj->proxy_alltoall_sync_epoch + 1);

  for (int source_pe = 0; source_pe < pe_size; source_pe++) {
    for (size_t offset = 0; offset < total; offset += tile_capacity) {
      const size_t count = min(tile_capacity, total - offset);
      internal_broadcast(scratch, src + offset, static_cast<int>(count), source_pe, 0, 1, pe_size, p_sync,
          sync_epoch);
      sync_epoch += proxy::kWorldBroadcastSyncPhases;
      const size_t begin = max(offset, wanted_begin);
      const size_t end = min(offset + count, wanted_end);
      if (begin < end) {
        T *target = dst + source_pe * nelems +
                    (begin - static_cast<size_t>(constmem.my_pe) * nelems);
        T *source = scratch + (begin - offset);
        memcpy_wg<MemcpyKind::Put>(target, source, (end - begin) * sizeof(T));
      }
      internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, sync_epoch++);
    }
  }
  if (is_thread_zero_in_block()) {
    team_obj->proxy_alltoall_sync_epoch = sync_epoch - 1;
  }
  __syncthreads();
}

// ############################################################################
// REDUCTION
// ############################################################################

/**
 * @brief Reduce via corresponding local proxy slots in complete IPC domains.
 * @param use_pxn_sync Use same-NIC WORLD epochs, or the original GDA sync.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ __noinline__ void GDAContext::internal_segmented_proxy_allreduce(T *dst, const T *src, int nelems,
    GDATeam *team_obj, ActiveWFInfo &wf_info, bool use_pxn_sync) {
  const int pe_start = team_obj->tinfo_wrt_world->pe_start;
  const int pe_size = team_obj->tinfo_wrt_world->size;
  const int local_size = constmem.ipc_shm_size;
  const int local_slot = ipcImpl_.shm_rank;
  const int ipc_domain_count = pe_size / local_size;
  const int root_ipc_domain_first_pe = pe_start;
  const bool on_root_ipc_domain = constmem.ipc_first_pe == root_ipc_domain_first_pe;
  long *p_sync = team_obj->reduce_pSync;
  T *p_wrk = reinterpret_cast<T *>(team_obj->pWrk);
  const int wg_id = get_flat_block_id();
  const int wg_size = get_flat_block_size();
  constexpr size_t MIN_BYTES_PER_PROXY = 64 * 1024;
  const size_t nbytes = static_cast<size_t>(nelems) * sizeof(T);
  const size_t useful_proxies = (nbytes + MIN_BYTES_PER_PROXY - 1) / MIN_BYTES_PER_PROXY;
  const int proxies = static_cast<int>(min(static_cast<size_t>(local_size), useful_proxies));
  const int proxy_chunk = (nelems + proxies - 1) / proxies;

  if (local_slot < proxies) {
    const int begin = local_slot * proxy_chunk;
    const int count = min(proxy_chunk, nelems - begin);
    internal_fused_ipc_reduce<T, Op>(dst + begin, src, begin, count, local_size, local_slot, wg_id, wg_size);
  }

  // Publish local partials before either transport reads them.
  threadfence_system();
  __syncthreads();
  const int qp_row = use_pxn_sync ? team_obj->pool_index_ % num_qps_per_pe
      : static_cast<int>(blockIdx.x) % num_qps_per_pe;
  int64_t epoch = use_pxn_sync ? static_cast<int64_t>(team_obj->proxy_reduce_sync_epoch) : 1;
  auto synchronize = [&] __device__ (int64_t phase) {
    if (use_pxn_sync) {
      internal_world_proxy_sync_wg(constmem.my_pe, pe_size, p_sync, wf_info, phase, qp_row);
    } else {
      internal_sync_wg(constmem.my_pe, pe_start, 1, pe_size, p_sync, wf_info);
    }
  };
  const int tile_elems = max(1,
      static_cast<int>(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T)));
  synchronize(epoch++);
  // Each local slot owns one segment and one scratch tile on the root IPC domain.
  // Serialize source IPC domains so incoming PUTs never overwrite an unread tile.
  for (int offset = 0; offset < proxy_chunk; offset += tile_elems) {
    const int begin = local_slot * proxy_chunk + offset;
    const int count = local_slot < proxies ? max(0, min(tile_elems,
        min(proxy_chunk - offset, nelems - begin))) : 0;
    for (int ipc_domain = 1; ipc_domain < ipc_domain_count; ipc_domain++) {
      if (constmem.ipc_first_pe == pe_start + ipc_domain * local_size && count > 0) {
        const int root_proxy = root_ipc_domain_first_pe + local_slot;
        internal_proxy_putmem_wg(p_wrk, dst + begin, static_cast<size_t>(count) * sizeof(T),
            root_proxy, qp_row, wf_info);
      }
      synchronize(epoch++);
      if (on_root_ipc_domain && count > 0) {
        // Reload NIC-written scratch instead of reusing cached prior tiles.
        for (int i = wg_id; i < count; i += wg_size) {
          T value = uncached_load(p_wrk + i);
          OpWrap<Op>::Calc(&value, dst + begin + i, 0);
        }
        __syncthreads();
      }
      synchronize(epoch++);
    }
    if (on_root_ipc_domain && count > 0) {
      for (int ipc_domain = 1; ipc_domain < ipc_domain_count; ipc_domain++) {
        const int peer = pe_start + ipc_domain * local_size + local_slot;
        internal_proxy_putmem_wg(dst + begin, dst + begin, static_cast<size_t>(count) * sizeof(T),
            peer, qp_row, wf_info);
      }
    }
    synchronize(epoch++);
  }
  for (int slot = 0; slot < proxies; slot++) {
    if (slot == local_slot) {
      continue;
    }
    const int begin = slot * proxy_chunk;
    const int count = min(proxy_chunk, nelems - begin);
    const int peer = constmem.ipc_first_pe + slot;
    internal_proxy_getmem_wg(dst + begin, dst + begin, static_cast<size_t>(count) * sizeof(T), peer, 0, wf_info);
  }
  synchronize(epoch++);
  if (use_pxn_sync && is_thread_zero_in_block()) {
    team_obj->proxy_reduce_sync_epoch = epoch;
  }
  __syncthreads();
}

/**
 * @brief Try the proxy allreduce path selected for the current NIC policy.
 *
 * @param team_obj Team and proxy layout metadata.
 * @param dest Reduction output buffer.
 * @param source Local contribution buffer.
 * @param nelems Number of elements.
 * @param wf_info Active workgroup metadata.
 * @return true when a proxy path completed the reduction and team barrier.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ bool GDAContext::internal_try_proxy_allreduce(GDATeam *team_obj, T *dest, const T *source,
    int nelems, ActiveWFInfo &wf_info) {
  if (nelems <= 0) {
    return false;
  }
  const auto *team_info = team_obj->tinfo_wrt_world;
  bool use_segmented = false;
  if (!constmem.pxn_enabled) {
    // The unrestricted proxy path requires large messages and complete IPC
    // IPC domains; smaller or irregular cases retain the original direct/ring path.
    const int local_size = constmem.ipc_shm_size;
    use_segmented = static_cast<size_t>(nelems) * sizeof(T) >= kHierarchicalMinBytes &&
        team_info->stride == 1 && local_size > 1 &&
        constmem.ipc_stride == 1 && team_info->pe_start % local_size == 0 &&
        team_info->size % local_size == 0 && team_info->size > local_size;
    // Preserve the non-PXN direct/ring fallback for ineligible messages.
    if (!use_segmented) {
      return false;
    }
  } else {
    const auto layout = proxy::SelectProxyCollectiveLayout(team_obj, team_info->pe_start, team_info->stride,
        team_obj->num_pes);
    // Matching local PE/NIC positions permit the segmented algorithm.
    // Other routed layouts use broadcast-based reduction instead.
    use_segmented = layout == proxy::ProxyCollectiveLayout::WorldAlignedData;
    if (!use_segmented && layout != proxy::ProxyCollectiveLayout::TeamProxyRoute &&
        layout != proxy::ProxyCollectiveLayout::WorldSyncOnly) {
      return false;
    }
  }

  if (use_segmented) {
    internal_segmented_proxy_allreduce<T, Op>(dest, source, nelems, team_obj, wf_info, constmem.pxn_enabled);
  } else {
    const bool world = team_info->pe_start == 0 && team_info->stride == 1 &&
                       team_obj->num_pes == constmem.num_pes;
    // In-place input must be captured before output updates can overwrite it.
    const bool aggregate_team = dest == source ||
        (team_obj->num_pes > team_obj->in_team_proxy_ipc_domain_count &&
         static_cast<size_t>(nelems) * sizeof(T) >= kHierarchicalMinBytes);
    if (world || aggregate_team) {
      internal_ipc_domain_proxy_reduce<T, Op, false>(team_obj, dest, source, nelems);
    } else {
      internal_broadcast_proxy_reduce<T, Op, false>(team_obj, dest, source, nelems);
    }
  }
  barrier_wg(reinterpret_cast<rocshmem_team_t>(team_obj));
  return true;
}

/**
 * @brief Select the shared ring or routed reduction for the available NIC policy.
 *
 * @param team_obj Team and proxy layout metadata.
 * @param dest This PE's reduced segment.
 * @param source Local vector containing one segment per destination.
 * @param nelems Elements in each destination segment.
 * @return true when a proxy path completed the reduction and team barrier.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ bool GDAContext::internal_try_proxy_reduce_scatter(GDATeam *team_obj, T *dest, const T *source,
    int nelems, int elem_offset, int elem_count) {
  if (nelems <= 0) {
    return false;
  }
  const auto *team_info = team_obj->tinfo_wrt_world;
  const auto layout = constmem.pxn_enabled
      ? proxy::SelectProxyCollectiveLayout(team_obj, team_info->pe_start, team_info->stride, team_obj->num_pes)
      : proxy::ProxyCollectiveLayout::None;
  const int local_size = constmem.ipc_shm_size;
  int local_slot = -1;
  const bool has_ipc = ipcImpl_.isIpcAvailable(constmem.my_pe, constmem.my_pe, &local_slot);
  const bool use_ring = has_ipc && local_size > 1 && team_info->stride == 1 &&
      constmem.ipc_stride == 1 && team_info->pe_start % local_size == 0 &&
      team_obj->num_pes > local_size && team_obj->num_pes % local_size == 0 &&
      nelems > 1024 && (!constmem.pxn_enabled || layout == proxy::ProxyCollectiveLayout::WorldAlignedData);
  if (use_ring) {
    const int status = hybrid_reduce_scatter_wg<T, Op>(team_obj, dest, source, nelems,
        elem_offset, elem_count, team_info->pe_start, team_info->stride, team_obj->num_pes, team_obj->my_pe);
    if (status != ROCSHMEM_SUCCESS) {
      LOGD_ERROR_ABORT("Proxy reduce-scatter ring failed");
    }
    barrier_wg(reinterpret_cast<rocshmem_team_t>(team_obj));
    return true;
  }
  if (!constmem.pxn_enabled ||
      (layout != proxy::ProxyCollectiveLayout::TeamProxyRoute &&
       layout != proxy::ProxyCollectiveLayout::WorldAlignedData &&
       layout != proxy::ProxyCollectiveLayout::WorldSyncOnly)) {
    return false;
  }
  // In-place input must be captured before output updates can overwrite it.
  const bool aggregate_team = dest == source ||
      (team_obj->num_pes > team_obj->in_team_proxy_ipc_domain_count &&
       static_cast<size_t>(nelems) * sizeof(T) >= kHierarchicalMinBytes);
  if (layout == proxy::ProxyCollectiveLayout::TeamProxyRoute && !aggregate_team) {
    internal_broadcast_proxy_reduce<T, Op, true>(team_obj, dest, source, nelems, elem_offset, elem_count);
  } else {
    internal_ipc_domain_proxy_reduce<T, Op, true>(team_obj, dest, source, nelems, elem_offset, elem_count);
  }
  barrier_wg(reinterpret_cast<rocshmem_team_t>(team_obj));
  return true;
}

/**
 * @brief Reduce local team contributions before exchanging domain partials.
 *
 * One IPC leader reduces each tile into the first half of pWrk. The second
 * half receives broadcasts, keeping local partials intact until all IPC domains
 * consume them. Reduce-scatter retains only the caller's output interval.
 */
template <typename T, ROCSHMEM_OP Op, bool ReduceScatter>
__device__ __noinline__ void GDAContext::internal_ipc_domain_proxy_reduce(GDATeam *team_obj, T *dest, const T *source,
    int nelems, int elem_offset, int elem_count) {
  constexpr size_t kScratchElements = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T);
  constexpr size_t kTileElements = kScratchElements / 2;
  static_assert(kTileElements > 0, "Proxy reduction needs two scratch tiles");
  T *partial = reinterpret_cast<T *>(team_obj->pWrk);
  T *received = partial + kTileElements;
  long *p_sync = team_obj->reduce_pSync;
  const int tid = get_flat_block_id();
  const int block_size = get_flat_block_size();
  const auto range = proxy::ProxyReductionRange::Make(nelems, team_obj->num_pes, ReduceScatter, elem_offset, elem_count);
  const size_t wanted_begin = ReduceScatter ? static_cast<size_t>(team_obj->my_pe) * nelems : 0;
  const size_t wanted_end = wanted_begin + nelems;
  const int qp_row = team_obj->pool_index_ % num_qps_per_pe;
  int64_t epoch = static_cast<int64_t>(team_obj->proxy_reduce_sync_epoch);
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  const auto *team_info = team_obj->tinfo_wrt_world;
  const bool world = team_info->pe_start == 0 && team_info->stride == 1 &&
      team_obj->num_pes == constmem.num_pes;
  const int ipc_domain_count = world ? constmem.world_proxy_ipc_domain_count : team_obj->in_team_proxy_ipc_domain_count;
  const int broadcast_phases = proxy::TeamBroadcastPhaseCount(team_obj);
  int local_proxy = constmem.ipc_first_pe;
  if (!world) {
    local_proxy = -1;
    for (int ipc_domain = 0; ipc_domain < ipc_domain_count; ipc_domain++) {
      if (team_obj->in_team_proxy_ipc_domain_first_pes[ipc_domain] == constmem.ipc_first_pe) {
        local_proxy = team_obj->in_team_proxy_pes[ipc_domain];
        break;
      }
    }
    if (local_proxy < 0) {
      LOGD_ERROR_ABORT("Partial-team reduction has no local proxy");
    }
    // Publish every member's input before its proxy reads the IPC mapping.
    internal_dispatch_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);
  }

  for (int segment = 0; segment < range.segment_count(team_obj->num_pes); segment++) {
    const size_t finish = range.segment_end(segment);
    for (size_t offset = range.segment_begin(segment); offset < finish; offset += kTileElements) {
      const int count = static_cast<int>(min(kTileElements, finish - offset));
      if (world && constmem.my_pe == local_proxy) {
        internal_fused_ipc_reduce<T, Op>(partial, source + offset, 0, count, constmem.ipc_shm_size,
            ipcImpl_.shm_rank, tid, block_size);
      } else if (!world && constmem.my_pe == local_proxy) {
        bool first = true;
        for (int rank = 0; rank < team_obj->num_pes; rank++) {
          const int member = team_obj->get_pe_in_world(rank);
          T *peer_source = const_cast<T *>(source + offset);
          if (member != constmem.my_pe) {
            int local_slot = -1;
            if (!ipcImpl_.isIpcAvailable(constmem.my_pe, member, &local_slot)) {
              continue;
            }
            peer_source = reinterpret_cast<T *>(get_local_ptr(source + offset, local_slot));
          }
          // uncached_load requires T*, but only reads this mapped source.
          // Fold only team members. Do not stage each peer through the receive
          // tile, which is reserved for the subsequent broadcasts between IPC domains.
          for (int i = tid; i < count; i += block_size) {
            T value = uncached_load(peer_source + i);
            if (first) {
              partial[i] = value;
            } else {
              OpWrap<Op>::Calc(&value, partial + i, 0);
            }
          }
          __syncthreads();
          first = false;
        }
        if (first) {
          LOGD_ERROR_ABORT("Partial-team reduction proxy has no local input");
        }
      }
      // Finish reading every local input before any PE overwrites an in-place output.
      internal_dispatch_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);
      const size_t begin = max(offset, wanted_begin);
      const size_t end = min(offset + count, wanted_end);
      for (int ipc_domain = 0; ipc_domain < ipc_domain_count; ipc_domain++) {
        const int root = world ? constmem.world_proxy_ipc_domain_first_pes[ipc_domain] : team_obj->in_team_proxy_pes[ipc_domain];
        internal_broadcast(received, partial, count, root, team_info->pe_start, team_info->stride,
            team_obj->num_pes, p_sync, epoch,
            qp_row, team_obj);
        for (size_t i = begin + tid; i < end; i += block_size) {
          T value = uncached_load(received + i - offset);
          if (ipc_domain == 0) {
            dest[i - wanted_begin] = value;
          } else {
            OpWrap<Op>::Calc(&value, dest + i - wanted_begin, 0);
          }
        }
        __syncthreads();
        // No source may reuse its partial, or receiver its tile, before consumption.
        epoch += broadcast_phases;
        internal_dispatch_proxy_sync_wg(team_obj, p_sync, wf_info, epoch++);
      }
    }
  }
  if (is_thread_zero_in_block()) {
    team_obj->proxy_reduce_sync_epoch = epoch;
  }
  __syncthreads();
}

/**
 * @brief Reduce per-PE broadcast tiles for allreduce or reduce-scatter.
 *
 * Reduce-scatter keeps only this PE's interval; allreduce keeps the whole
 * vector. In-place calls stage inputs within each IPC domain, so initializing
 * the output here cannot overwrite another PE's pending input.
 */
template <typename T, ROCSHMEM_OP Op, bool ReduceScatter>
__device__ __noinline__ void GDAContext::internal_broadcast_proxy_reduce(GDATeam *team_obj, T *dest, const T *source,
    int nelems, int elem_offset, int elem_count) {
  T *scratch = reinterpret_cast<T *>(team_obj->pWrk);
  long *p_sync = team_obj->reduce_pSync;
  const int pe_size = team_obj->num_pes;
  const int tid = get_flat_block_id();
  const int block_size = get_flat_block_size();
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  const size_t capacity = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T);
  const auto range = proxy::ProxyReductionRange::Make(nelems, pe_size, ReduceScatter, elem_offset, elem_count);
  const size_t wanted_begin = ReduceScatter ? static_cast<size_t>(team_obj->my_pe) * nelems : 0;
  const size_t wanted_end = wanted_begin + nelems;
  int64_t epoch = static_cast<int64_t>(team_obj->proxy_reduce_sync_epoch);
  const int broadcast_phases = proxy::TeamBroadcastPhaseCount(team_obj);

  if constexpr (ReduceScatter) {
    memcpy_wg<MemcpyKind::Put>(dest + elem_offset, const_cast<T *>(source + wanted_begin + elem_offset),
        range.element_count * sizeof(T));
    __syncthreads();
  }
  for (int segment = 0; segment < range.segment_count(pe_size); segment++) {
    const size_t finish = range.segment_end(segment);
    for (size_t offset = range.segment_begin(segment); offset < finish; offset += capacity) {
      const int count = static_cast<int>(min(capacity, finish - offset));
      if constexpr (!ReduceScatter) {
        memcpy_wg<MemcpyKind::Put>(dest + offset, const_cast<T *>(source + offset),
            static_cast<size_t>(count) * sizeof(T));
        __syncthreads();
      }
      const size_t begin = max(offset, wanted_begin);
      const size_t end = min(offset + count, wanted_end);
      for (int rank = 0; rank < pe_size; rank++) {
        const int source_pe = team_obj->get_pe_in_world(rank);
        internal_broadcast(scratch, source + offset, count, source_pe, team_obj->tinfo_wrt_world->pe_start,
            team_obj->tinfo_wrt_world->stride, pe_size, p_sync, epoch, team_obj->pool_index_ % num_qps_per_pe,
            team_obj);
        if (source_pe != constmem.my_pe && begin < end) {
          for (size_t i = begin + tid; i < end; i += block_size) {
            // RDMA writes bypass GPU caches; reload each source's receive tile.
            T value = uncached_load(scratch + (i - offset));
            OpWrap<Op>::Calc(&value, dest + (i - wanted_begin), 0);
          }
        }
        __syncthreads();
        internal_dispatch_proxy_sync_wg(team_obj, p_sync, wf_info, epoch + broadcast_phases);
        epoch += broadcast_phases + 1;
      }
    }
  }
  if (is_thread_zero_in_block()) {
    team_obj->proxy_reduce_sync_epoch = epoch;
  }
  __syncthreads();
}

}  // namespace rocshmem

#endif  // USE_DAS

#endif  // LIBRARY_SRC_GDA_PROXY_COLLECTIVE_DEVICE_HPP_
