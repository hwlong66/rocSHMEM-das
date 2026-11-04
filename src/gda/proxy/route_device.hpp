// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_
#define LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_

namespace rocshmem::proxy {

/** @brief Match WORLD membership, including teams with the same members. */
__device__ __forceinline__ bool IsWorldMembership(int pe_start, int stride, int pe_size) {
  return pe_start == 0 && stride == 1 && pe_size == constmem.num_pes;
}

/** @brief Return whether a team has one selected proxy in multiple IPC domains. */
__device__ __forceinline__ bool HasInTeamProxyRoute(const GDATeam *team) {
  return team != nullptr && team->in_team_proxy_ipc_domain_count > 1;
}

/**
 * @brief Check whether the calling PE can use WORLD proxy synchronization.
 * @return true when WORLD membership and IPC layout match the host plan.
 */
__device__ __forceinline__ bool IsWorldProxySyncEligible(int pe, int pe_start, int stride, int pe_size) {
  const int local_pe_count = constmem.ipc_shm_size;
  return constmem.world_proxy_sync_available && pe_start == 0 && stride == 1 &&
         pe_size == constmem.num_pes && local_pe_count > 0 &&
         (local_pe_count == 1 || constmem.ipc_stride == 1) &&
         constmem.world_proxy_ipc_domain_count > 1 &&
         pe >= constmem.ipc_first_pe &&
         pe < constmem.ipc_first_pe + local_pe_count;
}

/**
 * @brief Check whether aligned WORLD data routing is safe for this team.
 * @return true when corresponding IPC slots own matching selected NICs.
 */
__device__ __forceinline__ bool IsAlignedWorldProxyDataEligible(int pe_start, int stride, int pe_size) {
  const int local_pe_count = constmem.ipc_shm_size;
  return constmem.world_proxy_aligned_data_available &&
         IsWorldMembership(pe_start, stride, pe_size) && local_pe_count > 1 &&
         constmem.ipc_stride == 1 && pe_size % local_pe_count == 0 &&
         constmem.ipc_first_pe ==
             (constmem.my_pe / local_pe_count) * local_pe_count;
}

enum class ProxyCollectiveLayout {
  None,               // No eligible PXN route; not a non-PXN mode.
  TeamProxyRoute,     // Partial team: one same-NIC team proxy per IPC domain.
  WorldAlignedData,   // WORLD: corresponding local PEs have matching NICs.
  WorldSyncOnly,      // WORLD: sync route exists; data NICs may not align.
};

/**
 * @brief Classify PXN routing guarantees without choosing an operation algorithm.
 *
 * @param team Team metadata, or nullptr for an internal WORLD operation.
 * @param pe_start First world PE in the operation.
 * @param stride World-rank stride between operation members.
 * @param pe_size Number of operation members.
 * @return The strongest available proxy layout; algorithms may impose
 *         additional message-size and scratch-capacity requirements.
 */
__device__ __forceinline__ ProxyCollectiveLayout SelectProxyCollectiveLayout(const GDATeam *team,
    int pe_start, int stride, int pe_size) {
  if (IsWorldMembership(pe_start, stride, pe_size)) {
    if (IsAlignedWorldProxyDataEligible(pe_start, stride, pe_size)) {
      return ProxyCollectiveLayout::WorldAlignedData;
    }
    if (IsWorldProxySyncEligible(constmem.my_pe, pe_start, stride, pe_size)) {
      return ProxyCollectiveLayout::WorldSyncOnly;
    }
    return ProxyCollectiveLayout::None;
  }
  return HasInTeamProxyRoute(team) ? ProxyCollectiveLayout::TeamProxyRoute
                                   : ProxyCollectiveLayout::None;
}

struct WorldProxyLane {
  int source_pe{-1};
  int target_pe{-1};
};

struct WorldProxyTransferPlan {
  int active_lanes{0};
  size_t segment_bytes{0};

  __device__ __forceinline__ size_t begin(int lane) const {
    return static_cast<size_t>(lane) * segment_bytes;
  }

  __device__ __forceinline__ size_t bytes(int lane, size_t total_bytes) const {
    return min(segment_bytes, total_bytes - begin(lane));
  }
};

/** @brief Return the exclusive world-PE end of an IPC domain. */
__device__ __forceinline__ int WorldProxyIpcDomainEnd(int ipc_domain_first_pe) {
  for (int ipc_domain = 0; ipc_domain < constmem.world_proxy_ipc_domain_count; ipc_domain++) {
    if (constmem.world_proxy_ipc_domain_first_pes[ipc_domain] == ipc_domain_first_pe) {
      return ipc_domain + 1 < constmem.world_proxy_ipc_domain_count
                 ? constmem.world_proxy_ipc_domain_first_pes[ipc_domain + 1]
                 : constmem.num_pes;
    }
  }
  return -1;
}

/** @brief Count PEs owning a NIC ID within a world-PE interval. */
__device__ __forceinline__ int CountNicProxies(int first, int end, int nic_name_id) {
  int count = 0;
  for (int pe = first; pe < end; pe++) {
    count += constmem.world_proxy_pe_nic_name_ids[pe] == nic_name_id;
  }
  return count;
}

/**
 * @brief Return a PE by its ordinal among owners of a given NIC ID.
 * @param first First world PE in the interval.
 * @param end Exclusive world PE end of the interval.
 * @param nic_name_id NIC ID to match.
 * @param proxy_ordinal Zero-based index among matching PEs, not a PE rank.
 * @return Matching world PE, or -1 if the ordinal is unavailable.
 */
__device__ __forceinline__ int NthNicProxy(int first, int end, int nic_name_id, int proxy_ordinal) {
  if (proxy_ordinal < 0) {
    return -1;
  }
  for (int pe = first; pe < end; pe++) {
    if (constmem.world_proxy_pe_nic_name_ids[pe] == nic_name_id) {
      if (proxy_ordinal == 0) {
        return pe;
      }
      proxy_ordinal--;
    }
  }
  return -1;
}

/** @brief Source proxy and scratch batch assigned to an all-to-all target. */
struct WorldAlltoallTargetAssignment {
  int source_proxy_pe{-1};  // Same-NIC source PE, or -1 if unreachable.
  int scratch_batch{-1};   // Reused scratch slot group, or -1 if unreachable.
};

/**
 * @brief Select the same-NIC source proxy for one all-to-all target.
 * @return Source proxy and scratch batch, or invalid values when unreachable.
 */
__device__ __forceinline__ WorldAlltoallTargetAssignment
SelectWorldAlltoallTargetAssignment(int source_ipc_domain_first_pe, int target_ipc_domain_first_pe, int target_pe) {
  const int nic_name_id = constmem.world_proxy_pe_nic_name_ids[target_pe];
  const int source_end = WorldProxyIpcDomainEnd(source_ipc_domain_first_pe);
  const int source_count = CountNicProxies(source_ipc_domain_first_pe, source_end, nic_name_id);
  if (source_count == 0) {
    return {};
  }
  int target_ordinal = 0;
  for (int pe = target_ipc_domain_first_pe; pe < target_pe; pe++) {
    target_ordinal += constmem.world_proxy_pe_nic_name_ids[pe] == nic_name_id;
  }
  return {NthNicProxy(source_ipc_domain_first_pe, source_end, nic_name_id, target_ordinal % source_count),
          target_ordinal / source_count};
}

/** @brief Count independent same-NIC lanes between two IPC domains. */
__device__ __forceinline__ int WorldProxyLaneCount(int source_ipc_domain_first_pe, int target_ipc_domain_first_pe) {
  const int source_end = WorldProxyIpcDomainEnd(source_ipc_domain_first_pe);
  const int target_end = WorldProxyIpcDomainEnd(target_ipc_domain_first_pe);
  if (source_end < 0 || target_end < 0) {
    return 0;
  }
  int count = 0;
  for (int nic_name_id = 0; nic_name_id < constmem.world_proxy_nic_name_count; nic_name_id++) {
    count += CountNicProxies(source_ipc_domain_first_pe, source_end, nic_name_id) > 0 &&
             CountNicProxies(target_ipc_domain_first_pe, target_end, nic_name_id) > 0;
  }
  return count;
}

/**
 * @brief Partition a message across the useful independent proxy lanes.
 * @return Active lane count and bytes assigned to each lane.
 */
__device__ __forceinline__ WorldProxyTransferPlan PlanWorldProxyTransfer(int source_ipc_domain_first_pe, int target_ipc_domain_first_pe,
    size_t bytes, size_t min_bytes_per_lane) {
  if (bytes == 0 || min_bytes_per_lane == 0) {
    return {};
  }
  const int lanes = WorldProxyLaneCount(source_ipc_domain_first_pe, target_ipc_domain_first_pe);
  if (lanes <= 0) {
    return {};
  }
  const int useful = static_cast<int>((bytes + min_bytes_per_lane - 1) / min_bytes_per_lane);
  const int active = min(lanes, max(1, useful));
  return {active, (bytes + active - 1) / active};
}

/**
 * @brief Select one lane while rotating NIC and endpoint PEs across operations.
 * @return Same-NIC source and target proxy PEs for the requested lane.
 */
__device__ __forceinline__ WorldProxyLane SelectWorldProxyLane(int source_ipc_domain_first_pe, int target_ipc_domain_first_pe, int lane,
    int64_t lane_rotation_index) {
  const int source_end = WorldProxyIpcDomainEnd(source_ipc_domain_first_pe);
  const int target_end = WorldProxyIpcDomainEnd(target_ipc_domain_first_pe);
  const int total = WorldProxyLaneCount(source_ipc_domain_first_pe, target_ipc_domain_first_pe);
  if (total <= 0) {
    return {};
  }
  int selected = static_cast<int>((lane + lane_rotation_index) % total);
  for (int nic_name_id = 0; nic_name_id < constmem.world_proxy_nic_name_count; nic_name_id++) {
    const int source_count = CountNicProxies(source_ipc_domain_first_pe, source_end, nic_name_id);
    const int target_count = CountNicProxies(target_ipc_domain_first_pe, target_end, nic_name_id);
    if (source_count == 0 || target_count == 0) {
      continue;
    }
    if (selected-- == 0) {
      const int round = static_cast<int>(lane_rotation_index / total);
      return {NthNicProxy(source_ipc_domain_first_pe, source_end, nic_name_id, round % source_count),
              NthNicProxy(target_ipc_domain_first_pe, target_end, nic_name_id, (round / source_count) % target_count)};
    }
  }
  return {};
}

}  // namespace rocshmem::proxy

#endif  // LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_
