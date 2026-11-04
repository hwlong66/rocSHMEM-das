// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_
#define LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_

namespace rocshmem::proxy {

/** @brief Return whether a team has one selected proxy in multiple islands. */
__device__ __forceinline__ bool HasInTeamProxyRoute(const GDATeam *team) {
  return team != nullptr && team->in_team_proxy_node_count > 1;
}

/**
 * @brief Check whether the calling PE can use WORLD proxy synchronization.
 * @return true when WORLD membership and IPC layout match the host plan.
 */
__device__ __forceinline__ bool IsWorldProxySyncEligible(int pe, int pe_start, int stride, int pe_size) {
  const int local_pe_count = constmem.ipc_shm_size;
  return constmem.world_proxy_sync_enabled && pe_start == 0 && stride == 1 &&
         pe_size == constmem.num_pes && local_pe_count > 0 &&
         (local_pe_count == 1 || constmem.ipc_stride == 1) &&
         constmem.world_proxy_node_count > 1 &&
         pe >= constmem.ipc_first_pe &&
         pe < constmem.ipc_first_pe + local_pe_count;
}

/**
 * @brief Check whether aligned WORLD data routing is safe for this team.
 * @return true when corresponding IPC slots own matching selected NICs.
 */
__device__ __forceinline__ bool IsAlignedWorldProxyDataEligible(int pe_start, int stride, int pe_size) {
  const int local_pe_count = constmem.ipc_shm_size;
  return constmem.aligned_world_proxy_data_enabled && pe_start == 0 &&
         stride == 1 && pe_size == constmem.num_pes && local_pe_count > 1 &&
         constmem.ipc_stride == 1 && pe_size % local_pe_count == 0 &&
         constmem.ipc_first_pe ==
             (constmem.my_pe / local_pe_count) * local_pe_count;
}

enum class ProxyCollectiveLayout {
  None,               // No eligible PXN route; not a non-PXN mode.
  TeamProxyRoute,     // Partial team: one same-NIC team proxy per island.
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
  if (pe_start == 0 && stride == 1 && pe_size == constmem.num_pes) {
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

/** @brief Return the exclusive world-PE end of an IPC island. */
__device__ __forceinline__ int WorldProxyNodeEnd(int node_first_pe) {
  for (int node = 0; node < constmem.world_proxy_node_count; node++) {
    if (constmem.world_proxy_node_first_pes[node] == node_first_pe) {
      return node + 1 < constmem.world_proxy_node_count
                 ? constmem.world_proxy_node_first_pes[node + 1]
                 : constmem.num_pes;
    }
  }
  return -1;
}

/** @brief Count PEs owning a NIC ID within a world-PE interval. */
__device__ __forceinline__ int CountNicProxies(int first, int end, int nic_id) {
  int count = 0;
  for (int pe = first; pe < end; pe++) {
    count += constmem.world_proxy_pe_nic_ids[pe] == nic_id;
  }
  return count;
}

/**
 * @brief Return a PE by its ordinal among owners of a given NIC ID.
 * @param first First world PE in the interval.
 * @param end Exclusive world PE end of the interval.
 * @param nic_id NIC ID to match.
 * @param proxy_ordinal Zero-based index among matching PEs, not a PE rank.
 * @return Matching world PE, or -1 if the ordinal is unavailable.
 */
__device__ __forceinline__ int NthNicProxy(int first, int end, int nic_id, int proxy_ordinal) {
  if (proxy_ordinal < 0) {
    return -1;
  }
  for (int pe = first; pe < end; pe++) {
    if (constmem.world_proxy_pe_nic_ids[pe] == nic_id) {
      if (proxy_ordinal == 0) {
        return pe;
      }
      proxy_ordinal--;
    }
  }
  return -1;
}

/** @brief Source proxy and scratch batch assigned to an all-to-all target. */
struct AlltoallTargetAssignment {
  int source_proxy_pe{-1};  // Same-NIC source PE, or -1 if unreachable.
  int scratch_batch{-1};   // Reused scratch slot group, or -1 if unreachable.
};

/**
 * @brief Select the same-NIC source proxy for one all-to-all target.
 * @return Source proxy and scratch batch, or invalid values when unreachable.
 */
__device__ __forceinline__ AlltoallTargetAssignment
SelectAlltoallTargetAssignment(int source_first, int target_first, int target_pe) {
  const int nic = constmem.world_proxy_pe_nic_ids[target_pe];
  const int source_end = WorldProxyNodeEnd(source_first);
  const int source_count = CountNicProxies(source_first, source_end, nic);
  if (source_count == 0) {
    return {};
  }
  int target_ordinal = 0;
  for (int pe = target_first; pe < target_pe; pe++) {
    target_ordinal += constmem.world_proxy_pe_nic_ids[pe] == nic;
  }
  return {NthNicProxy(source_first, source_end, nic, target_ordinal % source_count),
          target_ordinal / source_count};
}

/** @brief Count independent same-NIC lanes between two IPC islands. */
__device__ __forceinline__ int WorldProxyLaneCount(int source_first, int target_first) {
  const int source_end = WorldProxyNodeEnd(source_first);
  const int target_end = WorldProxyNodeEnd(target_first);
  if (source_end < 0 || target_end < 0) {
    return 0;
  }
  int count = 0;
  for (int nic = 0; nic < constmem.world_proxy_nic_count; nic++) {
    count += CountNicProxies(source_first, source_end, nic) > 0 &&
             CountNicProxies(target_first, target_end, nic) > 0;
  }
  return count;
}

/**
 * @brief Partition a message across the useful independent proxy lanes.
 * @return Active lane count and bytes assigned to each lane.
 */
__device__ __forceinline__ WorldProxyTransferPlan PlanWorldProxyTransfer(int source_first, int target_first,
    size_t bytes, size_t min_bytes_per_lane) {
  if (bytes == 0 || min_bytes_per_lane == 0) {
    return {};
  }
  const int lanes = WorldProxyLaneCount(source_first, target_first);
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
__device__ __forceinline__ WorldProxyLane SelectWorldProxyLane(int source_first, int target_first, int lane,
    int64_t operation_index) {
  const int source_end = WorldProxyNodeEnd(source_first);
  const int target_end = WorldProxyNodeEnd(target_first);
  const int total = WorldProxyLaneCount(source_first, target_first);
  if (total <= 0) {
    return {};
  }
  int selected = static_cast<int>((lane + operation_index) % total);
  for (int nic = 0; nic < constmem.world_proxy_nic_count; nic++) {
    const int source_count = CountNicProxies(source_first, source_end, nic);
    const int target_count = CountNicProxies(target_first, target_end, nic);
    if (source_count == 0 || target_count == 0) {
      continue;
    }
    if (selected-- == 0) {
      const int round = static_cast<int>(operation_index / total);
      return {NthNicProxy(source_first, source_end, nic, round % source_count),
              NthNicProxy(target_first, target_end, nic, (round / source_count) % target_count)};
    }
  }
  return {};
}

}  // namespace rocshmem::proxy

#endif  // LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_
