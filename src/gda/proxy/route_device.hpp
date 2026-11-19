// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_
#define LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_

#include "constants.hpp"

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
  return constmem.world_proxy_sync_available && pe_start == 0 && stride == 1 && pe_size == constmem.num_pes &&
         local_pe_count > 0 && (local_pe_count == 1 || constmem.ipc_stride == 1) &&
         constmem.world_proxy_ipc_domain_count > 1 && pe >= constmem.ipc_first_pe &&
         pe < constmem.ipc_first_pe + local_pe_count;
}

/**
 * @brief Check whether aligned WORLD data routing is safe for this team.
 * @return true when corresponding IPC slots own matching selected NICs.
 */
__device__ __forceinline__ bool IsAlignedWorldProxyDataEligible(int pe_start, int stride, int pe_size) {
  const int local_pe_count = constmem.ipc_shm_size;
  return constmem.world_proxy_aligned_data_available && IsWorldMembership(pe_start, stride, pe_size) &&
         local_pe_count > 1 && constmem.ipc_stride == 1 && pe_size % local_pe_count == 0 &&
         constmem.ipc_first_pe == (constmem.my_pe / local_pe_count) * local_pe_count;
}

enum class ProxyCollectiveLayout {
  None,              // No eligible PXN route; not a non-PXN mode.
  TeamProxyRoute,    // Partial team: one same-NIC team proxy per IPC domain.
  WorldAlignedData,  // WORLD: corresponding local PEs have matching NICs.
  WorldSyncOnly,     // WORLD: sync route exists; data NICs may not align.
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
__device__ __forceinline__ ProxyCollectiveLayout SelectProxyCollectiveLayout(const GDATeam *team, int pe_start,
                                                                             int stride, int pe_size) {
  if (IsWorldMembership(pe_start, stride, pe_size)) {
    if (IsAlignedWorldProxyDataEligible(pe_start, stride, pe_size)) {
      return ProxyCollectiveLayout::WorldAlignedData;
    }

    if (IsWorldProxySyncEligible(constmem.my_pe, pe_start, stride, pe_size)) {
      return ProxyCollectiveLayout::WorldSyncOnly;
    }
    return ProxyCollectiveLayout::None;
  }
  return HasInTeamProxyRoute(team) ? ProxyCollectiveLayout::TeamProxyRoute : ProxyCollectiveLayout::None;
}

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

// Membership and routing capabilities are independent of the collective algorithm.
struct ProxyTeamView {
  const GDATeam *team;
  bool is_world;
  int ipc_domain_count;

  __device__ int domain_first(int domain) const {
    return is_world ? constmem.world_proxy_ipc_domain_first_pes[domain]
                    : team->in_team_proxy_ipc_domain_first_pes[domain];
  }

  __device__ int domain_end(int domain) const {
    if (is_world) {
      return domain + 1 < ipc_domain_count ? constmem.world_proxy_ipc_domain_first_pes[domain + 1]
                                          : constmem.num_pes;
    }
    return WorldProxyIpcDomainEnd(domain_first(domain));
  }
};

__device__ inline ProxyTeamView GetProxyTeamView(const GDATeam *team, int pe_start, int stride, int pe_size) {
  const bool world = IsWorldMembership(pe_start, stride, pe_size);
  return {team, world,
          world ? constmem.world_proxy_ipc_domain_count : (team != nullptr ? team->in_team_proxy_ipc_domain_count : 0)};
}

struct ProxyDomainMembers {
  int first_rank;  // First team rank in the domain, or -1 when empty.
  int count;
};

/** @brief Derive the contiguous rank range of a strided team's domain members. */
__device__ inline ProxyDomainMembers GetProxyDomainMembers(const ProxyTeamView &view, int domain) {
  const int start = view.team != nullptr ? view.team->tinfo_wrt_world->pe_start : 0;
  const int stride = view.team != nullptr ? view.team->tinfo_wrt_world->stride : 1;
  const int size = view.team != nullptr ? view.team->num_pes : constmem.num_pes;
  if (stride <= 0 || domain < 0 || domain >= view.ipc_domain_count) {
    return {-1, 0};
  }

  const int64_t begin_distance = static_cast<int64_t>(view.domain_first(domain)) - start;
  const int64_t end_distance = static_cast<int64_t>(view.domain_end(domain)) - start;
  const int first = static_cast<int>(min(static_cast<int64_t>(size),
      max(int64_t{0}, (begin_distance + stride - 1) / stride)));
  const int limit = static_cast<int>(min(static_cast<int64_t>(size),
      max(int64_t{0}, (end_distance + stride - 1) / stride)));
  const int count = max(0, limit - first);
  return {count > 0 ? first : -1, count};
}

// All matching-mode paths share this compatibility rule. IDs are global NIC
// names; a plane ID of -1 excludes an unlisted NIC from PXN routing.
__device__ __forceinline__ bool AreProxyNicsCompatible(int source, int target) {
  if (source < 0 || target < 0 || source >= constmem.world_proxy_nic_name_count ||
      target >= constmem.world_proxy_nic_name_count) {
    return false;
  }
  if (constmem.proxy_network_planes_enabled) {
    return constmem.proxy_nic_plane_ids[source] >= 0 &&
           constmem.proxy_nic_plane_ids[source] == constmem.proxy_nic_plane_ids[target];
  }
  return source == target;
}

struct ProxyDataRoute {
  int source_pe{-1};
  int target_pe{-1};
};

// WORLD and partial teams differ only in candidate membership. Fixed endpoints
// preserve tasks whose input or final output already belongs to a particular PE.
struct ProxyDataRouteCandidates {
  const GDATeam *team;
  int source_first;
  int source_end;
  int target_first;
  int target_end;
  bool require_matching_nic;
  int fixed_source{-1};
  int fixed_target{-1};
};

__device__ __forceinline__ bool IsProxyDataMember(const GDATeam *team, int pe) {
  if (team == nullptr) {
    return true;
  }

  const auto *info = team->tinfo_wrt_world;
  const int distance = pe - info->pe_start;

  return info->stride > 0 && distance >= 0 && distance % info->stride == 0 &&
         distance / info->stride < info->size;
}

// All planners share membership, NIC filtering and endpoint exclusion.
__device__ inline bool IsProxyDataEndpoint(const GDATeam *team, int pe, int nic, int fixed_pe,
                                           bool compatible_group, const int *used, int used_count) {
  if ((fixed_pe >= 0 && fixed_pe != pe) || !IsProxyDataMember(team, pe)) {
    return false;
  }

  const int pe_nic = constmem.world_proxy_pe_nic_name_ids[pe];
  if (nic >= 0 && !(compatible_group ? AreProxyNicsCompatible(nic, pe_nic) : nic == pe_nic)) {
    return false;
  }
  for (int index = 0; index < used_count; index++) {
    if (used[index] == pe) {
      return false;
    }
  }

  return true;
}

__device__ inline int CountProxyDataEndpoints(
    const GDATeam *team, int first, int end, int nic, int fixed_pe,
    bool compatible_group = false, const int *used = nullptr, int used_count = 0) {
  int count = 0;
  for (int pe = first; pe < end; pe++) {
    count += IsProxyDataEndpoint(team, pe, nic, fixed_pe, compatible_group, used, used_count);
  }

  return count;
}

__device__ inline int SelectProxyDataEndpoint(
    const GDATeam *team, int first, int end, int nic, int fixed_pe, uint64_t ordinal,
    bool compatible_group = false, const int *used = nullptr, int used_count = 0) {
  for (int pe = first; pe < end; pe++) {
    if (IsProxyDataEndpoint(team, pe, nic, fixed_pe, compatible_group, used, used_count) && ordinal-- == 0) {
      return pe;
    }
  }

  return -1;
}

// Remaining endpoint counts belong to one plan and its fixed candidate set.
struct ProxyEndpointCounts {
  int source[ROCSHMEM_PROXY_MAX_NIC_NAMES];
  int target[ROCSHMEM_PROXY_MAX_NIC_NAMES];
  uint64_t compatible_targets[ROCSHMEM_PROXY_NIC_MASK_BITS];  // Cached legal NIC pairs for up to 64 NICs.
};

/** @brief Initialize counts and compatibility once for one candidate set. */
__device__ inline void InitProxyEndpointCounts(
    const ProxyDataRouteCandidates &candidates, ProxyEndpointCounts &counts,
    const int *used_sources = nullptr, const int *used_targets = nullptr, int used_count = 0) {
  const int nics = constmem.world_proxy_nic_name_count;
  for (int nic = 0; nic < nics; nic++) {
    counts.source[nic] = CountProxyDataEndpoints(
        candidates.team, candidates.source_first, candidates.source_end, nic, candidates.fixed_source,
        false, used_sources, used_count);
    counts.target[nic] = CountProxyDataEndpoints(
        candidates.team, candidates.target_first, candidates.target_end, nic, candidates.fixed_target,
        false, used_targets, used_count);
    if (nics <= ROCSHMEM_PROXY_NIC_MASK_BITS) {
      uint64_t targets = 0;
      for (int target = 0; target < nics; target++) {
        if (!candidates.require_matching_nic || AreProxyNicsCompatible(nic, target)) {
          targets |= uint64_t{1} << target;
        }
      }
      counts.compatible_targets[nic] = targets;
    }
  }
}

// Equal turns per NIC, then equal turns among that NIC's endpoint PEs. The
// index identifies an existing task/operation, never a PE-local counter.
__device__ __forceinline__ ProxyDataRoute SelectProxyDataRouteImpl(const GDATeam *team, int source_first,
                                                                   int source_end, int target_first, int target_end,
                                                                   bool require_matching_nic, int fixed_source,
                                                                   int fixed_target, uint64_t transfer_index,
                                                                   const int *used_sources, const int *used_targets,
                                                                   int used_count,
                                                                   const ProxyEndpointCounts *cached_counts) {
  if (!constmem.world_proxy_nic_layout_available || source_first < 0 || target_first < 0 ||
      source_end > constmem.num_pes || target_end > constmem.num_pes) {
    return {};
  }

  // A fixed target has only one NIC: enumerate source NICs, not all NIC pairs.
  if (fixed_target >= 0) {
    if (fixed_target < target_first || fixed_target >= target_end) {
      return {};
    }

    const int target_nic = constmem.world_proxy_pe_nic_name_ids[fixed_target];
    if (target_nic < 0 || target_nic >= constmem.world_proxy_nic_name_count) {
      return {};
    }
    const int targets = cached_counts != nullptr ? cached_counts->target[target_nic] :
        CountProxyDataEndpoints(team, target_first, target_end, target_nic, fixed_target,
                                false, used_targets, used_count);
    if (targets == 0) {
      return {};
    }

    int groups = 0;
    for (int nic = 0; nic < constmem.world_proxy_nic_name_count; nic++) {
      if (!require_matching_nic || AreProxyNicsCompatible(nic, target_nic)) {
        const int count = cached_counts != nullptr ? cached_counts->source[nic] :
            CountProxyDataEndpoints(team, source_first, source_end, nic, fixed_source,
                                    false, used_sources, used_count);
        groups += count > 0;
      }
    }
    if (groups == 0) {
      return {};
    }

    int selected = transfer_index % groups;
    const uint64_t round = transfer_index / groups;
    for (int nic = 0; nic < constmem.world_proxy_nic_name_count; nic++) {
      if (require_matching_nic && !AreProxyNicsCompatible(nic, target_nic)) {
        continue;
      }

      const int count = cached_counts != nullptr ? cached_counts->source[nic] :
          CountProxyDataEndpoints(team, source_first, source_end, nic, fixed_source,
                                  false, used_sources, used_count);
      if (count > 0 && selected-- == 0) {
        return {SelectProxyDataEndpoint(team, source_first, source_end, nic, fixed_source,
                                        round % count, false, used_sources, used_count),
                fixed_target};
      }
    }

    return {};
  }

  const int nics = constmem.world_proxy_nic_name_count;
  if (nics <= 0 || nics > ROCSHMEM_PROXY_MAX_NIC_NAMES) {
    return {};
  }

  ProxyEndpointCounts local_counts;
  if (cached_counts == nullptr) {
    InitProxyEndpointCounts(
        {team, source_first, source_end, target_first, target_end,
         require_matching_nic, fixed_source, fixed_target},
        local_counts, used_sources, used_targets, used_count);
    cached_counts = &local_counts;
  }

  // Iterate only cached legal pairs. Keep ascending NIC order and the same
  // group/endpoint rotation as the general path below.
  if (nics <= ROCSHMEM_PROXY_NIC_MASK_BITS) {
    uint64_t available_targets = 0;
    for (int nic = 0; nic < nics; nic++) {
      if (cached_counts->target[nic] > 0) {
        available_targets |= uint64_t{1} << nic;
      }
    }
    int groups = 0;
    for (int nic = 0; nic < nics; nic++) {
      if (cached_counts->source[nic] > 0) {
        groups += __builtin_popcountll(cached_counts->compatible_targets[nic] & available_targets);
      }
    }
    if (groups == 0) {
      return {};
    }

    int selected = transfer_index % groups;
    const uint64_t round = transfer_index / groups;
    for (int source_nic = 0; source_nic < nics; source_nic++) {
      const int sources = cached_counts->source[source_nic];
      if (sources == 0) {
        continue;
      }

      uint64_t targets = cached_counts->compatible_targets[source_nic] & available_targets;
      const int count = __builtin_popcountll(targets);
      if (selected >= count) {
        selected -= count;
        continue;
      }
      while (selected-- > 0) {
        targets &= targets - 1;
      }
      const int target_nic = __builtin_ctzll(targets);
      return {SelectProxyDataEndpoint(team, source_first, source_end, source_nic, fixed_source,
                                      round % sources, false, used_sources, used_count),
              SelectProxyDataEndpoint(team, target_first, target_end, target_nic, fixed_target,
                                      round % cached_counts->target[target_nic],
                                      false, used_targets, used_count)};
    }

    return {};
  }

  const bool diagonal = require_matching_nic && !constmem.proxy_network_planes_enabled;
  const int pairs = diagonal ? nics : nics * nics;
  int groups = 0;

  for (int pair = 0; pair < pairs; pair++) {
    const int source_nic = diagonal ? pair : pair / nics;
    const int target_nic = diagonal ? pair : pair % nics;
    const int sources = cached_counts->source[source_nic];
    if (require_matching_nic && !AreProxyNicsCompatible(source_nic, target_nic)) {
      continue;
    }
    const int targets = cached_counts->target[target_nic];
    groups += sources > 0 && targets > 0;
  }

  if (groups == 0) {
    return {};
  }

  int selected = transfer_index % groups;
  const uint64_t round = transfer_index / groups;

  for (int pair = 0; pair < pairs; pair++) {
    const int source_nic = diagonal ? pair : pair / nics;
    const int target_nic = diagonal ? pair : pair % nics;
    const int sources = cached_counts->source[source_nic];
    if (require_matching_nic && !AreProxyNicsCompatible(source_nic, target_nic)) {
      continue;
    }
    const int targets = cached_counts->target[target_nic];

    if (sources == 0 || targets == 0) {
      continue;
    }

    if (selected-- == 0) {
      return {SelectProxyDataEndpoint(team, source_first, source_end, source_nic, fixed_source,
                                      round % sources, false, used_sources, used_count),
              SelectProxyDataEndpoint(team, target_first, target_end, target_nic, fixed_target,
                                      round % targets, false, used_targets, used_count)};
    }
  }

  return {};
}

/**
 * @brief Select a NIC pair and proxy endpoints using one shared rotation.
 *
 * Fixed-target tasks include the target's ordinal among compatible members,
 * so adjacent tasks rotate through proxies instead of repeatedly selecting
 * the same endpoint. Unconstrained tasks use the supplied index directly.
 */
__device__ __forceinline__ ProxyDataRoute SelectProxyDataRoute(const ProxyDataRouteCandidates &candidates,
                                                               uint64_t transfer_index,
                                                               const int *used_sources = nullptr,
                                                               const int *used_targets = nullptr, int used_count = 0,
                                                               const ProxyEndpointCounts *cached_counts = nullptr) {
  if (candidates.fixed_target >= 0) {
    if (candidates.fixed_target < candidates.target_first ||
        candidates.fixed_target >= candidates.target_end ||
        candidates.target_first < 0 || candidates.target_end > constmem.num_pes ||
        !IsProxyDataMember(candidates.team, candidates.fixed_target)) {
      return {};
    }

    const int target_nic = constmem.world_proxy_pe_nic_name_ids[candidates.fixed_target];
    for (int pe = candidates.target_first; pe < candidates.fixed_target; pe++) {
      transfer_index += IsProxyDataMember(candidates.team, pe) &&
                        (!candidates.require_matching_nic ||
                         AreProxyNicsCompatible(constmem.world_proxy_pe_nic_name_ids[pe], target_nic));
    }
  }

  return SelectProxyDataRouteImpl(candidates.team, candidates.source_first, candidates.source_end,
                                  candidates.target_first, candidates.target_end, candidates.require_matching_nic,
                                  candidates.fixed_source, candidates.fixed_target, transfer_index,
                                  used_sources, used_targets, used_count, cached_counts);
}

/**
 * @brief Build parallel routes using the common endpoint-exclusion filter.
 *
 * As with domain lanes, each selection excludes all endpoints already assigned
 * to previous lanes. Pair compatibility and fixed endpoints remain constraints.
 */
__device__ inline __noinline__ int BuildProxyDataRouteLanes(
    const ProxyDataRouteCandidates &candidates, uint64_t operation_index,
    int capacity, int *source_pes, int *target_pes) {
  const int nics = constmem.world_proxy_nic_name_count;
  if (!constmem.world_proxy_nic_layout_available || capacity <= 0 ||
      nics <= 0 || nics > ROCSHMEM_PROXY_MAX_NIC_NAMES ||
      candidates.source_first < 0 || candidates.source_end > constmem.num_pes ||
      candidates.target_first < 0 || candidates.target_end > constmem.num_pes) {
    return 0;
  }

  ProxyEndpointCounts counts;
  InitProxyEndpointCounts(candidates, counts);

  int lanes = 0;
  for (; lanes < capacity; lanes++) {
    const auto route = SelectProxyDataRoute(
        candidates, operation_index + lanes, source_pes, target_pes, lanes, &counts);
    if (route.source_pe < 0 || route.target_pe < 0) {
      break;
    }

    source_pes[lanes] = route.source_pe;
    target_pes[lanes] = route.target_pe;
    counts.source[constmem.world_proxy_pe_nic_name_ids[route.source_pe]]--;
    counts.target[constmem.world_proxy_pe_nic_name_ids[route.target_pe]]--;
  }

  return lanes;
}

/** @brief Read a host-validated domain proxy; never perform dynamic selection. */
__device__ inline int GetValidatedDomainProxy(const GDATeam *team, int domain_first) {
  const bool world = team == nullptr || IsWorldMembership(
      team->tinfo_wrt_world->pe_start, team->tinfo_wrt_world->stride, team->tinfo_wrt_world->size);
  if (world && constmem.world_proxy_sync_available &&
      WorldProxyIpcDomainEnd(domain_first) > domain_first) {
    for (int domain = 0; domain < constmem.world_proxy_ipc_domain_count; domain++) {
      if (constmem.world_proxy_ipc_domain_first_pes[domain] == domain_first) {
        return constmem.world_proxy_sync_pes[domain];
      }
    }
  }

  if (team != nullptr) {
    for (int domain = 0; domain < team->in_team_proxy_ipc_domain_count; domain++) {
      if (team->in_team_proxy_ipc_domain_first_pes[domain] == domain_first) {
        return team->in_team_proxy_pes[domain];
      }
    }
  }

  return -1;
}

// Build data candidates without modifying the fixed synchronization layout.
__device__ __forceinline__ ProxyDataRoute SelectTeamProxyDataRoute(const GDATeam *team, int source_first,
                                                                   int target_first, uint64_t transfer_index) {
  if (constmem.world_proxy_nic_layout_available) {
    return SelectProxyDataRoute(
        {team, source_first, WorldProxyIpcDomainEnd(source_first),
         target_first, WorldProxyIpcDomainEnd(target_first), constmem.pxn_enabled}, transfer_index);
  }

  return {GetValidatedDomainProxy(team, source_first), GetValidatedDomainProxy(team, target_first)};
}

/** @brief Canonical representative of a same-name or configured-plane group. */
__device__ inline bool IsProxyNicGroupRepresentative(int nic) {
  if (!AreProxyNicsCompatible(nic, nic)) {
    return false;
  }
  for (int previous = 0; previous < nic; previous++) {
    if (AreProxyNicsCompatible(nic, previous)) {
      return false;
    }
  }

  return true;
}

struct ProxyNicGroupSelection {
  int nic{-1};
  uint64_t round{0};
};

/** @brief Select a compatibility group present in every participating domain. */
__device__ inline __noinline__ ProxyNicGroupSelection SelectCommonProxyNicGroup(
    const ProxyTeamView &view, uint64_t operation, const int *used = nullptr,
    int used_stride = 0, int used_count = 0) {
  if (!constmem.world_proxy_nic_layout_available || view.ipc_domain_count <= 0) {
    return {};
  }

  int groups = 0;
  for (int pass = 0; pass < 2; pass++) {
    int selected = groups > 0 ? operation % groups : 0;
    for (int nic = 0; nic < constmem.world_proxy_nic_name_count; nic++) {
      if (!IsProxyNicGroupRepresentative(nic)) {
        continue;
      }

      bool common = true;
      for (int domain = 0; domain < view.ipc_domain_count; domain++) {
        const int first = view.domain_first(domain);
        common &= CountProxyDataEndpoints(
                      view.team, first, WorldProxyIpcDomainEnd(first), nic, -1, true,
                      used != nullptr ? used + domain * used_stride : nullptr, used_count) > 0;
      }
      if (!common) {
        continue;
      }
      if (pass == 0) {
        groups++;
      } else if (selected-- == 0) {
        return {nic, operation / groups};
      }
    }
    if (groups == 0) {
      break;
    }
  }

  return {};
}

/** @brief Rotate NICs first, then available endpoints within the selected group. */
__device__ inline int SelectProxyEndpointInGroup(
    const GDATeam *team, int first, int end, const ProxyNicGroupSelection &group,
    const int *used = nullptr, int used_count = 0) {
  if (group.nic < 0) {
    return -1;
  }

  int nics = 0;
  for (int nic = 0; nic < constmem.world_proxy_nic_name_count; nic++) {
    nics += AreProxyNicsCompatible(group.nic, nic) &&
            CountProxyDataEndpoints(team, first, end, nic, -1, false, used, used_count) > 0;
  }
  if (nics == 0) {
    return -1;
  }

  int selected = group.round % nics;
  for (int nic = 0; nic < constmem.world_proxy_nic_name_count; nic++) {
    if (!AreProxyNicsCompatible(group.nic, nic)) {
      continue;
    }

    const int count = CountProxyDataEndpoints(team, first, end, nic, -1, false, used, used_count);
    if (count > 0 && selected-- == 0) {
      return SelectProxyDataEndpoint(
          team, first, end, nic, -1, (group.round / nics) % count, false, used, used_count);
    }
  }

  return -1;
}

// Bound the per-plan workspace instead of allocating a domains-by-NICs matrix.
struct ProxyDomainEndpointCounts {
  uint16_t counts[ROCSHMEM_PROXY_DOMAIN_CACHE_CAPACITY];  // Domain-major remaining NIC counts.
  int16_t group_nic[ROCSHMEM_PROXY_DOMAIN_CACHE_CAPACITY];  // NIC -> canonical representative, -1 if excluded.
  uint16_t remaining[ROCSHMEM_PROXY_DOMAIN_CACHE_CAPACITY];  // Representative -> remaining common lanes.
  int ends[ROCSHMEM_PROXY_DOMAIN_CACHE_CAPACITY];
};

/** @brief Reuse per-domain NIC counts and decrement one endpoint per selected domain. */
__device__ inline __noinline__ int BuildCachedProxyDomainLanes(const ProxyTeamView &view,
         uint64_t operation, int capacity, int stride, int *owners) {
  ProxyDomainEndpointCounts cache;
  const int nics = constmem.world_proxy_nic_name_count;
  int groups = 0;

  for (int nic = 0; nic < nics; nic++) {
    cache.group_nic[nic] = -1;
    cache.remaining[nic] = 0;
    if (!AreProxyNicsCompatible(nic, nic)) {
      continue;
    }
    cache.group_nic[nic] = nic;
    for (int previous = 0; previous < nic; previous++) {
      if (AreProxyNicsCompatible(nic, previous)) {
        cache.group_nic[nic] = cache.group_nic[previous];
        break;
      }
    }
  }

  for (int domain = 0; domain < view.ipc_domain_count; domain++) {
    const int first = view.domain_first(domain);
    cache.ends[domain] = view.domain_end(domain);
    for (int nic = 0; nic < nics; nic++) {
      cache.counts[domain * nics + nic] = CountProxyDataEndpoints(
          view.team, first, cache.ends[domain], nic, -1);
    }
  }

  for (int group = 0; group < nics; group++) {
    if (cache.group_nic[group] != group) {
      continue;
    }

    int common = constmem.num_pes;
    for (int domain = 0; domain < view.ipc_domain_count; domain++) {
      int count = 0;
      for (int nic = 0; nic < nics; nic++) {
        if (cache.group_nic[nic] == group) {
          count += cache.counts[domain * nics + nic];
        }
      }
      common = min(common, count);
    }
    cache.remaining[group] = common;
    groups += common > 0;
  }

  int lanes = 0;
  for (; lanes < capacity && groups > 0; lanes++) {
    const uint64_t index = operation + lanes;
    int selected = index % groups;
    int group = 0;
    for (; group < nics; group++) {
      if (cache.remaining[group] > 0 && selected-- == 0) {
        break;
      }
    }

    const uint64_t round = index / groups;
    for (int domain = 0; domain < view.ipc_domain_count; domain++) {
      int available_nics = 0;
      for (int nic = 0; nic < nics; nic++) {
        available_nics += cache.group_nic[nic] == group && cache.counts[domain * nics + nic] > 0;
      }

      int turn = round % available_nics;
      for (int nic = 0; nic < nics; nic++) {
        auto &count = cache.counts[domain * nics + nic];
        if (cache.group_nic[nic] != group || count == 0 || turn-- != 0) {
          continue;
        }

        owners[domain * stride + lanes] = SelectProxyDataEndpoint(
            view.team, view.domain_first(domain), cache.ends[domain], nic, -1,
            (round / available_nics) % count, false, owners + domain * stride, lanes);
        count--;
        break;
      }
    }

    if (--cache.remaining[group] == 0) {
      groups--;
    }
  }

  return lanes;
}

/**
 * @brief Plan unique per-domain endpoints sharing one compatibility group per lane.
 *
 * Used endpoints are excluded through the same candidate filter as pair routes.
 * Algorithms retain responsibility for payload partitioning and synchronization.
 */
__device__ inline __noinline__ int BuildProxyDomainLanes(
    const ProxyTeamView &view, uint64_t operation, int capacity, int stride, int *owners) {
  if (view.ipc_domain_count <= 0 || capacity <= 0 || stride < capacity) {
    return 0;
  }

  // Cache counts and update them per lane when the workspace fits.
  const int nics = constmem.world_proxy_nic_name_count;
  if (constmem.world_proxy_nic_layout_available && nics > 0 &&
      nics <= ROCSHMEM_PROXY_DOMAIN_CACHE_CAPACITY / view.ipc_domain_count &&
      nics <= 32767 && constmem.num_pes <= 65535) {
    return BuildCachedProxyDomainLanes(view, operation, capacity, stride, owners);
  }

  // General path: query each lane when the cache cannot be used.
  int lanes = 0;
  for (; lanes < capacity; lanes++) {
    const auto group = SelectCommonProxyNicGroup(view, operation + lanes, owners, stride, lanes);
    if (group.nic < 0) {
      break;
    }

    for (int domain = 0; domain < view.ipc_domain_count; domain++) {
      const int first = view.domain_first(domain);
      owners[domain * stride + lanes] = SelectProxyEndpointInGroup(
          view.team, first, WorldProxyIpcDomainEnd(first), group, owners + domain * stride, lanes);
    }
  }

  return lanes;
}

// One WG reuses this plan for eligibility, tiling and transport.
struct WorldAlltoallPlan {
  bool valid;
  int max_local_pe_count;
  int max_batches;
  int source_proxies[ROCSHMEM_PROXY_MAX_PES];  // Target PE -> local source proxy.
  int scratch_batches[ROCSHMEM_PROXY_MAX_PES];  // Target PE -> scratch batch.
  int domain_batches[ROCSHMEM_PROXY_MAX_IPC_DOMAINS];  // Target domain -> global batch count.
};

/** @brief Select a source proxy while keeping the destination PE fixed. */
__device__ __forceinline__ int SelectWorldAlltoallSourceProxy(
    int source_ipc_domain_first_pe, int target_ipc_domain_first_pe, int target_pe,
    uint64_t operation_index) {
  const auto route = SelectProxyDataRoute(
      {nullptr, source_ipc_domain_first_pe, WorldProxyIpcDomainEnd(source_ipc_domain_first_pe),
       target_ipc_domain_first_pe, WorldProxyIpcDomainEnd(target_ipc_domain_first_pe),
       true, -1, target_pe},
      operation_index);
  return route.source_pe;
}

/** @brief Select each directed target once and derive all scratch batches. */
__device__ inline __noinline__ void BuildWorldAlltoallPlan(
    int local_first, uint64_t operation, WorldAlltoallPlan &plan) {
  plan.valid = false;
  plan.max_local_pe_count = 0;
  plan.max_batches = 0;
  const int domains = constmem.world_proxy_ipc_domain_count;
  if (!constmem.world_proxy_nic_layout_available || domains <= 1 ||
      domains > ROCSHMEM_PROXY_MAX_IPC_DOMAINS || constmem.num_pes > ROCSHMEM_PROXY_MAX_PES ||
      WorldProxyIpcDomainEnd(local_first) <= local_first) {
    return;
  }

  for (int pe = 0; pe < constmem.num_pes; pe++) {
    plan.source_proxies[pe] = -1;
    plan.scratch_batches[pe] = -1;
  }
  for (int domain = 0; domain < domains; domain++) {
    const int first = constmem.world_proxy_ipc_domain_first_pes[domain];
    const int end = WorldProxyIpcDomainEnd(first);
    plan.max_local_pe_count = max(plan.max_local_pe_count, end - first);
    plan.domain_batches[domain] = 0;
  }

  int proxy_load[ROCSHMEM_PROXY_MAX_PES];
  for (int source = 0; source < domains; source++) {
    const int source_first = constmem.world_proxy_ipc_domain_first_pes[source];
    const int source_end = WorldProxyIpcDomainEnd(source_first);
    for (int target = 0; target < domains; target++) {
      if (source == target) {
        continue;
      }

      for (int pe = source_first; pe < source_end; pe++) {
        proxy_load[pe] = 0;
      }
      const int target_first = constmem.world_proxy_ipc_domain_first_pes[target];
      const int target_end = WorldProxyIpcDomainEnd(target_first);
      for (int pe = target_first; pe < target_end; pe++) {
        const int proxy = SelectWorldAlltoallSourceProxy(source_first, target_first, pe, operation);
        if (proxy < source_first || proxy >= source_end) {
          return;
        }

        const int batch = proxy_load[proxy]++;
        plan.domain_batches[target] = max(plan.domain_batches[target], batch + 1);
        plan.max_batches = max(plan.max_batches, batch + 1);
        if (source_first == local_first) {
          plan.source_proxies[pe] = proxy;
          plan.scratch_batches[pe] = batch;
        }
      }
    }
  }

  plan.valid = plan.max_batches > 0;
}

}  // namespace rocshmem::proxy

#if defined(USE_DAS)

namespace rocshmem {

// Shared proxy route plans.
// Share one type-independent plan across fcollect and both reduction outputs.
__device__ inline __noinline__ const int *GDAContext::internal_build_collective_proxy_plan_wg(GDATeam *team_obj,
                                                                                              uint64_t operation) {
  const auto *info = team_obj->tinfo_wrt_world;
  const auto view = proxy::GetProxyTeamView(team_obj, info->pe_start, info->stride, team_obj->num_pes);
  __shared__ int proxies[ROCSHMEM_PROXY_MAX_IPC_DOMAINS];

  if (is_thread_zero_in_block()) {
    if (constmem.world_proxy_nic_layout_available) {
      if (proxy::BuildProxyDomainLanes(view, operation, 1, 1, proxies) == 0) {
        LOGD_ERROR_ABORT("PXN collective has no common compatible-NIC group");
      }
    } else {
      for (int domain = 0; domain < view.ipc_domain_count; domain++) {
        proxies[domain] = proxy::GetValidatedDomainProxy(view.team, view.domain_first(domain));
        if (proxies[domain] < 0) {
          LOGD_ERROR_ABORT("PXN collective has no usable domain data proxy");
        }
      }
    }
  }
  __syncthreads();
  return proxies;
}

// Reuse one owner table for all reduction tiles. Every lane has one distinct
// owner per IPC domain, so simultaneous transfers cannot overwrite scratch.
__device__ inline __noinline__ const int *GDAContext::internal_build_reduction_proxy_plan_wg(GDATeam *team_obj,
                                                                                             uint64_t operation,
                                                                                             int requested_lanes,
                                                                                             int &lanes) {
  const auto *info = team_obj->tinfo_wrt_world;
  const auto view = proxy::GetProxyTeamView(team_obj, info->pe_start, info->stride, team_obj->num_pes);
  __shared__ int owners[ROCSHMEM_PROXY_PLAN_CAPACITY];
  __shared__ int lane_count;

  if (is_thread_zero_in_block()) {
    if (view.ipc_domain_count <= 0 || requested_lanes <= 0 ||
        requested_lanes > ROCSHMEM_PROXY_PLAN_CAPACITY / view.ipc_domain_count) {
      LOGD_ERROR_ABORT("PXN reduction proxy plan exceeds owner capacity");
    }

    if (constmem.world_proxy_nic_layout_available) {
      lane_count = proxy::BuildProxyDomainLanes(view, operation, requested_lanes, requested_lanes, owners);
      if (lane_count == 0) {
        LOGD_ERROR_ABORT("PXN reduction has no common compatible-NIC group");
      }
    } else {
      for (int domain = 0; domain < view.ipc_domain_count; domain++) {
        owners[domain * requested_lanes] = proxy::GetValidatedDomainProxy(view.team, view.domain_first(domain));
        if (owners[domain * requested_lanes] < 0) {
          LOGD_ERROR_ABORT("PXN reduction has no usable domain data proxy");
        }
      }
      lane_count = 1;
    }
  }

  __syncthreads();
  lanes = lane_count;
  return owners;
}

// Build one type-independent plan in LDS for the complete WG operation.
__device__ inline __noinline__ const proxy::WorldAlltoallPlan *
GDAContext::internal_build_alltoall_proxy_plan_wg(GDATeam *team_obj, int source_ipc_domain_first_pe) {
  __shared__ proxy::WorldAlltoallPlan plan;
  if (is_thread_zero_in_block()) {
    proxy::BuildWorldAlltoallPlan(
        source_ipc_domain_first_pe, team_obj->proxy_data_route_sequence + team_obj->pool_index_, plan);
  }
  __syncthreads();

  return &plan;
}

}  // namespace rocshmem

#endif  // USE_DAS

#endif  // LIBRARY_SRC_GDA_PROXY_ROUTE_DEVICE_HPP_
