// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_PROXY_LAYOUT_HOST_HPP_
#define LIBRARY_SRC_GDA_PROXY_LAYOUT_HOST_HPP_

#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gda/proxy/candidate_host.hpp"
#include "constants.hpp"
#include "gda/proxy/network_plane.hpp"

namespace rocshmem::proxy {

/*
 * An IPC domain is a node-local set of PEs that can communicate directly through IPC.
 * Domain identifiers are world-PE based and do not represent physical host IDs.
 */

struct DomainProxyLayout {
  std::string nic_name;
  std::map<int, int> proxy_pe_by_ipc_domain_first_pe;
  std::map<int, std::string> nic_by_ipc_domain_first_pe;

  explicit operator bool() const { return !nic_name.empty() && !proxy_pe_by_ipc_domain_first_pe.empty(); }
};

struct WorldProxySyncLayout {
  std::vector<int> ipc_domain_first_pes;
  std::vector<int> proxy_pes;

  explicit operator bool() const { return ipc_domain_first_pes.size() > 1; }
};

/**
 * @brief Validate the selected per-domain proxies used by WORLD synchronization.
 *
 * @param candidates Globally exchanged NIC candidates.
 * @param sync_layout Selected stable endpoint and NIC for each participating domain.
 * @param num_pes Number of world PEs.
 * @param max_sync_slots Available world-rank synchronization slots.
 * @return Ordered IPC domain first PEs, or an invalid layout.
 */
inline WorldProxySyncLayout BuildWorldProxySyncLayout(const std::vector<ProxyNicCandidate> &candidates,
                                                      const DomainProxyLayout &sync_layout,
                                                      int num_pes, int max_sync_slots, const NetworkPlanePolicy &policy = {}) {
  WorldProxySyncLayout layout;
  if (num_pes < 2 || num_pes > max_sync_slots || num_pes > ROCSHMEM_PROXY_MAX_PES) {
    return {};
  }

  // Key is the first world PE in an IPC domain, not a hardware node ID.
  std::map<int, std::set<int>> ipc_domain_members;
  for (const auto &candidate : candidates) {
    if (candidate.ipc_domain_first_pe < 0 || candidate.proxy_pe < 0 || candidate.proxy_pe >= num_pes ||
        candidate.ipc_domain_first_pe > candidate.proxy_pe) {
      return {};
    }
    ipc_domain_members[candidate.ipc_domain_first_pe].insert(candidate.proxy_pe);
  }

  std::string reference_nic;
  int next_pe = 0;

  for (const auto &ipc_domain : ipc_domain_members) {
    const int ipc_domain_first_pe = ipc_domain.first;
    const int local_pe_count = static_cast<int>(ipc_domain.second.size());
    if (ipc_domain_first_pe != next_pe || local_pe_count < 1 || local_pe_count > num_pes - next_pe) {
      return {};
    }

    for (int pe = next_pe; pe < next_pe + local_pe_count; pe++) {
      if (!ipc_domain.second.count(pe)) {
        return {};
      }
    }

    layout.ipc_domain_first_pes.push_back(ipc_domain_first_pe);
    next_pe += local_pe_count;
    const auto proxy = sync_layout.proxy_pe_by_ipc_domain_first_pe.find(ipc_domain_first_pe);
    const auto nic = sync_layout.nic_by_ipc_domain_first_pe.find(ipc_domain_first_pe);
    if (proxy == sync_layout.proxy_pe_by_ipc_domain_first_pe.end() ||
        nic == sync_layout.nic_by_ipc_domain_first_pe.end() || !ipc_domain.second.count(proxy->second)) {
      return {};
    }
    bool advertised = false;
    for (const auto &candidate : candidates) {
      advertised |= candidate.ipc_domain_first_pe == ipc_domain_first_pe &&
                    candidate.proxy_pe == proxy->second && candidate.nic_name == nic->second &&
                    candidate.nic_priority == 0;
    }
    if (reference_nic.empty()) reference_nic = nic->second;
    if (!advertised || !policy.allows(reference_nic, nic->second)) return {};
    layout.proxy_pes.push_back(proxy->second);
  }

  if (next_pe != num_pes || layout.ipc_domain_first_pes.size() < 2 ||
      layout.ipc_domain_first_pes.size() > ROCSHMEM_PROXY_MAX_IPC_DOMAINS ||
      sync_layout.proxy_pe_by_ipc_domain_first_pe.size() != ipc_domain_members.size() ||
      sync_layout.nic_by_ipc_domain_first_pe.size() != ipc_domain_members.size()) {
    return {};
  }

  return layout;
}

/**
 * @brief Check whether equal local PE slots form valid WORLD data routes.
 *
 * @param candidates Globally exchanged NIC candidates.
 * @param num_pes Number of world PEs.
 * @param local_pe_count Number of PEs in each required equal-size IPC domain.
 * @return true when corresponding slots own compatible NICs.
 */
inline bool CanUseAlignedWorldProxyData(const std::vector<ProxyNicCandidate> &candidates, int num_pes,
                                        int local_pe_count, const NetworkPlanePolicy &policy = {}) {
  if (local_pe_count <= 1 || num_pes <= local_pe_count || num_pes % local_pe_count != 0) {
    return false;
  }

  std::map<int, std::set<int>> ipc_domain_members;
  std::map<int, std::string> selected_nics;
  for (const auto &candidate : candidates) {
    ipc_domain_members[candidate.ipc_domain_first_pe].insert(candidate.proxy_pe);
    if (candidate.nic_priority == 0) {
      selected_nics[candidate.proxy_pe] = candidate.nic_name;
    }
  }

  if (static_cast<int>(selected_nics.size()) != num_pes ||
      static_cast<int>(ipc_domain_members.size()) != num_pes / local_pe_count) {
    return false;
  }

  for (int ipc_domain_first_pe = 0; ipc_domain_first_pe < num_pes; ipc_domain_first_pe += local_pe_count) {
    auto ipc_domain = ipc_domain_members.find(ipc_domain_first_pe);
    if (ipc_domain == ipc_domain_members.end() || static_cast<int>(ipc_domain->second.size()) != local_pe_count) {
      return false;
    }

    for (int pe = ipc_domain_first_pe; pe < ipc_domain_first_pe + local_pe_count; pe++) {
      if (!ipc_domain->second.count(pe)) {
        return false;
      }
    }
  }

  for (int pe = local_pe_count; pe < num_pes; pe++) {
    const auto &local_nic = selected_nics[pe];
    const auto &first_ipc_domain_nic = selected_nics[pe % local_pe_count];
    if (local_nic.empty() || !policy.allows(local_nic, first_ipc_domain_nic)) {
      return false;
    }
  }

  return true;
}

struct WorldNicLayout {
  std::vector<std::string> nic_names;
  std::vector<int> pe_nic_name_ids;

  explicit operator bool() const { return !nic_names.empty(); }
};

/**
 * @brief Build deterministic NIC IDs for WORLD multi-lane data routing.
 *
 * This layout accepts exactly one candidate per world PE, all with priority
 * zero. NIC names are sorted so every PE assigns the same ID to a given name.
 *
 * @param candidates Globally exchanged NIC candidates.
 * @param world_pe_count Number of world PEs.
 * @param device_layout_capacity Capacity of the device-side PE/NIC arrays.
 * @return NIC names and one NIC ID per PE, or an invalid layout.
 */
inline WorldNicLayout BuildWorldNicLayout(const std::vector<ProxyNicCandidate> &candidates,
                                          int world_pe_count, int device_layout_capacity) {
  if (world_pe_count < 1 || world_pe_count > device_layout_capacity) {
    return {};
  }

  std::vector<std::string> nic_name_by_pe(world_pe_count);
  std::vector<int> candidate_count_by_pe(world_pe_count, 0);
  std::set<std::string> unique_nic_names;

  for (const auto &candidate : candidates) {
    if (candidate.proxy_pe < 0 || candidate.proxy_pe >= world_pe_count || candidate.nic_priority != 0 ||
        candidate.nic_name.empty()) {
      return {};
    }

    nic_name_by_pe[candidate.proxy_pe] = candidate.nic_name;
    candidate_count_by_pe[candidate.proxy_pe]++;
    unique_nic_names.insert(candidate.nic_name);
  }

  if (unique_nic_names.size() > ROCSHMEM_PROXY_MAX_NIC_NAMES) {
    return {};
  }

  // The PE count is already capacity-checked, and there is one NIC per PE.
  for (int count : candidate_count_by_pe) {
    if (count != 1) {
      return {};
    }
  }

  WorldNicLayout layout;
  layout.nic_names.assign(unique_nic_names.begin(), unique_nic_names.end());
  std::map<std::string, int> nic_id_by_name;

  for (size_t i = 0; i < layout.nic_names.size(); i++) {
    nic_id_by_name[layout.nic_names[i]] = static_cast<int>(i);
  }

  for (const auto &nic_name : nic_name_by_pe) {
    layout.pe_nic_name_ids.push_back(nic_id_by_name[nic_name]);
  }

  return layout;
}

// Each domain uses one stable endpoint in a compatibility group shared by
// all participating domains. Protocol constraints only change eligibility.
enum class ProxyEndpointConstraint {
  TeamMembers,  // Any team member in the domain may act as proxy.  // Any team member in the domain may act as the sync proxy.
  DomainFirst,  // Only the domain-first PE using its preferred NIC may proxy.  // Only the domain's first PE, using its preferred NIC, may proxy.
};


/**
 * @brief Test whether a world PE belongs to a strided team.
 *
 * @param pe_start First world PE in the team.
 * @param stride World-rank stride between team members.
 * @param team_size Number of team members.
 * @param world_pe World PE to test.
 * @return true when world_pe is a member of the team.
 */
inline bool TeamContainsPe(int pe_start, int stride, int team_size, int world_pe) {
  if (stride <= 0 || team_size <= 0 || world_pe < pe_start) {
    return false;
  }

  const int distance = world_pe - pe_start;
  return distance % stride == 0 && distance / stride < team_size;
}

/**
 * @brief Build fixed in-team proxies for synchronization and data-route fallback.
 *
 * @param candidates Globally exchanged NIC candidates.
 * @param pe_start First world PE in the team.
 * @param stride World-rank stride between team members.
 * @param team_size Number of team members.
 * @return The lowest-priority-sum compatible layout, or an invalid layout.
 */
inline DomainProxyLayout BuildFixedDomainProxyLayout(const std::vector<ProxyNicCandidate> &candidates,
                                                int pe_start, int stride, int team_size,
                                                const NetworkPlanePolicy &policy = {},
                                                ProxyEndpointConstraint endpoints = ProxyEndpointConstraint::TeamMembers) {
  if (pe_start < 0 || stride <= 0 || team_size <= 0) {
    return {};
  }

  std::set<int> participating_ipc_domains;
  std::set<std::string> nic_names;
  for (const auto &candidate : candidates) {
    if (!TeamContainsPe(pe_start, stride, team_size, candidate.proxy_pe)) {
      continue;
    }

    participating_ipc_domains.insert(candidate.ipc_domain_first_pe);
    if (endpoints == ProxyEndpointConstraint::DomainFirst &&
        (candidate.proxy_pe != candidate.ipc_domain_first_pe || candidate.nic_priority != 0)) {
      continue;
    }
    if (!candidate.nic_name.empty() && candidate.nic_priority >= 0) {
      const auto group = policy.group(candidate.nic_name);
      if (!group.empty()) {
        nic_names.insert(group);
      }
    }
  }

  if (participating_ipc_domains.empty()) {
    return {};
  }

  DomainProxyLayout best;
  int best_priority = std::numeric_limits<int>::max();

  for (const auto &nic_name : nic_names) {
    std::map<int, int> proxies;
    std::map<int, std::string> selected_nics;
    int total_priority = 0;
    for (int ipc_domain_first_pe : participating_ipc_domains) {
      const ProxyNicCandidate *selected = nullptr;
      for (const auto &candidate : candidates) {
        if (endpoints == ProxyEndpointConstraint::DomainFirst &&
            (candidate.proxy_pe != ipc_domain_first_pe || candidate.nic_priority != 0)) {
          continue;
        }
        if (candidate.ipc_domain_first_pe != ipc_domain_first_pe || policy.group(candidate.nic_name) != nic_name ||
            candidate.nic_priority < 0 || !TeamContainsPe(pe_start, stride, team_size, candidate.proxy_pe)) {
          continue;
        }

        if (selected == nullptr || candidate.nic_priority < selected->nic_priority ||
            (candidate.nic_priority == selected->nic_priority && candidate.proxy_pe < selected->proxy_pe)) {
          selected = &candidate;
        }
      }

      if (selected == nullptr) {
        proxies.clear();
        break;
      }
      proxies[ipc_domain_first_pe] = selected->proxy_pe;
      selected_nics[ipc_domain_first_pe] = selected->nic_name;
      total_priority += selected->nic_priority;
    }

    if (proxies.size() != participating_ipc_domains.size()) {
      continue;
    }

    if (!best || total_priority < best_priority) {
      best.nic_name = selected_nics.begin()->second;
      best.nic_by_ipc_domain_first_pe = std::move(selected_nics);
      best.proxy_pe_by_ipc_domain_first_pe = std::move(proxies);
      best_priority = total_priority;
    }
  }

  return best;
}

/**
 * @brief Host-owned PXN discovery result shared by backend setup stages.
 *
 * All layouts derive from the same globally exchanged candidate set.
 */
struct ProxyRoutingState {
  std::vector<ProxyNicCandidate> candidates;  // Selected-NIC candidates gathered from all PEs.
  NetworkPlanePolicy network_planes;  // Configured NIC compatibility policy; unset means same-name.
  std::vector<int> nic_plane_ids;  // NIC-name ID -> plane ID; -1 means not listed.

  bool world_sync_available = false;  // Host-validated WORLD synchronization layout is usable.
  bool world_aligned_data_available = false;  // Equal-sized contiguous domains have compatible matching slots.
  std::vector<int> world_sync_proxy_pes;  // Selected synchronization PE per IPC domain.
  std::vector<int> world_ipc_domain_first_pes;  // Sorted first world PE of each IPC domain.
  std::vector<int> world_pe_nic_name_ids;  // World PE -> global NIC-name ID, not a local device index.
  int world_nic_name_count = 0;  // Number of distinct NIC names in the WORLD mapping.

  bool has_world_nic_layout() const { return !world_pe_nic_name_ids.empty(); }
};

}  // namespace rocshmem::proxy

#endif  // LIBRARY_SRC_GDA_PROXY_LAYOUT_HOST_HPP_