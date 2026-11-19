// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_PROXY_ROUTE_PLAN_HOST_HPP_
#define LIBRARY_SRC_GDA_PROXY_ROUTE_PLAN_HOST_HPP_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "gda/proxy/network_plane.hpp"

namespace rocshmem::proxy {

// A PE may advertise more than one selected NIC.
struct ProxyNicCandidate {
  int proxy_pe;             // PE that owns this NIC and can act as its proxy.
  int ipc_domain_first_pe;  // World PE of this IPC domain's first member.
  std::string nic_name;
  int nic_priority;  // Position in select_nics() output; zero is preferred.
};

struct ProxyRoute {
  int source_proxy_pe = -1;  // Sender IPC domain's selected NIC owner.
  int target_proxy_pe = -1;  // Receiver IPC domain's compatible NIC owner.
  std::string nic_name;  // Source NIC; retained for existing diagnostics.
  std::string target_nic_name;

  explicit operator bool() const { return source_proxy_pe >= 0; }
};

/**
 * @brief Host-owned PXN discovery result shared by backend setup stages.
 *
 * All layouts derive from the same globally exchanged candidate set.
 */
struct ProxyRoutingState {
  std::vector<ProxyNicCandidate> candidates;
  NetworkPlanePolicy network_planes;
  std::vector<int> nic_plane_ids;

  bool world_sync_available = false;
  bool world_aligned_data_available = false;
  std::vector<int> world_ipc_domain_first_pes;
  std::vector<int> world_pe_nic_name_ids;
  int world_nic_name_count = 0;

  bool has_world_nic_layout() const { return !world_pe_nic_name_ids.empty(); }
};

}  // namespace rocshmem::proxy

#endif  // LIBRARY_SRC_GDA_PROXY_ROUTE_PLAN_HOST_HPP_
