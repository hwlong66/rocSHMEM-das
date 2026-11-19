// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocshmem::proxy {

// Host-owned compatibility policy. An empty map preserves same-name routing.
// A configured plane is a complete whitelist, not a connectivity probe.
struct NetworkPlanePolicy {
  bool configured{false};
  std::map<std::string, int> plane_by_nic;
  std::string canonical;

  std::string group(const std::string &nic) const {
    if (!configured) {
      return nic;
    }
    const auto it = plane_by_nic.find(nic);
    return it == plane_by_nic.end() ? std::string{} : std::to_string(it->second);
  }

  bool allows(const std::string &source, const std::string &target) const {
    const auto source_group = group(source);
    return !source_group.empty() && source_group == group(target);
  }
};

inline NetworkPlanePolicy ParseNetworkPlanes(const std::string &value) {
  NetworkPlanePolicy policy;
  policy.configured = true;
  std::vector<std::vector<std::string>> planes;
  std::set<std::string> names;
  size_t begin = 0;

  do {
    const size_t end = value.find('|', begin);
    const std::string plane = value.substr(begin, end == std::string::npos ? end : end - begin);
    std::vector<std::string> members;
    size_t member_begin = 0;
    do {
      const size_t member_end = plane.find(',', member_begin);
      std::string nic = plane.substr(member_begin, member_end == std::string::npos ? member_end : member_end - member_begin);
      const auto first = nic.find_first_not_of(" \t\r\n");
      const auto last = nic.find_last_not_of(" \t\r\n");
      nic = first == std::string::npos ? "" : nic.substr(first, last - first + 1);
      if (nic.empty() || nic.size() >= 64 || nic.find_first_of(" \t\r\n") != std::string::npos ||
          !names.insert(nic).second) {
        throw std::invalid_argument("NET_PLANE contains an empty, repeated or invalid NIC name");
      }
      members.push_back(nic);
      if (member_end == std::string::npos) {
        break;
      }
      member_begin = member_end + 1;
    } while (true);
    std::sort(members.begin(), members.end());
    planes.push_back(std::move(members));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  } while (true);

  std::sort(planes.begin(), planes.end());
  for (size_t id = 0; id < planes.size(); id++) {
    if (id) {
      policy.canonical += '|';
    }
    for (size_t member = 0; member < planes[id].size(); member++) {
      const auto &nic = planes[id][member];
      policy.plane_by_nic[nic] = static_cast<int>(id);
      policy.canonical += (member ? "," : "") + nic;
    }
  }
  return policy;
}

// Configuration parsing is local. Discovery validates status and canonical
// contents collectively alongside NIC candidates.
template <typename LogInfo>
void LogNetworkPlanePolicy(const NetworkPlanePolicy &planes, bool pxn_enabled,
                           bool explicitly_configured, LogInfo log_info) {
  const std::string policy = !pxn_enabled ? "unrestricted" : (planes.configured ? "same-plane" : "same-name");
  const std::string message = "Proxy NIC policy=" + policy + "; NET_PLANE=" +
                              (planes.configured ? planes.canonical : "unset");
  log_info(message.c_str());
  if (!pxn_enabled && explicitly_configured) {
    log_info("ROCSHMEM_NET_PLANE ignored because PXN is disabled");
  }
}

}  // namespace rocshmem::proxy
