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

// Validate parse status before exchanging contents, so no rank leaves peers
// inside the next collective when its local configuration is invalid.
template <typename AllGather>
NetworkPlanePolicy GatherNetworkPlanes(const char *value, bool pxn_enabled, int my_pe, int num_pes,
                                       AllGather all_gather) {
  NetworkPlanePolicy policy;
  int config_size = pxn_enabled ? 0 : -2;
  if (pxn_enabled && value != nullptr) {
    try {
      policy = ParseNetworkPlanes(value);
      config_size = policy.canonical.size() <= 16384 ? static_cast<int>(policy.canonical.size()) : -1;
    } catch (const std::invalid_argument&) {
      config_size = -1;
    }
  }
  std::vector<int> sizes(num_pes, 0);
  sizes[my_pe] = config_size;
  all_gather(sizes.data(), sizeof(int));
  for (int size : sizes) {
    if (size == -1 || size != config_size) {
      throw std::invalid_argument("invalid or inconsistent PXN/NET_PLANE configuration across PEs");
    }
  }
  if (config_size > 0) {
    const size_t width = static_cast<size_t>(config_size) + 1;
    std::vector<char> configs(width * num_pes, 0);
    std::memcpy(configs.data() + width * my_pe, policy.canonical.c_str(), width);
    all_gather(configs.data(), width);
    for (int pe = 0; pe < num_pes; pe++) {
      if (std::memcmp(configs.data() + width * pe, policy.canonical.c_str(), width) != 0) {
        throw std::invalid_argument("NET_PLANE differs between PEs");
      }
    }
  }
  return policy;
}

// Keep environment/logging dependencies at the backend boundary. The callback
// receives complete messages, so parsing remains independently testable.
template <typename AllGather, typename LogInfo>
NetworkPlanePolicy LoadNetworkPlanes(const char *value, bool pxn_enabled, int my_pe, int num_pes,
                                     AllGather all_gather, LogInfo log_info) {
  auto planes = GatherNetworkPlanes(value, pxn_enabled, my_pe, num_pes, all_gather);
  if (my_pe == 0) {
    const std::string policy = !pxn_enabled ? "unrestricted" : (planes.configured ? "same-plane" : "same-name");
    const std::string message = "Proxy NIC policy=" + policy + "; NET_PLANE=" +
                                (planes.configured ? planes.canonical : "unset");
    log_info(message.c_str());
    if (!pxn_enabled && value != nullptr) {
      log_info("ROCSHMEM_NET_PLANE ignored because PXN is disabled");
    }
  }
  return planes;
}

}  // namespace rocshmem::proxy
