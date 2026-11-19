// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_SRC_GDA_PROXY_CANDIDATE_HOST_HPP_
#define LIBRARY_SRC_GDA_PROXY_CANDIDATE_HOST_HPP_

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gda/proxy/network_plane.hpp"

namespace rocshmem::proxy {

// Host-side advertisement of an actual selected NIC.
struct ProxyNicCandidate {
  int proxy_pe;             // PE that owns this NIC and can act as its proxy.
  int ipc_domain_first_pe;  // World PE of this IPC domain's first member.
  std::string nic_name;
  int nic_priority;  // Position in select_nics() output; zero is preferred.
};

/**
 * @brief Build this PE's proxy candidates from the NICs selected by GDA.
 *
 * @param selected_nics NIC names in the priority order returned by select_nics().
 * @param my_pe This PE's world rank.
 * @param ipc_domain_first_pe First world PE in this PE's IPC domain.
 * @return One candidate for each selected NIC.
 */
inline std::vector<ProxyNicCandidate> BuildLocalNicCandidates(const std::vector<std::string> &selected_nics,
                                                              int my_pe, int ipc_domain_first_pe) {
  std::vector<ProxyNicCandidate> candidates;

  for (size_t nic_index = 0; nic_index < selected_nics.size(); nic_index++) {
    // Use the same order as select_nics(): it is the actual QP/NIC set and
    // already reflects OAM auto-topology for the default single-NIC path.
    candidates.push_back({my_pe, ipc_domain_first_pe, selected_nics[nic_index], static_cast<int>(nic_index)});
  }

  return candidates;
}

// Exchanged with the existing heap-rkey bootstrap record, before IPC discovery.
struct ProxyExchangeInfo {
  int candidate_count{0};
  int plane_config_size{0};  // -2: PXN disabled; -1: invalid local configuration.
};

inline ProxyExchangeInfo PrepareProxyExchange(const std::vector<std::string> &nics, const char *value,
                                              bool pxn_enabled, NetworkPlanePolicy &planes) {
  planes = {};
  if (!pxn_enabled) {
    return {0, -2};
  }

  ProxyExchangeInfo metadata;
  if (nics.empty() || nics.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    metadata.candidate_count = -1;
  } else {
    metadata.candidate_count = static_cast<int>(nics.size());
    for (const auto &nic : nics) {
      if (nic.empty() || nic.size() >= 64) {
        metadata.candidate_count = -1;
      }
    }
  }

  if (value != nullptr) {
    try {
      planes = ParseNetworkPlanes(value);
      metadata.plane_config_size = planes.canonical.size() <= 16384
                                       ? static_cast<int>(planes.canonical.size()) : -1;
    } catch (const std::invalid_argument &) {
      metadata.plane_config_size = -1;
    }
  }

  return metadata;
}

inline void ValidateProxyExchangeInfo(
    const std::vector<ProxyExchangeInfo> &metadata, int plane_config_size) {
  for (const auto &peer : metadata) {
    if (peer.plane_config_size == -1 || peer.plane_config_size != plane_config_size) {
      throw std::invalid_argument("invalid or inconsistent PXN/NET_PLANE configuration across PEs");
    }
    if (peer.candidate_count < 0 || (plane_config_size >= 0 && peer.candidate_count == 0)) {
      throw std::invalid_argument("invalid proxy NIC advertisement");
    }
  }
}

/**
 * @brief Exchange NIC candidates and plane contents in one post-IPC collective.
 *
 * Counts and parse status have already been exchanged with heap rkeys.
 * No PE enters this collective with an unvalidated local configuration.
 */
template <typename AllGather>
std::vector<ProxyNicCandidate> GatherProxyCandidates(
    const std::vector<ProxyNicCandidate> &local_candidates, const NetworkPlanePolicy &planes,
    const std::vector<ProxyExchangeInfo> &metadata, int my_pe, AllGather all_gather) {
  if (metadata.empty() || my_pe < 0 || static_cast<size_t>(my_pe) >= metadata.size()) {
    throw std::invalid_argument("invalid proxy discovery PE");
  }
  const int config_size = planes.configured ? static_cast<int>(planes.canonical.size()) : 0;
  ValidateProxyExchangeInfo(metadata, config_size);
  if (local_candidates.size() != static_cast<size_t>(metadata[my_pe].candidate_count)) {
    throw std::invalid_argument("proxy candidate count changed after rkey exchange");
  }

  struct ProxyNicCandidateWire {
    int proxy_pe;
    int ipc_domain_first_pe;
    int nic_priority;
    char nic_name[64];
  };

  int max_candidates = 0;
  for (const auto &peer : metadata) {
    max_candidates = std::max(max_candidates, peer.candidate_count);
  }
  const size_t config_offset = static_cast<size_t>(max_candidates) * sizeof(ProxyNicCandidateWire);
  const size_t width = config_offset + static_cast<size_t>(config_size) + 1;
  if (width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      metadata.size() > std::numeric_limits<size_t>::max() / width) {
    throw std::overflow_error("proxy discovery exchange exceeds capacity");
  }

  std::vector<char> buffer(width * metadata.size(), 0);
  char *local = buffer.data() + width * my_pe;
  for (size_t index = 0; index < local_candidates.size(); index++) {
    const auto &candidate = local_candidates[index];
    ProxyNicCandidateWire wire{};
    if (candidate.nic_name.size() >= sizeof(wire.nic_name)) {
      throw std::invalid_argument("NIC name exceeds proxy wire capacity");
    }
    wire.proxy_pe = candidate.proxy_pe;
    wire.ipc_domain_first_pe = candidate.ipc_domain_first_pe;
    wire.nic_priority = candidate.nic_priority;
    std::memcpy(wire.nic_name, candidate.nic_name.c_str(), candidate.nic_name.size() + 1);
    std::memcpy(local + index * sizeof(wire), &wire, sizeof(wire));
  }
  std::memcpy(local + config_offset, planes.canonical.c_str(), config_size + 1);
  all_gather(buffer.data(), width);

  std::vector<ProxyNicCandidate> candidates;
  for (size_t pe = 0; pe < metadata.size(); pe++) {
    const char *record = buffer.data() + pe * width;
    if (std::memcmp(record + config_offset, planes.canonical.c_str(), config_size + 1) != 0) {
      throw std::invalid_argument("NET_PLANE differs between PEs");
    }
    for (int index = 0; index < metadata[pe].candidate_count; index++) {
      ProxyNicCandidateWire wire;
      std::memcpy(&wire, record + index * sizeof(wire), sizeof(wire));
      if (wire.proxy_pe != static_cast<int>(pe) || wire.ipc_domain_first_pe < 0 ||
          static_cast<size_t>(wire.ipc_domain_first_pe) >= metadata.size() ||
          std::memchr(wire.nic_name, 0, sizeof(wire.nic_name)) == nullptr) {
        throw std::runtime_error("invalid proxy discovery record");
      }
      candidates.push_back({wire.proxy_pe, wire.ipc_domain_first_pe, wire.nic_name, wire.nic_priority});
    }
  }

  return candidates;
}

}  // namespace rocshmem::proxy

#endif  // LIBRARY_SRC_GDA_PROXY_CANDIDATE_HOST_HPP_
