// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef LIBRARY_SRC_GDA_PROXY_SYNC_DEVICE_HPP_
#define LIBRARY_SRC_GDA_PROXY_SYNC_DEVICE_HPP_

#include "gda/proxy/route_device.hpp"

namespace rocshmem {

/**
 * @brief Synchronize WORLD PEs through one same-NIC proxy per IPC island.
 *
 * @param pe Calling world PE.
 * @param PE_size WORLD team size.
 * @param pSync Team synchronization storage indexed by world PE.
 * @param wf_info Active workgroup metadata.
 * @param epoch Unique synchronization epoch.
 * @param proxy_qp_row Preferred QP row, or -1 for automatic selection.
 */
__device__ void GDAContext::internal_world_proxy_sync_thread(int pe, [[maybe_unused]] int PE_size, int64_t *pSync,
    ActiveWFInfo &wf_info, int64_t epoch, int proxy_qp_row) {
  const int local_pe_count = constmem.ipc_shm_size;
  // BuildWorldDataLayout enables the NIC layout only when every PE owns one
  // selected NIC, so every QP row on that PE reaches the same fabric. In
  // that validated case concurrent workgroups may use separate QP rows;
  // merged-NIC layouts stay on row zero until an explicit NIC-to-row map is
  // available.
  const int qp_row = !constmem.world_proxy_nic_layout_enabled ? 0
      : proxy_qp_row >= 0 ? proxy_qp_row % num_qps_per_pe
                          : static_cast<int>(blockIdx.x) % num_qps_per_pe;
  const int qp_row_offset = qp_row * constmem.num_pes;

  // Eligibility guarantees that the first PE owns the selected NIC route.
  // Arbitrary-team proxies are not necessarily first PEs.
  const int local_proxy_pe = constmem.ipc_first_pe;
  const int64_t arrived = epoch;

  if (pe != local_proxy_pe) {
    // RDMA sources must live in the registered symmetric heap, not a GPU
    // thread's private stack. The same pSync slot also works for IPC puts.
    pSync[pe] = arrived;
    __threadfence_system();
    internal_putmem(&pSync[pe], &pSync[pe], sizeof(arrived), local_proxy_pe, qp_row_offset + local_proxy_pe,
        wf_info);
    pSync[pe] = ROCSHMEM_SYNC_VALUE;
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, arrived);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
  } else {
    // Stage 1: collect all local participants before a node reports ready.
    for (int local_pe = local_proxy_pe + 1; local_pe < local_proxy_pe + local_pe_count; local_pe++) {
      wait_until(&pSync[local_pe], ROCSHMEM_CMP_EQ, arrived);
      pSync[local_pe] = ROCSHMEM_SYNC_VALUE;
    }
    __threadfence_system();

    if (local_proxy_pe != 0) {
      // Stage 2: only matching-NIC node proxies exchange cross-node signals.
      pSync[local_proxy_pe] = arrived;
      __threadfence_system();
      internal_putmem(&pSync[local_proxy_pe], &pSync[local_proxy_pe], sizeof(arrived), 0, qp_row_offset,
          wf_info);
      pSync[local_proxy_pe] = ROCSHMEM_SYNC_VALUE;
      wait_until(&pSync[0], ROCSHMEM_CMP_EQ, arrived);
      pSync[0] = ROCSHMEM_SYNC_VALUE;
    } else {
      for (int node = 1; node < constmem.world_proxy_node_count; node++) {
        const int remote_proxy_pe = constmem.world_proxy_node_first_pes[node];
        wait_until(&pSync[remote_proxy_pe], ROCSHMEM_CMP_EQ, arrived);
        pSync[remote_proxy_pe] = ROCSHMEM_SYNC_VALUE;
      }
      __threadfence_system();
      pSync[0] = arrived;
      __threadfence_system();
      for (int node = 1; node < constmem.world_proxy_node_count; node++) {
        const int remote_proxy_pe = constmem.world_proxy_node_first_pes[node];
        internal_putmem(&pSync[0], &pSync[0], sizeof(arrived), remote_proxy_pe,
            qp_row_offset + remote_proxy_pe, wf_info);
      }
    }

    pSync[0] = arrived;
    __threadfence_system();
    // Stage 3: release all local participants through IPC.
    for (int local_pe = local_proxy_pe + 1; local_pe < local_proxy_pe + local_pe_count; local_pe++) {
      internal_putmem(&pSync[0], &pSync[0], sizeof(arrived), local_pe, qp_row_offset + local_pe, wf_info);
    }
    pSync[0] = ROCSHMEM_SYNC_VALUE;
  }
  __threadfence_system();
}

__device__ void GDAContext::internal_world_proxy_sync_wg(int pe, [[maybe_unused]] int PE_size, int64_t *pSync,
    ActiveWFInfo &wf_info, int64_t epoch, int proxy_qp_row) {
  __syncthreads();
  if (is_thread_zero_in_block()) {
    internal_world_proxy_sync_thread(pe, PE_size, pSync, wf_info, epoch, proxy_qp_row);
  }
  __threadfence_system();
  __syncthreads();
}

/**
 * @brief Synchronize a partial team through its selected in-team proxies.
 *
 * @param team_obj Partial-team membership and proxy layout.
 * @param pSync Team synchronization storage indexed by world PE.
 * @param wf_info Active workgroup metadata.
 * @param epoch Unique synchronization epoch.
 */
__device__ void GDAContext::internal_team_proxy_sync_thread(GDATeam *team_obj, int64_t *pSync,
    ActiveWFInfo &wf_info, int64_t epoch) {

  int local_node = -1;
  for (int node = 0; node < team_obj->in_team_proxy_node_count; node++) {
    if (team_obj->in_team_proxy_node_first_pes[node] == constmem.ipc_first_pe) {
      local_node = node;
      break;
    }
  }
  assert(local_node >= 0);

  const int my_pe = constmem.my_pe;
  const int local_proxy = team_obj->in_team_proxy_pes[local_node];
  const int root_proxy = team_obj->in_team_proxy_pes[0];
  const int node_first = constmem.ipc_first_pe;
  const int node_end = node_first + constmem.ipc_shm_size;
  const int qp_row = team_obj->pool_index_ % num_qps_per_pe;
  const int qp_offset = qp_row * constmem.num_pes;

  if (my_pe != local_proxy) {
    pSync[my_pe] = epoch;
    __threadfence_system();
    internal_putmem(&pSync[my_pe], &pSync[my_pe], sizeof(epoch), local_proxy, qp_offset + local_proxy,
        wf_info);
    pSync[my_pe] = ROCSHMEM_SYNC_VALUE;
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, epoch);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
  } else {
    for (int rank = 0; rank < team_obj->num_pes; rank++) {
      const int member = team_obj->get_pe_in_world(rank);
      if (member == local_proxy || member < node_first || member >= node_end) {
        continue;
      }
      wait_until(&pSync[member], ROCSHMEM_CMP_EQ, epoch);
      pSync[member] = ROCSHMEM_SYNC_VALUE;
    }
    __threadfence_system();

    if (local_proxy != root_proxy) {
      pSync[local_proxy] = epoch;
      __threadfence_system();
      internal_putmem(&pSync[local_proxy], &pSync[local_proxy], sizeof(epoch), root_proxy,
          qp_offset + root_proxy, wf_info);
      pSync[local_proxy] = ROCSHMEM_SYNC_VALUE;
      wait_until(&pSync[0], ROCSHMEM_CMP_EQ, epoch);
      pSync[0] = ROCSHMEM_SYNC_VALUE;
    } else {
      for (int node = 1; node < team_obj->in_team_proxy_node_count; node++) {
        const int proxy_pe = team_obj->in_team_proxy_pes[node];
        wait_until(&pSync[proxy_pe], ROCSHMEM_CMP_EQ, epoch);
        pSync[proxy_pe] = ROCSHMEM_SYNC_VALUE;
      }
      pSync[0] = epoch;
      __threadfence_system();
      for (int node = 1; node < team_obj->in_team_proxy_node_count; node++) {
        const int proxy_pe = team_obj->in_team_proxy_pes[node];
        internal_putmem(&pSync[0], &pSync[0], sizeof(epoch), proxy_pe, qp_offset + proxy_pe, wf_info);
      }
    }

    pSync[0] = epoch;
    __threadfence_system();
    for (int rank = 0; rank < team_obj->num_pes; rank++) {
      const int member = team_obj->get_pe_in_world(rank);
      if (member == local_proxy || member < node_first || member >= node_end) {
        continue;
      }
      internal_putmem(&pSync[0], &pSync[0], sizeof(epoch), member, qp_offset + member, wf_info);
    }
    pSync[0] = ROCSHMEM_SYNC_VALUE;
  }
  __threadfence_system();
}

__device__ void GDAContext::internal_team_proxy_sync_wg(GDATeam *team_obj, int64_t *pSync,
    ActiveWFInfo &wf_info, int64_t epoch) {
  __syncthreads();
  if (is_thread_zero_in_block()) {
    internal_team_proxy_sync_thread(team_obj, pSync, wf_info, epoch);
  }
  __threadfence_system();
  __syncthreads();
}

/**
 * @brief Dispatch routed-collective synchronization to WORLD or team routing.
 *
 * @param team_obj Team whose route is active.
 * @param pSync Team synchronization storage.
 * @param wf_info Active workgroup metadata.
 * @param epoch Unique synchronization epoch.
 */
__device__ void GDAContext::internal_dispatch_proxy_sync_wg(GDATeam *team_obj, int64_t *pSync,
    ActiveWFInfo &wf_info, int64_t epoch) {
  const auto *team_info = team_obj->tinfo_wrt_world;
  if (team_info->pe_start == 0 && team_info->stride == 1 && team_obj->num_pes == constmem.num_pes) {
    internal_world_proxy_sync_wg(constmem.my_pe, team_obj->num_pes, pSync, wf_info, epoch,
        team_obj->pool_index_ % num_qps_per_pe);
  } else {
    internal_team_proxy_sync_wg(team_obj, pSync, wf_info, epoch);
  }
}

// The same signal core can synchronize a single participating wave without
// imposing a block barrier on other waves in its workgroup.
__device__ void GDAContext::internal_dispatch_proxy_sync_wave(GDATeam *team_obj, int64_t *p_sync,
    ActiveWFInfo &wf_info, int64_t epoch) {
  __threadfence_system();
  __builtin_amdgcn_wave_barrier();
  if (is_thread_zero_in_wave()) {
    const auto *info = team_obj->tinfo_wrt_world;
    if (info->pe_start == 0 && info->stride == 1 && team_obj->num_pes == constmem.num_pes) {
      internal_world_proxy_sync_thread(constmem.my_pe, team_obj->num_pes, p_sync, wf_info, epoch,
          team_obj->pool_index_ % num_qps_per_pe);
    } else {
      internal_team_proxy_sync_thread(team_obj, p_sync, wf_info, epoch);
    }
  }
  __threadfence_system();
  __builtin_amdgcn_wave_barrier();
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_PROXY_SYNC_DEVICE_HPP_
