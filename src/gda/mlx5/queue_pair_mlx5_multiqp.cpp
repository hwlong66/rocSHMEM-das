// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "gda/queue_pair.hpp"

namespace rocshmem {

__device__ void QueuePair::mlx5_post_wqe_rma_dp_single_lane(
    int32_t size, uintptr_t laddr, uintptr_t raddr, uint8_t opcode) {
  // Reuse normal queue-space and completion handling, with keys selected
  // from the actual source/destination heap (or registered local buffer).
  uint32_t local_key = size <= inline_threshold && opcode == gda_op_rdma_write
      ? 0 : get_lkey(laddr);
  mlx5_post_wqe_rma_single(size, laddr, local_key, raddr, get_rkey(raddr), opcode, true);
}

__device__ void QueuePair::mlx5_post_wqe_amo_dp_single_lane(
    int32_t size, uintptr_t raddr, uint8_t opcode, int64_t atomic_data,
    int64_t atomic_cmp, bool fetching) {
  (void)size;
  mlx5_post_wqe_amo_single(raddr, get_rkey(raddr), opcode, atomic_data,
                            atomic_cmp, fetching, false);
}

__device__ void QueuePair::mlx5_quiet_dp_single_lane() {
  mlx5_quiet_single();
}

}  // namespace rocshmem
