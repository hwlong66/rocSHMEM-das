// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include <rocshmem/rocshmem.hpp>
#include "util.h"
#include <cstring>
#include <cstdlib>
#include <vector>
using namespace rocshmem;

// One block deliberately exercises both default and explicitly created contexts.
__global__ void transfer(int* dst, const int* src, size_t n, uint64_t* sig,
                        int pe, int mode, bool user_ctx, bool dp) {
  __shared__ rocshmem_ctx_t ctx;
  if (user_ctx) {
    if (rocshmem_wg_ctx_create(0, &ctx)) abort();
  } else if (threadIdx.x == 0) {
    ctx = ROCSHMEM_CTX_DEFAULT;
  }
  __syncthreads();
  const size_t bytes = n * sizeof(int);
  switch (mode) {
    case 0: if (!threadIdx.x) rocshmem_ctx_putmem(ctx, dst, src, bytes, pe); break;
    case 1: if (!threadIdx.x) rocshmem_ctx_putmem_nbi(ctx, dst, src, bytes, pe); break;
    case 2: rocshmem_ctx_putmem_wave(ctx, dst, src, bytes, pe); break;
    case 3: rocshmem_ctx_putmem_nbi_wave(ctx, dst, src, bytes, pe); break;
    case 4: rocshmem_ctx_putmem_wg(ctx, dst, src, bytes, pe); break;
    case 5: rocshmem_ctx_putmem_nbi_wg(ctx, dst, src, bytes, pe); break;
    case 6: if (!threadIdx.x) rocshmem_ctx_getmem(ctx, dst, src, bytes, pe); break;
    case 7: if (!threadIdx.x) rocshmem_ctx_getmem_nbi(ctx, dst, src, bytes, pe); break;
    case 8: rocshmem_ctx_getmem_wave(ctx, dst, src, bytes, pe); break;
    case 9: rocshmem_ctx_getmem_nbi_wave(ctx, dst, src, bytes, pe); break;
    case 10: rocshmem_ctx_getmem_wg(ctx, dst, src, bytes, pe); break;
    case 11: rocshmem_ctx_getmem_nbi_wg(ctx, dst, src, bytes, pe); break;
    case 12: if (!threadIdx.x) rocshmem_ctx_putmem_signal(ctx, dst, src, bytes, sig, 7, ROCSHMEM_SIGNAL_SET, pe); break;
    case 13: if (!threadIdx.x) rocshmem_ctx_putmem_signal_nbi(ctx, dst, src, bytes, sig, 7, ROCSHMEM_SIGNAL_SET, pe); break;
    case 14: rocshmem_ctx_putmem_signal_wave(ctx, dst, src, bytes, sig, 7, ROCSHMEM_SIGNAL_SET, pe); break;
    case 15: rocshmem_ctx_putmem_signal_nbi_wave(ctx, dst, src, bytes, sig, 7, ROCSHMEM_SIGNAL_SET, pe); break;
    case 16: rocshmem_ctx_putmem_signal_wg(ctx, dst, src, bytes, sig, 7, ROCSHMEM_SIGNAL_SET, pe); break;
    case 17: rocshmem_ctx_putmem_signal_nbi_wg(ctx, dst, src, bytes, sig, 7, ROCSHMEM_SIGNAL_SET, pe); break;
    case 18:
      if (!threadIdx.x) {
        rocshmem_ctx_ulonglong_atomic_fetch_add(ctx, reinterpret_cast<unsigned long long*>(sig), 7, pe);
      }
      break;
    case 19: rocshmem_ctx_int_broadcast_wg(ctx, ROCSHMEM_TEAM_WORLD, dst, src, n, 0); break;
    case 20: rocshmem_ctx_int_sum_reduce_wg(ctx, ROCSHMEM_TEAM_WORLD, dst, src, n); break;
    case 21: rocshmem_ctx_int_alltoall_wg(ctx, ROCSHMEM_TEAM_WORLD, dst, src, n); break;
    case 22:
      if (dp) rocshmem_ctx_schar_put_nbi_wave_dp(ctx, reinterpret_cast<signed char*>(dst),
          reinterpret_cast<const signed char*>(src), bytes, pe, pe);
      break;
    case 23:
      if (dp && !threadIdx.x) rocshmem_ctx_longlong_atomic_add_dp(ctx, reinterpret_cast<long long*>(sig), 7, pe, pe);
      break;
  }
  __syncthreads();
  if (!threadIdx.x) rocshmem_ctx_quiet(ctx);
  __syncthreads();
  if (user_ctx) rocshmem_wg_ctx_destroy(&ctx);
}

int main(int argc, char** argv) {
  bool dp = false, host = false, verbose = false;
  int mode_filter = -1;
  for (int i = 1; i < argc; ++i) {
    dp |= !std::strcmp(argv[i], "--dp");
    host |= !std::strcmp(argv[i], "--host");
    verbose |= !std::strcmp(argv[i], "--verbose");
    if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) mode_filter = std::atoi(argv[++i]);
  }
  if (mode_filter < -1 || mode_filter > 23 || (mode_filter > 21 && !dp)) {
    std::fprintf(stderr, "Invalid --mode (22/23 require --dp)\n");
    return 2;
  }
  CHECK_HIP(hipSetDevice(get_launcher_local_rank()));
  rocshmem_init();
  const int rank = rocshmem_my_pe(), pes = rocshmem_n_pes();
  const int next = (rank + 1) % pes, prev = (rank + pes - 1) % pes;
  // Larger than the inline threshold, and distinct offsets within each heap.
  constexpr size_t n = 4096, pad = 17;
  const size_t count = n * pes + pad;
  int* src[2] = {static_cast<int*>(rocshmem_malloc(count * sizeof(int))),
                 static_cast<int*>(rocshmem_align_hdp(256, count * sizeof(int)))};
  int* dst[2] = {static_cast<int*>(rocshmem_malloc(count * sizeof(int))),
                 static_cast<int*>(rocshmem_malloc_hdp(count * sizeof(int)))};
  uint64_t* sig[2] = {static_cast<uint64_t*>(rocshmem_malloc(sizeof(uint64_t))),
                     static_cast<uint64_t*>(rocshmem_malloc_hdp(sizeof(uint64_t)))};
  for (int d = 0; d < 2; ++d) if (!src[d] || !dst[d] || !sig[d]) rocshmem_global_exit(2);
  std::vector<int> input(count), result(n * pes);
  for (size_t i = 0; i < count; ++i) input[i] = rank * 100000 + i;
  for (auto ptr : src) CHECK_HIP(hipMemcpy(ptr, input.data(), count * sizeof(int), hipMemcpyHostToDevice));
  bool pass = true;
  for (int source_domain = 0; source_domain < 2; ++source_domain) {
    for (int dest_domain = 0; dest_domain < 2; ++dest_domain) {
      for (int user = 0; user < 2; ++user) {
        for (int mode = 0; mode <= (dp ? 23 : 21); ++mode) {
          if (mode_filter >= 0 && mode != mode_filter) continue;
          if (verbose && rank == 0) {
            std::printf("BEGIN src=%d dst=%d ctx=%d mode=%d\n", source_domain, dest_domain, user, mode);
            std::fflush(stdout);
          }
          const size_t elems = mode == 21 ? n * pes : n;
          CHECK_HIP(hipMemset(dst[dest_domain], 0, count * sizeof(int)));
          CHECK_HIP(hipMemset(sig[dest_domain], 0, sizeof(uint64_t)));
          rocshmem_barrier_all();
          transfer<<<1, 64>>>(dst[dest_domain] + pad, src[source_domain] + pad,
              n, sig[dest_domain], next, mode, user, dp);
          CHECK_HIP(hipGetLastError());
          CHECK_HIP(hipDeviceSynchronize());
          rocshmem_barrier_all();
          CHECK_HIP(hipMemcpy(result.data(), dst[dest_domain] + pad, elems * sizeof(int), hipMemcpyDeviceToHost));
          uint64_t signal = 0;
          CHECK_HIP(hipMemcpy(&signal, sig[dest_domain], sizeof(signal), hipMemcpyDeviceToHost));
          if ((mode >= 12 && mode <= 18) || mode == 23) pass &= signal == 7;
          if (mode != 18 && mode != 23) {
            for (size_t i = 0; i < elems; ++i) {
              int expected = prev * 100000 + pad + i;
              if (mode >= 6 && mode <= 11) expected = next * 100000 + pad + i;
              if (mode == 19) expected = pad + i;
              if (mode == 20) expected = 100000 * pes * (pes - 1) / 2 + pes * (pad + i);
              if (mode == 21) expected = (i / n) * 100000 + pad + rank * n + i % n;
              if (result[i] != expected) {
                std::fprintf(stderr, "PE %d src=%d dst=%d ctx=%d mode=%d i=%zu got=%d expected=%d\n",
                    rank, source_domain, dest_domain, user, mode, i, result[i], expected);
                pass = false;
                break;
              }
            }
          }
        }
      }
      // Optional MPI host path: select the remote window from the object's heap.
      if (host) {
        rocshmem_putmem_nbi(dst[dest_domain] + pad, src[source_domain] + pad, n * sizeof(int), next);
        rocshmem_quiet();
        rocshmem_barrier_all();
        CHECK_HIP(hipMemcpy(result.data(), dst[dest_domain] + pad, n * sizeof(int), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) pass &= result[i] == prev * 100000 + pad + i;
        rocshmem_getmem_nbi(dst[dest_domain] + pad, src[source_domain] + pad, n * sizeof(int), next);
        rocshmem_quiet();
        CHECK_HIP(hipMemcpy(result.data(), dst[dest_domain] + pad, n * sizeof(int), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) pass &= result[i] == next * 100000 + pad + i;
        rocshmem_barrier_all();
      }
    }
  }
  std::printf("PE %d xdp_split: %s\n", rank, pass ? "PASS" : "FAIL");
  if (!pass) rocshmem_global_exit(1);
  for (int d = 0; d < 2; ++d) {
    rocshmem_free(src[d]); rocshmem_free(dst[d]); rocshmem_free(sig[d]);
  }
  rocshmem_finalize();
  return 0;
}
