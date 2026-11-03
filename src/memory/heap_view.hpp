// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef ROCSHMEM_MEMORY_HEAP_VIEW_HPP_
#define ROCSHMEM_MEMORY_HEAP_VIEW_HPP_
#include <cstddef>
#include <cstdint>
namespace rocshmem {
// Both heaps contain independent objects. Preserve the object's heap when
// translating symmetric addresses, regardless of the operation's direction.
__host__ __device__ inline bool heap_contains(const void* ptr, const void* base,
                                             size_t size) {
  auto p = reinterpret_cast<uintptr_t>(ptr);
  auto b = reinterpret_cast<uintptr_t>(base);
  return p >= b && p - b < size;
}
__host__ __device__ inline char* heap_translate(const void* ptr, int from, int to,
    char* const* bases, char* const* hdp_bases, size_t hdp_size) {
  auto selected = hdp_bases && heap_contains(ptr, hdp_bases[from], hdp_size)
      ? hdp_bases : bases;
  auto offset = reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(selected[from]);
  return reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(selected[to]) + offset);
}
}  // namespace rocshmem
#endif
