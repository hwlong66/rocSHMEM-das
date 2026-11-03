// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// c++ -std=c++17 -D__host__= -D__device__= -Isrc tests/standalone/heap_view_test.cpp -o heap_view_test
#include "memory/heap_view.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>

int main() {
  // Deliberately different gaps and orderings: no common XDP->HDP delta
  // may be assumed across PEs (or across IPC imports).
  char storage[16384]{};
  char* xdp[] = {storage + 1024, storage + 8192, storage + 4096};
  char* hdp[] = {storage + 6144, storage + 256, storage + 12288};
  constexpr size_t size = 512;
  for (int from = 0; from < 3; ++from) {
    for (int to = 0; to < 3; ++to) {
      for (size_t off : {size_t(0), size_t(7), size - 1}) {
        assert(rocshmem::heap_translate(xdp[from] + off, from, to, xdp, hdp, size) == xdp[to] + off);
        assert(rocshmem::heap_translate(hdp[from] + off, from, to, xdp, hdp, size) == hdp[to] + off);
        assert(rocshmem::heap_translate(xdp[from] + off, from, to, xdp, nullptr, 0) == xdp[to] + off);
      }
    }
    assert(rocshmem::heap_contains(hdp[from], hdp[from], size));
    assert(rocshmem::heap_contains(hdp[from] + size - 1, hdp[from], size));
    assert(!rocshmem::heap_contains(hdp[from] + size, hdp[from], size));
    assert(!rocshmem::heap_contains(hdp[from] - 1, hdp[from], size));
    assert(!rocshmem::heap_contains(hdp[from], hdp[from], 0));
  }
  std::puts("heap_view: PASS (independent heaps, peer permutations, boundaries, fallback)");
}
