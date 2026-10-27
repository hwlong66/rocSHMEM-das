/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "ibv_wrapper.hpp"
#include "envvar.hpp"
#include "log.hpp"
#include "util.hpp"
#include "memory/default_allocator.hpp"
#include "ibv_abi.hpp"

#include "rocshmem/rocshmem.hpp"
#include <dlfcn.h>
#include <sys/utsname.h>
#include <cstring>
#include <unistd.h> // close(fd)

namespace rocshmem {

IBVWrapper ibv;

IBVWrapper::IBVWrapper() {
  int err;

  ibv_handle = dlopen("libibverbs.so.1", RTLD_NOW);

  if (!ibv_handle) ibv_handle = dlopen("libibverbs.so", RTLD_NOW);

  if (!ibv_handle) {
    // Try hard-coded PATH
    ibv_handle = dlopen("/usr/lib/x86_64-linux-gnu/libibverbs.so", RTLD_NOW);

    if (!ibv_handle) {
      LOG_WARN("Could not open libibverbs. Disabled.");
      return;
    }
  }

  // The SHCA SDK adds six SDC callbacks to ibv_context_ops. Identify that
  // extension on the actual loaded library, not from ROCSHMEM_GDA_PROVIDER.
  const char* sdc_symbols[] = {"ibv_sdc_create_dct", "ibv_sdc_create_dci",
      "ibv_sdc_destroy_dct", "ibv_sdc_destroy_dci", "ibv_sdc_modify_dci"};
  unsigned sdc_count = 0;
  for (const char* symbol : sdc_symbols)
    sdc_count += dlsym(ibv_handle, symbol) != nullptr;
  if (sdc_count != 0 && sdc_count != 5) {
    LOG_WARN("Unrecognized partial SHCA verbs ABI (%u/5 SDC symbols); disabled",
             sdc_count);
    return;
  }
#if defined(GDA_SHCA)
  needs_abi_conversion_ = sdc_count == 0;
#else
  if (sdc_count != 0) {
    LOG_WARN("SHCA verbs ABI requires a GDA_SHCA build; disabled");
    return;
  }
#endif

  err = init_function_table();
  if (err != ROCSHMEM_SUCCESS) {
    LOG_WARN("Could not construct InfiniBand Verbs function table. Disabled.");
    return;
  }

  init_dmabuf_support_flag();

  is_initialized = true;
}

IBVWrapper::~IBVWrapper() {
  is_initialized = false;
  if (ibv_handle != nullptr) {
    dlclose(ibv_handle);
  }
}

void IBVWrapper::init_dmabuf_support_flag() {
  const char kernel_opt1[] = "CONFIG_DMABUF_MOVE_NOTIFY=y";
  const char kernel_opt2[] = "CONFIG_PCI_P2PDMA=y";
  int found_opt1           = 0;
  int found_opt2           = 0;
  FILE *fp;
  struct utsname utsname;
  char kernel_conf_file[128];
  char buf[256];

  if (false == envvar::gda::enable_dmabuf) {
    dmabuf_is_supported = 0;
    return;
  }

  if (ibv.reg_dmabuf_mr == NULL) {
    LOG_TRACE("ibv_reg_dmabuf_mr not present in verbs library");
    dmabuf_is_supported = 0;
    return;
  }

  if (uname(&utsname) == -1) {
    LOG_TRACE("could not get kernel name");
    dmabuf_is_supported = 0;
    return;
  }

  snprintf(kernel_conf_file, sizeof(kernel_conf_file),
           "/boot/config-%s", utsname.release);
  fp = fopen(kernel_conf_file, "r");
  if (fp == NULL) {
    LOG_TRACE("could not open kernel conf file %s error: %m",
            kernel_conf_file);
    dmabuf_is_supported = 0;
    return;
  }

  while (fgets(buf, sizeof(buf), fp) != NULL) {
    if (strstr(buf, kernel_opt1) != NULL) {
      found_opt1 = 1;
    }
    if (strstr(buf, kernel_opt2) != NULL) {
      found_opt2 = 1;
    }
    if (found_opt1 && found_opt2) {
      dmabuf_is_supported = 1;
      fclose(fp);
      return;
    }
  }
  fclose(fp);

  dmabuf_is_supported = 0;
  return;
}

int IBVWrapper::is_dmabuf_supported() {
  return dmabuf_is_supported;
}

int IBVWrapper::init_function_table() {
  DLSYM_HELPER(ibv, ibv_, ibv_handle, get_device_list);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, free_device_list);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, open_device);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, close_device);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, get_device_name);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, query_device);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, query_port);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, query_gid);
  DLSYM_HELPER(ibv, _ibv_, ibv_handle, query_gid_table); // This is not a typo
  DLSYM_HELPER(ibv, ibv_, ibv_handle, alloc_pd);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, dealloc_pd);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, reg_mr);
  DLSYM_OPT_HELPER(ibv, ibv_, ibv_handle, reg_dmabuf_mr);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, reg_mr_iova2);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, dereg_mr);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, destroy_cq);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, create_qp);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, modify_qp);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, destroy_qp);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, create_ah);
  DLSYM_HELPER(ibv, ibv_, ibv_handle, destroy_ah);
  return ROCSHMEM_SUCCESS;
}

struct ibv_device** IBVWrapper::get_device_list(int *num_devices) {
  return ibv.get_device_list(num_devices);
}

void IBVWrapper::free_device_list(struct ibv_device **list) {
  ibv.free_device_list(list);
}

struct ibv_context* IBVWrapper::open_device(struct ibv_device *device) {
  return ibv.open_device(device);

}

IBVWrapper::ContextInfo IBVWrapper::context_info(ibv_context* context) const {
  if (needs_abi_conversion_) {
    const auto* native = verbs_abi::native_context(context);
    return {reinterpret_cast<::ibv_device*>(native->device), native->cmd_fd, native->async_fd,
            native->num_comp_vectors, native->abi_compat};
  }
  return {context->device, context->cmd_fd, context->async_fd,
          context->num_comp_vectors, context->abi_compat};
}

int IBVWrapper::close_device(struct ibv_context *context) {
  return ibv.close_device(context);
}

const char* IBVWrapper::get_device_name(struct ibv_device *device) {
  return ibv.get_device_name(device);
}

int IBVWrapper::query_device(struct ibv_context *context, struct ibv_device_attr *device_attr) {
  return ibv.query_device(context, device_attr);
}

int IBVWrapper::query_port(struct ibv_context* context, uint8_t port_num,
                           struct ibv_port_attr* port_attr) {
  if (needs_abi_conversion_) {
    verbs_abi::PortAttr native{};
    auto query = verbs_abi::operation(context, &verbs_abi::VerbsContext::query_port);
    int rc;
    if (query) {
      rc = query(verbs_abi::native_context(context), port_num, &native, sizeof(native));
    } else {
      using Query = int (*)(ibv_context*, uint8_t, verbs_abi::PortAttr*);
      rc = reinterpret_cast<Query>(ibv.query_port)(context, port_num, &native);
    }
    if (rc) return rc;
    verbs_abi::port_to_internal(native, port_attr);
    return 0;
  }
  // Passthrough function for ibv_query_port macro in verbs.h
  struct verbs_context *vctx = verbs_get_ctx_op(context, query_port);

  if (!vctx) {
    int rc;

    memset(port_attr, 0, sizeof(*port_attr));

    rc = ibv.query_port(context, port_num, port_attr);
    return rc;
  }

  return vctx->query_port(context, port_num, port_attr, sizeof(*port_attr));
}

ssize_t IBVWrapper::query_gid_table(struct ibv_context *context,
                                    struct ibv_gid_entry *entries,
                                    size_t max_entries, uint32_t flags) {
  // Passthrough function for ibv_query_gid_table macro in verbs.h
  return ibv.query_gid_table(context, entries, max_entries, flags, sizeof(*entries));
}

int IBVWrapper::query_gid(struct ibv_context *context, uint8_t port_num, int index,
                          union ibv_gid *gid) {
  return ibv.query_gid(context, port_num, index, gid);
}

struct ibv_pd* IBVWrapper::alloc_pd(struct ibv_context *context) {
  return ibv.alloc_pd(context);
}

struct ibv_pd * IBVWrapper::alloc_parent_domain(struct ibv_context *context,
                                                struct ibv_parent_domain_init_attr *attr) {
  if (needs_abi_conversion_) {
    auto create = verbs_abi::operation(context,
        &verbs_abi::VerbsContext::alloc_parent_domain);
    if (!create) { errno = EOPNOTSUPP; return nullptr; }
    return reinterpret_cast<::ibv_pd*>(create(verbs_abi::native_context(context),
        reinterpret_cast<verbs_abi::ParentDomainAttr*>(attr)));
  }
  // Passthrough function for ibv_alloc_parent_domain macro in verbs.h
  return ibv_alloc_parent_domain(context, attr);
}

int IBVWrapper::dealloc_pd(struct ibv_pd *pd) {
  return ibv.dealloc_pd(pd);
}

struct ibv_mr* IBVWrapper::reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access, HIPAllocator *allocator) {
  hipPointerAttribute_t attr;
  bool is_device_ptr = false;

  CHECK_HIP(hipPointerGetAttributes(&attr, addr));

  is_device_ptr = attr.type == hipMemoryTypeDevice;

  if (is_dmabuf_supported() && is_device_ptr) {
    struct ibv_mr *mr;
    uint64_t offset = 0;
    int fd = 0;

    LOG_TRACE("Using ibv_reg_dmabuf_mr()");

    // Use provided allocator or fall back to default allocator
    HIPAllocator* alloc = (allocator != nullptr) ? allocator : get_default_allocator();
    hipError_t err = alloc->GetDmabufHandle(addr, length, &fd, &offset);
    if (err != hipSuccess) {
      LOG_ERROR("Failed to get dmabuf handle: %s", hipGetErrorString(err));
      return nullptr;
    }

    mr = ibv.reg_dmabuf_mr(pd, offset, length, (uint64_t) addr, fd, access);

    dmabuf_fd_map[(uintptr_t) mr] = fd;

    return mr;
  } else {
    LOG_TRACE("Using ibv_reg_mr(%p, %zd)", addr, length);

    // Passthrough function for ibv_reg_mr macro in verbs.h
    int is_access_const = __builtin_constant_p(((int)(access) & IBV_ACCESS_OPTIONAL_RANGE) == 0);

    if (is_access_const && (access & IBV_ACCESS_OPTIONAL_RANGE) == 0)
      return ibv.reg_mr(pd, addr, length, (int)access);
    else
      return ibv.reg_mr_iova2(pd, addr, length, (uintptr_t)addr, access);
  }
}

struct ibv_mr* IBVWrapper::reg_mr_iova2(struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, int access) {
  return ibv.reg_mr_iova2(pd, addr, length, iova, access);
}

struct ibv_mr* IBVWrapper::reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access) {
  return ibv.reg_dmabuf_mr(pd, offset, length, iova, fd, access);
}

int IBVWrapper::dereg_mr(struct ibv_mr *mr) {
  auto it = dmabuf_fd_map.find((uintptr_t) mr);
  if (it != dmabuf_fd_map.end()) {
    close(it->second);
    dmabuf_fd_map.erase(it);
  }
  return ibv.dereg_mr(mr);
}

struct ibv_cq_ex *IBVWrapper::create_cq_ex(struct ibv_context *context,
                                           struct ibv_cq_init_attr_ex *cq_attr) {
  if (needs_abi_conversion_) {
    auto create = verbs_abi::operation(context,
        &verbs_abi::VerbsContext::create_cq_ex);
    if (!create) { errno = EOPNOTSUPP; return nullptr; }
    return reinterpret_cast<::ibv_cq_ex*>(create(verbs_abi::native_context(context),
        reinterpret_cast<verbs_abi::CqInitAttr*>(cq_attr)));
  }
  // Passthrough function for ibv_create_cq_ex macro in verbs.h
  return ibv_create_cq_ex(context, cq_attr);
}

struct ibv_cq* IBVWrapper::cq_ex_to_cq(struct ibv_cq_ex *cq) {
  // Passthrough function for ibv_create_cq_ex macro in verbs.h
  return ibv_cq_ex_to_cq(cq);
}

int IBVWrapper::destroy_cq(struct ibv_cq *cq) {
  return ibv.destroy_cq(cq);
}

struct ibv_qp * IBVWrapper::create_qp_ex(struct ibv_context *context,
                                         struct ibv_qp_init_attr_ex *qp_init_attr_ex) {
  // Passthrough function for ibv_create_qp_ex macro in verbs.h
  struct verbs_context *vctx;
  uint32_t mask = qp_init_attr_ex->comp_mask;

  if (mask == IBV_QP_INIT_ATTR_PD)
    return ibv.create_qp(qp_init_attr_ex->pd, (struct ibv_qp_init_attr *)qp_init_attr_ex);

  if (needs_abi_conversion_) {
    auto create = verbs_abi::operation(context,
        &verbs_abi::VerbsContext::create_qp_ex);
    if (!create) { errno = EOPNOTSUPP; return nullptr; }
    return reinterpret_cast<::ibv_qp*>(create(verbs_abi::native_context(context),
        reinterpret_cast<verbs_abi::QpInitAttr*>(qp_init_attr_ex)));
  }

  vctx = verbs_get_ctx_op(context, create_qp_ex);
  if (!vctx) {
    errno = EOPNOTSUPP;
    return NULL;
  }
  return vctx->create_qp_ex(context, qp_init_attr_ex);
}

int IBVWrapper::modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) {
  if (needs_abi_conversion_) {
    verbs_abi::QpAttr native{};
    int rc = verbs_abi::qp_from_internal(*attr, attr_mask, &native);
    if (rc) { errno = rc; return rc; }
    using Modify = int (*)(ibv_qp*, verbs_abi::QpAttr*, int);
    return reinterpret_cast<Modify>(ibv.modify_qp)(qp, &native, attr_mask);
  }
  return ibv.modify_qp(qp, attr, attr_mask);
}

int IBVWrapper::destroy_qp(struct ibv_qp *qp) {
  return ibv.destroy_qp(qp);
}

uint16_t IBVWrapper::flow_label_to_udp_sport(uint32_t fl) {
  // Passthrough function for ibv_flow_label_to_udp_sport inline function in verbs.h
  return ibv_flow_label_to_udp_sport(fl);
}

struct ibv_ah* IBVWrapper::create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr) {
  if (needs_abi_conversion_) {
    verbs_abi::AhAttr native{};
    int rc = verbs_abi::ah_from_internal(*attr, &native);
    if (rc) { errno = rc; return nullptr; }
    using Create = ibv_ah* (*)(ibv_pd*, verbs_abi::AhAttr*);
    return reinterpret_cast<Create>(ibv.create_ah)(pd, &native);
  }
  return ibv.create_ah(pd, attr);
}

int IBVWrapper::destroy_ah(struct ibv_ah *ah) {
  return ibv.destroy_ah(ah);
}

} // namespace rocshmem
