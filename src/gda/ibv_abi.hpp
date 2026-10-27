/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2004, 2011-2012 Intel Corporation.  All rights reserved.
 * Copyright (c) 2005, 2006, 2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 * Copyright (c) 2020 Intel Corporation.  All rights reserved.
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
// Bridge the build SDK and standard verbs ABIs using the existing core types.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <pthread.h>
#include "ibv_core.hpp"

namespace rocshmem::verbs_abi {
using PortAttr = ibv_port_attr;
using AhAttr = ibv_ah_attr;
using QpAttr = ibv_qp_attr;
using Context = ibv_context;
using VerbsContext = verbs_context;
using ParentDomainAttr = ibv_parent_domain_init_attr;
using CqInitAttr = ibv_cq_init_attr_ex;
using QpInitAttr = ibv_qp_init_attr_ex;

inline Context* native_context(::ibv_context* context) {
  return reinterpret_cast<Context*>(context);
}
template <typename F>
inline F operation(::ibv_context* context, F VerbsContext::*member) {
  auto* native = native_context(context);
  if (native->abi_compat != __VERBS_ABI_IS_EXTENDED) return nullptr;
  auto* ext = reinterpret_cast<VerbsContext*>(reinterpret_cast<char*>(native) -
                                            offsetof(VerbsContext, context));
  const auto offset = reinterpret_cast<char*>(&(ext->*member)) -
                      reinterpret_cast<char*>(ext);
  if (ext->sz < sizeof(VerbsContext) - offset) return nullptr;
  return ext->*member;
}

inline void port_to_internal(const PortAttr& src, ::ibv_port_attr* dst) {
  *dst = {};
#define COPY_PORT(field) dst->field = static_cast<decltype(dst->field)>(src.field)
  COPY_PORT(state); COPY_PORT(max_mtu); COPY_PORT(active_mtu);
  COPY_PORT(gid_tbl_len); COPY_PORT(port_cap_flags); COPY_PORT(max_msg_sz);
  COPY_PORT(bad_pkey_cntr); COPY_PORT(qkey_viol_cntr); COPY_PORT(pkey_tbl_len);
#if defined(GDA_SHCA)
  dst->lid = u32_to_17(src.lid);
  dst->sm_lid = u32_to_17(src.sm_lid);
#else
  COPY_PORT(lid); COPY_PORT(sm_lid);
#endif
  COPY_PORT(lmc); COPY_PORT(max_vl_num); COPY_PORT(sm_sl);
  COPY_PORT(subnet_timeout); COPY_PORT(init_type_reply); COPY_PORT(active_width);
  COPY_PORT(active_speed); COPY_PORT(phys_state); COPY_PORT(link_layer);
  COPY_PORT(flags); COPY_PORT(port_cap_flags2);
#undef COPY_PORT
}

inline int ah_from_internal(const ::ibv_ah_attr& src, AhAttr* dst) {
  *dst = {};
#if defined(GDA_SHCA)
  const uint32_t lid = u17_to_32(src.dlid);
  if (lid > UINT16_MAX) return EOVERFLOW;
  dst->dlid = static_cast<uint16_t>(lid);
#else
  dst->dlid = src.dlid;
#endif
  static_assert(sizeof(dst->grh) == sizeof(src.grh));
  std::memcpy(&dst->grh, &src.grh, sizeof(dst->grh));
  dst->sl = src.sl;
  dst->src_path_bits = src.src_path_bits;
  dst->static_rate = src.static_rate;
  dst->is_global = src.is_global;
  dst->port_num = src.port_num;
  return 0;
}

inline int qp_from_internal(const ::ibv_qp_attr& src, int mask, QpAttr* dst) {
  *dst = {};
#define COPY_QP(flag, field) if (mask & flag) dst->field = static_cast<decltype(dst->field)>(src.field)
  COPY_QP(IBV_QP_STATE, qp_state); COPY_QP(IBV_QP_CUR_STATE, cur_qp_state);
  COPY_QP(IBV_QP_PATH_MTU, path_mtu);
  COPY_QP(IBV_QP_PATH_MIG_STATE, path_mig_state);
  COPY_QP(IBV_QP_QKEY, qkey); COPY_QP(IBV_QP_RQ_PSN, rq_psn);
  COPY_QP(IBV_QP_SQ_PSN, sq_psn); COPY_QP(IBV_QP_DEST_QPN, dest_qp_num);
  COPY_QP(IBV_QP_ACCESS_FLAGS, qp_access_flags);
  if (mask & IBV_QP_CAP) {
    dst->cap.max_send_wr = src.cap.max_send_wr;
    dst->cap.max_recv_wr = src.cap.max_recv_wr;
    dst->cap.max_send_sge = src.cap.max_send_sge;
    dst->cap.max_recv_sge = src.cap.max_recv_sge;
    dst->cap.max_inline_data = src.cap.max_inline_data;
  }
  COPY_QP(IBV_QP_PKEY_INDEX, pkey_index);
  COPY_QP(IBV_QP_EN_SQD_ASYNC_NOTIFY, en_sqd_async_notify);
  COPY_QP(IBV_QP_MAX_QP_RD_ATOMIC, max_rd_atomic);
  COPY_QP(IBV_QP_MAX_DEST_RD_ATOMIC, max_dest_rd_atomic);
  COPY_QP(IBV_QP_MIN_RNR_TIMER, min_rnr_timer); COPY_QP(IBV_QP_PORT, port_num);
  COPY_QP(IBV_QP_TIMEOUT, timeout); COPY_QP(IBV_QP_RETRY_CNT, retry_cnt);
  COPY_QP(IBV_QP_RNR_RETRY, rnr_retry); COPY_QP(IBV_QP_RATE_LIMIT, rate_limit);
#undef COPY_QP
  if (mask & IBV_QP_AV) {
    int err = ah_from_internal(src.ah_attr, &dst->ah_attr);
    if (err) return err;
  }
  if (mask & IBV_QP_ALT_PATH) {
    int err = ah_from_internal(src.alt_ah_attr, &dst->alt_ah_attr);
    if (err) return err;
    dst->alt_pkey_index = src.alt_pkey_index;
    dst->alt_port_num = src.alt_port_num;
    dst->alt_timeout = src.alt_timeout;
  }
  return 0;
}
} // namespace rocshmem::verbs_abi
