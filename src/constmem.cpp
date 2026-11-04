#include "constmem.hpp"
#include "backend_bc.hpp"
#include "envvar.hpp"
#include "device_module_registry.hpp"
#if defined(USE_GDA)
#include "gda/backend_gda.hpp"
#endif

namespace rocshmem {

extern Backend *backend;

void init_constant_memory(void) {
  std::string envstr;
  constmem_t constmem_values;

  memset(&constmem_values, 0, sizeof(constmem_t));

  envstr = envvar::gda::alltoallv_wg_algo;

  if (envstr.empty() || envstr.find("GET") != std::string::npos) {
    constmem_values.alltoall_wg_algo = gda::ALLTOALLV_WG_ALGO_GET;
  } else {
    constmem_values.alltoall_wg_algo = gda::ALLTOALLV_WG_ALGO_COPY;
  }

  constmem_values.my_pe = backend->getMyPE();
  constmem_values.num_pes = backend->getNumPEs();

  constmem_values.ipc_first_pe = backend->ipcImpl.ipc_first_pe;
  constmem_values.ipc_stride = backend->ipcImpl.ipc_stride;
  // ipc_shm_size == 0 means IPC disabled (fast early return on device).
  // Non-zero when IPC is available, regardless of stride pattern.
  constmem_values.ipc_shm_size = (backend->ipcImpl.pes_with_ipc_avail != nullptr)
                                 ? backend->ipcImpl.shm_size : 0;
  constmem_values.heap_base =
      reinterpret_cast<uintptr_t>(backend->heap.get_local_heap_base());
  constmem_values.heap_size = backend->heap.get_size();

  constmem_values.backend_type = backend->get_type();
#if defined(USE_GDA)
  if (constmem_values.backend_type == BackendType::GDA_BACKEND) {
    constmem_values.gda_provider = static_cast<GDABackend*>(backend)->get_gda_provider();
    constmem_values.pxn_enabled = !envvar::disable_pxn;
    constmem_values.world_proxy_sync_enabled =
        static_cast<GDABackend*>(backend)->world_proxy_sync_enabled();
    constmem_values.aligned_world_proxy_data_enabled =
        static_cast<GDABackend*>(backend)->aligned_world_proxy_data_enabled();
    const auto& node_first_pes =
        static_cast<GDABackend*>(backend)->world_proxy_node_first_pes();
    constmem_values.world_proxy_node_count =
        static_cast<int>(node_first_pes.size());
    for (size_t i = 0; i < node_first_pes.size(); i++) {
      constmem_values.world_proxy_node_first_pes[i] = node_first_pes[i];
    }
    auto* gda_backend = static_cast<GDABackend*>(backend);
    constmem_values.world_proxy_nic_layout_enabled =
        gda_backend->world_proxy_nic_layout_enabled();
    constmem_values.world_proxy_nic_count =
        gda_backend->world_proxy_nic_count();
    const auto& pe_nic_ids = gda_backend->world_proxy_pe_nic_ids();
    for (size_t i = 0; i < pe_nic_ids.size(); i++) {
      constmem_values.world_proxy_pe_nic_ids[i] = pe_nic_ids[i];
    }
  }
#endif

  CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(constmem), &constmem_values, sizeof(constmem_t)));
  sync_device_modules();
}

}
