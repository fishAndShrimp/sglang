#include "acl_batch.h"

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace py = pybind11;
using sglang::moe::Batch;
using sglang::moe::check_acl;

namespace {
void validate_device(int device) {
  if (device < 0 || device > 127) {
    throw std::invalid_argument("NPU device index is outside c10 DeviceIndex range");
  }
}

void submit(const std::shared_ptr<Batch>& batch, int64_t expected_stream_id) {
  c10_npu::OptionalNPUGuard guard;
  guard.set_index(batch->device());
  const auto stream = c10_npu::getCurrentNPUStream(batch->device());
  if (stream.id() != expected_stream_id) {
    throw std::runtime_error("ACL submit requires the caller's current torch_npu stream");
  }
  // stream(true) drains host queue entries (including prior event waits) into
  // the device stream. It does NOT wait for device execution. The pipeline owns
  // exclusive submission rights to this stream until this function returns.
  // The following Python event.record is submitted after this direct ACL call.
  batch->submit(stream.stream(true));
}

void validate_host_range(uint64_t address, uint64_t nbytes) {
  if (address == 0 || address % 4096 != 0 || nbytes == 0 ||
      address > UINTPTR_MAX - nbytes) {
    throw std::invalid_argument("Host registration requires a nonempty 4 KiB-aligned range");
  }
}

uint64_t register_host(uint64_t address, uint64_t nbytes, int device, bool mapped) {
  validate_host_range(address, nbytes);
  validate_device(device);
  c10_npu::OptionalNPUGuard guard;
  guard.set_index(device);
#if defined(ACL_HOST_REG_PINNED) && defined(ACL_HOST_REG_MAPPED)
  const uint32_t flags = ACL_HOST_REG_PINNED | (mapped ? ACL_HOST_REG_MAPPED : 0);
  void* host = reinterpret_cast<void*>(address);
  check_acl(aclrtHostRegisterV2(host, nbytes, flags), "aclrtHostRegisterV2");
  if (!mapped) {
    return 0;
  }
  void* device_address = nullptr;
  const auto status = aclrtHostGetDevicePointer(host, &device_address, 0);
  if (status != ACL_SUCCESS || device_address == nullptr) {
    const auto cleanup = aclrtHostUnregister(host);
    throw std::runtime_error(
        "aclrtHostGetDevicePointer failed, status=" + std::to_string(status) +
        "; unregister status=" + std::to_string(cleanup) +
        "; retain host allocation if unregister failed");
  }
  return reinterpret_cast<uint64_t>(device_address);
#else
  throw std::runtime_error(
      "Installed CANN headers do not support pinned HostRegisterV2; no flag fallback");
#endif
}

void unregister_host(uint64_t address, int device) {
  if (address == 0) throw std::invalid_argument("Host address cannot be null");
  validate_device(device);
  c10_npu::OptionalNPUGuard guard;
  guard.set_index(device);
  check_acl(aclrtHostUnregister(reinterpret_cast<void*>(address)), "aclrtHostUnregister");
}

uint64_t malloc_host(size_t nbytes, int device) {
  if (nbytes == 0) throw std::invalid_argument("Host allocation must be nonempty");
  validate_device(device);
  c10_npu::OptionalNPUGuard guard;
  guard.set_index(device);
  void* address = nullptr;
  check_acl(aclrtMallocHost(&address, nbytes), "aclrtMallocHost");
  if (address == nullptr) throw std::runtime_error("aclrtMallocHost returned null");
  return reinterpret_cast<uint64_t>(address);
}

void free_host(uint64_t address, int device) {
  if (address == 0) throw std::invalid_argument("Host address cannot be null");
  validate_device(device);
  c10_npu::OptionalNPUGuard guard;
  guard.set_index(device);
  check_acl(aclrtFreeHost(reinterpret_cast<void*>(address)), "aclrtFreeHost");
}

void memcpy_sync(uint64_t destination, size_t destination_capacity, uint64_t source,
                 size_t nbytes, const std::string& direction, int device) {
  if (source == 0 || destination == 0 || nbytes > destination_capacity ||
      source > UINTPTR_MAX - nbytes || destination > UINTPTR_MAX - destination_capacity) {
    throw std::invalid_argument("Invalid synchronous byte copy range");
  }
  aclrtMemcpyKind kind;
  validate_device(device);
  if (direction == "host_to_device") kind = ACL_MEMCPY_HOST_TO_DEVICE;
  else if (direction == "device_to_host") kind = ACL_MEMCPY_DEVICE_TO_HOST;
  else throw std::invalid_argument("Only H2D/D2H byte initialization copies are supported");
  c10_npu::OptionalNPUGuard guard;
  guard.set_index(device);
  check_acl(aclrtMemcpy(reinterpret_cast<void*>(destination), destination_capacity,
                       reinterpret_cast<void*>(source), nbytes, kind), "aclrtMemcpy");
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<Batch, std::shared_ptr<Batch>>(module, "Batch", py::module_local())
      .def(py::init<const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    const std::vector<size_t>&, const std::vector<size_t>&, uint32_t,
                    const std::string&, uint32_t>());
  // No Python objects are touched while GIL is released. Caller retains Batch
  // and every allocation owner before submit, through completion or failure drain.
  module.def("submit", &submit, py::call_guard<py::gil_scoped_release>());
  module.def("register_host", &register_host, py::arg("address"), py::arg("nbytes"),
             py::arg("device_index"), py::arg("mapped") = false,
             py::call_guard<py::gil_scoped_release>());
  module.def("unregister_host", &unregister_host, py::call_guard<py::gil_scoped_release>());
  module.def("malloc_host", &malloc_host, py::call_guard<py::gil_scoped_release>());
  module.def("free_host", &free_host, py::call_guard<py::gil_scoped_release>());
  module.def("memcpy_sync", &memcpy_sync, py::call_guard<py::gil_scoped_release>());
  module.def("api_info", [] {
    py::dict result;
    result["batch_abi"] = SGLANG_ACL_BATCH_ABI;
    result["queue_ordering"] = "torch_npu_stream_flush";
    result["device_synchronize"] = false;
#if defined(ACL_HOST_REG_PINNED) && defined(ACL_HOST_REG_MAPPED)
    result["pinned_registration"] = true;
#else
    result["pinned_registration"] = false;
#endif
    return result;
  });
}
