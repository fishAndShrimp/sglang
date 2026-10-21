#pragma once

#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sglang::moe {

// Selected by compiling exact pointer casts against the installed CANN header.
// An unknown signature is an error: never reinterpret_cast a runtime symbol.
#if SGLANG_ACL_BATCH_ABI == 10
using BatchFunction = aclError (*)(void**, size_t*, void**, size_t*, size_t,
                                  aclrtMemcpyBatchAttr*, size_t*, size_t, size_t*,
                                  aclrtStream);
#elif SGLANG_ACL_BATCH_ABI == 9
using BatchFunction = aclError (*)(void**, size_t*, void**, size_t*, size_t,
                                  aclrtMemcpyBatchAttr*, size_t*, size_t,
                                  aclrtStream);
#else
#error "Probe the installed CANN aclrtMemcpyBatchAsync signature before building"
#endif
inline constexpr BatchFunction batch_function =
    static_cast<BatchFunction>(&aclrtMemcpyBatchAsync);

inline void check_acl(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed, ACL status=" +
                             std::to_string(status));
  }
}

inline aclrtMemLocation host_location(const std::string& name, uint32_t id) {
  aclrtMemLocation result{};
  if (name == "host") {
    result.type = ACL_MEM_LOCATION_TYPE_HOST;
  } else if (name == "unregistered") {
    result.type = ACL_MEM_LOCATION_TYPE_UNREGISTERED;
  } else if (name == "host_numa") {
    result.type = ACL_MEM_LOCATION_TYPE_HOST_NUMA;
  } else {
    throw std::invalid_argument("Unknown ACL host allocation location");
  }
  result.id = id;
  return result;
}

class Batch {
 public:
  Batch(const std::vector<uint64_t>& sources,
        const std::vector<uint64_t>& destinations,
        const std::vector<size_t>& sizes,
        const std::vector<size_t>& capacities, uint32_t device,
        const std::string& source_location, uint32_t source_location_id)
      : sizes_(sizes), capacities_(capacities), device_(device) {
    if (device > 127) {
      throw std::invalid_argument("NPU device index is outside c10 DeviceIndex range");
    }
    const size_t count = sources.size();
    if (count == 0 || count != destinations.size() || count != sizes.size() ||
        count != capacities.size()) {
      throw std::invalid_argument("Native ACL arrays must have equal nonzero lengths");
    }
    std::vector<std::pair<uint64_t, uint64_t>> writes;
    for (size_t index = 0; index < count; ++index) {
      if (sources[index] == 0 || destinations[index] == 0 || sizes[index] == 0 ||
          sizes[index] > capacities[index] ||
          sources[index] > UINTPTR_MAX - sizes[index] ||
          destinations[index] > UINTPTR_MAX - capacities[index]) {
        throw std::invalid_argument("Invalid native ACL byte range/capacity");
      }
      sources_.push_back(reinterpret_cast<void*>(sources[index]));
      destinations_.push_back(reinterpret_cast<void*>(destinations[index]));
      writes.emplace_back(destinations[index], destinations[index] + sizes[index]);
    }
    std::sort(writes.begin(), writes.end());
    for (size_t index = 1; index < count; ++index) {
      if (writes[index].first < writes[index - 1].second) {
        throw std::invalid_argument("Native ACL destination ranges overlap");
      }
    }
    // One SDK-defined, zero-initialized attribute covers every entry. Reserved
    // bytes stay zero and the attribute start index is exactly zero.
    attribute_.srcLoc = host_location(source_location, source_location_id);
    attribute_.dstLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    attribute_.dstLoc.id = device;
  }

  uint32_t device() const { return device_; }

  void submit(aclrtStream stream) {
    if (stream == nullptr) {
      throw std::runtime_error("torch_npu host queue flush returned a null stream");
    }
    fail_index_ = std::numeric_limits<size_t>::max();
#if SGLANG_ACL_BATCH_ABI == 10
    const auto status = batch_function(
        destinations_.data(), capacities_.data(), sources_.data(), sizes_.data(),
        sizes_.size(), &attribute_, &attribute_index_, 1, &fail_index_, stream);
#else
    const auto status = batch_function(
        destinations_.data(), capacities_.data(), sources_.data(), sizes_.data(),
        sizes_.size(), &attribute_, &attribute_index_, 1, stream);
#endif
    if (status != ACL_SUCCESS) {
      throw std::runtime_error(
          "aclrtMemcpyBatchAsync failed, ACL status=" + std::to_string(status) +
          ", fail_index=" +
          (fail_index_ == std::numeric_limits<size_t>::max()
               ? std::string("unavailable") : std::to_string(fail_index_)) +
          "; submission may be partial; retain and drain before reuse");
    }
  }

 private:
  std::vector<void*> sources_, destinations_;
  std::vector<size_t> sizes_, capacities_;
  uint32_t device_;
  aclrtMemcpyBatchAttr attribute_{};
  size_t attribute_index_ = 0;
  size_t fail_index_ = std::numeric_limits<size_t>::max();
};

}  // namespace sglang::moe
