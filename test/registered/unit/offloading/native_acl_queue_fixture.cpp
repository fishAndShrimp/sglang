// Compile the real shim against deterministic external SDK/task-queue stubs.
// No NPU behavior is reimplemented: the test controls launch and device queues.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using aclError = int;
using aclrtStream = void*;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_MEM_LOCATION_TYPE_HOST = 0;
constexpr int ACL_MEM_LOCATION_TYPE_UNREGISTERED = 1;
constexpr int ACL_MEM_LOCATION_TYPE_HOST_NUMA = 2;
constexpr int ACL_MEM_LOCATION_TYPE_DEVICE = 3;
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST };
struct aclrtMemLocation { int type{}; uint32_t id{}; };
struct aclrtMemcpyBatchAttr { aclrtMemLocation srcLoc, dstLoc; };

namespace fake {
std::deque<std::function<void()>> launches, device;
std::vector<std::string> executed;
bool queue_enabled = true;
int flushes = 0, failure = 0;
size_t failure_index = 0;
int64_t current_stream = 17;
aclrtStream launched_stream = nullptr;
std::function<void()> nested;
void drain(std::deque<std::function<void()>>& queue) {
  while (!queue.empty()) {
    auto task = std::move(queue.front());
    queue.pop_front();
    task();
  }
}
void enqueue(std::function<void()> task) {
  if (queue_enabled) launches.push_back(std::move(task));
  else task();
}
aclError copy(void** dst, void** src, size_t* sizes, size_t count,
              size_t* failed, aclrtStream stream) {
  launched_stream = stream;
  const auto status = failure;
  if (failed) *failed = failure_index;
  if (nested) {
    auto callback = std::move(nested);
    nested = nullptr;
    callback();
  }
  if (status) return status;
  for (size_t i = 0; i < count; ++i) {
    device.push_back([d=dst[i], s=src[i], n=sizes[i]] {
      std::memcpy(d, s, n);
      executed.push_back("copy");
    });
  }
  return ACL_SUCCESS;
}
}  // namespace fake

#if SGLANG_ACL_BATCH_ABI == 10
aclError aclrtMemcpyBatchAsync(void** dst, size_t*, void** src, size_t* sizes,
                              size_t count, aclrtMemcpyBatchAttr*, size_t*,
                              size_t, size_t* failed, aclrtStream stream) {
  return fake::copy(dst, src, sizes, count, failed, stream);
}
#else
aclError aclrtMemcpyBatchAsync(void** dst, size_t*, void** src, size_t* sizes,
                              size_t count, aclrtMemcpyBatchAttr*, size_t*,
                              size_t, aclrtStream stream) {
  return fake::copy(dst, src, sizes, count, nullptr, stream);
}
#endif
aclError aclrtHostUnregister(void*) { return ACL_SUCCESS; }
aclError aclrtMallocHost(void** ptr, size_t) { *ptr = reinterpret_cast<void*>(1); return 0; }
aclError aclrtFreeHost(void*) { return ACL_SUCCESS; }
aclError aclrtMemcpy(void* dst, size_t, const void* src, size_t size, aclrtMemcpyKind) {
  std::memcpy(dst, src, size);
  return ACL_SUCCESS;
}

namespace c10_npu {
struct OptionalNPUGuard { void set_index(int) {} };
struct NPUStream {
  int64_t value;
  int64_t id() const { return value; }
  aclrtStream stream(bool flush = true) const {
    if (flush) { ++fake::flushes; fake::drain(fake::launches); }
    return reinterpret_cast<void*>(value);
  }
};
NPUStream getCurrentNPUStream(int) { return {fake::current_stream}; }
}  // namespace c10_npu
namespace at_npu::native {
struct OpCommand {
  static void RunOpApi(const std::string&, std::function<int()> handler, bool = false) {
    fake::enqueue([handler = std::move(handler)] {
      if (handler() != ACL_SUCCESS) throw std::runtime_error("queued ACL failure");
    });
  }
};
}  // namespace at_npu::native

// Binding declarations only; native functions above are the code under test.
namespace pybind11 {
struct gil_scoped_release {};
template<class... T> struct call_guard {};
struct arg { explicit arg(const char*) {} template<class T> arg operator=(T) { return *this; } };
struct module_local {};
struct dict {
  struct item { template<class T> void operator=(T&&) {} };
  item operator[](const char*) { return {}; }
};
struct module_ { template<class... T> void def(T&&...) {} };
template<class... T> struct init {};
template<class... T> struct class_ {
  template<class... A> class_(A&&...) {}
  template<class... A> class_& def(A&&...) { return *this; }
};
}  // namespace pybind11
#define PYBIND11_MODULE(name, module) void unused_bindings(pybind11::module_& module)
#include SGLANG_NATIVE_ACL_SOURCE

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
std::shared_ptr<Batch> descriptor(int& source, int& destination) {
  return std::make_shared<Batch>(
      std::vector<uint64_t>{reinterpret_cast<uint64_t>(&source)},
      std::vector<uint64_t>{reinterpret_cast<uint64_t>(&destination)},
      std::vector<size_t>{sizeof(int)}, std::vector<size_t>{sizeof(int)},
      0, "host", 0);
}
void ordered_submission(bool queued) {
  fake::queue_enabled = queued;
  int source = 42, bank = 7, reader = -1, ready = -1;
  // A preceding wait protects a reader of the bank being replaced.
  fake::enqueue([&] {
    fake::device.push_back([&] {
      reader = bank;
      fake::executed.push_back("reader");
    });
  });
  auto batch = descriptor(source, bank);
  std::weak_ptr<Batch> lifetime = batch;
  submit(batch, 17);
  batch.reset();
  if (queued) {
    require(fake::flushes == 0, "submit flushed the producer's launch queue");
    require(fake::device.empty(), "submit overtook queued bank dependency");
    require(!lifetime.expired(), "queued descriptor was released before launch");
  }
  // This models Python event.record immediately following module.submit.
  fake::enqueue([&] {
    fake::device.push_back([&] {
      ready = bank;
      fake::executed.push_back("ready");
    });
  });
  fake::current_stream = 99;  // queue worker TLS must not select the target stream
  fake::drain(fake::launches);
  require(lifetime.expired(), "launch queue leaked the descriptor");
  require(fake::launched_stream == reinterpret_cast<void*>(17), "wrong stream at launch");
  require(bank == 7 && ready == -1, "submission synchronized the device");
  fake::drain(fake::device);
  require(reader == 7 && bank == 42 && ready == 42, "bank read/copy/ready ordering broken");
  require(fake::executed == std::vector<std::string>{"reader", "copy", "ready"},
          "device dependency ordering broken");
}
void submission_failure(bool queued) {
  fake::queue_enabled = queued;
  fake::failure = 71;
  fake::failure_index = 3;
  int source = 42, bank = 7;
  auto batch = descriptor(source, bank);
  std::weak_ptr<Batch> lifetime = batch;
  bool raised = false;
  try {
    submit(batch, 17);
    require(queued, "synchronous launch swallowed an ACL error");
    batch.reset();
    require(!lifetime.expired(), "failing queued launch lost descriptor");
    fake::drain(fake::launches);
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    require(message.find("ACL status=71") != std::string::npos, "ACL error not preserved");
#if SGLANG_ACL_BATCH_ABI == 10
    require(message.find("fail_index=3") != std::string::npos, "failure index not preserved");
#else
    require(message.find("fail_index=unavailable") != std::string::npos, "9-argument ABI guessed failure index");
#endif
    raised = true;
  }
  require(raised, "launch error disappeared");
  require(bank == 7, "failed launch changed bank");
  if (queued) require(lifetime.expired(), "failed queued launch leaked descriptor");
}
void per_launch_failure_state() {
#if SGLANG_ACL_BATCH_ABI == 10
  int source = 42, bank = 7;
  auto batch = descriptor(source, bank);
  fake::failure = 71;
  fake::failure_index = 3;
  // Deterministic overlapping invocations of one cached Batch, as allowed once
  // queued work outlives the Python submission lock. Inner failure must not
  // overwrite the outer invocation's failure index.
  fake::nested = [&] {
    fake::failure = 72;
    fake::failure_index = 5;
    try { batch->submit(reinterpret_cast<void*>(17)); }
    catch (const std::runtime_error& error) {
      require(std::string(error.what()).find("fail_index=5") != std::string::npos,
              "inner failure index lost");
    }
  };
  bool raised = false;
  try { batch->submit(reinterpret_cast<void*>(17)); }
  catch (const std::runtime_error& error) {
    require(std::string(error.what()).find("fail_index=3") != std::string::npos,
            "cached descriptor shared mutable failure state");
    raised = true;
  }
  require(raised, "outer failure disappeared");
#endif
}
int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected one test case");
    const std::string test = argv[1];
    if (test == "queued") ordered_submission(true);
    else if (test == "immediate") ordered_submission(false);
    else if (test == "queued_failure") submission_failure(true);
    else if (test == "immediate_failure") submission_failure(false);
    else if (test == "failure_state") per_launch_failure_state();
    else throw std::runtime_error("unknown test case");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
