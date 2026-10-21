"""Compile the real ACL shim against explicit fake SDK headers on CPU.

This verifies C++ argument lowering, host-queue ordering, errors and Python
lifetime bridging; it does not certify linking against CANN or NPU hardware.
"""

import ctypes
import importlib.util
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.offloading.moe import native_acl
from sglang.srt.offloading.moe.native_acl import (
    NativeAclBatchSubmitter,
    probe_batch_abi,
)
from sglang.srt.offloading.moe.npu_transfer import AclCopyBatch, DeviceMemoryRegion
from sglang.srt.offloading.moe.storage import HostMemoryCapability
from sglang.srt.offloading.transfer import (
    HostCopyPlan,
    HostCopyRegion,
    ImmediateCompletion,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_SOURCE = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/offloading/moe/native"
)

_ACL_HEADER = r"""
#pragma once
#include <cstdint>
#include <cstddef>
using aclError = int;
using aclrtStream = void*;
constexpr int ACL_SUCCESS = 0;
enum aclrtMemLocationType { ACL_MEM_LOCATION_TYPE_HOST=0, ACL_MEM_LOCATION_TYPE_DEVICE=1,
  ACL_MEM_LOCATION_TYPE_UNREGISTERED=2, ACL_MEM_LOCATION_TYPE_HOST_NUMA=4 };
struct aclrtMemLocation { uint32_t id; aclrtMemLocationType type; };
struct aclrtMemcpyBatchAttr { aclrtMemLocation dstLoc, srcLoc; uint8_t rsv[16]; };
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE=1, ACL_MEMCPY_DEVICE_TO_HOST=2 };
#define ACL_HOST_REG_PINNED 0x10000000UL
#define ACL_HOST_REG_MAPPED 0x2UL
#if FAKE_ARITY == 10
aclError aclrtMemcpyBatchAsync(void**,size_t*,void**,size_t*,size_t,aclrtMemcpyBatchAttr*,size_t*,size_t,size_t*,aclrtStream);
#elif FAKE_ARITY == 9
aclError aclrtMemcpyBatchAsync(void**,size_t*,void**,size_t*,size_t,aclrtMemcpyBatchAttr*,size_t*,size_t,aclrtStream);
#else
aclError aclrtMemcpyBatchAsync(int unsupported);
#endif
aclError aclrtHostRegisterV2(void*,uint64_t,uint32_t);
aclError aclrtHostGetDevicePointer(void*,void**,uint32_t);
aclError aclrtHostUnregister(void*);
aclError aclrtMallocHost(void**,size_t);
aclError aclrtFreeHost(void*);
aclError aclrtMemcpy(void*,size_t,const void*,size_t,aclrtMemcpyKind);
extern int fake_device;
extern int64_t fake_stream_id;
extern bool fake_null_stream;
void fake_flush(bool);
"""

_NPU_STREAM = r"""
#pragma once
#include <acl/acl.h>
namespace c10_npu {
struct NPUStream {
  int64_t id() const { return fake_stream_id; }
  aclrtStream stream(bool need_empty) const {
    fake_flush(need_empty);
    return fake_null_stream ? nullptr : reinterpret_cast<void*>(1234);
  }
};
inline NPUStream getCurrentNPUStream(int device) { fake_device=device; return {}; }
}
"""

_NPU_GUARD = r"""
#pragma once
#include <acl/acl.h>
namespace c10_npu {
struct OptionalNPUGuard { void set_index(int device) { fake_device=device; } };
}
"""

_FAKE_RUNTIME = r"""
#include <acl/acl.h>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>
int fake_device=0;
int64_t fake_stream_id=5;
bool fake_null_stream=false;
static std::vector<int> trace;
static int failure=0, delay=0;
static uint32_t flags=0;
void fake_flush(bool enabled) { trace.push_back(enabled ? 1 : -1); }
#pragma GCC visibility push(default)
extern "C" {
void fake_reset() { trace.clear();failure=0;delay=0;fake_null_stream=false; }
void fake_fail(int status) { failure=status; }
void fake_set_null(int value) { fake_null_stream=value; }
void fake_delay(int value) { delay=value; }
int fake_trace_size() { return static_cast<int>(trace.size()); }
int fake_trace_at(int index) { return trace.at(index); }
uint32_t fake_flags() { return flags; }
}
#pragma GCC visibility pop
static aclError do_batch(void**dst,size_t*cap,void**src,size_t*len,size_t n,
  aclrtMemcpyBatchAttr*attrs,size_t*index,size_t attrn,aclrtStream stream) {
  trace.push_back(2);
  if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  if (attrn!=1 || index[0]!=0 || attrs[0].dstLoc.type!=ACL_MEM_LOCATION_TYPE_DEVICE
      || attrs[0].dstLoc.id!=static_cast<uint32_t>(fake_device) || stream==nullptr) return 71;
  for (auto byte: attrs[0].rsv) if(byte!=0) return 72;
  trace.push_back(10 + attrs[0].srcLoc.type);
  if(failure) return failure;
  for(size_t i=0;i<n;++i) { if(len[i]>cap[i]) return 73; std::memcpy(dst[i],src[i],len[i]); }
  return 0;
}
#if FAKE_ARITY == 10
aclError aclrtMemcpyBatchAsync(void**d,size_t*c,void**s,size_t*l,size_t n,
  aclrtMemcpyBatchAttr*a,size_t*i,size_t an,size_t*fail,aclrtStream st) {
  *fail=1;return do_batch(d,c,s,l,n,a,i,an,st);
}
#else
aclError aclrtMemcpyBatchAsync(void**d,size_t*c,void**s,size_t*l,size_t n,
  aclrtMemcpyBatchAttr*a,size_t*i,size_t an,aclrtStream st) {
  return do_batch(d,c,s,l,n,a,i,an,st);
}
#endif
aclError aclrtHostRegisterV2(void*,uint64_t,uint32_t f) { flags=f;trace.push_back(3);return failure; }
aclError aclrtHostGetDevicePointer(void*p,void**d,uint32_t) { *d=reinterpret_cast<void*>(reinterpret_cast<uint64_t>(p)+(1ULL<<40));trace.push_back(4);return failure; }
aclError aclrtHostUnregister(void*) {trace.push_back(5);return failure;}
aclError aclrtMallocHost(void**p,size_t n) {return posix_memalign(p,4096,n);}
aclError aclrtFreeHost(void*p) {std::free(p);return 0;}
aclError aclrtMemcpy(void*d,size_t cap,const void*s,size_t n,aclrtMemcpyKind k) {
 if(n>cap || (k!=ACL_MEMCPY_HOST_TO_DEVICE && k!=ACL_MEMCPY_DEVICE_TO_HOST)) return 74;
 std::memcpy(d,s,n);return 0;
}
"""


class TestNativeAcl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("c++") is None:
            raise unittest.SkipTest("C++ compiler unavailable")
        torch_spec = importlib.util.find_spec("torch")
        include = Path(torch_spec.origin).parent / "include"
        if not (include / "pybind11/pybind11.h").is_file():
            raise unittest.SkipTest("PyTorch pybind11 development headers unavailable")
        python_include = Path(sysconfig.get_paths()["include"])
        if not (python_include / "Python.h").is_file():
            python_include = Path(sys.prefix) / "Headers"
        if not (python_include / "Python.h").is_file():
            raise unittest.SkipTest("Python development headers unavailable")
        cls.temp = tempfile.TemporaryDirectory(prefix="native-acl-test-")
        cls.modules = []
        for arity in (10, 9):
            root = Path(cls.temp.name) / str(arity)
            (root / "acl").mkdir(parents=True)
            (root / "torch_npu/csrc/core/npu").mkdir(parents=True)
            (root / "acl/acl.h").write_text(
                f"#define FAKE_ARITY {arity}\n" + _ACL_HEADER
            )
            (root / "torch_npu/csrc/core/npu/NPUStream.h").write_text(_NPU_STREAM)
            (root / "torch_npu/csrc/core/npu/NPUGuard.h").write_text(_NPU_GUARD)
            (root / "fake.cpp").write_text(_FAKE_RUNTIME)
            actual_abi = probe_batch_abi(root)
            if actual_abi != arity:
                raise AssertionError("Compile probe chose an incorrect CANN signature")
            name = f"_native_acl_test_{arity}"
            output = root / (name + sysconfig.get_config_var("EXT_SUFFIX"))
            command = [
                "c++",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-fvisibility=hidden",
                "-O0",
                f"-DTORCH_EXTENSION_NAME={name}",
                f"-DSGLANG_ACL_BATCH_ABI={arity}",
                "-I",
                str(root),
                "-I",
                str(include),
                "-I",
                str(python_include),
                str(_SOURCE / "acl_batch.cpp"),
                str(root / "fake.cpp"),
                "-o",
                str(output),
            ]
            if sys.platform == "darwin":
                command += ["-undefined", "dynamic_lookup"]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(
                    "Actual native shim compile failed:\n" + result.stderr
                )
            spec = importlib.util.spec_from_file_location(name, output)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            control = ctypes.CDLL(
                str(output)
            )  # Explicit ABI of this test's fake runtime.
            cls.modules.append((module, control, arity))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        for _, control, _ in self.modules:
            control.fake_reset()

    @staticmethod
    def arrays():
        source = ctypes.create_string_buffer(b"native-byte-tail!")
        destination = ctypes.create_string_buffer(len(source))
        return source, destination

    def batch(self, module, source, destination, location="unregistered"):
        return module.Batch(
            [ctypes.addressof(source)],
            [ctypes.addressof(destination)],
            [len(source)],
            [len(destination)],
            0,
            location,
            0,
        )

    @staticmethod
    def trace(control):
        return [control.fake_trace_at(i) for i in range(control.fake_trace_size())]

    def test_actual_cpp_supports_both_header_signatures_and_flush_before_copy(self):
        for module, control, arity in self.modules:
            with self.subTest(arity=arity):
                source, destination = self.arrays()
                module.submit(self.batch(module, source, destination), 5)
                self.assertEqual(destination.raw, source.raw)
                self.assertEqual(self.trace(control), [1, 2, 12])
                self.assertEqual(module.api_info()["batch_abi"], arity)
                self.assertFalse(module.api_info()["device_synchronize"])

    def test_unknown_sdk_signature_fails_probe(self):
        root = Path(self.temp.name) / "unknown"
        (root / "acl").mkdir(parents=True)
        (root / "acl/acl.h").write_text("#define FAKE_ARITY 7\n" + _ACL_HEADER)
        with self.assertRaisesRegex(RuntimeError, "no supported"):
            probe_batch_abi(root)

    def test_mismatched_stream_and_failed_flush_cannot_submit_copy(self):
        for module, control, _ in self.modules:
            source, destination = self.arrays()
            batch = self.batch(module, source, destination)
            with self.assertRaisesRegex(RuntimeError, "current torch_npu stream"):
                module.submit(batch, 9)
            self.assertEqual(self.trace(control), [])
            control.fake_set_null(1)
            with self.assertRaisesRegex(RuntimeError, "null stream"):
                module.submit(batch, 5)
            self.assertEqual(self.trace(control), [1])

    def test_failure_reports_real_status_and_available_fail_index(self):
        for module, control, arity in self.modules:
            source, destination = self.arrays()
            control.fake_fail(19)
            with self.assertRaisesRegex(RuntimeError, "ACL status=19") as caught:
                module.submit(self.batch(module, source, destination), 5)
            self.assertIn(
                "fail_index=" + ("1" if arity == 10 else "unavailable"),
                str(caught.exception),
            )
            self.assertIn("partial", str(caught.exception))

    def test_cpp_checks_lengths_capacity_overlap_and_location(self):
        module = self.modules[0][0]
        for args in (
            ([1], [2], [], [3]),
            ([1], [2], [4], [3]),
            ([100, 200], [300, 302], [4, 4], [4, 4]),
        ):
            with self.assertRaises(ValueError):
                module.Batch(*args, 0, "host", 0)
        with self.assertRaisesRegex(ValueError, "location"):
            module.Batch([100], [200], [4], [4], 0, "made-up", 0)

    def test_native_submission_releases_gil(self):
        module, control, _ = self.modules[0]
        source, destination = self.arrays()
        control.fake_delay(120)
        ran = threading.Event()
        worker = threading.Thread(target=lambda: (time.sleep(0.02), ran.set()))
        worker.start()
        module.submit(self.batch(module, source, destination), 5)
        self.assertTrue(ran.is_set(), "Host queue wait must not retain the Python GIL")
        worker.join()

    def test_real_registration_calls_use_sdk_pinned_macro_and_keep_mapping_distinct(
        self,
    ):
        for module, control, _ in self.modules:
            address = module.malloc_host(4096, 0)
            try:
                self.assertEqual(module.register_host(address, 4096, 0, False), 0)
                self.assertEqual(control.fake_flags(), 0x10000000)
                module.unregister_host(address, 0)
                mapped = module.register_host(address, 4096, 0, True)
                self.assertNotEqual(mapped, address)
                self.assertEqual(control.fake_flags(), 0x10000002)
                module.unregister_host(address, 0)
                with self.assertRaisesRegex(ValueError, "4 KiB"):
                    module.register_host(address + 1, 4096, 0, False)
            finally:
                module.free_host(address, 0)

    def test_native_sync_init_copy_preserves_raw_byte_tail(self):
        module = self.modules[0][0]
        source, destination = self.arrays()
        module.memcpy_sync(
            ctypes.addressof(destination),
            len(destination),
            ctypes.addressof(source),
            len(source),
            "device_to_host",
            0,
        )
        self.assertEqual(destination.raw, source.raw)
        with self.assertRaisesRegex(ValueError, "H2D/D2H"):
            module.memcpy_sync(
                ctypes.addressof(destination),
                len(destination),
                ctypes.addressof(source),
                len(source),
                "device_to_device",
                0,
            )

    def test_python_bridge_retains_native_batch_before_submit_even_on_error(self):
        module, control, _ = self.modules[0]
        source, destination = self.arrays()
        retained = []
        submit = NativeAclBatchSubmitter(
            module, retained.append, source_location="unregistered"
        )
        device = SimpleNamespace(type="npu", index=0)
        target = DeviceMemoryRegion(
            ctypes.addressof(destination), len(destination), device, destination
        )
        batch = AclCopyBatch(
            (ctypes.addressof(source),),
            (ctypes.addressof(destination),),
            (len(source),),
            (len(destination),),
            ((source, target),),
        )
        stream = SimpleNamespace(
            device=SimpleNamespace(type="npu", index=0), stream_id=5
        )
        control.fake_fail(23)
        with self.assertRaisesRegex(RuntimeError, "ACL status=23"):
            submit(batch, stream)
        self.assertEqual(len(retained), 1)
        self.assertIs(retained[0][1], batch)
        self.assertIs(retained[0][2], stream)

    def test_python_bridge_rejects_wrong_destination_device_before_submit(self):
        module, control, _ = self.modules[0]
        source, destination = self.arrays()
        target = DeviceMemoryRegion(
            ctypes.addressof(destination),
            len(destination),
            SimpleNamespace(type="npu", index=1),
            destination,
        )
        batch = AclCopyBatch(
            (ctypes.addressof(source),),
            (ctypes.addressof(destination),),
            (len(source),),
            (len(destination),),
            ((source, target),),
        )
        stream = SimpleNamespace(
            device=SimpleNamespace(type="npu", index=0), stream_id=5
        )
        retained = []
        with self.assertRaisesRegex(ValueError, "stream device"):
            NativeAclBatchSubmitter(module, retained.append)(batch, stream)
        self.assertEqual(retained, [])
        self.assertEqual(self.trace(control), [])

    def test_factory_connects_full_host_plan_to_real_cpp_and_completion(self):
        module = self.modules[0][0]
        source, destination = self.arrays()
        device = SimpleNamespace(type="npu", index=0)
        stream = SimpleNamespace(device=device, stream_id=5)
        host = HostMemoryCapability(
            ctypes.addressof(source), len(source), True, None, source
        )
        target = DeviceMemoryRegion(
            ctypes.addressof(destination), len(destination), device, destination
        )
        plan = HostCopyPlan(
            (
                HostCopyRegion(
                    host, target, 0, 0, len(source), len(source), len(destination)
                ),
            )
        )
        retained, events = [], []

        def complete(actual_stream):
            self.assertIs(actual_stream, stream)
            self.assertEqual(destination.raw, source.raw)
            events.append("record")
            return ImmediateCompletion()

        with (
            patch.object(
                native_acl, "_npu_stream_context", lambda stream: nullcontext()
            ),
            patch.object(native_acl, "_npu_completion", complete),
        ):
            ops = native_acl.make_acl_host_ops(
                module=module, retain=retained.append, source_location="unregistered"
            )
            ticket = ops.materialize_into(plan, None, stream)
        self.assertEqual(events, ["record"])
        self.assertIs(retained[-1], ticket)
        self.assertTrue(
            any(
                isinstance(item, tuple) and isinstance(item[0], module.Batch)
                for item in retained
            )
        )

    def test_loader_reuses_same_module_for_normalized_directory(self):
        expected = object()
        with (
            patch.dict(sys.modules, {"torch_npu": SimpleNamespace()}),
            patch.object(native_acl, "_LOADED", {}),
            patch.object(native_acl, "_load_native_acl", return_value=expected) as load,
        ):
            first = native_acl.get_native_acl(self.temp.name)
            second = native_acl.get_native_acl(Path(self.temp.name) / ".")
        self.assertIs(first, expected)
        self.assertIs(second, first)
        load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
