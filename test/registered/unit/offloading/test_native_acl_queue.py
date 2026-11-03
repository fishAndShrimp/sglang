"""Compile the production ACL shim to check queue ordering without an NPU SDK."""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNativeAclQueue(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="sglang-acl-queue-")
        build = Path(cls.directory.name)
        # The C++ fixture models only external SDK and queue boundaries. Include
        # and compile the production source intact, including both CANN ABIs.
        for header in (
            "acl/acl.h",
            "pybind11/pybind11.h",
            "pybind11/stl.h",
            "torch_npu/csrc/core/npu/NPUGuard.h",
            "torch_npu/csrc/core/npu/NPUStream.h",
            "torch_npu/csrc/framework/OpCommand.h",
        ):
            path = build / header
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        fixture = Path(__file__).with_name("native_acl_queue_fixture.cpp")
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/offloading/moe/npu/native/acl_batch.cpp"
        )
        cls.executables = {}
        for abi in (9, 10):
            executable = build / f"queue-{abi}"
            command = shlex.split(os.environ.get("CXX", "c++")) + [
                "-std=c++17",
                f"-I{build}",
                f"-DSGLANG_ACL_BATCH_ABI={abi}",
                f'-DSGLANG_NATIVE_ACL_SOURCE="{source}"',
                str(fixture),
                "-o",
                str(executable),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            cls.executables[abi] = executable

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "directory"):
            cls.directory.cleanup()

    def test_queued_copy_preserves_reader_and_ready_event_without_host_flush(self):
        """Old direct submissions drain the model's queue at every bank refill."""
        self._run("queued")

    def test_disabled_queue_preserves_async_device_order(self):
        self._run("immediate")

    def test_launch_errors_survive_queue_and_immediate_modes(self):
        for mode in ("queued_failure", "immediate_failure"):
            self._run(mode)

    def test_cached_descriptor_keeps_failure_output_per_invocation(self):
        self._run("failure_state", abis=(10,))

    def _run(self, case, abis=(9, 10)):
        for abi in abis:
            with self.subTest(case=case, abi=abi):
                result = subprocess.run(
                    [str(self.executables[abi]), case],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
