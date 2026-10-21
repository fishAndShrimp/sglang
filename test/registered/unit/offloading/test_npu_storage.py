import ctypes
import os
import types
import unittest
from unittest import mock

from sglang.srt.offloading.moe.npu_storage import (
    MemFabricHostPool,
    _check_host_registration_platform,
)
from sglang.srt.offloading.moe.resources import AllocationLedger, ResourceKind


class FakeOffload:
    Scene = types.SimpleNamespace(LOCAL=0)
    OffloadConfig = types.SimpleNamespace

    def __init__(self):
        self.calls = []
        self.bad_mapping = False
        self.fail_initialize = False

    def initialize(self, config):
        self.calls.append(("initialize", config))
        return 1 if self.fail_initialize else 0

    def malloc(self, nbytes):
        self.calls.append(("malloc", nbytes))
        page = os.sysconf("SC_PAGE_SIZE")
        self.buffer = ctypes.create_string_buffer(nbytes + page)
        self.address = (ctypes.addressof(self.buffer) + page - 1) // page * page
        return self.address

    def get_dva(self, address):
        return (
            address + 65536 + (1 if self.bad_mapping and address != self.address else 0)
        )

    def free(self, address):
        self.calls.append(("free", address))

    def uninitialize(self):
        self.calls.append(("uninitialize",))


class FakeAcl:
    def __init__(self):
        self.calls = []
        self.fail_register = self.fail_unregister = False

    def register_host(self, address, nbytes, device, mapped):
        self.calls.append(("register", address, nbytes, device, mapped))
        if self.fail_register:
            raise RuntimeError("register failed")
        return 0

    def unregister_host(self, address, device):
        self.calls.append(("unregister", address, device))
        if self.fail_unregister:
            raise RuntimeError("unregister failed")


class HostPoolTests(unittest.TestCase):
    def tearDown(self):
        active = MemFabricHostPool._active
        if active is not None:
            active.native_acl.fail_unregister = False
            active.close()

    def pool(self, total=16384, **kwargs):
        offload, acl = FakeOffload(), FakeAcl()
        return MemFabricHostPool(
            total,
            0,
            physical_bytes=1024**3,
            offload=offload,
            native_acl=acl,
            platform_info=("Linux", "6.8.0"),
            **kwargs,
        )

    def test_one_allocation_and_registration_all_views_share_master(self):
        pool = self.pool()
        a, b = pool.allocate(100), pool.allocate(99)
        self.assertEqual(b.tensor.data_ptr() - a.tensor.data_ptr(), 128)
        self.assertIs(a.capability, b.capability)
        a.tensor.fill_(17)
        self.assertEqual(ctypes.c_ubyte.from_address(a.tensor.data_ptr()).value, 17)
        self.assertEqual(
            a.capability.device_pointer(b.tensor.data_ptr(), 99),
            b.tensor.data_ptr() + 65536,
        )
        self.assertTrue(a.capability.acl_async_readable)
        self.assertEqual([x[0] for x in pool.offload.calls], ["initialize", "malloc"])
        self.assertFalse(pool.native_acl.calls[0][-1])
        self.assertEqual(pool.allocations()[0].nbytes, 1024**3)
        ledger = AllocationLedger(
            {ResourceKind.HOST_WEIGHTS: 1024**3}, pool.allocations()
        )
        self.assertEqual(ledger.usage[ResourceKind.HOST_WEIGHTS], 1024**3)

    def test_explicit_budget_and_alignment(self):
        for value in (1, 0, 1024**3 + 1):
            with self.assertRaises(ValueError):
                MemFabricHostPool(
                    1024,
                    0,
                    physical_bytes=value,
                    offload=FakeOffload(),
                    native_acl=FakeAcl(),
                )
        with self.assertRaises(ValueError):
            MemFabricHostPool(
                1024**3 + 1,
                0,
                physical_bytes=1024**3,
                offload=FakeOffload(),
                native_acl=FakeAcl(),
            )
        pool = self.pool(1)
        self.assertEqual(pool.nbytes, os.sysconf("SC_PAGE_SIZE"))
        pool.allocate(pool.nbytes)
        with self.assertRaises(MemoryError):
            pool.allocate(1)
        with self.assertRaises(ValueError):
            pool.allocate(1, alignment=3)

    def test_single_owner_and_idempotent_close(self):
        pool = self.pool()
        with self.assertRaises(RuntimeError):
            self.pool()
        pool.close()
        pool.close()
        self.assertEqual(
            [x[0] for x in pool.native_acl.calls], ["register", "unregister"]
        )
        self.assertEqual(
            [x[0] for x in pool.offload.calls],
            ["initialize", "malloc", "free", "uninitialize"],
        )
        with self.assertRaises(RuntimeError):
            pool.allocate(1)
        self.pool().close()

    def test_registration_or_mapping_failure_frees_without_certification(self):
        for failure in ("register", "mapping", "initialize"):
            offload, acl = FakeOffload(), FakeAcl()
            acl.fail_register = failure == "register"
            offload.bad_mapping = failure == "mapping"
            offload.fail_initialize = failure == "initialize"
            with self.assertRaises(RuntimeError):
                MemFabricHostPool(
                    1024,
                    0,
                    physical_bytes=1024**3,
                    offload=offload,
                    native_acl=acl,
                    platform_info=("Linux", "6.8.0"),
                )
            self.assertIsNone(MemFabricHostPool._active)
            self.assertNotIn("unregister", [x[0] for x in acl.calls])
            self.assertEqual(
                "free" in [x[0] for x in offload.calls], failure != "initialize"
            )

    def test_failed_unregister_preserves_allocation_until_retry(self):
        pool = self.pool()
        pool.native_acl.fail_unregister = True
        with self.assertRaises(RuntimeError):
            pool.close()
        self.assertNotIn("free", [x[0] for x in pool.offload.calls])
        self.assertIs(MemFabricHostPool._active, pool)
        pool.native_acl.fail_unregister = False
        pool.close()
        self.assertIn("free", [x[0] for x in pool.offload.calls])

    def test_unsupported_kernel_rejected_before_native_initialization(self):
        for system, release in (
            ("Darwin", "26.0"),
            ("Linux", "5.10.99"),
            ("Linux", "4.19.0"),
            ("Linux", "unknown"),
        ):
            with self.assertRaises(RuntimeError):
                _check_host_registration_platform(system, release)
        _check_host_registration_platform("Linux", "5.15.0-generic")
        _check_host_registration_platform("Linux", "6.8.0")

    def test_hicache_singleton_conflict_rejected(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_HICACHE_HOST_MEM_BACKEND": "memfabric"}
        ):
            with self.assertRaisesRegex(ValueError, "HiCache"):
                self.pool()
        self.assertIsNone(MemFabricHostPool._active)


if __name__ == "__main__":
    unittest.main()
