"""CPU tests for explicitly registered and physically budgeted host storage."""

import unittest
from dataclasses import replace

import torch
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.resources import ResourceKind
from sglang.srt.offloading.moe.storage import HostExpertStore, HostMemoryCapability
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def capability(tensor, *, mapped=True, extra=0):
    storage = tensor.untyped_storage()
    return HostMemoryCapability(
        storage.data_ptr(),
        storage.nbytes(),
        True,
        storage.data_ptr() + 4096 if mapped else None,
        tensor,
        storage.nbytes() + extra,
    )


class TestHostCapabilities(unittest.TestCase):
    def test_external_master_requires_explicit_capability_and_preserves_aliases(self):
        master = torch.arange(64, dtype=torch.uint8)
        store = HostExpertStore()
        with self.assertRaises(ValueError):
            store.add(3, "w13", master)
        cap = capability(master, extra=4096)
        store.add(3, "w13", master[:32], capability=cap)
        store.add(3, "w2", master[32:], capability=cap)
        store.seal()
        store.require_device_access()
        allocations = store.allocations()
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].nbytes, 4160)
        self.assertEqual(
            cap.device_pointer(master.data_ptr() + 7, 3), cap.device_address + 7
        )
        self.assertIs(
            store.components[(3, "w13")].untyped_storage(), master.untyped_storage()
        )
        with self.assertRaises(TypeError):
            store.capabilities[(3, "w13")] = None

    def test_pinning_and_device_mapping_are_distinct_capabilities(self):
        master = torch.zeros(8)
        store = HostExpertStore()
        store.add(3, "w", master, capability=capability(master, mapped=False))
        store.seal()
        with self.assertRaisesRegex(ValueError, "device mapping"):
            store.require_device_access()

    def test_bad_ranges_registration_and_conflicting_alias_are_rejected(self):
        master = torch.zeros(8)
        cap = capability(master)
        store = HostExpertStore()
        for bad in (replace(cap, nbytes=1), replace(cap, acl_async_readable=False)):
            with self.assertRaises(ValueError):
                store.add(3, "w", master, capability=bad)
        with self.assertRaises(ValueError):
            replace(cap, registration=None)
        with self.assertRaises(ValueError):
            replace(cap, allocation_nbytes=1)
        with self.assertRaises(ValueError):
            cap.device_pointer(master.data_ptr() + cap.nbytes, 1)
        store.add(3, "w", master, capability=cap)
        with self.assertRaises(ValueError):
            store.add(
                3,
                "scale",
                master,
                capability=replace(cap, device_address=cap.device_address + 1),
            )

    def test_physical_allocator_rounding_requires_explicit_host_budget(self):
        from test_moe_framework import FakeRuntime, catalog, config

        runtime = FakeRuntime()
        runtime.resources = tuple(
            (
                replace(a, nbytes=a.nbytes + 64)
                if a.kind is ResourceKind.HOST_WEIGHTS
                else a
            )
            for a in runtime.resources
        )
        with self.assertRaises(ValueError):
            MoEOffloadOwner(config(), catalog(), runtime)
        cfg = replace(config(), host_allocation_limit_bytes=192)
        owner = MoEOffloadOwner(cfg, catalog(), runtime)
        self.assertEqual(owner.allocation_ledger.usage[ResourceKind.HOST_WEIGHTS], 192)
        owner.close()
        with self.assertRaises(ValueError):
            MoEOffloadOwner(
                replace(config(), host_allocation_limit_bytes=1),
                catalog(),
                FakeRuntime(),
            )


if __name__ == "__main__":
    unittest.main()
