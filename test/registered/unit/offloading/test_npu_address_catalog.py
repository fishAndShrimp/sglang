"""Real catalog/binding allocation and lowering with CPU metadata at the device seam.

Only NPU allocation, stream execution and the native copy are substituted. The
address catalog, scratch buffers, aliases, byte accounting and tensor lowering
are the production implementations; this is not an NPU graph certification.
"""

import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.offloading.moe.npu.transfer import (
    DeviceMemoryRegion,
    MemFabricAddressCatalog,
    MemFabricCopyComponent,
    MemFabricDeviceTransferOps,
    MemFabricTransferBindings,
)
from sglang.srt.offloading.moe.resources import AllocationLedger, ResourceKind
from sglang.srt.offloading.moe.storage import HostMemoryCapability
from sglang.srt.offloading.transfer import (
    CopySourceKind,
    DeviceCopyPlan,
    ImmediateCompletion,
)


class TestNpuAddressCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Naming the PrivateUse1 device does not initialize an accelerator.
        backend = torch._C._get_privateuse1_backend_name()
        if backend == "privateuseone":
            torch.utils.rename_privateuse1_backend("npu")
        elif backend != "npu":
            raise unittest.SkipTest("PrivateUse1 belongs to a different backend")

    def setUp(self):
        self.device = torch.device("npu:0")
        original = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original(*args, **kwargs)

        def validate(binding, tensor, shape, dtype):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError("Invalid metadata shape/dtype/layout")

        allocation = mock.patch.object(torch, "tensor", side_effect=cpu_tensor)
        validation = mock.patch.object(
            MemFabricTransferBindings, "_validate_tensor", validate
        )
        allocation.start()
        validation.start()
        self.addCleanup(validation.stop)
        self.addCleanup(allocation.stop)
        self.owners = [object(), object()]
        self.components = tuple(self.component(i) for i in range(2))
        self.catalog = MemFabricAddressCatalog(self.components, device=self.device)
        self.generations = torch.tensor([3, 4], dtype=torch.int64)
        self.error = torch.zeros((), dtype=torch.int32)

    def component(self, index):
        base = 100000 * (index + 1)
        owner = self.owners[index]
        return MemFabricCopyComponent(
            tuple(
                HostMemoryCapability(base + 64 * i, 64, True, base + 64 * i, owner)
                for i in range(3)
            ),
            tuple(
                DeviceMemoryRegion(base + 10000 + 64 * i, 64, self.device, owner)
                for i in range(2)
            ),
            tuple(
                DeviceMemoryRegion(base + 20000 + 64 * i, 64, self.device, owner)
                for i in range(2)
            ),
            64,
            f"component{index}",
        )

    def bind(self, name, components=None, **kwargs):
        return MemFabricTransferBindings(
            self.components if components is None else components,
            capacity=2,
            hot_generations=self.generations,
            error_status=self.error,
            device=self.device,
            name=name,
            address_catalog=self.catalog,
            **kwargs,
        )

    def test_tables_share_storage_but_all_writable_scratch_is_independent(self):
        first = self.bind("first")
        # A separately constructed Python bank view of the same allocation shares.
        views = tuple(
            replace(
                c,
                destinations=tuple(replace(r, owner=object()) for r in c.destinations),
            )
            for c in self.components
        )
        second = self.bind("second", views)
        for left, right in zip(first.tables, second.tables):
            for a, b in zip(left, right):
                self.assertIs(a, b)
                self.assertEqual(
                    a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr()
                )
        first_addresses = {t.data_ptr() for t in first._tensors}
        second_addresses = {t.data_ptr() for t in second._tensors}
        self.assertFalse(first_addresses & second_addresses)
        first.mask.fill_(True)
        self.assertFalse(bool(second.mask.any()))
        allocations = (
            self.catalog.allocations() + first.allocations() + second.allocations()
        )
        ledger = AllocationLedger({ResourceKind.METADATA: 1 << 20}, allocations)
        self.assertEqual(len(self.catalog.allocations()), 6)
        self.assertEqual(
            sum(a.nbytes for a in self.catalog.allocations()), 8 * 2 * (3 + 2 + 2 + 3)
        )
        self.assertEqual(
            ledger.usage[ResourceKind.METADATA], sum(a.nbytes for a in allocations)
        )

    def test_new_destination_view_reuses_sources_and_caches_alias_validation(self):
        first = self.bind("execution")
        update_components = tuple(
            replace(c, destinations=c.hot_sources) for c in self.components
        )
        update = self.bind("update", update_components)
        again = self.bind("again", update_components)
        self.assertFalse(first.hot_targets_alias)
        self.assertTrue(update.hot_targets_alias)
        self.assertEqual(len(self.catalog.allocations()), 8)
        for original, changed, repeated in zip(
            first.tables, update.tables, again.tables
        ):
            self.assertIs(original[0], changed[0])
            self.assertIs(original[1], changed[1])
            self.assertIs(changed[2], repeated[2])
            self.assertIsNot(original[2], changed[2])

    def test_seal_preserves_cached_views_and_rejects_late_table_allocation(self):
        first = self.bind("before_capture")
        self.catalog.seal()
        again = self.bind("same_view")
        self.assertIs(first.tables[0][2], again.tables[0][2])
        update = tuple(replace(c, destinations=c.hot_sources) for c in self.components)
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            self.bind("late_update", update)

    def test_rejects_changed_sources_and_overlapping_or_host_alias_destinations(self):
        changed = replace(
            self.components[0], host_sources=self.components[0].host_sources[1:]
        )
        with self.assertRaisesRegex(ValueError, "sources differ"):
            self.bind("bad_source", (changed, self.components[1]))
        overlapping = replace(
            self.components[0], destinations=(self.components[0].destinations[0],) * 2
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.bind("overlap", (overlapping, self.components[1]))
        host_alias = replace(
            self.components[0],
            destinations=(
                replace(
                    self.components[0].destinations[0],
                    address=self.components[0].host_sources[0].device_address,
                ),
                self.components[0].destinations[1],
            ),
        )
        with self.assertRaisesRegex(ValueError, "host storage overlaps"):
            self.bind("host_alias", (host_alias, self.components[1]))

    def test_lowering_changes_only_scratch_and_faults_suppress_all_components(self):
        binding = self.bind("decode")
        other = self.bind("next")
        before = tuple(t.clone() for tables in binding.tables for t in tables)
        submitted, retained = [], []
        ops = MemFabricDeviceTransferOps(
            retain=retained.append,
            group_pack_copy=lambda src, dst, lengths, scan, groups, packed, device: submitted.append(
                (src.clone(), dst.clone(), groups.clone())
            ),
            record_completion=lambda stream: ImmediateCompletion(),
            stream_context=lambda stream: nullcontext(),
        )
        plan = DeviceCopyPlan(
            torch.tensor([int(CopySourceKind.HOST), int(CopySourceKind.HOT_SLOT)]),
            torch.tensor([2, 1]),
            torch.tensor([0, 1]),
            torch.tensor([True, True]),
            torch.tensor([-1, 4]),
            torch.tensor(2, dtype=torch.int32),
            2,
        )
        stream = SimpleNamespace(device=self.device, stream_id=1)
        ops.materialize_into(plan, binding, stream)
        self.assertEqual(len(submitted), 2)
        for component, (src, dst, groups) in zip(self.components, submitted):
            self.assertEqual(
                src.tolist(),
                [
                    component.host_sources[2].device_address,
                    component.hot_sources[1].address,
                ],
            )
            self.assertEqual(dst.tolist(), [r.address for r in component.destinations])
            self.assertEqual(groups.tolist(), [1, 1])
        self.assertFalse(bool(other.mask.any()))
        for a, b in zip(before, (t for tables in binding.tables for t in tables)):
            self.assertTrue(torch.equal(a, b))
        submitted.clear()
        plan.expected_generation[1] = 99
        ops.materialize_into(plan, binding, stream)
        self.assertNotEqual(int(self.error), 0)
        self.assertTrue(all(not bool(groups.any()) for _, _, groups in submitted))


if __name__ == "__main__":
    unittest.main()
