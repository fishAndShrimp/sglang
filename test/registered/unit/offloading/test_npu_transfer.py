"""CPU byte-copy/ABI contracts. These tests do not certify NPU execution/graphs."""

import itertools
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.offloading.moe.npu_transfer import (
    AclBatchHostTransferOps,
    DeviceMemoryRegion,
    MemFabricCopyComponent,
    MemFabricDeviceTransferOps,
    MemFabricTransferBindings,
    TransferSubmissionError,
)
from sglang.srt.offloading.moe.resources import AllocationLedger, ResourceKind
from sglang.srt.offloading.moe.storage import HostMemoryCapability
from sglang.srt.offloading.transfer import (
    CopySourceKind,
    DeviceCopyPlan,
    HostCopyPlan,
    HostCopyRegion,
    ImmediateCompletion,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestAclBatchTransfer(unittest.TestCase):
    def setUp(self):
        self.retained, self.calls = [], []
        self.npu = SimpleNamespace(type="npu", index=0)
        self.source = HostMemoryCapability(1000, 128, True, 100000, object())
        self.destination = DeviceMemoryRegion(10000, 128, self.npu, object())

    def region(self, source_offset=0, destination_offset=0, nbytes=8):
        return HostCopyRegion(
            self.source,
            self.destination,
            source_offset,
            destination_offset,
            nbytes,
            128,
            128,
        )

    def ops(self, native=None, record=None, **kwargs):
        def submit(batch, stream):
            self.assertTrue(self.retained, "Descriptors must be retained before submit")
            self.calls.append(("submit", batch, stream))
            return 0

        def complete(stream):
            self.calls.append(("record", stream))
            return ImmediateCompletion()

        return AclBatchHostTransferOps(
            native_submit=native or submit,
            record_completion=record or complete,
            retain=self.retained.append,
            queue_ordering_verified=True,
            **kwargs,
        )

    def test_batches_are_bounded_and_completion_covers_all(self):
        plan = HostCopyPlan(tuple(self.region(i * 8, i * 9) for i in range(5)))
        ticket = self.ops(max_batch_entries=2).materialize_into(plan, "bank", "copy")
        batches = [call[1] for call in self.calls if call[0] == "submit"]
        self.assertEqual([len(b.sizes) for b in batches], [2, 2, 1])
        self.assertEqual(batches[0].source_addresses, (1000, 1008))
        self.assertEqual(batches[0].destination_addresses, (10000, 10009))
        self.assertEqual(batches[0].destination_capacities, (128, 119))
        self.assertEqual(batches[0].direction, "host_to_device")
        self.assertEqual(self.calls[-1], ("record", "copy"))
        self.assertIs(self.retained[-1], ticket)
        self.assertIs(ticket.keepalive[0], plan)

    def test_native_shim_must_declare_queue_ordering(self):
        with self.assertRaisesRegex(ValueError, "queue-ordered"):
            AclBatchHostTransferOps(
                native_submit=lambda *a: 0,
                record_completion=lambda *a: ImmediateCompletion(),
                retain=self.retained.append,
            )

    def test_invalid_registration_direction_capacities_and_overlap_rejected(self):
        bad_sources = [object(), replace(self.source, acl_async_readable=False)]
        cases = [replace(self.region(), source=s) for s in bad_sources]
        cases += [
            replace(self.region(), source_capacity=129),
            replace(self.region(), source_capacity=128.0),
            replace(self.region(), destination_capacity=129),
            replace(self.region(), destination=replace(self.destination, device="cpu")),
        ]
        for region in cases:
            with (
                self.subTest(region=region),
                self.assertRaises((TypeError, ValueError)),
            ):
                self.ops().materialize_into(HostCopyPlan((region,)), None, "copy")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.ops(max_batch_entries=1).materialize_into(
                HostCopyPlan((self.region(), self.region(destination_offset=4))),
                None,
                "copy",
            )
        self.assertEqual(self.calls, [])

    def test_zero_length_regions_are_safe_noops_with_completion(self):
        ticket = self.ops().materialize_into(
            HostCopyPlan((self.region(nbytes=0),)), None, "copy"
        )
        self.assertTrue(ticket.completion.done())
        self.assertEqual(self.calls, [("record", "copy")])

    def test_rejects_device_destination_aliasing_mapped_host_source(self):
        aliased = replace(self.destination, address=self.source.device_address)
        with self.assertRaisesRegex(ValueError, "immutable host"):
            self.ops().materialize_into(
                HostCopyPlan((replace(self.region(), destination=aliased),)),
                None,
                "copy",
            )
        self.assertEqual(self.calls, [])

    def test_partial_failure_retains_every_descriptor_and_records_drain_event(self):
        submitted = []

        def submit(batch, stream):
            self.assertTrue(self.retained)
            submitted.append(batch)
            if len(submitted) == 2:
                raise RuntimeError("partially submitted")

        plan = HostCopyPlan(tuple(self.region(i * 8, i * 8) for i in range(3)))
        with self.assertRaises(TransferSubmissionError) as caught:
            self.ops(native=submit, max_batch_entries=1).materialize_into(
                plan, None, "copy"
            )
        self.assertEqual(len(submitted), 2)
        self.assertIsNotNone(caught.exception.completion_ticket)
        self.assertIs(self.retained[0][0], plan)
        self.assertEqual(len(self.retained[0][2]), 3)
        self.assertEqual(self.calls[-1], ("record", "copy"))

    def test_event_failure_preserves_resources_without_claiming_completion(self):
        def fail_record(stream):
            raise RuntimeError("record failed")

        plan = HostCopyPlan((self.region(),))
        with self.assertRaises(TransferSubmissionError) as caught:
            self.ops(record=fail_record).materialize_into(plan, None, "copy")
        self.assertIsNone(caught.exception.completion_ticket)
        self.assertIs(self.retained[0][0], plan)

    def test_device_region_tensor_views_have_bounded_extent(self):
        base = torch.empty(20, dtype=torch.uint8)
        region = DeviceMemoryRegion.from_tensor(base[3:9])
        self.assertEqual(region.address, base.data_ptr() + 3)
        self.assertEqual(region.nbytes, 6)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            DeviceMemoryRegion.from_tensor(base[::2])


class FakeGroupPackCopy:
    """Exact source[i] -> destination[j] native compaction, including stale tails."""

    def __init__(self, registry):
        self.registry = registry
        self.calls = []

    def __call__(self, src, dst, lengths, scan, groups, packed, device):
        assert src.dtype == dst.dtype == groups.dtype == packed.dtype == torch.int64
        assert lengths.dtype == scan.dtype == torch.uint32
        count = int(scan.numpy())
        self.calls.append((src.clone(), dst.clone(), lengths.clone(), groups.clone()))
        output = 0
        for index in range(count):
            if groups.numpy()[index] == 0:
                continue
            source = int(src.numpy()[index])
            destination = int(dst.numpy()[output])
            nbytes = int(lengths.numpy()[index])
            self.registry[destination][:nbytes].copy_(self.registry[source][:nbytes])
            packed[output] = groups[index]
            output += 1
        return 0


class TestMemFabricTransfer(unittest.TestCase):
    def fixture(
        self,
        capacity=5,
        host_count=6,
        hot_count=2,
        destination_count=9,
        batch=2,
        lengths=(13, 5),
    ):
        self.registry, self.retained, self.host, self.hot, self.targets = (
            {},
            [],
            [],
            [],
            [],
        )
        components = []
        for component_index, length in enumerate(lengths):
            hosts = [
                torch.full((length,), 10 + i + component_index * 30, dtype=torch.uint8)
                for i in range(host_count)
            ]
            hots = [
                torch.full((length,), 100 + i + component_index * 30, dtype=torch.uint8)
                for i in range(hot_count)
            ]
            targets = [
                torch.full((length,), 255, dtype=torch.uint8)
                for _ in range(destination_count)
            ]
            host_regions = []
            for value in hosts:
                # Mapped device address is deliberately distinct from CPU data_ptr.
                mapped = value.data_ptr() + (1 << 40)
                self.registry[mapped] = value
                host_regions.append(
                    HostMemoryCapability(value.data_ptr(), length, True, mapped, value)
                )
            for value in hots + targets:
                self.registry[value.data_ptr()] = value
            components.append(
                MemFabricCopyComponent(
                    tuple(host_regions),
                    tuple(DeviceMemoryRegion.from_tensor(t) for t in hots),
                    tuple(DeviceMemoryRegion.from_tensor(t) for t in targets),
                    length,
                    name=f"component{component_index}",
                )
            )
            self.host.append(hosts)
            self.hot.append(hots)
            self.targets.append(targets)
        self.kernel = FakeGroupPackCopy(self.registry)
        self.generations = torch.arange(hot_count, dtype=torch.int64) + 7
        self.error = torch.zeros((), dtype=torch.int32)
        self.bindings = MemFabricTransferBindings(
            components,
            capacity=capacity,
            hot_generations=self.generations,
            error_status=self.error,
            device="cpu",
            allow_cpu=True,
            max_batch_entries=batch,
        )
        self.ops = MemFabricDeviceTransferOps(
            retain=self.retained.append,
            group_pack_copy=self.kernel,
            record_completion=lambda stream: ImmediateCompletion(),
            allow_cpu=True,
        )
        return self.bindings

    def plan(self, kinds, sources, destinations, valid=None, generations=None):
        if valid is None:
            valid = [kind != CopySourceKind.NONE for kind in kinds]
        if generations is None:
            generations = [
                -1 if kind != CopySourceKind.HOT_SLOT else 7 + src
                for kind, src in zip(kinds, sources)
            ]
        return DeviceCopyPlan(
            torch.tensor(kinds, dtype=torch.int64),
            torch.tensor(sources, dtype=torch.int64),
            torch.tensor(destinations, dtype=torch.int64),
            torch.tensor(valid, dtype=torch.bool),
            torch.tensor(generations, dtype=torch.int64),
            torch.tensor(sum(valid), dtype=torch.int32),
            len(kinds),
        )

    def test_sparse_nonmonotonic_destinations_copy_all_components_and_tails(self):
        self.fixture()
        plan = self.plan([1, 0, 2, 1, 0], [3, -1, 1, 0, -1], [7, -1, 2, 0, -1])
        ticket = self.ops.materialize_into(plan, self.bindings, "copy")
        self.assertEqual(self.error, 0)
        self.assertEqual(len(self.kernel.calls), 6)  # 3 bounded batches x 2 components
        for component in range(2):
            for slot, expected in (
                (7, self.host[component][3]),
                (2, self.hot[component][1]),
                (0, self.host[component][0]),
            ):
                torch.testing.assert_close(self.targets[component][slot], expected)
            for slot in (1, 3, 4, 5, 6, 8):
                self.assertTrue((self.targets[component][slot] == 255).all())
        self.assertIs(self.retained[-1], ticket)

    def test_every_sparse_mask_preserves_native_packed_destination_order(self):
        self.fixture(capacity=5, batch=3, lengths=(7,))
        destinations = [7, 2, 8, 0, 4]
        for mask in itertools.product([False, True], repeat=5):
            for value in self.targets[0]:
                value.fill_(255)
            plan = self.plan([1] * 5, list(range(5)), destinations, list(mask))
            self.ops.materialize_into(plan, self.bindings, "copy")
            for index, slot in enumerate(destinations):
                expected = (
                    self.host[0][index]
                    if mask[index]
                    else torch.full((7,), 255, dtype=torch.uint8)
                )
                torch.testing.assert_close(self.targets[0][slot], expected)
            self.assertEqual(self.error, 0)

    def test_replay_empty_mask_clears_packed_tail_and_reuses_metadata_addresses(self):
        self.fixture(capacity=3, batch=3)
        plan = self.plan([1, 1, 1], [0, 1, 2], [7, 2, 0])
        pointers = [
            allocation.storage_key for allocation in self.bindings.allocations()
        ]
        self.ops.materialize_into(plan, self.bindings, "copy")
        plan.valid.zero_()
        plan.count.zero_()
        self.ops.materialize_into(plan, self.bindings, "copy")
        for batches in self.bindings.batches:
            _, _, src, dst, _, _, groups, packed = batches[0]
            for tensor in (src, dst, groups, packed):
                self.assertFalse(tensor.any())
        self.assertEqual(pointers, [a.storage_key for a in self.bindings.allocations()])

    def test_invalid_dynamic_metadata_poison_whole_bundle_before_any_write(self):
        corruptions = [
            lambda p: p.source_kind.fill_(99),
            lambda p: p.source_index.fill_(999),
            lambda p: p.destination_slot.fill_(-1),
            lambda p: p.destination_slot.fill_(0),
            lambda p: p.count.fill_(1),
            lambda p: p.expected_generation.fill_(0),
        ]
        for corrupt in corruptions:
            self.fixture(capacity=2)
            plan = self.plan([2, 1], [0, 0], [0, 1])
            corrupt(plan)
            self.ops.materialize_into(plan, self.bindings, "copy")
            self.assertNotEqual(self.error, 0)
            for component in self.targets:
                self.assertTrue(all((value == 255).all() for value in component))
            for _, _, _, groups in self.kernel.calls:
                self.assertFalse(groups.any())

    def test_preexisting_error_blocks_copy_and_is_preserved(self):
        self.fixture(capacity=1)
        self.error.fill_(8)
        self.ops.materialize_into(self.plan([1], [0], [0]), self.bindings, "copy")
        self.assertEqual(self.error, 8)
        self.assertTrue((self.targets[0][0] == 255).all())

    def test_empty_hot_pool_empty_plan_and_invalid_hot_source_are_safe(self):
        self.fixture(capacity=0, hot_count=0)
        self.ops.materialize_into(self.plan([], [], []), self.bindings, "copy")
        self.assertEqual(self.kernel.calls, [])
        self.fixture(capacity=1, hot_count=0)
        self.ops.materialize_into(self.plan([2], [0], [0]), self.bindings, "copy")
        self.assertNotEqual(self.error, 0)

    def test_no_dynamic_host_scalar_or_list_conversion(self):
        self.fixture(capacity=3)
        plan = self.plan([1, 0, 2], [0, -1, 1], [7, -1, 2])
        with (
            patch.object(
                torch.Tensor, "item", side_effect=AssertionError("host scalar read")
            ),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("host list read")
            ),
        ):
            self.ops.materialize_into(plan, self.bindings, "copy")

    def test_binding_rejects_unmapped_unbounded_mistyped_and_aliasing_regions(self):
        self.fixture(capacity=1)
        component = self.bindings.components[0]
        with self.assertRaisesRegex(ValueError, "mapping"):
            replace(
                component,
                host_sources=(replace(component.host_sources[0], device_address=None),),
            )
        with self.assertRaisesRegex(ValueError, "uint32"):
            replace(component, nbytes=2**32)
        with self.assertRaisesRegex(ValueError, "asynchronous ACL"):
            replace(
                component,
                host_sources=(
                    replace(component.host_sources[0], acl_async_readable=False),
                ),
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            MemFabricTransferBindings(
                [replace(component, destinations=(component.destinations[0],) * 2)],
                capacity=1,
                hot_generations=self.generations,
                error_status=self.error,
                device="cpu",
                allow_cpu=True,
            )
        plan = self.plan([1], [0], [0])
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self.ops.materialize_into(
                replace(plan, count=plan.count.to(torch.int64)), self.bindings, "copy"
            )
        with self.assertRaisesRegex(ValueError, "NPU"):
            MemFabricTransferBindings(
                [component],
                capacity=1,
                hot_generations=self.generations,
                error_status=self.error,
                device="cpu",
            )

    def test_hot_arena_update_allows_host_but_rejects_hot_copy_cycles(self):
        self.fixture(capacity=1)
        components = [
            replace(c, destinations=c.hot_sources) for c in self.bindings.components
        ]
        bindings = MemFabricTransferBindings(
            components,
            capacity=1,
            hot_generations=self.generations,
            error_status=self.error,
            device="cpu",
            allow_cpu=True,
        )
        self.ops.materialize_into(self.plan([1], [0], [0]), bindings, "copy")
        self.assertEqual(self.error, 0)
        torch.testing.assert_close(self.hot[0][0], self.host[0][0])
        self.ops.materialize_into(self.plan([2], [0], [1]), bindings, "copy")
        self.assertNotEqual(self.error, 0)

    def test_multiple_bindings_metadata_combine_in_single_allocation_ledger(self):
        self.fixture(capacity=3)
        second = MemFabricTransferBindings(
            self.bindings.components,
            capacity=3,
            hot_generations=self.generations,
            error_status=self.error,
            device="cpu",
            allow_cpu=True,
        )
        allocations = self.bindings.allocations() + second.allocations()
        total = sum(allocation.nbytes for allocation in allocations)
        ledger = AllocationLedger({ResourceKind.METADATA: total}, allocations)
        self.assertEqual(ledger.usage[ResourceKind.METADATA], total)

    def test_native_error_retains_all_components_before_first_submit(self):
        self.fixture(capacity=3)
        plan = self.plan([1, 1, 1], [0, 1, 2], [0, 1, 2])

        def failed(*args):
            self.assertTrue(self.retained)
            self.assertIs(self.retained[0][1], self.bindings)
            self.kernel(*args)  # partial bytes can already have been written
            return 31

        self.ops.group_pack_copy = failed
        with self.assertRaises(TransferSubmissionError) as caught:
            self.ops.materialize_into(plan, self.bindings, "copy")
        self.assertIsNotNone(caught.exception.completion_ticket)
        self.assertEqual(len(self.kernel.calls), 1)


if __name__ == "__main__":
    unittest.main()
