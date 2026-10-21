"""CPU assembly contracts: real residency, transfer metadata and allocation ledger.

Only native allocation, device events, byte-copy kernels and EP/FFN construction
are simulated. These tests do not claim accelerator or collective correctness.
"""

import ctypes
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.offloading.moe.config import (
    ExpertCatalog,
    ExpertLayerSpec,
    ExpertWave,
    MoEOffloadConfig,
    Phase,
)
from sglang.srt.offloading.moe.execution import ExecutionRequest
from sglang.srt.offloading.moe.npu import backend as npu_backend
from sglang.srt.offloading.moe.npu import experts as npu_experts
from sglang.srt.offloading.moe.npu import native_acl
from sglang.srt.offloading.moe.npu.backend import NpuMoEBackend
from sglang.srt.offloading.moe.npu.transfer import (
    DeviceMemoryRegion,
    MemFabricDeviceTransferOps,
)
from sglang.srt.offloading.moe.owner import MoEOffloadOwner
from sglang.srt.offloading.moe.policy import TensorLFUPolicy
from sglang.srt.offloading.moe.resources import Allocation, ResourceKind
from sglang.srt.offloading.moe.storage import HostExpertStore, HostMemoryCapability
from sglang.srt.offloading.transfer import ImmediateCompletion, TransferTicket


@dataclass
class QuantInfo:
    w13_weight: object
    w2_weight: object
    w13_weight_scale: object
    w2_weight_scale: object


class Bank:
    def __init__(self, tensors, kind, name, owner=None):
        self.tensors, self.kind, self.name, self.owner = tensors, kind, name, owner
        self.capacity = next(iter(tensors.values())).shape[0]
        self.component_regions = {
            name: tuple(DeviceMemoryRegion.from_tensor(row) for row in tensor)
            for name, tensor in tensors.items()
        }

    def view(self, start, count):
        if start < 0 or count < 0 or start + count > self.capacity:
            raise ValueError("Bank slice exceeds physical allocation")
        return Bank(
            {
                name: tensor.narrow(0, start, count)
                for name, tensor in self.tensors.items()
            },
            self.kind,
            self.name,
            owner=self,
        )

    def quant_info(self):
        return QuantInfo(*(self.tensors[name] for name in Format.component_nbytes))

    def allocations(self):
        if self.owner is not None:
            return ()
        return tuple(
            Allocation(
                f"{self.name}.{name}",
                self.kind,
                tensor.untyped_storage().nbytes(),
                (str(tensor.device), tensor.untyped_storage().data_ptr()),
                tensor,
            )
            for name, tensor in self.tensors.items()
            if tensor.numel()
        )


class Format:
    component_nbytes = {
        "w13_weight": 8,
        "w2_weight": 8,
        "w13_weight_scale_inv": 4,
        "w2_weight_scale_inv": 4,
    }
    expert_nbytes = sum(component_nbytes.values())

    def __init__(self):
        self.banks = []

    def allocate_bank(self, capacity, kind, name):
        bank = Bank(
            {
                key: torch.zeros(capacity, size, dtype=torch.uint8)
                for key, size in self.component_nbytes.items()
            },
            kind,
            name,
        )
        self.banks.append(bank)
        return bank


class HostPool:
    def __init__(self, nbytes, events):
        self.tensor = torch.zeros(nbytes, dtype=torch.uint8)
        self.events = events
        self.capability = HostMemoryCapability(
            self.tensor.data_ptr(), nbytes, True, self.tensor.data_ptr(), self
        )

    def allocations(self):
        return (
            Allocation(
                "host.master",
                ResourceKind.HOST_WEIGHTS,
                self.tensor.numel(),
                ("cpu", self.tensor.data_ptr()),
                self,
            ),
        )

    def close(self):
        self.events.append("pool.close")


class Storage:
    def __init__(self, spec, pool, offset, events):
        self.spec, self.events = spec, events
        self.host_store = HostExpertStore()
        self.component_capabilities = {}
        for component, nbytes in Format.component_nbytes.items():
            view = pool.tensor.narrow(0, offset, spec.local_experts * nbytes)
            self.host_store.add(
                spec.layer_id, component, view, capability=pool.capability
            )
            capabilities = []
            for local in range(spec.local_experts):
                # Distinct complete expert bytes make incorrect layer flattening
                # or bank offsets visible to copy tests.
                view[local * nbytes : (local + 1) * nbytes].fill_(
                    spec.layer_id * 10 + local
                )
                address = view.data_ptr() + local * nbytes
                capabilities.append(
                    HostMemoryCapability(address, nbytes, True, address, pool)
                )
            self.component_capabilities[component] = tuple(capabilities)
            offset += view.numel()
        self.host_store.seal()
        self.end = offset

    def close(self):
        self.events.append(f"storage.{self.spec.layer_id}.close")


class Device:
    def __init__(self, events):
        self.events, self.copy_stream, self.compute = events, object(), object()

    def current_stream(self):
        return self.compute

    def record(self, stream):
        return TransferTicket(ImmediateCompletion(), (stream,))

    def stream_scope(self, stream):
        return nullcontext()

    def synchronize(self):
        self.events.append("device.synchronize")


class HostOps:
    def __init__(self, retain):
        self.retain, self.plans = retain, []

    def materialize_into(self, plan, bindings, stream):
        self.retain((plan, bindings))
        self.plans.append(plan)
        for region in plan.regions:
            source, destination = region.source, region.destination
            source.validate_range(
                source.host_address + region.source_offset, region.nbytes
            )
            if region.destination_offset + region.nbytes > destination.nbytes:
                raise ValueError("Physical destination overflow")
            ctypes.memmove(
                destination.address + region.destination_offset,
                source.host_address + region.source_offset,
                region.nbytes,
            )
        return TransferTicket(ImmediateCompletion(), (plan, bindings))


class DeviceOps:
    """Test-only byte transport; production never enables CPU MemFabric."""

    def __init__(self, retain):
        self.retain = retain

    def materialize_into(self, plan, binding, stream):
        assert plan.count.dtype == torch.int32
        self.retain((plan, binding))
        MemFabricDeviceTransferOps._mask(plan, binding)
        for lane in range(plan.capacity):
            if not bool(binding.mask[lane]):
                continue
            index = int(plan.source_index[lane])
            destination = int(plan.destination_slot[lane])
            for component in binding.components:
                if int(plan.source_kind[lane]) == 1:
                    source = component.host_sources[index].host_address
                else:
                    source = component.hot_sources[index].address
                ctypes.memmove(
                    component.destinations[destination].address,
                    source,
                    component.nbytes,
                )
        return TransferTicket(ImmediateCompletion(), (plan, binding))


class Assembly:
    """Native seams are scoped; all metadata/ownership code is the real code."""

    def __init__(self, ratio=0.5, slots=4, group=2, layer_count=3, local_experts=5):
        self.events, self.bindings, self.builds = [], [], []
        self.catalog = ExpertCatalog(
            tuple(
                ExpertLayerSpec(layer, local_experts * 2, 2, 1)
                for layer in range(3, 3 + layer_count)
            )
        )
        self.config = MoEOffloadConfig(
            ratio, slots, 2, 1, workspace_limit_bytes=2**20, metadata_limit_bytes=2**20
        )
        self.policy = TensorLFUPolicy()
        backend = self.backend = NpuMoEBackend.__new__(NpuMoEBackend)
        backend.settings = SimpleNamespace(
            prefill_group_size=group,
            prefill_microbatch_size=1,
            prefill_max_inflight=2,
            max_dispatch_rows=32,
        )
        backend.config, backend.device = self.config, torch.device("cpu")
        backend.format = Format()
        backend.host_pool = HostPool(
            self.catalog.num_experts * Format.expert_nbytes, self.events
        )
        backend.device_runtime = Device(self.events)
        backend.layers, backend._unbound_resources = {}, []
        backend.runtime, backend._closed, backend.native_acl = None, False, object()
        self.layers, offset = [], 0
        for spec in self.catalog.offloaded_layers:
            storage = Storage(spec, backend.host_pool, offset, self.events)
            offset = storage.end
            backend.layers[spec.layer_id] = storage
            expert = SimpleNamespace(
                layer_id=spec.layer_id, _moe_offload_storage=storage
            )
            self.layers.append(
                (SimpleNamespace(mlp=SimpleNamespace(experts=expert)), spec)
            )
        self.shared_workspace = torch.zeros(32, dtype=torch.uint8)
        self.fail_layer = None

    def bind(self, *args, **kwargs):
        components = tuple(args[0])
        first = components[0]
        result = SimpleNamespace(
            **kwargs,
            components=components,
            host_count=len(first.host_sources),
            hot_count=len(first.hot_sources),
            destination_count=len(first.destinations),
            hot_targets_alias=first.hot_sources == first.destinations,
            mask=torch.zeros(kwargs["capacity"], dtype=torch.bool),
            destination_writes=torch.zeros(
                len(first.destinations) + 1, dtype=torch.int64
            ),
            allocations=lambda: (),
        )
        self.bindings.append(result)
        return result

    def build(self, experts, **kwargs):
        if experts.layer_id == self.fail_layer:
            raise RuntimeError("injected EP builder failure")
        shared = kwargs["shared_layout"]
        layout = (
            shared
            if shared is not None
            else SimpleNamespace(workspace=self.shared_workspace)
        )
        allocation = Allocation(
            "dispatch.shared",
            ResourceKind.WORKSPACE,
            self.shared_workspace.numel(),
            ("cpu", self.shared_workspace.data_ptr()),
            self.shared_workspace,
        )
        result = SimpleNamespace(
            prefill=object(),
            decode=object(),
            layout=layout,
            allocations=lambda: (allocation,),
            **{key: value for key, value in kwargs.items() if key != "shared_layout"},
        )
        self.builds.append((experts.layer_id, result, shared))
        return result

    def host_ops(self, **kwargs):
        self.host_arguments = kwargs
        self.host = HostOps(kwargs["retain"])
        return self.host

    def finalize(self):
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    npu_backend, "MemFabricTransferBindings", side_effect=self.bind
                )
            )
            stack.enter_context(
                mock.patch.object(
                    npu_backend,
                    "MemFabricDeviceTransferOps",
                    side_effect=lambda **kwargs: DeviceOps(**kwargs),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    native_acl, "make_acl_host_ops", side_effect=self.host_ops
                )
            )
            stack.enter_context(
                mock.patch.object(
                    npu_experts, "build_npu_layer_programs", side_effect=self.build
                )
            )
            return self.backend.finalize(
                self.layers, self.catalog, self.config, self.policy
            )

    def close(self):
        if self.backend.runtime is not None:
            self.backend.runtime.close()
        else:
            self.backend.close_unbound()


class NpuBackendAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.assemblies = []

    def tearDown(self):
        for assembly in self.assemblies:
            assembly.close()

    def assemble(self, **kwargs):
        assembly = Assembly(**kwargs)
        self.assemblies.append(assembly)
        return assembly

    def test_physical_resources_are_counted_once_across_layers_and_bank_views(self):
        for ratio in (0.0, 0.25, 0.75, 1.0):
            with self.subTest(ratio=ratio):
                assembly = self.assemble(ratio=ratio)
                runtime = assembly.finalize()
                owner = MoEOffloadOwner(assembly.config, assembly.catalog, runtime)
                usage = owner.allocation_ledger.usage
                self.assertEqual(
                    usage[ResourceKind.HOST_WEIGHTS], 15 * Format.expert_nbytes
                )
                self.assertEqual(
                    usage[ResourceKind.HOT_WEIGHTS],
                    int(15 * ratio) * Format.expert_nbytes,
                )
                self.assertEqual(
                    usage[ResourceKind.EXECUTION_WEIGHTS], 4 * Format.expert_nbytes
                )
                self.assertEqual(usage[ResourceKind.WORKSPACE], 32)
                self.assertEqual(len(assembly.backend.format.banks), 2)
                self.assertEqual(len(assembly.bindings), 7)  # tail + 2 waves per layer
                self.assertEqual(
                    [x.capacity for x in assembly.bindings[1:]], [4, 1] * 3
                )
                self.assertTrue(
                    all(
                        result.layout is assembly.builds[0][1].layout
                        for _, result, _ in assembly.builds
                    )
                )
                self.assertEqual(
                    assembly.host_arguments["source_location"], "unregistered"
                )
                self.assertIs(
                    assembly.host_arguments["module"], assembly.backend.native_acl
                )

    def test_all_bindings_share_fault_and_generation_state_and_catalog_order(self):
        assembly = self.assemble()
        runtime = assembly.finalize()
        for binding in assembly.bindings:
            self.assertIs(binding.error_status, runtime.ops.error_status)
            self.assertIs(binding.hot_generations, runtime.residency.slot_generation)
            for name, component in zip(Format.component_nbytes, binding.components):
                expected = tuple(
                    cap
                    for spec in assembly.catalog.offloaded_layers
                    for cap in assembly.backend.layers[
                        spec.layer_id
                    ].component_capabilities[name]
                )
                self.assertEqual(component.host_sources, expected)
        for _, program, _ in assembly.builds:
            self.assertIs(program.residency, runtime.ops)

    def test_prefill_bank_offsets_copy_correct_layer_and_combine_adjacent_views(self):
        for ratio in (0.0, 0.5):
            with self.subTest(ratio=ratio):
                assembly = self.assemble(ratio=ratio)
                runtime = assembly.finalize()
                context = runtime._new_context(Phase.PREFILL)
                runtime._context = context
                pipeline = runtime.pipelines[1]
                access = pipeline.access
                leases = []
                for expert in range(2):
                    wave = ExpertWave(expert, expert, expert + 1, 0)
                    request = pipeline.make_request(4, wave, expert, context)
                    lease = access.stage(request, context)
                    leases.append(lease)
                    for tensor in lease.bindings.bank.tensors.values():
                        self.assertTrue(torch.all(tensor == 40 + expert))
                whole = ExpertWave(0, 0, 2, 0)
                combined = pipeline.combine_bindings(4, whole, leases, context)
                self.assertEqual(combined.bank.capacity, 2)
                for tensor in combined.bank.tensors.values():
                    self.assertEqual(tensor[:, 0].tolist(), [40, 41])
                with self.assertRaisesRegex(ValueError, "adjacent"):
                    pipeline.combine_bindings(4, whole, list(reversed(leases)), context)
                a = combined.bank
                b = pipeline.make_request(
                    4, ExpertWave(1, 2, 4, 1), 0, context
                ).local_demand.binding.bank
                for name in Format.component_nbytes:
                    a_ranges = {
                        (region.address, region.nbytes)
                        for region in a.component_regions[name]
                    }
                    b_ranges = {
                        (region.address, region.nbytes)
                        for region in b.component_regions[name]
                    }
                    self.assertTrue(a_ranges.isdisjoint(b_ranges))
                for lease in leases:
                    access.release_after(
                        lease, context.record_compute_completion(), context
                    )

    def test_decode_transfer_and_restore_reuse_fixed_bank_storage(self):
        assembly = self.assemble(ratio=0.25)
        runtime = assembly.finalize()
        context = runtime._new_context(Phase.DECODE, capturing=True)
        runtime._context = context
        runtime.ops.begin_step(context)
        access = runtime.pipelines[0].access
        targets = torch.tensor([0, 5, 14], dtype=torch.int64)
        access.apply_update(targets, context)
        hot, execution = assembly.backend.format.banks
        before = {name: tensor.data_ptr() for name, tensor in hot.tensors.items()}
        for tensor in hot.tensors.values():
            self.assertEqual(tensor[:, 0].tolist(), [30, 40, 54])
        wave = assembly.builds[0][1].decode_waves[0]
        lease = access.stage(
            ExecutionRequest(3, torch.tensor([2, 0, 0, 1, 0]), wave, Phase.DECODE),
            context,
        )
        for tensor in execution.tensors.values():
            self.assertEqual(tensor[[0, 3], 0].tolist(), [30, 33])
        self.assertEqual(runtime.residency.slot_to_key.tolist(), [0, 5, 14])
        access.release_after(lease, context.record_compute_completion(), context)
        runtime.reload_weights(
            SimpleNamespace(
                slot_to_key=torch.tensor([4, 9, 10]),
                slot_valid=torch.ones(3, dtype=torch.bool),
            ),
            context,
        )
        for name, tensor in hot.tensors.items():
            self.assertEqual(tensor[:, 0].tolist(), [34, 44, 50])
            self.assertEqual(tensor.data_ptr(), before[name])
        self.assertEqual(int(runtime.ops.error_status), 0)

    def test_finalize_failure_drains_before_retiring_host_views_and_pool(self):
        assembly = self.assemble()
        assembly.fail_layer = 4
        with self.assertRaisesRegex(RuntimeError, "EP builder"):
            assembly.finalize()
        self.assertFalse(assembly.backend._closed)
        self.assertIsNone(assembly.backend.runtime)
        self.assertEqual(assembly.events, [])
        assembly.backend.close_unbound()
        self.assertEqual(
            assembly.events,
            [
                "device.synchronize",
                "storage.3.close",
                "storage.4.close",
                "storage.5.close",
                "pool.close",
            ],
        )
        self.assertEqual(assembly.backend._unbound_resources, [])
        assembly.backend.close_unbound()
        self.assertEqual(len(assembly.events), 5)

    def test_incomplete_or_unsealed_host_master_fails_before_bank_allocation(self):
        assembly = self.assemble()
        removed = assembly.backend.layers.pop(5)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            assembly.finalize()
        self.assertEqual(assembly.backend.format.banks, [])
        assembly.backend.layers[5] = removed
        removed.host_store._sealed = False
        with self.assertRaisesRegex(RuntimeError, "Seal"):
            assembly.finalize()
        self.assertEqual(assembly.backend.format.banks, [])

    def test_insufficient_prefill_storage_rejects_then_cleanup_is_safe(self):
        assembly = self.assemble(ratio=0.0, slots=3, group=2)
        with self.assertRaisesRegex(ValueError, "Prefill requires"):
            assembly.finalize()
        assembly.backend.close_unbound()
        self.assertEqual(assembly.events[-1], "pool.close")

    def test_zero_hot_pool_restore_has_no_hidden_weight_allocation(self):
        assembly = self.assemble(ratio=0.0)
        runtime = assembly.finalize()
        context = runtime._new_context(Phase.DECODE, capturing=True)
        runtime._context = context
        runtime.ops.begin_step(context)
        runtime.pipelines[0].access.apply_update(
            torch.empty(0, dtype=torch.int64), context
        )
        runtime.reload_weights(
            SimpleNamespace(
                slot_to_key=torch.empty(0, dtype=torch.int64),
                slot_valid=torch.empty(0, dtype=torch.bool),
            ),
            context,
        )
        self.assertEqual(assembly.backend.format.banks[0].allocations(), ())
        self.assertEqual(len(assembly.backend.format.banks), 2)
        self.assertEqual(int(runtime.ops.error_status), 0)

    def test_restore_uses_shared_sticky_fault_to_suppress_all_component_writes(self):
        assembly = self.assemble(ratio=0.25)
        runtime = assembly.finalize()
        context = runtime._new_context(Phase.DECODE, capturing=True)
        runtime._context = context
        runtime.ops.error_status.fill_(7)
        hot = assembly.backend.format.banks[0]
        for tensor in hot.tensors.values():
            tensor.fill_(199)
        runtime.reload_weights(
            SimpleNamespace(
                slot_to_key=torch.tensor([0, 5, 14]),
                slot_valid=torch.ones(3, dtype=torch.bool),
            ),
            context,
        )
        self.assertEqual(int(runtime.ops.error_status), 7)
        self.assertTrue(
            all(bool(torch.all(tensor == 199)) for tensor in hot.tensors.values())
        )


if __name__ == "__main__":
    unittest.main()
