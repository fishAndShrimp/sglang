import ctypes
import concurrent.futures
import unittest

import torch

from sglang.srt.offloading.moe.config import ExpertLayerSpec
from sglang.srt.offloading.moe.npu_storage import MemFabricHostPool
from sglang.srt.offloading.moe.resources import ResourceKind
from sglang.srt.offloading.moe.v4_format import (
    NpuV4FP4Format,
    V4LayerStorage,
    _scale_view,
)
from test_npu_storage import FakeAcl, FakeOffload


class FakeFormatAdapter:
    """Physical-byte transport oracle, deliberately not an emulation of NZ."""

    def __init__(self):
        self.transfers, self.prepares, self.events, self.copies = [], [], [], []
        self.fail_copy = False
        self.change_shape = False
        self.padding = 0

    def to_device(self, tensor):
        self.transfers.append(tuple(tensor.shape))
        return tensor.clone()

    def prepare(self, weight, scale):
        self.prepares.append(tuple(weight.shape))
        physical = torch.full((weight.numel() + self.padding,), 165, dtype=torch.uint8)
        result = torch.as_strided(
            physical, weight.shape, (physical.numel(), weight.shape[-1], 1)
        )
        result.copy_(weight.flip(-1))
        result = result.transpose(-1, -2)
        if self.change_shape:
            result = result[:, :-1]
        return result, _scale_view(scale)

    def synchronize(self):
        self.events.append(len(self.prepares))

    def physical_nbytes(self, tensor):
        return tensor.untyped_storage().nbytes()

    def empty_component(self, name, layout, capacity):
        if "scale" not in name:
            physical = torch.empty(capacity * layout.nbytes, dtype=torch.uint8)
            return torch.as_strided(
                physical,
                (capacity,) + layout.raw_shape[1:],
                (layout.nbytes, layout.raw_shape[-1], 1),
            ).transpose(-1, -2)
        raw = torch.empty((capacity,) + layout.raw_shape[1:], dtype=torch.uint8)
        return _scale_view(raw) if "scale" in name else raw.transpose(-1, -2)

    def copy_to_host(self, address, source, nbytes):
        if self.fail_copy:
            raise RuntimeError("partial copy failed")
        if self.events[-1] != len(self.prepares):
            raise AssertionError("copy before format completion")
        self.copies.append((address, nbytes))
        ctypes.memmove(address, source.untyped_storage().data_ptr(), nbytes)


class V4FormatTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeFormatAdapter()
        self.format = NpuV4FP4Format(64, 128, "cpu", adapter=self.adapter)
        self.pool = MemFabricHostPool(
            self.format.host_nbytes(2),
            0,
            physical_bytes=1024**3,
            offload=FakeOffload(),
            native_acl=FakeAcl(),
            platform_info=("Linux", "6.8.0"),
        )
        self.layer = torch.nn.Module()
        self.spec = ExpertLayerSpec(4, 4, 2, 1)
        self.storage = V4LayerStorage(
            self.layer, self.spec, self.format, self.pool, {"custom_attr": 123}
        )

    def tearDown(self):
        self.storage.close()
        self.pool.close()

    def piece(self, name, shard, intermediate=96):
        if name.startswith("w13"):
            shape = (intermediate, 2 if "scale" in name else 32)
        else:
            shape = (64, intermediate // (32 if "scale" in name else 2))
        return (torch.arange(shape[0] * shape[1]).reshape(shape) % 251 + 1).to(
            torch.uint8
        )

    def load(self, **kwargs):
        for global_id in (2, 3):
            for name in self.format.layouts:
                param = getattr(self.layer, name)
                for shard in ("w1", "w3") if name.startswith("w13") else ("w2",):
                    param.weight_loader(
                        param, self.piece(name, shard, **kwargs), name, shard, global_id
                    )

    def test_native_metadata_and_workspace_are_single_expert(self):
        self.assertTrue(all(shape[0] == 1 for shape in self.adapter.prepares))
        self.assertEqual(self.format.expert_nbytes, 13056)
        self.assertEqual(self.format.host_nbytes(2), 26112)
        self.assertEqual(self.format.workspace_nbytes, 16896)
        self.assertEqual(self.format.allocations(), ())

    def test_direct_parameter_alias_loader_padding_and_native_repack(self):
        self.load()
        before = {
            name: tensor.clone() for name, tensor in self.storage.raw_components.items()
        }
        for name, raw in self.storage.raw_components.items():
            param = getattr(self.layer, name)
            self.assertEqual(param.data_ptr(), raw.data_ptr())
            self.assertEqual(param.custom_attr, 123)
            if name.startswith("w13"):
                self.assertFalse(bool(raw[:, 96:128].any()))
                self.assertFalse(bool(raw[:, 224:256].any()))
            else:
                tail = 3 if "scale" in name else 48
                self.assertFalse(bool(raw[:, :, tail:].any()))
        self.storage.finalize()
        self.storage.host_store.require_device_access()
        self.assertEqual(len(self.adapter.copies), 8)
        self.assertTrue(all(shape[0] == 1 for shape in self.adapter.prepares))
        for name, raw in self.storage.raw_components.items():
            expected = before[name] if "scale" in name else before[name].flip(-1)
            self.assertTrue(torch.equal(raw, expected))
        copies = len(self.adapter.copies)
        self.storage.finalize()
        self.assertEqual(len(self.adapter.copies), copies)
        self.assertEqual(self.storage.allocations(), ())
        with self.assertRaises(RuntimeError):
            self.layer.w2_weight.weight_loader(
                self.layer.w2_weight, self.piece("w2_weight", "w2"), None, "w2", 2
            )

    def test_banks_views_byte_regions_and_zero_capacity(self):
        bank = self.format.allocate_bank(5, ResourceKind.HOT_WEIGHTS, "hot")
        view = bank.view(2, 2)
        self.assertEqual(view.allocations(), ())
        self.assertEqual(
            sum(a.nbytes for a in bank.allocations()), 5 * self.format.expert_nbytes
        )
        for name in self.format.layouts:
            self.assertEqual(
                view.component_regions[name][0].address,
                bank.component_regions[name][2].address,
            )
            self.assertEqual(
                view.component_regions[name][1].address,
                bank.component_regions[name][3].address,
            )
            self.assertEqual(
                view.tensors[name].untyped_storage().data_ptr(),
                bank.tensors[name].untyped_storage().data_ptr(),
            )
        self.assertEqual(bank.slice(1, 4).capacity, 3)
        for start, count in ((-1, 1), (1, 5), (6, 0)):
            with self.assertRaises(ValueError):
                bank.view(start, count)
        empty = self.format.allocate_bank(0, ResourceKind.HOT_WEIGHTS, "empty")
        self.assertEqual(empty.allocations(), ())
        self.assertTrue(
            all(not regions for regions in empty.component_regions.values())
        )

    def test_missing_duplicate_wrong_ownership_and_wrong_shapes(self):
        p = self.layer.w13_weight
        # A foreign rank is ignored without consuming a completion record.
        p.weight_loader(p, self.piece("w13_weight", "w1"), None, "w1", 0)
        self.assertFalse(self.storage._loaded)
        for expert, shard, tensor in (
            (4, "w1", self.piece("w13_weight", "w1")),
            (2, "w2", self.piece("w13_weight", "w1")),
            (2, "w1", torch.zeros(96, 31, dtype=torch.uint8)),
            (2, "w1", torch.zeros(96, 32, dtype=torch.float32)),
        ):
            with self.assertRaises(ValueError):
                p.weight_loader(p, tensor, None, shard, expert)
        p.weight_loader(p, self.piece("w13_weight", "w1"), None, "w1", 2)
        with self.assertRaises(ValueError):
            p.weight_loader(p, self.piece("w13_weight", "w1"), None, "w1", 2)
        with self.assertRaises(RuntimeError):
            self.storage.finalize()
        self.assertEqual(len(self.adapter.copies), 0)

    def test_scale_bytes_and_padding_mismatch(self):
        p = self.layer.w13_weight
        p.weight_loader(p, self.piece("w13_weight", "w1"), None, "w1", 2)
        scale = self.layer.w13_weight_scale_inv
        with self.assertRaises(ValueError):
            scale.weight_loader(
                scale, self.piece("w13_weight_scale_inv", "w1", 64), None, "w1", 2
            )
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if e8m0 is not None:
            value = self.piece("w13_weight_scale_inv", "w1").view(e8m0)
            scale.weight_loader(scale, value, None, "w1", 2)
            self.assertTrue(torch.equal(scale[0, :96], value.view(torch.uint8)))

    def test_failure_poison_prevents_reinterpreting_partially_repacked_master(self):
        self.load()
        self.adapter.fail_copy = True
        with self.assertRaisesRegex(RuntimeError, "partial copy"):
            self.storage.finalize()
        self.adapter.fail_copy = False
        with self.assertRaisesRegex(RuntimeError, "failed repack"):
            self.storage.finalize()
        with self.assertRaises(RuntimeError):
            self.storage.host_store.require_device_access()

    def test_conversion_abi_drift_fails_before_copy(self):
        self.load()
        self.adapter.change_shape = True
        with self.assertRaisesRegex(RuntimeError, "changed after"):
            self.storage.finalize()
        self.assertEqual(self.adapter.copies, [])

    def test_format_budget_and_dimension_rejection(self):
        for hidden, intermediate in ((63, 128), (64, 0), (64, 96)):
            with self.assertRaises(ValueError):
                NpuV4FP4Format(hidden, intermediate, "cpu", adapter=self.adapter)
        adapter = FakeFormatAdapter()
        with self.assertRaises(ValueError):
            NpuV4FP4Format(64, 128, "cpu", adapter=adapter, workspace_limit_bytes=1)
        self.assertEqual(adapter.transfers, [])
        with self.assertRaises(ValueError):
            NpuV4FP4Format(64, 128, "cpu", adapter=adapter, workspace_limit_bytes=9000)

    def test_close_retires_module_parameter_views(self):
        self.storage.close()
        self.storage.close()
        self.assertEqual(self.layer.w13_weight.numel(), 0)
        self.assertFalse(hasattr(self.layer.w13_weight, "weight_loader"))

    def test_physical_padding_is_copied_and_bank_slices_keep_full_stride(self):
        self.storage.close()
        self.pool.close()
        self.adapter.padding = 128
        self.format = NpuV4FP4Format(64, 128, "cpu", adapter=self.adapter)
        self.pool = MemFabricHostPool(
            self.format.host_nbytes(2),
            0,
            physical_bytes=1024**3,
            offload=FakeOffload(),
            native_acl=FakeAcl(),
            platform_info=("Linux", "6.8.0"),
        )
        self.layer = torch.nn.Module()
        self.storage = V4LayerStorage(self.layer, self.spec, self.format, self.pool, {})
        self.load()
        self.storage.finalize()
        for name in ("w13_weight", "w2_weight"):
            cap = self.storage.component_capabilities[name][1]
            padding = (ctypes.c_ubyte * 128).from_address(
                cap.host_address + cap.nbytes - 128
            )
            self.assertEqual(list(padding), [165] * 128)
        bank = self.format.allocate_bank(3, ResourceKind.HOT_WEIGHTS, "padded")
        view = bank.view(1, 2)
        for name in self.format.layouts:
            self.assertEqual(
                view.component_regions[name][0].address,
                bank.component_regions[name][1].address,
            )

    def test_concurrent_duplicate_checkpoint_shards_are_atomic(self):
        param = self.layer.w13_weight

        def submit():
            try:
                param.weight_loader(
                    param, self.piece("w13_weight", "w1"), None, "w1", 2
                )
                return True
            except ValueError:
                return False

        with concurrent.futures.ThreadPoolExecutor(8) as executor:
            results = list(executor.map(lambda _: submit(), range(16)))
        self.assertEqual(sum(results), 1)

    def test_partial_initialization_retires_already_attached_views(self):
        layer = torch.nn.Module()
        real_allocate = self.pool.allocate
        calls = []

        def allocate(nbytes):
            calls.append(nbytes)
            if len(calls) == 2:
                raise MemoryError("injected midway failure")
            # The existing slab is already full. Reuse a test region only for
            # this failed constructor; no repack or production alias is allowed.
            self.pool._offset = 0
            return real_allocate(nbytes)

        self.pool.allocate = allocate
        with self.assertRaises(MemoryError):
            V4LayerStorage(layer, self.spec, self.format, self.pool, {})
        self.assertFalse(hasattr(layer, "_moe_offload_storage"))
        self.assertEqual(layer.w13_weight.numel(), 0)
        self.assertFalse(hasattr(layer.w13_weight, "weight_loader"))


if __name__ == "__main__":
    unittest.main()
