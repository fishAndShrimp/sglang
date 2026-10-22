"""CPU byte-layout/loading contracts; native NZ and NPU graph need hardware tests."""

import ast
import ctypes
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.moe.w8a8 import NPUSwigluInt8Quant
from sglang.srt.offloading.moe.config import ExpertLayerSpec
from sglang.srt.offloading.moe.npu import v4_format, w8a8_format
from sglang.srt.offloading.moe.resources import ResourceKind
from sglang.srt.offloading.moe.storage import HostMemoryCapability


class FormatAdapter:
    """Native boundary substitute: ordinary typed CPU tensors, actual byte copies."""

    def __init__(self, device, native_acl):
        self.conversions = 0
        self.fail = False

    def begin_workspace(self):
        return 0

    def end_workspace(self, baseline):
        return 0

    def to_device(self, x):
        return x.clone()

    def prepare(self, weight, scale, offset):
        if self.fail:
            raise RuntimeError("injected format failure")
        self.conversions += 1
        return (
            weight.transpose(1, 2).contiguous(),
            scale.squeeze(-1).to(torch.bfloat16),
            offset.squeeze(-1),
        )

    def synchronize(self):
        pass

    def physical_nbytes(self, value):
        return value.untyped_storage().nbytes()

    def empty_component(self, name, layout, capacity):
        return torch.empty((capacity,) + layout.runtime_shape[1:], dtype=layout.dtype)

    def copy_to_host(self, address, value, nbytes):
        ctypes.memmove(address, value.data_ptr(), nbytes)


class HostPool:
    def __init__(self, size):
        self.bytes = torch.zeros(size, dtype=torch.uint8)
        self.offset = 0
        self.capability = HostMemoryCapability(
            self.bytes.data_ptr(), size, True, self.bytes.data_ptr(), self
        )

    def allocate(self, size):
        value = self.bytes[self.offset : self.offset + size]
        self.offset += size
        if value.numel() != size:
            raise ValueError("Host pool overflow")
        return SimpleNamespace(tensor=value, capability=self.capability)


def physical_tensor(region, dtype, shape):
    buffer = (ctypes.c_ubyte * region.nbytes).from_address(region.host_address)
    return torch.frombuffer(buffer, dtype=dtype).reshape(shape)


class TestW8A8Format(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(
            w8a8_format, "_TorchNpuW8A8FormatAdapter", FormatAdapter
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.format = w8a8_format.NpuV4W8A8Format(
            64, 128, "cpu", workspace_limit_bytes=2**20
        )
        self.spec = ExpertLayerSpec(3, 4, 2, 1)
        self.pool = HostPool(self.format.host_nbytes(self.spec.local_experts))
        self.layer = torch.nn.Module()
        self.layer.dispatcher = SimpleNamespace(set_quant_config=mock.Mock())
        self.storage = w8a8_format.W8A8LayerStorage(
            self.layer, self.spec, self.format, self.pool, {}
        )
        self.addCleanup(self.storage.close)

    def component(self, projection, suffix, expert):
        n = 128 if projection in ("w1", "w3") else 64
        k = 64 if projection in ("w1", "w3") else 128
        if suffix == "weight":
            salt = {"w1": 3, "w3": 17, "w2": 31}[projection] + expert
            return ((torch.arange(n * k).reshape(n, k) + salt) % 127 - 63).to(
                torch.int8
            )
        if suffix == "weight_scale":
            return torch.arange(1, n + 1).float().unsqueeze(1) / (10 + expert)
        return torch.zeros(n, 1)

    def load_one(self, expert, projection, suffix, value=None):
        prefix = "w13" if projection in ("w1", "w3") else "w2"
        parameter = getattr(self.layer, prefix + "_" + suffix)
        parameter.weight_loader(
            parameter,
            self.component(projection, suffix, expert) if value is None else value,
            shard_id=projection,
            expert_id=expert,
        )

    def load_all(self):
        for expert in (2, 3):
            for projection in ("w1", "w3", "w2"):
                for suffix in ("weight", "weight_scale", "weight_offset"):
                    self.load_one(expert, projection, suffix)

    def test_typed_load_repack_and_physical_bank_offsets(self):
        self.load_all()
        self.storage.finalize()
        self.assertEqual(self.pool.offset, self.format.host_nbytes(2))
        for local, expert in enumerate((2, 3)):
            expected = {
                "w13_weight": torch.cat(
                    [self.component(p, "weight", expert) for p in ("w1", "w3")]
                )
                .t()
                .contiguous(),
                "w2_weight": self.component("w2", "weight", expert).t().contiguous(),
                "w13_weight_scale": torch.cat(
                    [self.component(p, "weight_scale", expert) for p in ("w1", "w3")]
                )
                .squeeze(-1)
                .bfloat16(),
                "w2_weight_scale": self.component("w2", "weight_scale", expert)
                .squeeze(-1)
                .bfloat16(),
                "w13_weight_offset": torch.zeros(256),
                "w2_weight_offset": torch.zeros(64),
            }
            for name, value in expected.items():
                region = self.storage.component_capabilities[name][local]
                actual = physical_tensor(region, value.dtype, value.shape)
                self.assertTrue(torch.equal(actual, value), name)
        bank = self.format.allocate_bank(2, ResourceKind.EXECUTION_WEIGHTS, "test")
        view = bank.view(1, 1)
        for name, tensor in view.tensors.items():
            self.assertEqual(
                tensor.data_ptr() - bank.tensors[name].data_ptr(),
                self.format.component_nbytes[name],
            )
        self.assertEqual(
            sum(x.nbytes for x in bank.allocations()), 2 * self.format.expert_nbytes
        )
        self.assertEqual(view.allocations(), ())
        self.layer.dispatcher.set_quant_config.assert_called_once_with(
            {
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "bf16",
            }
        )
        conversions = self.format.adapter.conversions
        self.storage.finalize()
        self.assertEqual(self.format.adapter.conversions, conversions)

    def test_host_padding_is_charged_and_distinct_from_runtime_scale_extent(self):
        name = "w13_weight_scale"
        self.assertEqual(
            self.format.host_component_nbytes[name],
            2 * self.format.component_nbytes[name],
        )
        regions = self.storage.component_capabilities[name]
        self.assertEqual(
            regions[1].host_address - regions[0].host_address,
            self.format.host_component_nbytes[name],
        )
        self.assertGreater(self.format.host_nbytes(2), 2 * self.format.expert_nbytes)

    def test_nonlocal_experts_are_ignored_and_completeness_is_required(self):
        self.load_one(0, "w1", "weight")
        self.assertEqual(len(self.storage._loaded), 0)
        self.load_one(2, "w1", "weight")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load_one(2, "w1", "weight")
        with self.assertRaisesRegex(RuntimeError, "nine components"):
            self.storage.finalize()

    def test_invalid_dtype_scale_offset_and_shape_are_rejected(self):
        for suffix, value, message in (
            ("weight", torch.zeros(128, 64, dtype=torch.uint8), "INT8"),
            ("weight_scale", torch.full((128, 1), float("nan")), "finite"),
            ("weight_scale", -torch.ones(128, 1), "nonnegative"),
            ("weight_offset", torch.ones(128, 1), "zero offsets"),
            ("weight_scale", torch.ones(127, 1), "shape"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.load_one(2, "w1", suffix, value)
        self.assertEqual(len(self.storage._loaded), 0)
        self.load_one(2, "w1", "weight_scale", torch.ones(128, dtype=torch.bfloat16))
        self.assertTrue(
            torch.equal(self.layer.w13_weight_scale[0, :128], torch.ones(128, 1))
        )

    def test_repack_failure_poisoning_prevents_reinterpreting_native_bytes(self):
        self.load_all()
        self.format.adapter.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.storage.finalize()
        self.format.adapter.fail = False
        with self.assertRaisesRegex(RuntimeError, "failed repack"):
            self.storage.finalize()
        with self.assertRaisesRegex(RuntimeError, "not writable"):
            self.load_one(2, "w1", "weight")

    def test_too_small_loader_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "workspace"):
            w8a8_format.NpuV4W8A8Format(64, 128, "cpu", workspace_limit_bytes=1)

    def test_quant_info_binds_every_runtime_component(self):
        bank = self.format.allocate_bank(2, ResourceKind.HOT_WEIGHTS, "hot")
        module = SimpleNamespace(AscendQuantInfo=SimpleNamespace)
        with mock.patch.dict(
            "sys.modules", {"sglang.srt.layers.moe.moe_runner.ascend": module}
        ):
            quant = bank.quant_info()
        self.assertEqual(set(vars(quant)), set(bank.tensors))
        for name, value in bank.tensors.items():
            self.assertIs(getattr(quant, name), value)


class FP4Adapter(FormatAdapter):
    def prepare(self, weight, scale):
        return weight.transpose(-1, -2), v4_format._scale_view(scale)

    def empty_component(self, name, layout, capacity):
        raw = torch.empty((capacity,) + layout.raw_shape[1:], dtype=layout.dtype)
        return (
            raw.transpose(-1, -2)
            if name.endswith("_weight")
            else v4_format._scale_view(raw)
        )


class TestFP4FormatRegression(unittest.TestCase):
    def test_shared_bank_and_host_loading_keep_fp4_layout(self):
        with mock.patch.object(v4_format, "_TorchNpuFormatAdapter", FP4Adapter):
            format = v4_format.NpuV4FP4Format(
                64, 128, "cpu", workspace_limit_bytes=2**20
            )
        pool = HostPool(format.host_nbytes(2))
        layer = torch.nn.Module()
        storage = v4_format.V4LayerStorage(
            layer, ExpertLayerSpec(3, 2, 1, 0), format, pool, {}
        )
        self.addCleanup(storage.close)
        for expert in range(2):
            for name, raw in storage.raw_components.items():
                parameter = getattr(layer, name)
                if name.startswith("w13_"):
                    for shard, value in (("w1", 1), ("w3", 2)):
                        parameter.weight_loader(
                            parameter,
                            torch.full_like(raw[0, :128], value),
                            expert_id=expert,
                            shard_id=shard,
                        )
                else:
                    parameter.weight_loader(
                        parameter,
                        torch.ones_like(raw[0]),
                        expert_id=expert,
                        shard_id="w2",
                    )
        storage.finalize()
        bank = format.allocate_bank(2, ResourceKind.HOT_WEIGHTS, "fp4")
        self.assertEqual(pool.offset, format.host_nbytes(2))
        for name, tensor in bank.view(1, 1).tensors.items():
            self.assertEqual(
                tensor.data_ptr() - bank.tensors[name].data_ptr(),
                format.component_nbytes[name],
            )
        self.assertEqual(layer.w13_weight_scale_inv.quant_method, "block")


class TestW8A8Binding(unittest.TestCase):
    def test_v4_clamp_selects_int8_activation_in_actual_runner_constructor(self):
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/layers/moe/moe_runner/ascend.py"
        )
        tree = ast.parse(source.read_text())
        core = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "AscendRunnerCore"
        )
        core.body = [
            n
            for n in core.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        ]

        class Base:
            def __init__(self, config):
                self.config = config

        class W8Kernel:
            pass

        namespace = dict(
            MoeRunnerCore=Base,
            MoeRunnerBackend=SimpleNamespace(ASCEND="ascend"),
            _uses_fused_gmm1=lambda *args: False,
            NPUW8A8Int8MoEMethod=W8Kernel,
            NPUW4A8MXFP4MoEMethod=type("FP4", (), {}),
            NPUSwigluInt8Quant=NPUSwigluInt8Quant,
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, core], type_ignores=[])
                ),
                str(source),
                "exec",
            ),
            namespace,
        )
        config = SimpleNamespace(
            layer=SimpleNamespace(w2_kernel=W8Kernel()), swiglu_limit=10
        )
        result = namespace["AscendRunnerCore"](config)
        self.assertIsInstance(result.activation, NPUSwigluInt8Quant)

    def test_backend_accepts_only_the_two_w8a8_schemes_and_marks_hash_layers(self):
        from sglang.srt.offloading.moe.npu.backend import NpuMoEBackend

        class FP4:
            pass

        class Method:
            pass

        class Scheme:
            pass

        modules = {
            "sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods": SimpleNamespace(
                NPUW4A8MXFP4FusedMoEMethod=FP4
            ),
            "sglang.srt.layers.quantization.modelslim.modelslim": SimpleNamespace(
                ModelSlimFusedMoEMethod=Method
            ),
            "sglang.srt.layers.quantization.modelslim.schemes": SimpleNamespace(
                ModelSlimW8A8Int8MoE=Scheme
            ),
        }
        backend = NpuMoEBackend.__new__(NpuMoEBackend)
        backend.resident_layers, backend.layer_ids = 3, (3,)
        with mock.patch.dict("sys.modules", modules):
            for layer_id in (0, 3):
                layer = SimpleNamespace(
                    layer_id=layer_id, w13_scheme=Scheme(), w2_scheme=Scheme()
                )
                self.assertEqual(backend.validate_quant_method(layer, Method()), "w8a8")
                self.assertTrue(layer._moe_offload_bf16_dispatch)
            layer.w2_scheme = object()
            with self.assertRaisesRegex(ValueError, "W8A8_DYNAMIC"):
                backend.validate_quant_method(layer, Method())


class TestW8A8Activation(unittest.TestCase):
    @staticmethod
    def quantize(x, *, dst_type):
        assert dst_type == torch.int8
        scale = (x.abs().amax(-1) / 127).clamp_min(1e-8)
        return (x / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8), scale

    @staticmethod
    def clipped(x, *, alpha, limit, bias, interleaved):
        assert alpha == 1 and bias == 0 and not interleaved
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)

    def test_clamp_padding_and_dynamic_device_counts(self):
        activation = NPUSwigluInt8Quant(10)
        x = torch.tensor(
            [[30.0, -3.0, 40.0, -20.0], [2.0, 3.0, 4.0, 5.0], [float("nan")] * 4]
        )
        with mock.patch.object(
            torch.ops.npu, "npu_dynamic_quant", self.quantize, create=True
        ), mock.patch.object(
            torch.ops.npu, "npu_clipped_swiglu", self.clipped, create=True
        ):
            for counts in (
                torch.tensor([1, 1]),
                torch.tensor([0, 0]),
                torch.tensor([0, 1]),
            ):
                q, scale = activation._apply_activation(x, counts, 1)
                valid = torch.arange(3) < counts.sum()
                clean = torch.where(valid[:, None], x, 0)
                gate, up = clean.chunk(2, -1)
                expected = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
                eq, es = self.quantize(expected, dst_type=torch.int8)
                self.assertTrue(torch.equal(q, eq))
                self.assertTrue(torch.equal(scale, es))
                self.assertFalse(torch.isnan(scale).any())
                self.assertTrue(q[~valid].eq(0).all())
            traced = torch.jit.trace(
                lambda a, b: activation._apply_activation(a, b, 1),
                (x, torch.tensor([1, 1])),
            )
            q, scale = traced(x, torch.tensor([0, 0]))
            self.assertTrue(q.eq(0).all())


if __name__ == "__main__":
    unittest.main()
