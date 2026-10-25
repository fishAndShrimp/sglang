"""Topology and model-entry contracts use the production binding on CPU."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.offloading.moe.config import (
    ExpertCatalog,
    ExpertLayerSpec,
    MoEOffloadConfig,
)
from sglang.srt.offloading.moe.models import DeepSeekV4Adapter, ModelAdapter
from sglang.srt.offloading.moe.offloader import MoEOffloaderV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class OtherModelAdapter(ModelAdapter):
    """A different model structure consumes the same entry/lifetime binding."""

    direct_layer_entries = ("compute",)

    def plan_model(self, hf_config, config):
        return ExpertCatalog((ExpertLayerSpec(7, 4, config.ep_size, config.ep_rank),))

    def describe(self, decoder):
        return ExpertLayerSpec(decoder.index, 4, 1, 0)

    def expert_module(self, decoder):
        return decoder.routed


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.index = 7
        self.routed = torch.nn.Linear(1, 1)

    def compute(self, value):
        return value + 1

    def forward(self, value):
        return self.compute(value)


class Runtime:
    def __init__(self):
        self.layer_programs = {7: object()}
        self.started = set()
        self.entries = []

    def _enter_layer(self, layer_id):
        if layer_id not in self.started:
            self.entries.append(layer_id)
            self.started.add(layer_id)


class ModelAdapterTests(unittest.TestCase):
    def test_preallocation_plan_and_constructed_layer_share_identity(self):
        config = MoEOffloadConfig(0.5, 2, 2, 1)
        adapter = DeepSeekV4Adapter()
        plan = adapter.plan_model(
            SimpleNamespace(
                model_type="deepseek_v4",
                num_hidden_layers=6,
                num_hash_layers=4,
                n_routed_experts=8,
            ),
            config,
        )
        self.assertEqual([s.layer_id for s in plan.offloaded_layers], [4, 5])
        self.assertEqual(plan.num_experts, 8)
        experts = SimpleNamespace(
            moe_tp_size=1,
            moe_ep_size=2,
            moe_ep_rank=1,
            _num_global_routed=8,
            _num_local_routed=4,
        )
        decoder = SimpleNamespace(
            mlp=SimpleNamespace(
                is_deepseek_v4=True, layer_id=4, is_hash=False, experts=experts
            )
        )
        self.assertEqual(adapter.describe(decoder), plan.layer(4))
        self.assertEqual(plan.key(5, 3), 7)
        self.assertIs(plan.offloaded_layers, plan.offloaded_layers)
        with self.assertRaisesRegex(ValueError, "resident"):
            plan.key(3, 0)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            plan.key(4, 4)
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            plan.layer(6)

    def test_unsupported_model_and_empty_offload_plan_fail_before_loading(self):
        config = MoEOffloadConfig(0.5, 2, 1, 0)
        adapter = DeepSeekV4Adapter()
        with self.assertRaisesRegex(ValueError, "deepseek_v4"):
            adapter.plan_model(SimpleNamespace(model_type="other"), config)
        with self.assertRaisesRegex(ValueError, "no offloadable"):
            adapter.plan_model(
                SimpleNamespace(
                    model_type="deepseek_v4", num_hidden_layers=3, n_routed_experts=8
                ),
                config,
            )

    def test_other_model_reuses_binding_and_restores_instance_lookup(self):
        decoder, runtime = Decoder(), Runtime()
        adapter = OtherModelAdapter()
        original = decoder.compute
        binding = adapter.bind_entries(runtime, ((decoder, adapter.describe(decoder)),))
        self.assertIs(decoder.routed._moe_offload_runtime, runtime)
        self.assertEqual(decoder(torch.tensor(2)), 3)
        self.assertEqual(runtime.entries, [7])
        runtime.started.clear()
        self.assertEqual(decoder.compute(torch.tensor(5)), 6)
        self.assertEqual(runtime.entries, [7, 7])
        binding.close()
        binding.close()
        self.assertNotIn("compute", decoder.__dict__)
        self.assertEqual(decoder.compute, original)
        self.assertFalse(hasattr(decoder.routed, "_moe_offload_runtime"))
        self.assertEqual(len(decoder._forward_pre_hooks), 0)

    def test_partial_install_rolls_back_all_entries(self):
        decoder, runtime = Decoder(), Runtime()
        adapter = OtherModelAdapter()
        spec = adapter.describe(decoder)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            adapter.bind_entries(runtime, ((decoder, spec), (decoder, spec)))
        self.assertNotIn("compute", decoder.__dict__)
        self.assertFalse(hasattr(decoder.routed, "_moe_offload_runtime"))
        self.assertEqual(len(decoder._forward_pre_hooks), 0)

    def test_binding_does_not_keep_runtime_alive_after_close_without_gc(self):
        enabled = gc.isenabled()
        gc.disable()
        try:
            decoder, runtime = Decoder(), Runtime()
            adapter = OtherModelAdapter()
            binding = adapter.bind_entries(
                runtime, ((decoder, adapter.describe(decoder)),)
            )
            probe = weakref.ref(runtime)
            binding.close()
            del runtime
            self.assertIsNone(probe())
        finally:
            if enabled:
                gc.enable()

    def test_native_runtime_installs_adapter_entries_and_retires_graph_hooks(self):
        from test_npu_backend import Assembly

        assembly = Assembly(layer_count=1)
        self.addCleanup(assembly.close)
        runtime = assembly.finalize()
        runtime.layer_programs = {7: next(iter(runtime.layer_programs.values()))}
        decoder, adapter = Decoder(), OtherModelAdapter()
        runner = SimpleNamespace(capture_head_hooks=[], capture_tail_hooks=[])
        offloader = SimpleNamespace(
            owner=None, adapter=adapter, _layers=((decoder, adapter.describe(decoder)),)
        )
        runtime.install_model_hooks(offloader, runner)
        self.assertIs(decoder.routed._moe_offload_runtime, runtime)
        self.assertEqual(len(runner.capture_head_hooks), 1)
        self.assertEqual(len(runner.capture_tail_hooks), 1)
        runtime.close()
        self.assertEqual(runner.capture_head_hooks, [])
        self.assertEqual(runner.capture_tail_hooks, [])
        self.assertFalse(hasattr(decoder.routed, "_moe_offload_runtime"))

    def test_graph_hook_install_failure_rolls_back_model_binding(self):
        from test_npu_backend import Assembly

        class FailedHooks(list):
            def append(self, value):
                raise RuntimeError("hook installation failed")

        assembly = Assembly(layer_count=1)
        self.addCleanup(assembly.close)
        runtime = assembly.finalize()
        runtime.layer_programs = {7: next(iter(runtime.layer_programs.values()))}
        decoder, adapter = Decoder(), OtherModelAdapter()
        runner = SimpleNamespace(
            capture_head_hooks=[], capture_tail_hooks=FailedHooks()
        )
        offloader = SimpleNamespace(
            owner=None, adapter=adapter, _layers=((decoder, adapter.describe(decoder)),)
        )
        with self.assertRaisesRegex(RuntimeError, "hook installation"):
            runtime.install_model_hooks(offloader, runner)
        self.assertEqual(runner.capture_head_hooks, [])
        self.assertFalse(hasattr(decoder.routed, "_moe_offload_runtime"))
        self.assertNotIn("compute", decoder.__dict__)

    def test_catalog_drift_rejected_during_registration_and_unbound_resources_closed(
        self,
    ):
        expected = ExpertLayerSpec(3, 8, 2, 1)
        plan = ExpertCatalog((expected, ExpertLayerSpec(4, 8, 2, 1)))
        for actual in (
            (ExpertLayerSpec(3, 8, 2, 1, resident=True),),
            (expected,),
            (ExpertLayerSpec(5, 8, 2, 1),),
        ):
            with self.subTest(actual=actual):
                backend = SimpleNamespace(
                    model_plan=plan, prepare_layer=Mock(), close_unbound=Mock()
                )
                adapter = SimpleNamespace(describe=lambda layer: layer)
                facade = MoEOffloaderV2(
                    MoEOffloadConfig(0.5, 2, 2, 1), backend, adapter=adapter
                )
                with self.assertRaises(ValueError):
                    facade.wrap_modules(iter(actual))
                backend.close_unbound.assert_called_once()
                self.assertTrue(facade._failed)
                if actual != (expected,):
                    backend.prepare_layer.assert_not_called()

    def test_facade_hook_failure_closes_native_runtime_and_failed_drain_can_retry(self):
        from test_npu_backend import Assembly

        class FailedHooks(list):
            def append(self, value):
                raise RuntimeError("hook installation failed")

        for fail_drain in (False, True):
            with self.subTest(fail_drain=fail_drain):
                assembly = Assembly(layer_count=1)
                self.addCleanup(assembly.close)
                runtime = assembly.finalize()
                decoder, adapter = Decoder(), OtherModelAdapter()
                spec = assembly.catalog.offloaded_layers[0]
                adapter.describe = lambda layer: spec
                backend = SimpleNamespace(
                    prepare_layer=Mock(),
                    finalize=Mock(return_value=runtime),
                    close_unbound=Mock(),
                )
                facade = MoEOffloaderV2(assembly.config, backend, adapter=adapter)
                facade.wrap_modules(iter((decoder,)))
                facade.post_init()
                runner = SimpleNamespace(
                    model=SimpleNamespace(),
                    capture_head_hooks=[],
                    capture_tail_hooks=FailedHooks(),
                )
                if fail_drain:
                    with patch.object(
                        runtime.device_runtime,
                        "synchronize",
                        side_effect=RuntimeError("drain failed"),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "drain failed"):
                            facade.bind_model_runner(runner)
                    self.assertFalse(facade._closed)
                    self.assertFalse(runtime._closed)
                    self.assertNotIn("pool.close", assembly.events)
                    facade.close()
                else:
                    with self.assertRaisesRegex(RuntimeError, "hook installation"):
                        facade.bind_model_runner(runner)
                self.assertTrue(facade._closed)
                self.assertTrue(runtime._closed)
                self.assertEqual(assembly.events.count("device.synchronize"), 1)
                self.assertEqual(assembly.events.count("pool.close"), 1)
                self.assertEqual(runner.capture_head_hooks, [])
                self.assertFalse(hasattr(decoder.routed, "_moe_offload_runtime"))
                self.assertNotIn("compute", decoder.__dict__)

    def test_facade_cleans_partial_native_finalize_without_caller_cleanup(self):
        from test_npu_backend import Assembly

        assembly = Assembly()
        assembly.fail_layer = 4
        specs = {id(module): spec for module, spec in assembly.layers}
        adapter = SimpleNamespace(describe=lambda layer: specs[id(layer)])
        facade = MoEOffloaderV2(
            assembly.config, assembly.backend, adapter=assembly.backend.model_adapter
        )
        # The physical seam's adapter also supplies registration descriptions.
        assembly.backend.model_adapter.describe = adapter.describe
        facade.wrap_modules(iter(module for module, _ in assembly.layers))
        with self.assertRaisesRegex(RuntimeError, "EP builder"):
            assembly.finalize(facade.post_init)
        self.assertTrue(assembly.backend._closed)
        self.assertEqual(assembly.backend._unbound_resources, [])
        self.assertEqual(assembly.events[-1], "pool.close")
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
        facade.close()

    def test_prepare_rejects_quant_or_geometry_before_new_storage(self):
        from sglang.srt.offloading.moe.npu.backend import NpuMoEBackend

        class Format:
            pass

        class OtherFormat:
            pass

        backend = NpuMoEBackend.__new__(NpuMoEBackend)
        backend.model_plan = ExpertCatalog(
            (ExpertLayerSpec(0, 8, 1, 0, True), ExpertLayerSpec(3, 8, 1, 0))
        )
        backend.runtime, backend._closed, backend.layers = None, False, {}
        backend.format, backend._dimensions, backend.host_pool = (
            Format(),
            (16, 32),
            object(),
        )
        layer = SimpleNamespace(layer_id=3, moe_tp_size=1, num_fused_shared_experts=0)
        args = dict(
            method=object(),
            num_experts=8,
            hidden_size=16,
            intermediate_size_per_partition=32,
            params_dtype=torch.bfloat16,
        )
        resolver = "sglang.srt.offloading.moe.npu.backend.resolve_npu_format"
        with patch(
            "sglang.srt.offloading.moe.npu.format.ExpertLayerStorage"
        ) as storage:
            with patch(resolver, side_effect=ValueError("unsupported quant")):
                with self.assertRaisesRegex(ValueError, "unsupported quant"):
                    backend.prepare_expert_weights(layer, **args)
            with patch(resolver, return_value=OtherFormat):
                with self.assertRaisesRegex(ValueError, "physical expert format"):
                    backend.prepare_expert_weights(layer, **args)
            with patch(resolver, return_value=Format):
                with self.assertRaisesRegex(ValueError, "physical expert format"):
                    backend.prepare_expert_weights(layer, **dict(args, hidden_size=32))
            storage.assert_not_called()
            layer.layer_id = 0
            with patch(resolver, return_value=None) as resolve:
                self.assertFalse(backend.prepare_expert_weights(layer, **args))
                self.assertTrue(resolve.call_args.kwargs["resident"])
            storage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
