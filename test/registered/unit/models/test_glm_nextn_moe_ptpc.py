"""CPU contracts for GLM-5.2 NextN quantization and bundled checkpoints.

The feature is off by default. Without a case that turns the flag on, CI never
touches the Quark scheme rewrite and cannot claim the path works. These tests
exercise that ON wiring on CPU without loading a 70B MXFP4 checkpoint.

A full serve+generate job still needs the MXFP4 weights in the runner cache;
register that separately as nightly if the checkpoint is present.

ModelSlim contracts cover dynamic draft-layer lookup and safetensors preflight;
device allocation is isolated, while production selection and validation run.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
from sglang.srt.models.glm4_moe import (
    GlmMoeDsaForCausalLMNextN,
    enable_glm_nextn_moe_ptpc,
    should_apply_glm_nextn_moe_ptpc,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

LAYER = 78
PREFIX = f"model.layers.{LAYER}"
EXPERT_LEAF = f"{PREFIX}.mlp.experts.0.w1"
SHARED_EXPERT_LEAF = f"{PREFIX}.mlp.shared_experts.gate_proj"
ATTN_LEAF = f"{PREFIX}.self_attn.q_proj"
EXPERT_WEIGHT = f"{PREFIX}.mlp.experts.0.gate_proj.weight"
SHARED_EXPERT_WEIGHT = f"{PREFIX}.mlp.shared_experts.gate_proj.weight"
_PTPC_ENV = "sglang.srt.models.glm4_moe.envs.SGLANG_GLM_NEXTN_MOE_PTPC.get"


def _quark_cfg(*, exclude=None, layer_quant=None):
    return SimpleNamespace(
        get_name=lambda: "quark",
        quant_config={"layer_quant_config": dict(layer_quant or {}), "exclude": []},
        exclude_layers=list(
            exclude if exclude is not None else [EXPERT_LEAF, ATTN_LEAF]
        ),
    )


class TestEnableGlmNextnMoePtpc(CustomTestCase):
    def test_off_by_default(self):
        self.assertFalse(enable_glm_nextn_moe_ptpc(_quark_cfg()))

    def test_on_requires_quark(self):
        with patch(
            _PTPC_ENV,
            return_value=True,
        ):
            self.assertTrue(enable_glm_nextn_moe_ptpc(_quark_cfg()))
            self.assertFalse(
                enable_glm_nextn_moe_ptpc(SimpleNamespace(get_name=lambda: "fp8"))
            )
            self.assertFalse(enable_glm_nextn_moe_ptpc(None))

    def test_apply_requires_excluded_mtp_experts(self):
        with patch(
            _PTPC_ENV,
            return_value=True,
        ):
            self.assertTrue(should_apply_glm_nextn_moe_ptpc(_quark_cfg(), LAYER))
            self.assertFalse(
                should_apply_glm_nextn_moe_ptpc(_quark_cfg(exclude=[ATTN_LEAF]), LAYER)
            )

    def test_ptpc_disables_shared_expert_fusion(self):
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLMNextN"],
            n_routed_experts=256,
            n_shared_experts=1,
            num_hidden_layers=LAYER,
        )
        # Shared expert BF16 (excluded) or still quantized: neither fits the slot.
        for exclude in ([EXPERT_LEAF, SHARED_EXPERT_LEAF], [EXPERT_LEAF]):
            with self.subTest(exclude=exclude), patch(_PTPC_ENV, return_value=True):
                reason = GlmMoeDsaForCausalLMNextN.shared_experts_fusion_disable_reason(
                    hf_config, _quark_cfg(exclude=exclude)
                )
                self.assertIn("PTPC", reason or "")


class TestResolveNextnQuantConfigPtpcOn(CustomTestCase):
    def _resolve(self, cfg, flag: bool):
        model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
        hf = SimpleNamespace(num_hidden_layers=LAYER)
        with patch(
            _PTPC_ENV,
            return_value=flag,
        ):
            return model._resolve_nextn_quant_config(hf, cfg)

    def test_flag_off_excludes_fused_experts(self):
        src = _quark_cfg()
        out = self._resolve(src, flag=False)
        self.assertIn("model.decoder.mlp.experts", out.exclude_layers)
        self.assertNotIn(
            "model.decoder.mlp.experts",
            out.quant_config.get("layer_quant_config", {}),
        )

    def test_flag_on_assigns_ptpc_scheme_instead_of_bf16_exclude(self):
        src = _quark_cfg()
        out = self._resolve(src, flag=True)
        self.assertNotIn("model.decoder.mlp.experts", out.exclude_layers)
        scheme = out.quant_config["layer_quant_config"]["model.decoder.mlp.experts"]
        self.assertEqual(scheme["weight"]["dtype"], "fp8_e4m3")
        self.assertEqual(scheme["weight"]["qscheme"], "per_channel")
        self.assertFalse(scheme["weight"]["is_dynamic"])
        self.assertEqual(scheme["input_tensors"]["dtype"], "fp8_e4m3")
        self.assertEqual(scheme["input_tensors"]["qscheme"], "per_channel")
        self.assertTrue(scheme["input_tensors"]["is_dynamic"])

    def test_flag_on_fused_experts_not_ignored_by_quark(self):
        self.assertTrue(
            should_ignore_layer(
                "model.decoder.mlp.experts",
                ignore=self._resolve(_quark_cfg(), flag=False).exclude_layers,
            )
        )
        self.assertFalse(
            should_ignore_layer(
                "model.decoder.mlp.experts",
                ignore=self._resolve(_quark_cfg(), flag=True).exclude_layers,
            )
        )

    def test_flag_on_does_not_mutate_caller_config(self):
        src = _quark_cfg()
        orig_exclude = list(src.exclude_layers)
        orig_layer = dict(src.quant_config.get("layer_quant_config") or {})
        self._resolve(src, flag=True)
        self.assertEqual(src.exclude_layers, orig_exclude)
        self.assertEqual(src.quant_config.get("layer_quant_config") or {}, orig_layer)

    def test_flag_on_skips_ptpc_when_experts_not_excluded(self):
        src = _quark_cfg(exclude=[ATTN_LEAF])
        out = self._resolve(src, flag=True)
        self.assertNotIn("model.decoder.mlp.experts", out.exclude_layers)
        self.assertNotIn(
            "model.decoder.mlp.experts",
            out.quant_config.get("layer_quant_config", {}),
        )

    def test_flag_on_skips_ptpc_when_mtp_not_in_exclude(self):
        src = _quark_cfg(exclude=[])
        out = self._resolve(src, flag=True)
        self.assertNotIn(
            "model.decoder.mlp.experts",
            out.quant_config.get("layer_quant_config", {}),
        )


class TestMaybeQuantGlmNextnMoeToPtpc(CustomTestCase):
    def _cast(self, cfg, flag: bool):
        loader = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
        loader.quant_config = cfg
        loader.config = SimpleNamespace(num_hidden_layers=LAYER)
        weights = [
            (EXPERT_WEIGHT, torch.ones(4, 8, dtype=torch.bfloat16)),
            (
                SHARED_EXPERT_WEIGHT,
                torch.full((4, 8), 1e-4, dtype=torch.bfloat16),
            ),
        ]
        with patch(
            _PTPC_ENV,
            return_value=flag,
        ):
            return list(loader._maybe_quant_glm_nextn_moe_to_ptpc(weights))

    def test_flag_on_casts_excluded_bf16_experts(self):
        out = self._cast(_quark_cfg(), flag=True)
        names = [name for name, _ in out]
        self.assertIn(EXPERT_WEIGHT, names)
        self.assertIn(EXPERT_WEIGHT[: -len("weight")] + "weight_scale", names)
        weight = dict(out)[EXPERT_WEIGHT]
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(dict(out)[SHARED_EXPERT_WEIGHT].dtype, torch.bfloat16)
        self.assertNotIn(SHARED_EXPERT_WEIGHT[: -len("weight")] + "weight_scale", names)

    def test_flag_on_does_not_cast_when_experts_not_excluded(self):
        src = _quark_cfg(exclude=[ATTN_LEAF])
        out = self._cast(src, flag=True)
        self.assertEqual(len(out), 2)
        self.assertEqual(dict(out)[EXPERT_WEIGHT].dtype, torch.bfloat16)
        self.assertEqual(dict(out)[SHARED_EXPERT_WEIGHT].dtype, torch.bfloat16)


class TestGlmNextnModelSlim(CustomTestCase):
    @staticmethod
    def _config(draft_scheme):
        from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

        config = object.__new__(ModelSlimConfig)
        target_scheme = (
            "W4A8_DYNAMIC" if draft_scheme == "W8A8_DYNAMIC" else "W8A8_DYNAMIC"
        )
        config.quant_description = {
            f"model.layers.{layer}.mlp.experts.0.{projection}.weight": scheme
            for layer, scheme in ((61, target_scheme), (78, draft_scheme))
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        config.quant_description["group_size"] = 64
        config.ignore = []
        config.packed_modules_mapping = {}
        return config

    @staticmethod
    def _build_npu_draft(config):
        from sglang.srt.models import deepseek_nextn

        def decoder(hf, layer_id, **kwargs):
            module = torch.nn.Module()
            module.layer_id = layer_id
            module.prefix = kwargs["prefix"]
            module.is_nextn = kwargs["is_nextn"]
            module.quant_config = kwargs["quant_config"]
            return module

        hf = SimpleNamespace(
            num_hidden_layers=78,
            vocab_size=16,
            hidden_size=8,
            rms_norm_eps=1e-6,
            qk_rope_head_dim=64,
        )
        # Attention, EP and NPU allocation are boundaries. The real constructor
        # chooses the layer ID/prefix and forwards the actual ModelSlim config.
        with (
            patch.object(deepseek_nextn, "_is_npu", True),
            patch.object(deepseek_nextn, "_is_cuda", False),
            patch.object(
                deepseek_nextn, "enable_nextn_moe_bf16_cast_to_fp8", return_value=False
            ),
            patch.object(
                deepseek_nextn,
                "get_model",
                return_value=SimpleNamespace(model_path="/test/glm52"),
            ),
            patch.object(
                deepseek_nextn,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_draft_model_path="/test/glm52"
                ),
            ),
            patch.object(deepseek_nextn.os.path, "isfile", return_value=False),
            patch.object(
                deepseek_nextn.envs.SGLANG_NPU_USE_MULTI_STREAM,
                "get",
                return_value=False,
            ),
            patch.object(deepseek_nextn, "get_embedding_tp_kwargs", return_value={}),
            patch.object(
                deepseek_nextn,
                "VocabParallelEmbedding",
                side_effect=lambda *a, **kw: torch.nn.Identity(),
            ),
            patch.object(
                deepseek_nextn,
                "RMSNorm",
                side_effect=lambda *a, **kw: torch.nn.Identity(),
            ),
            patch.object(deepseek_nextn, "DeepseekV2DecoderLayer", side_effect=decoder),
            patch.object(deepseek_nextn, "layer_stack", side_effect=nullcontext),
        ):
            return deepseek_nextn.DeepseekModelNextN(hf, config, prefix="model")

    def test_same_checkpoint_npu_draft_uses_its_own_dynamic_layer_scheme(self):
        """GLM layer 78 must not resolve the inherited DeepSeek layer-61 scheme."""
        from sglang.srt.hardware_backend.npu.quantization import moe_methods
        from sglang.srt.offloading.moe.npu.model_adapter import get_quant_spec

        for scheme, expected in (("W8A8_DYNAMIC", "w8a8"), ("W4A8_DYNAMIC", "w4a8")):
            with self.subTest(scheme=scheme):
                config = self._config(scheme)
                before = dict(config.quant_description)
                model = self._build_npu_draft(config)
                self.assertEqual(model.decoder.prefix, "model.layers.78")
                self.assertEqual(model.decoder.layer_id, 0)
                self.assertTrue(model.decoder.is_nextn)
                with patch.object(
                    moe_methods,
                    "HiddenStatesDynamicQuant",
                    return_value=SimpleNamespace(),
                    create=True,
                ):
                    first, second = model.decoder.quant_config.get_moe_scheme(
                        model.decoder, model.decoder.prefix + ".mlp.experts"
                    )
                self.assertEqual(get_quant_spec(first, second).kind, expected)
                if expected == "w4a8":
                    self.assertEqual(first.group_size, 64)
                    self.assertFalse(first.is_per_channel_weight)
                self.assertEqual(config.quant_description, before)
                mapper = GlmMoeDsaForCausalLMNextN.get_hf_to_sglang_mapper(
                    SimpleNamespace(num_hidden_layers=78)
                )
                key = "model.layers.61.mlp.experts.0.gate_proj.weight"
                self.assertEqual(mapper._map_name(key), key)

    def test_draft_layer_zero_remains_sparse_despite_targets_dense_prefix(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        config = SimpleNamespace(
            n_routed_experts=256, first_k_dense_replace=3, moe_layer_freq=1
        )
        self.assertFalse(DeepseekV2DecoderLayer._is_layer_sparse(config, 0, False))
        self.assertTrue(DeepseekV2DecoderLayer._is_layer_sparse(config, 0, True))


class TestGlmMtpCheckpointValidation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        path = (
            Path(__file__).resolve().parents[4]
            / "scripts/moe_offload/glm52_mtp_config.py"
        )
        spec = importlib.util.spec_from_file_location("glm52_mtp_config_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.validate = staticmethod(module.validate_mtp_checkpoint)

    @staticmethod
    def _fixture(kind="FLOAT", *, layer=78):
        config = {
            "model_type": "glm_moe_dsa",
            "architectures": ["GlmMoeDsaForCausalLM"],
            "num_hidden_layers": layer,
            "num_nextn_predict_layers": 1,
            "n_routed_experts": 2,
            "n_shared_experts": 0,
            "q_lora_rank": None,
            "topk_method": "noaux_tc",
        }
        prefix = f"model.layers.{layer}."
        modules = (
            "eh_proj",
            "enorm",
            "hnorm",
            "shared_head.norm",
            "input_layernorm",
            "post_attention_layernorm",
            "mlp.gate",
            "self_attn.q_proj",
            "self_attn.kv_a_proj_with_mqa",
            "self_attn.kv_a_layernorm",
            "self_attn.kv_b_proj",
            "self_attn.o_proj",
            "self_attn.indexer.wq_b",
            "self_attn.indexer.wk",
            "self_attn.indexer.weights_proj",
            "self_attn.indexer.k_norm",
        )
        dtypes = {prefix + name + ".weight": "BF16" for name in modules}
        dtypes[prefix + "self_attn.indexer.k_norm.bias"] = "BF16"
        dtypes[prefix + "mlp.gate.e_score_correction_bias"] = "BF16"
        description = {"model.layers.61.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC"}
        if kind == "W4A8_DYNAMIC":
            description["group_size"] = 64
        for expert in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                stem = f"{prefix}mlp.experts.{expert}.{projection}."
                dtypes[stem + "weight"] = "BF16" if kind == "FLOAT" else "I8"
                description[stem + "weight"] = kind
                if kind != "FLOAT":
                    suffixes = ["weight_scale", "weight_offset"]
                    if kind == "W4A8_DYNAMIC":
                        suffixes += [
                            "scale_bias",
                            "weight_scale_second",
                            "weight_offset_second",
                        ]
                    dtypes.update({stem + suffix: "F32" for suffix in suffixes})
        return config, dtypes, description

    @staticmethod
    def _write(root, config, dtypes, description, *, indexed=True):
        root = Path(root)
        (root / "config.json").write_text(json.dumps(config))
        if description is not None:
            (root / "quant_model_description.json").write_text(json.dumps(description))
        sizes = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "I32": 4}
        offset, header = 0, {}
        for name, dtype in dtypes.items():
            end = offset + sizes[dtype]
            header[name] = {"dtype": dtype, "shape": [1], "data_offsets": [offset, end]}
            offset = end
        encoded = json.dumps(header).encode()
        shard = "mtp.safetensors" if indexed else "model.safetensors"
        (root / shard).write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + bytes(offset)
        )
        if indexed:
            weights = {name: shard for name in dtypes}
            # This deliberately absent file proves validation never opens
            # target shards while checking the bundled draft.
            weights["model.layers.61.mlp.experts.0.gate_proj.weight"] = (
                "target-unopened.safetensors"
            )
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weights})
            )

    def test_float_draft_uses_tensor_headers_without_reading_target_shards(self):
        for indexed, layer in ((True, 78), (False, 92)):
            with self.subTest(
                indexed=indexed, layer=layer
            ), tempfile.TemporaryDirectory() as root:
                config, dtypes, _ = self._fixture(layer=layer)
                self._write(root, config, dtypes, None, indexed=indexed)
                result = self.validate(root)
                self.assertEqual(result["mtp_layer_id"], layer)
                self.assertEqual(result["expert_quantization"], "FLOAT")
                self.assertEqual(
                    result["checked_shards"],
                    ["mtp.safetensors" if indexed else "model.safetensors"],
                )

    def test_quantized_draft_requires_matching_mode_and_complete_projection_metadata(
        self,
    ):
        for kind in ("W8A8_DYNAMIC", "W4A8_DYNAMIC"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                config, dtypes, description = self._fixture(kind)
                self._write(root, config, dtypes, description)
                result = self.validate(root, "modelslim")
                self.assertEqual(result["expert_quantization"], kind)
                with self.assertRaisesRegex(ValueError, "quantized|floating"):
                    self.validate(root, "unquant")
                missing = "model.layers.78.mlp.experts.1.down_proj." + (
                    "scale_bias" if kind == "W4A8_DYNAMIC" else "weight_scale"
                )
                del dtypes[missing]
                self._write(root, config, dtypes, description)
                with self.assertRaisesRegex(ValueError, "Missing floating"):
                    self.validate(root, "modelslim")

    def test_missing_mtp_and_integer_weights_without_metadata_fail_closed(self):
        for mutation in (
            "no_mtp",
            "wrong_layer",
            "missing_projection",
            "missing_indexer",
            "missing_gate_bias",
            "integer_without_metadata",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                config, dtypes, _ = self._fixture()
                if mutation == "no_mtp":
                    config["num_nextn_predict_layers"] = 0
                elif mutation == "wrong_layer":
                    config["num_hidden_layers"] = 79
                elif mutation == "missing_projection":
                    del dtypes["model.layers.78.mlp.experts.1.down_proj.weight"]
                elif mutation == "missing_indexer":
                    del dtypes["model.layers.78.self_attn.indexer.k_norm.bias"]
                elif mutation == "missing_gate_bias":
                    del dtypes["model.layers.78.mlp.gate.e_score_correction_bias"]
                else:
                    dtypes["model.layers.78.mlp.experts.1.down_proj.weight"] = "I8"
                self._write(root, config, dtypes, None)
                with self.assertRaises(ValueError):
                    self.validate(root, "unquant")

    def test_quantized_eh_proj_and_mixed_expert_schemes_are_rejected(self):
        for mutation in (
            "eh_proj",
            "mixed_experts",
            "missing_description",
            "float_experts",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                config, dtypes, description = self._fixture("W8A8_DYNAMIC")
                if mutation == "eh_proj":
                    key = "model.layers.78.eh_proj.weight"
                    description[key], dtypes[key] = "W8A8_DYNAMIC", "I8"
                elif mutation == "mixed_experts":
                    description["model.layers.78.mlp.experts.1.down_proj.weight"] = (
                        "W4A8_DYNAMIC"
                    )
                elif mutation == "missing_description":
                    description = None
                else:
                    for key in description:
                        if key.startswith("model.layers.78."):
                            description[key], dtypes[key] = "FLOAT", "BF16"
                self._write(root, config, dtypes, description)
                with self.assertRaises(ValueError):
                    self.validate(root, "modelslim")

    def test_hf_blob_symlinks_are_readable_but_index_parent_traversal_is_rejected(self):
        """HF snapshot shard symlinks legitimately point outside the snapshot root."""
        for indexed in (False, True):
            with self.subTest(indexed=indexed), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "snapshot"
                root.mkdir()
                config, dtypes, description = self._fixture()
                self._write(root, config, dtypes, description, indexed=indexed)
                shard = "mtp.safetensors" if indexed else "model.safetensors"
                blob = Path(temp) / "cached-blob"
                (root / shard).rename(blob)
                (root / shard).symlink_to(blob)
                self.assertEqual(self.validate(root)["expert_quantization"], "FLOAT")
                if indexed:
                    index = {name: "../cached-blob" for name in dtypes}
                    (root / "model.safetensors.index.json").write_text(
                        json.dumps({"weight_map": index})
                    )
                    with self.assertRaises(ValueError):
                        self.validate(root)


if __name__ == "__main__":
    unittest.main()
