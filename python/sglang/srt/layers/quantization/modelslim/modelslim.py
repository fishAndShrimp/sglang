from __future__ import annotations

import logging
from copy import copy, deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _NPULinearMethodBase,
)
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimMXFP4Scheme,
    ModelSlimMXFP4W4A8Scheme,
    ModelSlimMXFP8MoEScheme,
    ModelSlimMXFP8Scheme,
    ModelSlimW4A4Int4,
    ModelSlimW4A4Int4MoE,
    ModelSlimW4A4MXFP4MoE,
    ModelSlimW4A8Int8MoE,
    ModelSlimW4A8MXFP4MoE,
    ModelSlimW8A8Int8,
    ModelSlimW8A8Int8MoE,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import apply_module_patch

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
    from sglang.srt.layers.quantization.modelslim.schemes import (
        ModelSlimLinearScheme,
    )

logger = logging.getLogger(__name__)


# func refers to RMSNorm.__init__
def npu_wrapper_rmsnorm_init(func):
    def init(self, hidden_size: int, **extra_args) -> None:
        func(self, hidden_size, **extra_args)
        self.ignore_anti = True
        # The Ascend w8a8_int8 quantization requires adding a bias in rmsnorm
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)

    return init


# func refers to RMSNorm.forward_oot
def npu_wrapper_rmsnorm_forward(func):
    def _rmsnorm_forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias

            out, residual_out = add_rmsnorm_bias(
                x,
                residual,
                self.weight.data,
                self.bias,
                self.variance_epsilon,
            )
            return out.to(x.dtype), residual_out

        out = torch.ops.npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
        out = out + self.bias
        return out.to(x.dtype)

    return _rmsnorm_forward_oot


class ModelSlimConfig(QuantizationConfig):
    """
    Config class for ModelSlim Quantization, a NPU-specific quantization type.
    """

    supports_kimi_k3_quantized_latent_projections = True

    # Hardcoded compatibility for models missing an MLA fusion declaration.
    _HARDCODED_PACKED_MODULES = {
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
    # Hardcoded vision aliases: runtime layer name -> checkpoint layer name.
    # These require the path and fusion guards in the vision resolver below.
    _HARDCODED_VISION_ALIASES = {
        "vision_model": {"proj": "out_proj"},
        "visual": {"qkv_proj": "qkv"},
    }

    def __init__(self, quant_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        quant_config = dict(quant_config or {})
        keys = [k for k in quant_config if isinstance(k, str)]
        is_dsv4 = any(k.startswith("hc_head_") for k in keys)
        if is_dsv4:
            from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

            remap = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
            quant_config = {
                (remap(k) if isinstance(k, str) else k): v
                for k, v in quant_config.items()
            }

        # Add an mlp.* alias for each block_sparse_moe.* key but KEEP the original,
        # so both module namings resolve
        for k in list(quant_config.keys()):
            if not isinstance(k, str):
                continue
            if "block_sparse_moe" in k:
                quant_config[
                    k.replace("block_sparse_moe.experts", "mlp.experts").replace(
                        "block_sparse_moe.shared_experts", "mlp.shared_experts"
                    )
                ] = quant_config[k]

        self.quant_description = quant_config
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )

        for name in self.quant_description.keys():
            if "norm.bias" in name:
                apply_module_patch(
                    "sglang.srt.layers.layernorm.RMSNorm",
                    "__init__",
                    [npu_wrapper_rmsnorm_init],
                )
                apply_module_patch(
                    "sglang.srt.layers.layernorm.RMSNorm",
                    "forward_npu",
                    [npu_wrapper_rmsnorm_forward],
                )
        # DSpark checkpoint weights use mtp.<stage>.*, while the runtime draft
        # model constructs canonical modules under stages.<stage>.*.  Keep
        # this transformation in sync with
        # DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name.  Merely
        # replacing ``mtp`` with ``stages`` is insufficient: it silently
        # misses ModelSlim lookups such as stages.0.mlp.experts and
        # stages.0.self_attn.
        dspark_quant_aliases = {}
        for name, scheme in quant_config.items():
            if not isinstance(name, str) or not name.startswith("mtp."):
                continue

            parts = name.split(".", 2)
            if len(parts) != 3:
                continue

            stage_id, rest = parts[1], parts[2]
            if not stage_id.isdigit():
                continue

            # The draft model attaches the target model's shared embedding and
            # LM head; mtp-local copies are not runtime draft modules.
            if rest.startswith(("embed.", "embed_tokens.", "head.", "lm_head.")):
                continue
            if rest.startswith("markov_head."):
                alias = f"markov_head.{rest[len('markov_head.') :]}"
            elif rest.startswith("confidence_head."):
                alias = f"confidence_head.{rest[len('confidence_head.') :]}"
            else:
                mapped_rest = rest
                if mapped_rest.startswith("attn."):
                    mapped_rest = "self_attn." + mapped_rest.removeprefix("attn.")
                elif mapped_rest.startswith("ffn."):
                    mapped_rest = "mlp." + mapped_rest.removeprefix("ffn.")
                elif mapped_rest.startswith("attn_norm."):
                    mapped_rest = "input_layernorm." + mapped_rest.removeprefix(
                        "attn_norm."
                    )
                elif mapped_rest.startswith("ffn_norm."):
                    mapped_rest = (
                        "post_attention_layernorm."
                        + mapped_rest.removeprefix("ffn_norm.")
                    )
                mapped_rest = mapped_rest.replace(".w1.", ".gate_proj.")
                mapped_rest = mapped_rest.replace(".w2.", ".down_proj.")
                mapped_rest = mapped_rest.replace(".w3.", ".up_proj.")
                mapped_rest = mapped_rest.replace(".gate.tid2eid", ".topk.tid2eid")
                mapped_rest = mapped_rest.replace(
                    ".gate.bias", ".gate.e_score_correction_bias"
                )
                alias = f"stages.{stage_id}.{mapped_rest}"

            dspark_quant_aliases[alias] = scheme

        quant_config = {
            **dspark_quant_aliases,
            **quant_config,
        }

        self.quant_description = quant_config

    @staticmethod
    def _iter_mapped_quant_descriptions(descriptions, mapper):
        """Apply the model mapper to parameter names or their module suffixes."""
        parameter_mapper = copy(mapper)
        parameter_mapper.orig_to_new_suffix = {}
        suffix_mapper = copy(mapper)
        suffix_mapper.orig_to_new_prefix = {}
        suffix_mapper.orig_to_new_substr = {}

        for original, value in descriptions.items():
            # Use the shared mapper for prefix and substring rules first.
            mapped = parameter_mapper.apply_list([original])
            if not mapped:
                continue
            name = mapped[0]
            field = None
            # An existing parameter-suffix rule applies to the full name.
            # Otherwise apply module-suffix rules without the final field.
            if not any(name.endswith(suffix) for suffix in mapper.orig_to_new_suffix):
                module, separator, parameter = name.rpartition(".")
                if separator:
                    name, field = module, parameter
            mapped = suffix_mapper.apply_list([name])
            if mapped:
                name = mapped[0] if field is None else f"{mapped[0]}.{field}"
                yield name, value

    def apply_weight_name_mapper(self, hf_to_sglang_mapper):
        # Keep checkpoint names as well as the model-declared runtime aliases.
        # Always map the original description, so repeated registration cannot
        # apply substring rules twice (qkv -> qkv_proj -> qkv_proj_proj).
        if not hasattr(self, "_unmapped_quant_description"):
            self._unmapped_quant_description = dict(self.quant_description)
        # Only tensor descriptions name parameters. Keep top-level format
        # metadata and structured settings out of the model's name mapper.
        descriptions = {
            name: value
            for name, value in self._unmapped_quant_description.items()
            if isinstance(name, str) and "." in name and isinstance(value, str)
        }
        aliases = dict(self.quant_description)
        for name, value in self._iter_mapped_quant_descriptions(
            descriptions, hf_to_sglang_mapper
        ):
            if name in aliases and aliases[name] != value:
                raise ValueError(f"Conflicting ModelSlim descriptions for {name}")
            aliases[name] = value
        self.quant_description = aliases

    def update_packed_modules_mapping(self, mapping: Dict[str, Any]) -> None:
        # Preserve dict.update semantics without modifying shared tables or lists.
        updated = deepcopy(self.packed_modules_mapping)
        updated.update(deepcopy(mapping))
        self.packed_modules_mapping = updated

    def _quant_prefix_candidates(self, prefix: str) -> List[str]:
        """Return checkpoint-name variants without copying the large config.

        Kimi-K3's upstream model uses ``mlp`` internally while its ModelSlim
        checkpoint retains the Hugging Face ``block_sparse_moe`` hierarchy.
        Some multimodal checkpoints also keep the outer ``language_model``
        prefix.  Resolve those layout-only differences at the quantization
        boundary.
        """
        candidates = [prefix]
        for candidate in list(candidates):
            if candidate.startswith(".") and not candidate.startswith(".."):
                candidates.append(candidate[1:])
        for candidate in list(candidates):
            if ".mlp." in candidate:
                candidates.append(candidate.replace(".mlp.", ".block_sparse_moe."))

        for candidate in list(candidates):
            if candidate.startswith("model.language_model."):
                candidates.append(
                    "model." + candidate.removeprefix("model.language_model.")
                )
            elif candidate.startswith("model."):
                candidates.append(
                    "model.language_model." + candidate.removeprefix("model.")
                )
            if candidate.startswith("model.visual."):
                candidates.append(candidate.removeprefix("model."))
            elif candidate.startswith("visual."):
                candidates.append("model." + candidate)
            if candidate.startswith("language_model."):
                candidates.append(candidate.removeprefix("language_model."))
            else:
                candidates.append(f"language_model.{candidate}")

        return list(dict.fromkeys(candidates))

    def _resolve_quant_prefix(self, prefix: str) -> str:
        for candidate in self._quant_prefix_candidates(prefix):
            if candidate + ".weight" in self.quant_description:
                return candidate
        return prefix

    def get_linear_method(self) -> ModelSlimLinearMethod:
        return ModelSlimLinearMethod(self)

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_name(cls) -> str:
        return "modelslim"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        filenames = ["quant_model_description.json"]
        return filenames

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelSlimConfig:
        return cls(config)

    def _get_packed_modules_mapping(self) -> Mapping[str, List[str]]:
        from sglang.srt.layers.quantization.utils import _FALLBACK_FUSED_SHARDS

        mapping = dict(_FALLBACK_FUSED_SHARDS)
        for name, parts in self.packed_modules_mapping.items():
            if isinstance(parts, Mapping):
                if name != "model":
                    raise ValueError(
                        f"Unsupported ModelSlim packed mapping group: {name!r}"
                    )
                continue
            mapping[name] = parts

        # KimiK3LinearForCausalLM wraps its ModelSlim mapping in a "model"
        # subtable. Other current model declarations use a flat mapping.
        model_mapping = self.packed_modules_mapping.get("model", {})
        if isinstance(model_mapping, Mapping):
            mapping.update(model_mapping)
        # Fill missing entries only; do not share the compatibility lists.
        for name, parts in self._HARDCODED_PACKED_MODULES.items():
            mapping.setdefault(name, list(parts))
        return mapping

    def _resolve_legacy_vision_quant_prefix(
        self, prefix: str, shards: List[str]
    ) -> Optional[str]:
        """Find vision aliases otherwise applied only during load_weights."""
        from sglang.srt.layers.quantization.utils import _FALLBACK_FUSED_SHARDS

        parent, separator, name = prefix.rpartition(".")

        # 1. Select aliases for this vision path.
        path = parent.split(".")
        if "vision_model" in path:
            aliases = self._HARDCODED_VISION_ALIASES["vision_model"]
        elif "visual" in path:
            aliases = self._HARDCODED_VISION_ALIASES["visual"]
        else:
            return None
        checkpoint_name = aliases.get(name)
        if checkpoint_name is None:
            return None

        # 2. Do not override custom fusion declarations.
        standard_shards = _FALLBACK_FUSED_SHARDS.get(name, [])
        if shards and shards != standard_shards:
            return None

        # 3. Use the alias only if its quantization description exists.
        quant_prefix = self._resolve_quant_prefix(parent + separator + checkpoint_name)
        if quant_prefix + ".weight" in self.quant_description:
            return quant_prefix
        return None

    def _linear_quant_prefixes(
        self,
        prefix: str,
        fused_mapping: Mapping[str, List[str]],
        output_sizes: Optional[List[int]] = None,
    ) -> List[str]:
        """Resolve the same checkpoint entries for scheme and FLOAT selection."""
        # Possibility 1: whole layer, e.g. qkv_proj (including name variants).
        direct = self._resolve_quant_prefix(prefix)
        whole_quant_prefix = (
            direct if direct + ".weight" in self.quant_description else None
        )

        # Possibility 2: split components, e.g. q_proj, k_proj, v_proj.
        # Check both layouts to detect ambiguity.
        parent, separator, name = prefix.rpartition(".")
        shards = fused_mapping.get(name, [])
        # QKV projections can omit V by declaring an output size of zero.
        # Only use sizes when they describe exactly the declared fusion.
        if output_sizes is not None and len(output_sizes) == len(shards):
            shards = [shard for shard, size in zip(shards, output_sizes) if size != 0]
        components = [
            self._resolve_quant_prefix(parent + separator + shard)
            for shard in shards
        ]
        present = [p + ".weight" in self.quant_description for p in components]
        if any(present):
            if not all(present):
                missing = [
                    p + ".weight" for p, found in zip(components, present) if not found
                ]
                raise ValueError(
                    f"Missing ModelSlim quantization descriptions for some components "
                    f"of fused layer {prefix}. Missing keys: {missing}"
                )
            schemes = {self.quant_description[p + ".weight"] for p in components}
            if len(schemes) != 1:
                entries = {
                    p + ".weight": self.quant_description[p + ".weight"]
                    for p in components
                }
                raise ValueError(
                    f"ModelSlim requires all components of fused layer {prefix} "
                    f"to use the same quantization scheme. Found: {entries}"
                )
        if whole_quant_prefix is None:
            whole_quant_prefix = self._resolve_legacy_vision_quant_prefix(prefix, shards)
        # A mapper may give the whole tensor a runtime alias; that is still
        # one layout. A separate set of component entries is another layout.
        # Even identical schemes cannot tell us which weights the loader uses.
        if any(present):
            if whole_quant_prefix is not None and components != [whole_quant_prefix]:
                raise ValueError(
                    f"Ambiguous ModelSlim layout for {prefix}: whole tensor "
                    f"{whole_quant_prefix}.weight and component tensors "
                    f"{[p + '.weight' for p in components]} are both described"
                )
            return components
        if whole_quant_prefix is not None:
            return [whole_quant_prefix]
        # Some checkpoints serialize the fused tensor itself. Do not infer
        # quantization from absent component entries, including for FLOAT.
        return [self._resolve_quant_prefix(prefix)]

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if "vision_tower" in prefix or "mm_projector" in prefix:
                prefix = prefix.replace(r"attn.qkv_proj", r"wqkv")
                prefix = prefix.replace(r"attn.proj", r"wo")
            packed_modules_mapping = self._get_packed_modules_mapping()
            prefixes = self._linear_quant_prefixes(
                prefix, packed_modules_mapping, getattr(layer, "output_sizes", None)
            )
            prefix_in_quant_config = prefixes[0]
            scheme_name = self.quant_description.get(prefix_in_quant_config + ".weight")
            if scheme_name == "FLOAT":
                return UnquantizedLinearMethod()
            layer.scheme = self.get_linear_scheme(layer, prefix_in_quant_config)
            if layer.scheme is None:
                return UnquantizedLinearMethod()
            return ModelSlimLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            moe_schemes = self.get_moe_scheme(layer, prefix)
            if moe_schemes is None:
                raise ValueError(f"No ModelSlim MoE scheme found for layer {prefix}")
            layer.w13_scheme, layer.w2_scheme = moe_schemes
            layer.w13_kernel, layer.w2_kernel = (
                layer.w13_scheme.kernel,
                layer.w2_scheme.kernel,
            )
            return ModelSlimFusedMoEMethod(self)
        return None

    def get_linear_scheme(
        self, layer: torch.nn.Module, prefix: Optional[str] = None
    ) -> Optional[ModelSlimLinearScheme]:
        """
        get_scheme method adjusted for modelslim, taken from
        python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
        """

        linear_quant_schemes = [
            ("W4A4_DYNAMIC", ModelSlimW4A4Int4),
            ("W8A8", ModelSlimW8A8Int8),
            ("W8A8S", ModelSlimW8A8Int8),
            ("W8A8_DYNAMIC", ModelSlimW8A8Int8),
            ("W8A8_MXFP8", ModelSlimMXFP8Scheme),
            ("W4A8_MXFP", ModelSlimMXFP4W4A8Scheme),
            ("W4A4_MXFP4", ModelSlimMXFP4Scheme),
        ]

        prefix = self._resolve_quant_prefix(prefix)
        key = prefix + ".weight"
        if key not in self.quant_description:
            logger.warning(
                "No ModelSlim description for layer %s; using unquantized weights. "
                "Tried: %s",
                prefix,
                [
                    candidate + ".weight"
                    for candidate in self._quant_prefix_candidates(prefix)
                ],
            )
            return None
        scheme_name = self.quant_description[key]
        if scheme_name == "FLOAT":
            return None

        for supported_name, scheme_class in linear_quant_schemes:
            if scheme_name == supported_name:
                logger.info_once(f"Using {scheme_class.__name__}")
                return scheme_class(quant_config=self.quant_description, prefix=prefix)

        raise ValueError(
            f"Unsupported ModelSlim Linear scheme {scheme_name!r} for layer {prefix} "
            f"(description key: {key})"
        )

    def get_moe_scheme(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ):
        moe_quant_schemes = [
            ("W4A4_MXFP4", ModelSlimW4A4MXFP4MoE),
            ("W4A8_MXFP", ModelSlimW4A8MXFP4MoE),
            ("W4A4_DYNAMIC", ModelSlimW4A4Int4MoE),
            ("W4A8_DYNAMIC", ModelSlimW4A8Int8MoE),
            ("W8A8_DYNAMIC", ModelSlimW8A8Int8MoE),
            ("W8A8_MXFP8", ModelSlimMXFP8MoEScheme),
        ]

        # Try multiple naming conventions:
        #   (gate_proj, up_proj, down_proj) – standard compressed-tensors format
        #   (w1, w3, w2)                      – MiniMax-M2.5 / some other models
        naming_conventions = [
            ("gate_proj", "up_proj", "down_proj"),
            ("w1", "w3", "w2"),
        ]

        w13_scheme_name = None
        w2_scheme_name = None
        resolved_prefix = prefix
        for candidate in self._quant_prefix_candidates(prefix):
            for gate_name, up_name, down_name in naming_conventions:
                w13_keys = [
                    f"{candidate}.0.{gate_name}.weight",
                    f"{candidate}.0.{up_name}.weight",
                ]
                w2_key = f"{candidate}.0.{down_name}.weight"
                w13_entries = {
                    key: self.quant_description[key]
                    for key in w13_keys
                    if key in self.quant_description
                }
                if w13_entries and w2_key in self.quant_description:
                    w13_names = list(w13_entries.values())
                    # For w13, both projections must agree on the scheme
                    unique_w13 = set(w13_names)
                    if len(unique_w13) > 1:
                        raise ValueError(
                            "Mismatched ModelSlim quantization for W13 in layer "
                            f"{prefix}: {w13_entries}"
                        )
                    w13_scheme_name = w13_names[0]
                    w2_scheme_name = self.quant_description[w2_key]
                    resolved_prefix = candidate
                    break
            if w13_scheme_name is not None:
                break

        if w13_scheme_name is None:
            # Build a helpful error message listing all attempted key patterns
            all_attempted = []
            for candidate in self._quant_prefix_candidates(prefix):
                for gate_name, up_name, down_name in naming_conventions:
                    w13_keys = [
                        f"{candidate}.0.{gate_name}.weight",
                        f"{candidate}.0.{up_name}.weight",
                    ]
                    w2_key = f"{candidate}.0.{down_name}.weight"
                    w13_found = any(k in self.quant_description for k in w13_keys)
                    w2_found = w2_key in self.quant_description
                    status = (
                        f"{candidate} "
                        f"({gate_name}/{up_name}="
                        f"{'found' if w13_found else 'missing'}, "
                        f"{down_name}={'found' if w2_found else 'missing'})"
                    )
                    all_attempted.append(status)
            raise ValueError(
                f"Missing ModelSlim MoE quantization description for layer {prefix}: "
                + "; ".join(all_attempted)
            )

        # Map scheme names to classes
        scheme_map = dict(
            moe_quant_schemes
        )  # dict: "W4A4_DYNAMIC" -> ModelSlimW4A4Int4MoE, etc.

        # Instantiate the schemes
        def instantiate(name, weight_group):
            cls = scheme_map.get(name)
            if cls is None:
                logger.warning(
                    f"Unsupported scheme '{name}' for layer {resolved_prefix}"
                )
                return None
            return cls(self, weight_group)

        w13_scheme = instantiate(w13_scheme_name, weight_group="w13")
        w2_scheme = instantiate(w2_scheme_name, weight_group="w2")
        if w13_scheme is None or w2_scheme is None:
            raise ValueError(
                f"Unsupported ModelSlim MoE schemes for layer {prefix}: "
                f"W13='{w13_scheme_name}', W2='{w2_scheme_name}'"
            )
        logger.info_once(f"Using {type(w13_scheme).__name__} for W13")
        logger.info_once(f"Using {type(w2_scheme).__name__} for W2")

        return w13_scheme, w2_scheme

    def is_layer_skipped(
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})
    ):
        resolved = self._linear_quant_prefixes(prefix, fused_mapping)[0]
        return self.quant_description.get(resolved + ".weight") == "FLOAT"

    def get_scaled_act_names(self) -> List[str]:
        return []


class ModelSlimLinearMethod(_NPULinearMethodBase):
    def __init__(self, quantization_config: ModelSlimConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the ModelSlimLinearScheme associated with the layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Use the output of create_weights and the ModelSlimLinearScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class ModelSlimFusedMoEMethod(FusedMoEMethodBase):
    """
    Fused MoE method for ModelSlim quantization on Ascend NPU.

    Delegates routing, activation, and finalization to the modular NPU MoE
    components introduced in the hardware backend refactoring.
    """

    def __init__(self, quantization_config: ModelSlimConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_scheme.process_weights_after_loading(layer)
        layer.w2_scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the ModelSlimMoEScheme associated with the layer to create
        the necessary parameters for the layer. See FusedMoEMethodBase for param
        details
        """
        layer.w13_scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            weight_prefix="w13",
            **extra_weight_attrs,
        )
        layer.w2_scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            weight_prefix="w2",
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        moe_runner_config.layer = layer
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.ASCEND
        self.runner = MoeRunner(backend, moe_runner_config)

    # ------------------------------------------------------------------
    # Main apply()
    # ------------------------------------------------------------------
    def apply(
        self,
        layer,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        quant_info = AscendQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight_scale=layer.w2_weight_scale,
            w13_weight_offset=getattr(layer, "w13_weight_offset", None),
            w2_weight_offset=getattr(layer, "w2_weight_offset", None),
            w13_scale_bias=getattr(layer, "w13_scale_bias", None),
            w2_scale_bias=getattr(layer, "w2_scale_bias", None),
            w13_weight_bias=getattr(layer, "w13_weight_bias", None),
            w2_weight_bias=getattr(layer, "w2_weight_bias", None),
        )
        return self.runner.run(dispatch_output, quant_info)
