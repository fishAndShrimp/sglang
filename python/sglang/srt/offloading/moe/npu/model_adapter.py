"""Model layer identity and homogeneous ModelSlim expert bundle selection."""

from dataclasses import dataclass

from ..config import ExpertCatalog, ExpertLayerSpec


def get_model_id(hf_config):
    model_type = getattr(hf_config, "model_type", None)
    if model_type == "deepseek_v4":
        return "DeepSeek-V4-Flash"
    if model_type == "glm_moe_dsa" and "GlmMoeDsaForCausalLM" in (
        getattr(hf_config, "architectures", None) or ()
    ):
        return "GLM-5.2"
    raise ValueError(
        "Ascend expert offload supports DeepSeek V4 and GLM-5.2 "
        "(glm_moe_dsa / GlmMoeDsaForCausalLM) target models"
    )


def build_expert_catalog(hf_config, ep_size, ep_rank):
    model_id = get_model_id(hf_config)
    layer_count = hf_config.num_hidden_layers
    if type(layer_count) is not int or layer_count <= 0:
        raise ValueError("num_hidden_layers must be a positive integer")
    if model_id == "GLM-5.2":
        first = hf_config.first_k_dense_replace
        frequency = hf_config.moe_layer_freq
        if type(first) is not int or first < 0:
            raise ValueError("first_k_dense_replace must be a nonnegative integer")
        if type(frequency) is not int or frequency <= 0:
            raise ValueError("moe_layer_freq must be a positive integer")
        # Match DeepseekV2DecoderLayer, which also implements GLM's DSA model.
        layer_ids = [i for i in range(first, layer_count) if i % frequency == 0]
        resident_ids = set(layer_ids[:1])
    else:
        layer_ids = range(layer_count)
        resident_ids = set(range(max(3, getattr(hf_config, "num_hash_layers", 0))))
    catalog = ExpertCatalog(
        tuple(
            ExpertLayerSpec(
                i, hf_config.n_routed_experts, ep_size, ep_rank, i in resident_ids
            )
            for i in layer_ids
        )
    )
    if not catalog.offloaded_layers:
        raise ValueError("Expert offload requires at least one nonresident MoE layer")
    return catalog


@dataclass(frozen=True)
class ExpertQuantSpec:
    kind: str
    group_size: int = 0
    tp_size: int = 1
    activation_use_clip: bool = False

    @property
    def supports_direct_hot(self):
        # A2/A3 GMM group_list_type=2 requires INT8 weights and activations.
        # W4's packed INT32 carrier must use the packed/count path instead.
        return self.kind == "w8a8"


def get_quant_spec(w13_scheme, w2_scheme):
    def describe(scheme):
        name = type(scheme).__name__
        if name == "ModelSlimW8A8Int8MoE":
            return ExpertQuantSpec("w8a8")
        if name != "ModelSlimW4A8Int8MoE":
            raise ValueError("Offload requires ModelSlim W8A8_DYNAMIC or W4A8_DYNAMIC")
        spec = ExpertQuantSpec(
            "w4a8", scheme.group_size, scheme.tp_size, scheme.activation_use_clip
        )
        if type(spec.group_size) is not int or spec.group_size < 0:
            raise ValueError("W4A8 group_size must be a nonnegative integer")
        if type(spec.tp_size) is not int or spec.tp_size <= 0 or 16 % spec.tp_size:
            raise ValueError("W4A8 scale-bias tp_size must divide 16")
        if spec.activation_use_clip:
            raise ValueError(
                "W4A8 MoE offload does not support activation_use_clip=True"
            )
        if scheme.is_per_channel_weight != (spec.group_size == 0):
            raise ValueError("Inconsistent W4A8 group and per-channel configuration")
        return spec

    w13, w2 = describe(w13_scheme), describe(w2_scheme)
    if w13 != w2:
        raise ValueError("Offload requires matching W13/W2 quantization formats")
    return w13
