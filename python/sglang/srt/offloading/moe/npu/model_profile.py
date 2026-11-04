"""Startup-only architecture facts, separate from runtime format support.

Recognizing a configuration does not enable its NPU offload adapter. In
particular MXFP4/SiTU cannot be treated as ModelSlim INT4/SiLU.
"""

from dataclasses import asdict, dataclass


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def identify_model(config):
    kind = _get(config, "model_type")
    identities = {
        "deepseek_v4": "DeepSeek-V4-Flash",
        "deepseek_v41": "DeepSeek-V4.1-Flash",
        "deepseek_v41_text": "DeepSeek-V4.1-Flash",
        "glm5_next": "GLM-5.3-Flash",
        "glm5_next_text": "GLM-5.3-Flash",
        "kimi_k3": "Kimi-K3",
    }
    if kind == "glm_moe_dsa" and "GlmMoeDsaForCausalLM" in (_get(config, "architectures", ()) or ()):
        return "GLM-5.2"
    if kind in identities:
        return identities[kind]
    raise ValueError(f"Unknown expert offload model architecture: {kind!r}")


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    global_experts: int
    top_k: int
    hidden_size: int
    expert_hidden_size: int
    intermediate_size: int
    activation: str
    moe_layer_ids: tuple
    resident_layer_ids: tuple
    runtime_supported: bool
    unsupported_reason: str

    def to_dict(self):
        return asdict(self)

    def require_runtime(self):
        if not self.runtime_supported:
            raise ValueError(f"{self.model_id} is recognized for capacity planning only: {self.unsupported_reason}")

    def local_experts(self, ep_size):
        _integer(ep_size, "ep_size")
        if self.global_experts % ep_size:
            raise ValueError("Routed experts must divide evenly across EP ranks")
        return self.global_experts // ep_size


def inspect_model(config):
    model = identify_model(config)
    text = _get(config, "text_config") or config
    kimi = model == "Kimi-K3"
    experts = _integer(_get(text, "num_experts" if kimi else "n_routed_experts"), "routed experts")
    top_k = _integer(_get(text, "num_experts_per_token" if kimi else "num_experts_per_tok"), "top-k")
    if top_k > experts:
        raise ValueError("top-k exceeds routed experts")
    hidden = _integer(_get(text, "hidden_size"), "hidden_size")
    expert_hidden = _integer(_get(text, "routed_expert_hidden_size", hidden) if kimi else hidden, "expert_hidden_size")
    intermediate = _integer(_get(text, "moe_intermediate_size"), "moe_intermediate_size")
    layers = _integer(_get(text, "num_hidden_layers"), "num_hidden_layers")
    activation = _get(text, "hidden_act")
    if not isinstance(activation, str) or not activation:
        raise ValueError("Expert activation must be explicit")
    if model in ("GLM-5.2", "GLM-5.3-Flash", "Kimi-K3"):
        first = _integer(_get(text, "first_k_dense_replace"), "first_k_dense_replace", 0)
        frequency = _integer(_get(text, "moe_layer_freq", 1), "moe_layer_freq")
        moe_ids = tuple(i for i in range(first, layers) if i % frequency == 0)
    else:
        moe_ids = tuple(range(layers))
    if not moe_ids:
        raise ValueError("Model has no routed expert layers")
    # Only the two validated adapters assign resident layers. Other profiles
    # expose a conservative catalog upper bound, not a fabricated residency policy.
    resident = ()
    if model == "DeepSeek-V4-Flash":
        hashes = _integer(_get(text, "num_hash_layers", 0), "num_hash_layers", 0)
        resident = tuple(range(min(layers, max(3, hashes))))
    elif model == "GLM-5.2":
        resident = moe_ids[:1]
    supported = model in ("DeepSeek-V4-Flash", "GLM-5.2")
    reason = ""
    if not supported:
        reason = ("NPU weight format, activation/layout and resident-layer policy need validation"
                  if not kimi else "MXFP4/SiTU/latent-MoE adapter is not implemented; ModelSlim INT4 is not interchangeable")
    if supported and activation != "silu":
        supported, reason = False, "current ModelSlim adapter requires SiLU expert activation"
    return ModelProfile(model, experts, top_k, hidden, expert_hidden, intermediate,
                        activation, moe_ids, resident, supported, reason)
