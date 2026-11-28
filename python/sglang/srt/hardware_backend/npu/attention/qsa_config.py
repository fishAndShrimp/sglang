from sglang.srt.layers.attention.qsa.config import QSAProfile


def supports_qsa_block_attention(profile: QSAProfile) -> bool:
    """Check whether the NPU QSA kernels support this block layout."""
    return (
        # LightningIndexerV2 takes 128-wide Q/K and 1 to 64 query heads.
        profile.head_dim == 128
        and profile.n_heads <= 64
        # KV compression and the FIA mask both group four tokens per block.
        and profile.compress_ratio == 4
        # qsa_sort_count_groups takes exactly 512 block IDs per query.
        and profile.block_topk == 512
    )
