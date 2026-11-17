"""QSA attention fallback for the standard CANN 9.2.0 runtime.

The toolkit GBSA accepts head dimension 128 and paged KV, while QSA needs
 dimension 256 and arbitrary physical token slots. Preserve the existing
 sparse attention implementation until a compatible interface is available.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _report_fallback():
    logger.warning(
        "QSA uses sparse attention fallback: standard CANN 9.2.0 GBSA "
        "does not support QSA head dimension 256."
    )


def qsa_prefill_attention(q, k, v, slots, softmax_scale=None):
    """Preserve QSA masks, empty rows, and slot semantics on CANN 9.2.0."""
    from sgl_kernel_npu.operators.attention import qsa_sparse_attention

    _report_fallback()
    return qsa_sparse_attention(q, k, v, slots, softmax_scale)


def qsa_decode_attention(q, k, v, slots, softmax_scale=None):
    """The experimental decode GBSA switch also falls back on this runtime."""
    return qsa_prefill_attention(q, k, v, slots, softmax_scale)
