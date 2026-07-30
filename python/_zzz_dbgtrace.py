"""
Debug trace module for SGLang development.

Only active when SGLANG_ZZZ_DBGTRACE=1 and SGLANG_ALLOW_ZZZ_DBGTRACE=1.
All debug/tracing code is isolated here — zero impact when disabled.

Usage:
    from _zzz_dbgtrace import DBGTRACE
    DBGTRACE(some_condition, lambda: f"expensive info: {compute()}", rank=0)
"""

import os
import time
import warnings
from array import array

# ═══════════════════════════════════════════════════════════════════
# Security guard — warns if imported without explicit authorization
# ═══════════════════════════════════════════════════════════════════

if os.environ.get("SGLANG_ALLOW_ZZZ_DBGTRACE") != "1":
    warnings.warn(
        "⚠️ _zzz_dbgtrace module imported without SGLANG_ALLOW_ZZZ_DBGTRACE=1. "
        "If this appears in CI/PR, debug code was not cleaned up!",
        RuntimeWarning,
        stacklevel=2,
    )

_DEBUG_ON = os.environ.get("SGLANG_ZZZ_DBGTRACE", "1") == "1"


# ═══════════════════════════════════════════════════════════════════
# Core trace function
# ═══════════════════════════════════════════════════════════════════

def DBGTRACE(condition, thunk, rank=None):
    """Lazy debug trace — only evaluates ``thunk()`` if both
    ``SGLANG_ZZZ_DBGTRACE=1`` and ``condition`` are truthy.

    Returns silently otherwise (zero overhead for disabled traces).
    """
    if _DEBUG_ON and condition:
        prefix = f"[rank{rank}] " if rank is not None else ""
        result = thunk()
        if result:  # skip empty / None results
            print(f"{prefix}[DBGTRACE] {result}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Radix tree dump utilities
# ═══════════════════════════════════════════════════════════════════

def dump_radix_tree(tree_cache) -> str:
    """Build a file-tree-like string of the radix tree showing
    tokens, KV cache, SSM/Mamba state, lock refs per node.

    Shows ALL nodes (including evicted) so the prefix structure is
    always visible; evicted nodes are marked [evicted].

    Works generically across RadixCache, MambaRadixCache, SWARadixCache,
    UnifiedRadixCache etc. by probing node attributes.
    """
    root = getattr(tree_cache, "root_node", None)
    if root is None:
        return "  (no root_node — ChunkCache or disabled)\n"

    lines = []

    def _fmt_tokens(key) -> str:
        t = key.token_ids
        ps = getattr(tree_cache, 'page_size', 1)
        head_n = min(max(ps, 25), len(t), 50)
        if len(t) <= head_n + 4:
            return str(list(t))
        ck = tuple(t[:ps])
        ck_hash = hex(hash(ck) & 0xFFFF)
        return f"ck={ck_hash} " + str(list(t[:head_n]) + ["…"] + list(t[-3:]))

    def _get_val_len(node):
        if hasattr(node, "value") and node.value is not None:
            try:
                return len(node.value)
            except Exception:
                return "?"
        if hasattr(node, "component_data"):
            for cd in node.component_data:
                v = getattr(cd, "value", None)
                if v is not None:
                    try:
                        return len(v)
                    except Exception:
                        return "?"
        return 0

    def _get_mamba_str(node):
        if hasattr(node, "mamba_value") and node.mamba_value is not None:
            return "✓"
        if hasattr(node, "component_data"):
            from sglang.srt.mem_cache.unified_cache_components import ComponentType
            if len(node.component_data) > 2:
                cd = node.component_data[ComponentType.MAMBA]
                v = getattr(cd, "value", None)
                if v is not None and (hasattr(v, "__len__") and len(v) > 0):
                    return "✓"
        return "✗"

    def _get_flk(node):
        for attr in ("full_lock_ref", "lock_ref"):
            v = getattr(node, attr, None)
            if isinstance(v, int):
                return v
        return 0

    def _get_mlk(node):
        v = getattr(node, "mamba_lock_ref", None)
        return v if isinstance(v, int) else 0

    def _is_evicted(node):
        if hasattr(node, "evicted"):
            e = node.evicted
            if isinstance(e, bool):
                return e
            if callable(e):
                return e()
        if hasattr(node, "component_data"):
            for cd in node.component_data:
                if getattr(cd, "value", None) is not None:
                    return False
            return node.parent is not None
        return False

    def _fmt_node(node) -> str:
        if node is root:
            return "ROOT"
        key = getattr(node, "key", None)
        tok_str = _fmt_tokens(key) if key is not None else "?"
        kv_len = _get_val_len(node)
        flk = _get_flk(node)
        mlk = _get_mlk(node)
        flk_icon = "🔒" if flk > 0 else "  "
        mba = _get_mamba_str(node)
        nid = getattr(node, "id", "?")
        ek = getattr(key, "extra_key", None) if key is not None else None
        ek_str = f" ek={ek}" if ek is not None else ""
        parts = [f"[id={nid}]{ek_str} {tok_str}  kv:{kv_len}"]
        parts.append(f"{flk_icon}lk:{flk}")
        if mlk or mba != "✗":
            mlk_icon = "🔒" if mlk > 0 else "  "
            parts.append(f"{mlk_icon}mlk:{mlk}  mba:{mba}")
        else:
            parts.append(f"mba:{mba}")
        if _is_evicted(node):
            parts.append("[evicted]")
        return "  ".join(parts)

    def _children_of(node):
        if hasattr(node, "children"):
            return list(node.children.values())
        return []

    # Collect stats
    total_nodes = 0
    total_tokens = 0
    total_evicted = 0

    def _collect(node):
        nonlocal total_nodes, total_tokens, total_evicted
        if node is not root:
            total_nodes += 1
            if hasattr(node, "key") and node.key is not None:
                total_tokens += len(node.key)
            if _is_evicted(node):
                total_evicted += 1

    stack = [root]
    while stack:
        node = stack.pop()
        _collect(node)
        for child in _children_of(node):
            stack.append(child)

    lines.append(
        f"Radix Tree  type={type(tree_cache).__name__}  "
        f"page_size={getattr(tree_cache, 'page_size', '?')}  "
        f"nodes={total_nodes}  tokens={total_tokens}  "
        f"evicted={total_evicted}"
    )

    def _draw(node, prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + _fmt_node(node))
        children = _children_of(node)
        for i, child in enumerate(children):
            extension = "    " if is_last else "│   "
            _draw(child, prefix + extension, i == len(children) - 1)

    children = _children_of(root)
    if children:
        for i, child in enumerate(children):
            _draw(child, "", i == len(children) - 1)
    else:
        lines.append("  (empty)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Composite formatters (produce final log strings)
# ═══════════════════════════════════════════════════════════════════

def fmt_pd_request_arrival(req, is_retracted, disaggregation_mode, tree_cache) -> str:
    """Format PD request arrival header + BEFORE radix tree dump."""
    from sglang.srt.disaggregation.utils import DisaggregationMode
    if disaggregation_mode == DisaggregationMode.NULL:
        return ""
    mode_tag = "PREFILL" if disaggregation_mode == DisaggregationMode.PREFILL else "DECODE"
    tree_str = dump_radix_tree(tree_cache)
    return (
        f"\n{'=' * 80}\n"
        f"  [PD-{mode_tag}] NEW REQUEST ARRIVED\n"
        f"    rid              = {req.rid}\n"
        f"    input_len        = {len(req.origin_input_ids)}\n"
        f"    output_len       = {len(req.output_ids)}\n"
        f"    bootstrap_room   = {req.bootstrap_room}\n"
        f"    bootstrap_host   = {req.bootstrap_host}\n"
        f"    is_retracted     = {is_retracted}\n"
        f"    max_new_tokens   = {req.sampling_params.max_new_tokens}\n"
        f"    priority         = {req.priority}\n"
        f"{'=' * 80}\n"
        f"\n╔{'═' * 78}╗\n"
        f"║{'   🌲 BEFORE ── RADIX TREE ── BEFORE ── 🌲'.center(80)}║\n"
        f"╚{'═' * 78}╝\n"
        f"{tree_str}"
    )


def fmt_tree_after_change(tree_cache, req, caller: str) -> str:
    """Format AFTER radix tree dump for cache insert operations."""
    tree_str = dump_radix_tree(tree_cache)
    rid_short = getattr(req, "rid", "?")[:8]
    title = f"   🏁 AFTER ── {caller} ── rid={rid_short}.."
    return (
        "\n"
        + "╔" + "═" * 78 + "╗\n"
        + "║" + title.ljust(80) + "║\n"
        + "╚" + "═" * 78 + "╝\n"
        + tree_str
    )


# ═══════════════════════════════════════════════════════════════════
# Mamba state tracking
# ═══════════════════════════════════════════════════════════════════

def collect_tree_mamba_states(tree_cache, req) -> list:
    """Walk the radix tree from req.last_node's parent up to root, collecting
    (prefix_token_count, node_id) for nodes that already had mamba state
    BEFORE this request. Excludes req.last_node itself (that's the request's
    own mamba, already sent as state=(MAMBA=1rows))."""
    from sglang.srt.mem_cache.unified_cache_components import ComponentType

    if req.last_node is None or req.last_node == tree_cache.root_node:
        return []

    # Walk up from last_node.parent to root, collecting ancestor nodes.
    # Radix tree keys are fragments, so we accumulate the real prefix length.
    ancestors = []
    node = req.last_node.parent
    while node is not None and node != tree_cache.root_node:
        ancestors.append(node)
        node = node.parent
    ancestors.reverse()  # now root→leaf order

    results = []
    cumulative = 0
    ct = ComponentType.MAMBA
    for node in ancestors:
        cumulative += len(node.key) if node.key else 0
        if len(node.component_data) > ct:
            cd_mamba = node.component_data[ct].value
            if cd_mamba is not None:
                mamba_len = len(cd_mamba) if hasattr(cd_mamba, "__len__") else 1
                results.append({
                    "node_id": node.id,
                    "len_node_key": len(node.key) if node.key else 0,
                    "prefix_cum": cumulative,
                    "mamba_rows": mamba_len,
                })
    return results


def fmt_tree_mamba_sent(
    req,
    start_idx,
    pf_tree_mamba_idx,
    pf_tree_mamba_prefix,
    pf_tree_mamba_node_id,
) -> str:
    """Debug log: tree mamba state (if any) being sent alongside this chunk."""
    rid_short = req.rid[:8] if req.rid else "?"
    W = 74

    def _box_top():
        return "╔" + "═" * W + "╗"

    def _box_bot():
        return "╚" + "═" * W + "╝"

    def _row(s):
        return "║  " + s.ljust(W - 2) + "║"

    lines = []
    lines.append(_box_top())
    lines.append(_row(f"🧬 TREE MAMBA CHECK  rid={rid_short}  start_idx={start_idx}"))
    lines.append("╠" + "═" * W + "╣")

    if pf_tree_mamba_idx is None:
        lines.append(_row("❌ No tree mamba found at this boundary"))
        if start_idx == 0:
            lines.append(_row("   (start_idx=0 → sending from beginning, no boundary)"))
        else:
            lines.append(_row(f"   (checked ancestors up to root — all None or TOMBSTONE)"))
    else:
        lines.append(_row(f"✅ Tree mamba FOUND  at prefix_cum={pf_tree_mamba_prefix}"))
        lines.append(_row(f"   node_id={pf_tree_mamba_node_id}"))
        idx_val = (
            int(pf_tree_mamba_idx[0]) if hasattr(pf_tree_mamba_idx, "__getitem__")
            else pf_tree_mamba_idx
        )
        lines.append(_row(f"   prefill-side pool index = {idx_val}"))
        lines.append(_row(f"   (decode-side index will differ — see decode logs)"))

    # req's own mamba info
    own_idx = getattr(req, "mamba_pool_idx", None)
    if own_idx is not None:
        lines.append("╟" + "─" * W + "╢")
        own_val = int(own_idx) if hasattr(own_idx, "__int__") else own_idx
        lines.append(_row(f"🔹 req's own mamba  prefill pool idx={own_val}"))

    lines.append(_box_bot())
    return "\n".join(lines)


def fmt_pd_send_mamba_states(tree_cache, req) -> str:
    """Format tree mamba states log line (only if states exist)."""
    info = collect_tree_mamba_states(tree_cache, req)
    if not info:
        return ""
    return f"📤 [PD-SEND] rid={req.rid[:8]}..  tree_mamba_states={info}"


def fmt_kv_send_detail(
    req,
    page_indices,
    start_idx,
    end_idx,
    last_chunk,
    state_indices,
    state_types,
    page_size,
    tree_cache,
) -> str:
    """Detailed dump of the KV cache pages being sent to decode,
    together with the radix-tree ancestor chain so you can see which
    tree nodes overlap the send range and whether they carry SSM state."""
    from sglang.srt.mem_cache.unified_cache_components import ComponentType

    rid_short = req.rid[:8] if req.rid else "?"
    W = 74  # inner content width

    # ── helpers ──
    def _box_top():
        return "╔" + "═" * W + "╗"

    def _box_sep():
        return "╠" + "═" * W + "╣"

    def _box_mid():
        return "╟" + "─" * W + "╢"

    def _box_bot():
        return "╚" + "═" * W + "╝"

    def _row(text: str) -> str:
        return "║  " + text.ljust(W - 2) + "║"

    lines = []

    # ═══════════════════ HEADER ═══════════════════
    lines.append(_box_top())
    chunk_tag = "🧩 LAST" if last_chunk else "📋 MID "
    lines.append(_row(f"📦 KV CACHE → DECODE   rid={rid_short}   {chunk_tag}"))
    lines.append(_box_sep())

    # ═══════════════════ SEND SUMMARY ═══════════════════
    n_tokens = end_idx - start_idx
    n_pages = len(page_indices)
    lines.append(_row(f"📤 Sending  tokens=[{start_idx} → {end_idx})  len={n_tokens}  pages={n_pages}"))

    if n_pages > 0 and n_pages <= 10:
        lines.append(_row(f"   page_ids={list(page_indices)}"))
    elif n_pages > 0:
        lines.append(_row(f"   page_ids={list(page_indices[:5])} … {list(page_indices[-3:])} (first 5 + last 3)"))
    else:
        lines.append(_row(f"   (no KV pages — state-only chunk)"))

    if last_chunk and state_indices:
        parts = []
        for st, si in zip(state_types, state_indices):
            if si is not None:
                parts.append(f"{st.name}={len(si)}rows")
        if parts:
            lines.append(_row(f"   state:  {', '.join(parts)}"))

    lines.append(_box_sep())
    lines.append(_row("🌲 TREE ANCESTOR CHAIN  (last_node.parent → root)"))
    lines.append(_box_sep())

    # ═══════════════════ ANCESTOR NODES ═══════════════════
    if req.last_node is None or req.last_node == tree_cache.root_node:
        lines.append(_row("(no prefix match — nothing cached on tree)"))
    else:
        ancestors = []
        node = req.last_node.parent
        while node is not None and node != tree_cache.root_node:
            ancestors.append(node)
            node = node.parent
        ancestors.reverse()

        ct_full = ComponentType.FULL
        ct_mamba = ComponentType.MAMBA
        cumulative = 0  # cumulative prefix at END of each node

        if not ancestors:
            lines.append(_row("(no ancestors — last_node is direct child of root)"))
        else:
            for node in ancestors:
                prev_cum = cumulative
                node_key_len = len(node.key) if node.key else 0
                cumulative += node_key_len
                node_start = prev_cum
                node_end = cumulative

                # ── chunk overlap badge ──
                if node_end <= start_idx:
                    badge = "⬅️  BEFORE  (already on decode / cached)"
                elif node_start >= end_idx:
                    badge = "➡️  AFTER   (future chunk)"
                elif node_start >= start_idx and node_end <= end_idx:
                    badge = "✅ IN THIS CHUNK"
                elif node_start < start_idx and node_end <= end_idx:
                    badge = "↗️  TAIL IN  (head before, tail in this chunk)"
                elif node_start >= start_idx and node_end > end_idx:
                    badge = "↘️  HEAD IN  (head in this chunk, tail after)"
                else:
                    badge = "↔️  SPANS    (starts before, ends after this chunk)"

                # KV slots on this node
                kv_slots = 0
                if len(node.component_data) > ct_full:
                    cd_full = node.component_data[ct_full]
                    if cd_full.value is not None:
                        kv_slots = len(cd_full.value)
                kv_pages = (kv_slots + page_size - 1) // page_size if kv_slots else 0

                # Mamba / SSM state
                mamba_val = None
                if len(node.component_data) > ct_mamba:
                    mamba_val = node.component_data[ct_mamba].value

                lines.append(_box_mid())
                lines.append(_row(f"  node id={node.id}   tokens=[{node_start}, {node_end})"))
                lines.append(_row(f"    len_node_key={node_key_len:<5}  prefix_cum={cumulative:<5}"))
                lines.append(_row(f"    kv_slots={kv_slots:<5}  (~{kv_pages} pages)"))

                if mamba_val is not None:
                    mr = len(mamba_val) if hasattr(mamba_val, "__len__") else 1
                    lines.append(_row(f"    🧬 SSM state: ✓  ({mr} rows)"))
                elif kv_slots > 0:
                    lines.append(_row(f"    🪦 SSM state: ✗  TOMBSTONE (lost on split)"))
                else:
                    lines.append(_row(f"    SSM state: ✗"))

                lines.append(_row(f"    {badge}"))

        # ── last_node itself (request's own) ──
        lines.append(_box_mid())
        last = req.last_node
        last_key_len = len(last.key) if last.key else 0
        prev_cum = cumulative
        last_cum = cumulative + last_key_len

        # badge for last_node
        if last_cum <= start_idx:
            last_badge = "⬅️  BEFORE this chunk"
        elif prev_cum >= end_idx:
            last_badge = "➡️  AFTER this chunk"
        elif prev_cum >= start_idx and last_cum <= end_idx:
            last_badge = "✅ IN THIS CHUNK"
        elif prev_cum < start_idx:
            last_badge = "↗️  overlaps START of this chunk"
        elif last_cum > end_idx:
            last_badge = "↘️  overlaps END of this chunk"
        else:
            last_badge = "↔️  spans across this chunk"

        mamba_val = None
        if len(last.component_data) > ct_mamba:
            mamba_val = last.component_data[ct_mamba].value
        mr = len(mamba_val) if (mamba_val is not None and hasattr(mamba_val, "__len__")) else 0

        lines.append(_row(f"  🔹 last_node  id={last.id}  tokens=[{prev_cum}, {last_cum})"))
        lines.append(_row(f"     len_node_key={last_key_len}  prefix_cum={last_cum}  SSM={'✓' if mr else '✗'}  {last_badge}"))
        lines.append(_row(f"     ⓘ  request's own state — already sent as part of the payload"))

    lines.append(_box_bot())
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# PD transfer logs
# ═══════════════════════════════════════════════════════════════════

def fmt_pd_send(req, page_indices, start_idx, end_idx, last_chunk,
                state_indices, state_types, page_size) -> str:
    """Format PD-SEND transfer log line."""
    state_desc = []
    if last_chunk and state_indices:
        for st, si in zip(state_types, state_indices):
            if si is not None:
                state_desc.append(f"{st.name}={len(si)}rows")
    total_len = len(req.origin_input_ids)
    page_aligned_end = ((end_idx + page_size - 1) // page_size) * page_size
    page_aligned_start = (start_idx // page_size) * page_size
    return (
        f"📤 [PD-SEND] rid={req.rid[:8]}..  "
        f"kv_pages={len(page_indices):>3}  "
        f"tokens=[{start_idx}:{end_idx}]/{total_len}  "
        f"page_aligned=[{page_aligned_start}:{page_aligned_end}]  "
        f"chunk={'LAST' if last_chunk else 'MID '}  "
        f"state=({', '.join(state_desc) if state_desc else 'KV-only'})"
    )


def fmt_pd_recv(req, page_size) -> str:
    """Format PD-RECV transfer log line."""
    kv_len = len(req.origin_input_ids) + len(req.output_ids)
    kv_pages = (
        (req.kv_committed_len + page_size - 1) // page_size
        if req.kv_committed_len
        else 0
    )
    return (
        f"📥 [PD-RECV] rid={req.rid[:8]}..  "
        f"kv_pages={kv_pages:>3}  "
        f"kv_committed={req.kv_committed_len}/{kv_len} tokens  "
        f"bootstrap_room={req.bootstrap_room}"
    )


# ═══════════════════════════════════════════════════════════════════
# Incoming request pretty-printer
# ═══════════════════════════════════════════════════════════════════

def fmt_recv_reqs(recv_reqs: list) -> str:
    """Pretty-print incoming requests — the main entry-point marker in logs."""
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput, TokenizedEmbeddingReqInput

    lines = []
    W = 74

    # ═══════════════════════════════════════════════════════════════
    #  BIG BANNER — request boundary marker
    # ═══════════════════════════════════════════════════════════════
    lines.append("")
    lines.append("")
    lines.append("█" * 80)
    lines.append("█" + " 🚀🚀🚀  INCOMING REQUEST  🚀🚀🚀".center(78) + "█")
    lines.append("█" + f" 📨 {len(recv_reqs)} request(s) received".ljust(78) + "█")
    lines.append("█" * 80)

    for i, req in enumerate(recv_reqs):
        if isinstance(req, TokenizedGenerateReqInput):
            tag = "🟢 GENERATE"
        elif isinstance(req, TokenizedEmbeddingReqInput):
            tag = "🔵 EMBED"
        else:
            tag = f"⚪ {type(req).__name__}"

        rid = getattr(req, "rid", None)
        rid_short = rid[:8] if rid else "?"
        ids = getattr(req, "input_ids", None)
        text = getattr(req, "input_text", None)
        sp = getattr(req, "sampling_params", None)

        # ── per-request section ──
        lines.append("")
        lines.append("╔" + "═" * W + "╗")
        header = f" [{i}] {tag}  rid={rid_short}.."
        lines.append("║" + header.ljust(W) + "║")
        lines.append("╠" + "═" * W + "╣")

        def _row(s):
            return "║  " + s.ljust(W - 2) + "║"

        # ── text ──
        if text is not None:
            if isinstance(text, str):
                preview = text[:300].replace("\n", "\\n")
                if len(text) > 300:
                    preview += f"… (total {len(text)} chars)"
                lines.append(_row(f"📝 text  ({len(text)} chars): {preview}"))
            else:
                lines.append(_row(f"📝 text  (type={type(text).__name__})"))

        # ── token_ids ──
        if ids is not None:
            n = len(ids)
            if n <= 24:
                ids_str = str(list(ids))
            else:
                ids_str = f"{list(ids[:12])} … {list(ids[-12:])}"
            lines.append(_row(f"🔢 ids   ({n:>5} tokens): {ids_str}"))

        # ── sampling params ──
        if sp is not None:
            sp_parts = []
            for attr in ("max_new_tokens", "temperature", "top_p", "top_k",
                         "frequency_penalty", "presence_penalty", "repetition_penalty"):
                v = getattr(sp, attr, None)
                if v is not None:
                    sp_parts.append(f"{attr}={v}")
            lines.append(_row(f"🎲 samp  : {', '.join(sp_parts)}"))

        # ── PD / session / misc ──
        extras = []
        for attr in ("bootstrap_host", "bootstrap_port", "bootstrap_room",
                     "session_id", "lora_id"):
            v = getattr(req, attr, None)
            if v is not None:
                extras.append(f"{attr}={v}")
        if extras:
            lines.append(_row(f"🏷️  extra : {', '.join(extras)}"))

        lines.append("╚" + "═" * W + "╝")

    lines.append("")
    lines.append("█" * 80)
    lines.append("█" + " 🏁 REQUEST LOG END 🏁".center(78) + "█")
    lines.append("█" * 80)
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Request prefix-match logger
# ═══════════════════════════════════════════════════════════════════

_req_history: dict[str, dict] = {}

def log_input_prefix_matches(recv_reqs: list) -> str:
    """For each incoming generate request, compute naive prefix-match length
    against all previously seen requests (by input_ids) and return a log string."""
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

    now = time.monotonic()
    lines = []
    for req in recv_reqs:
        if not isinstance(req, TokenizedGenerateReqInput):
            continue
        rid = req.rid
        ids = req.input_ids
        if ids is None:
            continue

        # Compute best prefix match against history
        best_match_len = 0
        best_match_rid = None
        for hist_rid, hist in _req_history.items():
            hist_ids = hist["input_ids"]
            ml = 0
            limit = min(len(ids), len(hist_ids))
            while ml < limit and ids[ml] == hist_ids[ml]:
                ml += 1
            if ml > best_match_len:
                best_match_len = ml
                best_match_rid = hist_rid

        # Store current request
        _req_history[rid] = {
            "text": req.input_text[:120] if req.input_text else "",
            "input_ids": array('q', ids),
            "len": len(ids),
            "time": now,
        }

        if best_match_rid is not None:
            lines.append(
                f" :: [PREFIX-MATCH] rid={rid[:8]}..  input_len={len(ids)}  "
                f"best_hist_match={best_match_len}/{len(ids)} tokens  "
                f"matched_rid={best_match_rid[:8]}..  "
                f"history_size={len(_req_history)}"
            )
        else:
            lines.append(
                f" :: [PREFIX-MATCH] rid={rid[:8]}..  input_len={len(ids)}  "
                f"best_hist_match=0 (no history yet)  "
                f"history_size={len(_req_history)}"
            )

    return "\n".join(lines)
