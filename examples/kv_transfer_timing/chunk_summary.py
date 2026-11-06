"""Aggregate complete per-chunk records without double-counting overlapping calls."""

import math
from collections import defaultdict

MAIN_FIELDS = (
    "prefill_ms",
    "queue_ms",
    "wait_ms",
    "send_ms",
    "mooncake_ms",
    "send_other_ms",
    "worker_other_ms",
    "notify_ms",
    "detect_ms",
    "cleanup_ms",
    "metric_ms",
)
DETAIL_FIELDS = (
    "kv_ms",
    "state_ms",
    "aux_ms",
    "mooncake_kv_ms",
    "mooncake_state_ms",
    "mooncake_aux_ms",
    "mooncake_call_sum_ms",
    "worker_ms",
    "post_send_ms",
    "last_submit_offset_ms",
    "send_overlap_prefill_ms",
    "metric_mooncake_ms",
    "metric_send_other_ms",
    "metric_worker_other_ms",
    "metric_queue_only_ms",
    "metric_other_ms",
)


def timestamp(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("missing or invalid timestamp")
    return value


def intervals_union(intervals, lower=None, upper=None):
    clipped = []
    for start, end in intervals:
        if lower is not None:
            start = max(start, lower)
        if upper is not None:
            end = min(end, upper)
        if end > start:
            clipped.append((start, end))
    merged = []
    for start, end in sorted(clipped):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals, lower=None, upper=None):
    return math.fsum(
        end - start for start, end in intervals_union(intervals, lower, upper)
    )


def make_chunk_row(chunks, finish):
    expected = finish.get("chunks")
    if type(expected) is not int or expected < 1:
        raise ValueError("no measured sender chunks")
    by_chunk = defaultdict(list)
    for chunk in chunks:
        if type(chunk.get("chunk")) is not int:
            raise ValueError("missing chunk ID")
        by_chunk[chunk["chunk"]].append(chunk)
    if set(by_chunk) != set(range(1, expected + 1)):
        raise ValueError(
            f"incomplete chunks: expected={expected}, found={len(by_chunk)}"
        )

    forward = timestamp(finish.get("forward"))
    prefill = timestamp(finish.get("prefill"))
    transfer = timestamp(finish.get("transfer"))
    poll = timestamp(finish.get("poll"))
    done = timestamp(finish.get("done"))
    if not forward <= prefill <= transfer <= poll <= done:
        raise ValueError("out-of-order scheduler boundaries")

    queues, workers, initial_enqueues = [], [], []
    phases, calls = defaultdict(list), defaultdict(list)
    call_sum, call_count = 0.0, 0
    for chunk_id, attempts in by_chunk.items():
        if any(type(a.get("attempt")) is not int for a in attempts):
            raise ValueError("missing attempt number")
        attempts.sort(key=lambda a: a["attempt"])
        if [a["attempt"] for a in attempts] != list(range(1, len(attempts) + 1)):
            raise ValueError("missing or duplicated chunk attempts")
        if [a.get("outcome") for a in attempts] != ["deferred"] * (
            len(attempts) - 1
        ) + ["complete"]:
            raise ValueError("chunk did not complete successfully")
        initial_enqueues.append(timestamp(attempts[0].get("enqueue")))
        for attempt in attempts:
            if attempt.get("last") is not (chunk_id == expected):
                raise ValueError("inconsistent final chunk marker")
            enqueue = timestamp(attempt.get("enqueue"))
            dequeue = timestamp(attempt.get("dequeue"))
            end = timestamp(attempt.get("work_end"))
            cleanup = timestamp(attempt.get("cleanup_end"))
            if not enqueue <= dequeue <= end <= cleanup or end > poll:
                raise ValueError("out-of-order worker boundaries")
            queues.append((enqueue, dequeue))
            workers.append((dequeue, end))
            local_phases = defaultdict(list)
            for name, start, stop in attempt.get("phases", []):
                start, stop = timestamp(start), timestamp(stop)
                if name not in ("wait", "kv", "state", "aux", "notify"):
                    raise ValueError("unknown worker phase")
                if not dequeue <= start <= stop <= end:
                    raise ValueError("phase outside worker interval")
                local_phases[name].append((start, stop))
                phases[name].append((start, stop))
            for name, start, stop, size, result in attempt.get("calls", []):
                start, stop = timestamp(start), timestamp(stop)
                if (
                    name not in ("kv", "state", "aux")
                    or not dequeue <= start <= stop <= end
                ):
                    raise ValueError("API call outside send phase")
                if not any(a <= start <= stop <= b for a, b in local_phases[name]):
                    raise ValueError("API call missing its enclosing phase")
                if result != 0 or not isinstance(size, int) or size < 0:
                    raise ValueError("failed or malformed API call")
                calls[name].append((start, stop))
                call_sum += stop - start
                call_count += 1

    sends = phases["kv"] + phases["state"] + phases["aux"]
    api = calls["kv"] + calls["state"] + calls["aux"]
    if not sends:
        raise ValueError("no send phases recorded")
    last_send_end = max(end for _, end in sends)
    send_s, api_s, worker_s = duration(sends), duration(api), duration(workers)
    covered_worker_s = duration(sends + phases["wait"] + phases["notify"])
    row = {
        "chunks": expected,
        "attempts": len(chunks),
        "mooncake_calls": call_count,
        "prefill_ms": (prefill - forward) * 1000,
        "queue_ms": duration(queues) * 1000,
        "wait_ms": duration(phases["wait"]) * 1000,
        "send_ms": send_s * 1000,
        "mooncake_ms": api_s * 1000,
        "send_other_ms": max(0.0, send_s - api_s) * 1000,
        "worker_other_ms": max(0.0, worker_s - covered_worker_s) * 1000,
        "notify_ms": duration(phases["notify"]) * 1000,
        "detect_ms": (poll - last_send_end) * 1000,
        "cleanup_ms": (done - poll) * 1000,
        "post_send_ms": (done - last_send_end) * 1000,
        "worker_ms": worker_s * 1000,
        "mooncake_call_sum_ms": call_sum * 1000,
        "last_submit_offset_ms": (max(initial_enqueues) - prefill) * 1000,
        "send_overlap_prefill_ms": duration(sends, forward, prefill) * 1000,
    }
    for name in ("kv", "state", "aux"):
        row[f"{name}_ms"] = duration(phases[name]) * 1000
        row[f"mooncake_{name}_ms"] = duration(calls[name]) * 1000

    # A disjoint partition of the *existing metric window*. Earlier chunk
    # transfers can overlap prefill and lie before this window. Queue waiting
    # for one chunk can also overlap processing another chunk of the request.
    api_window = duration(api, transfer, done)
    send_window = duration(api + sends, transfer, done)
    work_window = duration(api + sends + workers, transfer, done)
    queue_window = duration(api + sends + workers + queues, transfer, done)
    row.update(
        {
            "metric_ms": (done - transfer) * 1000,
            "metric_mooncake_ms": api_window * 1000,
            "metric_send_other_ms": max(0.0, send_window - api_window) * 1000,
            "metric_worker_other_ms": max(0.0, work_window - send_window) * 1000,
            "metric_queue_only_ms": max(0.0, queue_window - work_window) * 1000,
            "metric_other_ms": max(0.0, done - transfer - queue_window) * 1000,
        }
    )
    return row
