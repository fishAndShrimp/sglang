"""Join P-side timing events by process and room; write one CSV row per sample."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from chunk_summary import DETAIL_FIELDS, MAIN_FIELDS, make_chunk_row

DURATIONS = (
    "prefill_ms",
    "submit_ms",
    "queue_ms",
    "send_ms",
    "post_send_ms",
    "detect_ms",
    "cleanup_ms",
    "metric_ms",
    "metric_start_after_enqueue_ms",
)


def make_row(worker, finish):
    if finish.get("chunks") != 1 or finish.get("chunked"):
        raise ValueError(
            "requires one send chunk: overlapping/chunked sends are not a five-stage split "
            f"(chunks={finish.get('chunks')}, chunked={finish.get('chunked')}); "
            "use --metric-only to include this request in the original metric"
        )
    if worker.get("attempts") != 1:
        raise ValueError("worker retried: queue/send intervals are not a single pass")
    fields = (
        finish.get("forward"),
        finish.get("prefill"),
        worker.get("enqueue"),
        worker.get("dequeue"),
        worker.get("ready"),
        finish.get("poll"),
        finish.get("done"),
        finish.get("transfer"),
    )
    if any(
        not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
        for x in fields
    ):
        raise ValueError("missing or invalid timestamps")
    forward, prefill, enqueue, dequeue, ready, poll, done, transfer = fields
    if not (forward <= prefill <= enqueue <= dequeue <= ready <= poll <= done):
        raise ValueError("overlapping or out-of-order boundaries")
    if not prefill <= transfer <= done:
        raise ValueError("metric interval is outside the expected boundaries")

    def ms(a, b):
        return (a - b) * 1000

    return {
        "pid": finish["pid"],
        "room": finish["room"],
        "rid": finish.get("rid", ""),
        "prefill_ms": ms(prefill, forward),
        "submit_ms": ms(enqueue, prefill),
        "queue_ms": ms(dequeue, enqueue),
        "send_ms": ms(ready, dequeue),
        "post_send_ms": ms(done, ready),
        "detect_ms": ms(poll, ready),
        "cleanup_ms": ms(done, poll),
        "metric_ms": ms(done, transfer),
        "metric_start_after_enqueue_ms": ms(transfer, enqueue),
    }


def make_metric_row(finish):
    """Reproduce Mooncake's metric without requiring serial transfer stages."""
    transfer, done = finish.get("transfer"), finish.get("done")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
        for value in (transfer, done)
    ):
        raise ValueError("missing or invalid metric timestamps")
    if done <= transfer:
        raise ValueError("non-positive metric interval")
    return {
        "pid": finish["pid"],
        "room": finish["room"],
        "rid": finish.get("rid", ""),
        "metric_ms": (done - transfer) * 1000,
    }


def read_rows(paths, exclude_rids, metric_only=False):
    records = defaultdict(lambda: defaultdict(list))
    invalid = 0
    for path in paths:
        with path.open(errors="replace") as stream:
            for line in stream:
                if "KV_TIMING " not in line:
                    continue
                payload = line.split("KV_TIMING ", 1)[1]
                try:
                    event, _ = json.JSONDecoder().raw_decode(payload)
                    kind = event["event"]
                    if kind not in ("worker", "finish", "chunk"):
                        raise ValueError("unknown event")
                    records[(event["pid"], event["room"])][kind].append(event)
                except (ValueError, KeyError, TypeError):
                    invalid += 1
    if invalid:
        print(f"Ignored {invalid} invalid timing lines", file=sys.stderr)
    rows = []
    for key, group in records.items():
        workers, finishes = group["worker"], group["finish"]
        is_v2 = len(finishes) == 1 and finishes[0].get("version") == 2
        if len(finishes) != 1 or (not metric_only and not is_v2 and len(workers) != 1):
            print(
                f"Skipped pid/room={key}: missing or repeated events "
                f"(worker={len(workers)}, finish={len(finishes)})",
                file=sys.stderr,
            )
            continue
        if finishes[0].get("rid") in exclude_rids:
            continue
        try:
            if is_v2 and not metric_only:
                finish = finishes[0]
                row = make_metric_row(finish)
                row.update(
                    {
                        key: finish.get(key)
                        for key in (
                            "tp_rank",
                            "dp_rank",
                            "cp_rank",
                            "pp_rank",
                            "chunks",
                        )
                    }
                )
                row["detail_status"] = "complete"
                try:
                    row.update(make_chunk_row(group["chunk"], finish))
                except (ValueError, KeyError, TypeError) as error:
                    # Keep the metric sample even when per-chunk logs are
                    # incomplete. Do not silently change its population again.
                    row["detail_status"] = str(error)
                    print(
                        f"Partial pid/room={key}: {error}; metric retained",
                        file=sys.stderr,
                    )
                rows.append(row)
                continue
            rows.append(
                make_metric_row(finishes[0])
                if metric_only
                else make_row(workers[0], finishes[0])
            )
        except ValueError as error:
            print(f"Skipped pid/room={key}: {error}", file=sys.stderr)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Also show per-phase API timings and the disjoint metric-window breakdown",
    )
    parser.add_argument(
        "--metric-only",
        action="store_true",
        help="Use finish events to reconcile the original metric, including multi-chunk requests",
    )
    parser.add_argument(
        "--exclude-rid",
        action="append",
        default=[],
        help="Exclude a known warmup request ID (repeatable)",
    )
    args = parser.parse_args()
    if len({path.resolve() for path in args.logs}) != len(args.logs):
        parser.error("Pass each log file only once")
    rows = read_rows(args.logs, set(args.exclude_rid), metric_only=args.metric_only)
    if not rows:
        raise SystemExit("No valid samples found for the selected mode")
    is_v2 = any("detail_status" in row for row in rows)
    if is_v2 and any("detail_status" not in row for row in rows):
        raise SystemExit(
            "Do not mix version 1 and version 2 logs in a detailed summary"
        )
    durations = (
        ("metric_ms",)
        if args.metric_only
        else MAIN_FIELDS + (DETAIL_FIELDS if args.details else ())
        if is_v2
        else DURATIONS
    )
    csv_durations = MAIN_FIELDS + DETAIL_FIELDS if is_v2 else durations
    metadata = ("pid", "room", "rid") + (
        (
            "tp_rank",
            "dp_rank",
            "cp_rank",
            "pp_rank",
            "chunks",
            "attempts",
            "mooncake_calls",
            "detail_status",
        )
        if is_v2
        else ()
    )
    print(
        "mode=metric-only (valid finish events; includes multi-chunk requests)"
        if args.metric_only
        else "mode=per-chunk-v2 (unions of intervals; see n for each field)"
        if is_v2
        else "mode=five-stage-v1 (single-chunk subset; use --metric-only to reconcile /metrics)",
        file=sys.stderr,
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=metadata + csv_durations)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: round(value, 6) if key in csv_durations else value
                for key, value in row.items()
            }
        )
    # Report separately by process. Multiple TP ranks can observe one request.
    by_pid = defaultdict(list)
    for row in rows:
        by_pid[row["pid"]].append(row)
    for pid, samples in sorted(by_pid.items()):
        print(
            f"pid={pid} samples={len(samples)} (rank-local observations)"
            + (
                f" detailed={sum(s.get('detail_status') == 'complete' for s in samples)}"
                f" tp_rank={samples[0].get('tp_rank')}"
                if is_v2
                else ""
            ),
            file=sys.stderr,
        )
        if is_v2:
            print(
                f"  chunks={sum(s.get('chunks') or 0 for s in samples)} "
                f"attempts={sum(s.get('attempts', 0) for s in samples)} "
                f"mooncake_calls={sum(s.get('mooncake_calls', 0) for s in samples)}",
                file=sys.stderr,
            )
        for field in durations:
            values = sorted(sample[field] for sample in samples if field in sample)
            if not values:
                print(f"  {field}: n=0 (details unavailable)", file=sys.stderr)
                continue
            p95 = values[math.ceil(0.95 * len(values)) - 1]
            print(
                f"  {field}: mean={statistics.mean(values):.3f} "
                f"p50={statistics.median(values):.3f} p95={p95:.3f} "
                f"max={values[-1]:.3f} sum={math.fsum(values):.3f} n={len(values)}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
