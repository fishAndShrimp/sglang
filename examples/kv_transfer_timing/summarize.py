"""Join P-side timing events by process and room; write one CSV row per sample."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

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
            "requires one send chunk: overlapping/chunked sends are not a five-stage split"
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


def read_rows(paths, exclude_rids):
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
                    if kind not in ("worker", "finish"):
                        raise ValueError("unknown event")
                    records[(event["pid"], event["room"])][kind].append(event)
                except (ValueError, KeyError, TypeError):
                    invalid += 1
    if invalid:
        print(f"Ignored {invalid} invalid timing lines", file=sys.stderr)
    rows = []
    for key, group in records.items():
        workers, finishes = group["worker"], group["finish"]
        if len(workers) != 1 or len(finishes) != 1:
            print(
                f"Skipped pid/room={key}: missing or repeated events "
                f"(worker={len(workers)}, finish={len(finishes)})",
                file=sys.stderr,
            )
            continue
        if finishes[0].get("rid") in exclude_rids:
            continue
        try:
            rows.append(make_row(workers[0], finishes[0]))
        except ValueError as error:
            print(f"Skipped pid/room={key}: {error}", file=sys.stderr)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument(
        "--exclude-rid",
        action="append",
        default=[],
        help="Exclude a known warmup request ID (repeatable)",
    )
    args = parser.parse_args()
    if len({path.resolve() for path in args.logs}) != len(args.logs):
        parser.error("Pass each log file only once")
    rows = read_rows(args.logs, set(args.exclude_rid))
    if not rows:
        raise SystemExit("No complete, unchunked request samples found")
    writer = csv.DictWriter(sys.stdout, fieldnames=("pid", "room", "rid") + DURATIONS)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: round(value, 6) if key in DURATIONS else value
                for key, value in row.items()
            }
        )
    # Report separately by process. Multiple TP ranks can observe one request.
    by_pid = defaultdict(list)
    for row in rows:
        by_pid[row["pid"]].append(row)
    for pid, samples in sorted(by_pid.items()):
        print(
            f"pid={pid} samples={len(samples)} (rank-local observations)",
            file=sys.stderr,
        )
        for field in DURATIONS:
            values = sorted(sample[field] for sample in samples)
            p95 = values[math.ceil(0.95 * len(values)) - 1]
            print(
                f"  {field}: mean={statistics.mean(values):.3f} "
                f"p50={statistics.median(values):.3f} p95={p95:.3f} "
                f"max={values[-1]:.3f}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
