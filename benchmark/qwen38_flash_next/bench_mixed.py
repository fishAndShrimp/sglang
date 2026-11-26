"""Reproduce the two frozen A032 rounds on a running TP8 server."""

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from statistics import mean

from unpack_mixed import main as unpack


def main():
    unpack()
    dataset = Path(__file__).resolve().parent / "serving_benchmark_v1"
    sys.path.insert(0, str(dataset))
    from bench import prepare

    base_url = f"http://127.0.0.1:{os.environ.get('PORT', '30000')}/v1"
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    models = urllib.request.Request(base_url + "/models", headers=headers)
    with client.open(models, timeout=30) as response:
        model = json.load(response)["data"][0]["id"]

    # Warm up with the longest D032 prompt before timing either A032 round.
    _, _, payload = max(prepare(dataset, "D032", model, False), key=lambda row: row[1])
    body = json.loads(payload)
    body.update(stream=False, max_tokens=32)
    body.pop("stream_options")
    warmup = urllib.request.Request(
        base_url + "/chat/completions", data=json.dumps(body).encode(), headers=headers
    )
    with client.open(warmup, timeout=300) as response:
        if json.load(response)["usage"]["completion_tokens"] != 32:
            raise SystemExit("Warm-up did not return 32 tokens")

    results = Path(tempfile.mkdtemp(prefix="qwen38-bench-"))
    print(f"Benchmark results: {results}", flush=True)
    rounds = []
    for number in (1, 2):
        output = results / f"round-{number}.json"
        subprocess.run(
            [
                sys.executable, str(dataset / "bench.py"),
                "--base-url", base_url, "--model", model,
                "--batch", "A032", "--spec-details", "--out", str(output),
            ],
            check=True,
        )
        rounds.append(json.loads(output.read_text()))

    metrics = (
        ("E2E (s)", "duration_s"),
        ("TTFT (s)", "mean_ttft_s"),
        ("TPOT (ms)", "mean_tpot_ms"),
        ("Output tok/s", "output_tps"),
        ("MTP acceptance length", "mtp_accept_length"),
    )
    for label, field in metrics:
        if all(field in result for result in rounds):
            print(f"{label} (two-round mean): {mean(r[field] for r in rounds):.3f}")


if __name__ == "__main__":
    main()
