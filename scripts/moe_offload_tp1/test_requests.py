"""Small HTTP checks, not sglang.bench_serving or a sustained-load benchmark."""

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

# 可在这里修改请求并发与重复次数；并发不等于设备每一步的实际 batch。
BATCHES, REPEATS = [1, 4, 8], 2

# 2026-10-29 旧 BS1/DeepEP 实验：前三题的错误已在 prefill 首 token 出现，
# 关闭快速路径仍复现；这定位了首次错误输出，不证明根因是 prefill 算子。
# 保留空格、÷、× 原样；英文题也曾错，无空格题作对照。不是当前版本的预期答案。
MATH = [
    ("计算(8+7)*3，只输出数字。", "45", "TP1 W4A8 修复前答24"),
    ("先把8与7相加，再把结果乘以3。只输出最终数字。", "45", "TP1 W4A8 修复前答21"),
    ("计算12*(15-7)，只输出数字。", "96", "复合运算"),
    ("计算125乘以8，只输出数字。", "1000", "乘法"),
    ("每袋有6个苹果，买了4袋，吃掉5个，还剩多少个？只输出数字。", "19", "简单应用题"),
    ("计算 17 加 25，只输出数字。", "42", "历史答32；首token已错"),
    ("计算 144 ÷ 12，只输出数字。", "12", "历史答11；首token已错"),
    ("计算 (8 + 7) × 3，只输出数字。", "45", "历史答11；首token已错"),
    ("What is 17 + 25? Output only the number.", "42", "历史答32"),
    ("计算17加25，只输出数字。", "42", "无空格对照，历史答42"),
]

# 与此前 diverse8 小规模对照相同的短自然语言提示，不使用随机 token。
PROMPTS = [
    "Explain how to organize a small software project. Give five practical steps and explain each step with a concrete example.",
    "Write a Python implementation of binary search for a sorted list. Include a docstring, explain the loop invariant, and provide three example tests.",
    "A shop buys 120 notebooks for 3 dollars each, sells 75 at 5 dollars each and the rest at 4 dollars each. Calculate revenue, cost, profit, and profit margin. Explain every step.",
    "Implement a small LRU cache in Python with get and put methods. Explain why the operations are constant time and show how an item is evicted.",
    "A database query filters orders by customer_id and sorts by created_at. Explain how to choose an index, how to inspect the query plan, and when the index may not help.",
    "Explain how TCP handles packet loss and out-of-order delivery. Use a concrete example with three packets and distinguish retransmission from application retries.",
    "请向刚学习计算机的同学解释缓存为什么能加速程序。用一个读书查资料的例子说明命中、未命中和淘汰，并解释为什么缓存不是越大就一定越快。",
    "A team has two engineers and three weeks to deliver a small expense-tracking application. Propose a realistic plan covering data storage, API design, tests, and deployment, and identify the main risks.",
]


def generate(base, model, prompt, bench, rank):
    body = dict(
        model=model,
        messages=[dict(role="user", content=prompt)],
        temperature=0,
        seed=1024,
        max_tokens=128,
        ignore_eos=bench,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"thinking": False},
        routed_dp_rank=rank,
    )
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.perf_counter()
    first = last = usage = None
    text, done = "", False
    with opener.open(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if content:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last, text = now, text + content
    if not done or not usage or first is None or not text.strip():
        raise RuntimeError("Incomplete stream, empty output or missing usage")
    tokens = usage["completion_tokens"]
    if bench and tokens != 128:
        raise RuntimeError(f"Expected 128 output tokens, received {tokens}")
    return dict(
        request=body,
        output=text,
        usage=usage,
        ttft_ms=(first - start) * 1000,
        # Historical client metric: first-to-last content chunk / (tokens - 1).
        # SSE chunks may contain several tokens, especially with MTP.
        tpot_ms=(last - first) * 1000 / (tokens - 1) if tokens > 1 else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true", help="默认检查历史错题")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + "/get_server_info", timeout=15) as response:
        info = json.load(response)
    model, dp_size = info["served_model_name"], info["attn_dp_size"]
    output = Path(f"/tmp/moe-check-{time.time_ns()}.jsonl")
    failed = False
    with output.open("x", encoding="utf-8") as log:

        def save(row):
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()

        if not args.bench:
            for prompt, expected, note in MATH:
                row = generate(base, model, prompt, False, 0)
                matched = row["output"].strip().rstrip("。.!") == expected
                failed |= not matched
                row.update(expected=expected, exact_match=matched, history=note)
                save(row)
                label = "PASS" if matched else "CHECK"
                print(f"{label}: {prompt} -> {row['output']!r}", flush=True)
        else:
            for batch in BATCHES:
                for repeat in range(REPEATS):
                    barrier = threading.Barrier(batch)

                    def run(i):
                        barrier.wait(timeout=30)
                        index = i + repeat
                        prompt = PROMPTS[index % len(PROMPTS)]
                        return generate(base, model, prompt, True, index % dp_size)

                    start = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=batch) as pool:
                        rows = list(pool.map(run, range(batch)))
                    elapsed = time.perf_counter() - start
                    for row in rows:
                        row.update(concurrency=batch, repeat=repeat)
                        save(row)
                    summary = dict(concurrency=batch, repeat=repeat)
                    summary["mean_ttft_ms"] = mean(r["ttft_ms"] for r in rows)
                    summary["mean_tpot_ms"] = mean(r["tpot_ms"] for r in rows)
                    summary["output_tokens_per_s"] = batch * 128 / elapsed
                    save(summary)
                    print(json.dumps(summary), flush=True)
    print(f"Saved {output}")
    if failed:
        raise SystemExit(
            "Numeric-only answer mismatch: inspect saved output before diagnosing accuracy."
        )


if __name__ == "__main__":
    main()
