#!/usr/bin/env python3
"""向 SGLang Router 发送长文本补全请求，流式输出结果

用法:
    python send_prompt.py                  # 默认发送序列 1
    python send_prompt.py --seq 2          # 发送序列 2
    python send_prompt.py --seq 3          # 发送序列 3
    python send_prompt.py --list           # 列出所有可用序列
"""

import argparse
import requests
import json
import sys
import time
import io

from send_prompt_seqs import get_sequence, list_sequences

ROUTER_URL = "http://127.0.0.1:61900"

# -- 强制 stdout 用 UTF-8，避免中文乱码 --
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="向 SGLang Router 发送长文本补全请求")
    parser.add_argument("--seq", type=int, default=1, help="选择序列编号 (1/2/3)，默认 1")
    parser.add_argument("--list", action="store_true", help="列出所有可用序列")
    args = parser.parse_args()

    if args.list:
        seqs = list_sequences()
        print("可用的长文本序列:")
        print("-" * 50)
        for seq_id, info in seqs.items():
            print(f"  [{seq_id}] {info['name']}  ({info['len']} 字符)")
        return

    PROMPT = get_sequence(args.seq)

    url = f"{ROUTER_URL}/v1/chat/completions"

    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "max_tokens": 400,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,
    }

    headers = {"Content-Type": "application/json"}

    print("=" * 60)
    print(f"📤 发送序列 [{args.seq}] ({len(PROMPT)} 字符) → {url}")
    print("=" * 60)

    t0 = time.time()
    first_token_time = None
    total_chunks = 0
    full_text = ""

    try:
        # 用 iter_content 逐字节收，手动按行切割，避免 UTF-8 多字节字符被截断
        resp = requests.post(
            url, json=payload, headers=headers, stream=True, timeout=300
        )
        resp.raise_for_status()

        print("\n📥 模型续写:\n")
        print("─" * 50)

        buf = b""
        for chunk in resp.iter_content(chunk_size=1):
            if not chunk:
                continue
            buf += chunk
            if buf.endswith(b"\n"):
                line = buf.decode("utf-8", errors="replace").rstrip("\r\n")
                buf = b""
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time() - t0
                                total_chunks += 1
                                full_text += content
                                print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass

        elapsed = time.time() - t0
        print("\n")
        print("─" * 50)
        print(f"\n✅ 完成！")
        if first_token_time is not None:
            print(f"   首 token 延迟:  {first_token_time:.2f}s")
        print(f"   总耗时:         {elapsed:.2f}s")
        print(f"   输出字数:       {len(full_text)}")
        if total_chunks > 0 and first_token_time is not None:
            gen_time = max(elapsed - first_token_time, 0.001)
            print(f"   吞吐:           {len(full_text) / gen_time:.1f} 字符/s (不含首token)")

    except requests.exceptions.ConnectionError:
        print(f"\n❌ 无法连接到 Router ({ROUTER_URL})")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"\n❌ 请求超时")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
