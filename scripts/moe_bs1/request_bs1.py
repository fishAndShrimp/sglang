"""服务 ready 后运行：python3 scripts/moe_bs1/request_bs1.py；只发送一次请求。"""
import json
import os
import urllib.request

# 修改端口和问题即可；端口要与 serve_tp4_bs1.sh 一致。
PORT = int(os.environ.get("SERVER_PORT", "30000"))
PROMPT = "Explain how to organize a small software project. Give five practical steps."

body = {
    "model": "dsv4f-bs1",
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0,
    "max_tokens": 128,
    "stream": False,
    "chat_template_kwargs": {"thinking": False},
}
request = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=600) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
