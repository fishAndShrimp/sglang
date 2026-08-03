#!/usr/bin/env python3
"""launch_all.py —— 统一管理 Prefill / Decode / Router 三个进程"""

import subprocess
import socket
import time
import os
import sys
import atexit

import port_check
from _zzz_config import MODEL_PATH, SGLANG_DEV_PATH


# ========== 共享配置(敏感值请修改 config.py)==========
TP_SIZE = 1

# ========== GPU 分配(从 TP_SIZE 自动推导)==========
PREFILL_BASE_GPU = 6
DECODE_BASE_GPU = PREFILL_BASE_GPU + TP_SIZE

# ========== 端口分组(按你的编号方案)==========
PREFILL_PORT = 61100
PREFILL_DIST_INIT_PORT = 61101
PREFILL_BOOTSTRAP_PORT = 61102
PREFILL_HCCL_NPU_SOCKET_PORT_RANGE = "61110-61199"

DECODE_PORT = 61200
DECODE_DIST_INIT_PORT = 61201
DECODE_HCCL_NPU_SOCKET_PORT_RANGE = "61210-61299"

STORE_PORT = 61320
ROUTER_PORT = 61900

# 开发时设为 sglang dev 包的路径，不需要则为 None
# SGLANG_DEV_PATH 已从 config.py 导入

MEM_FRACTION_STATIC = 0.8  # Prefill / Decode 显存占用比例

# ========== debugpy 远程调试(None=关闭, 设端口号=开启) ==========
PREFILL_DEBUGPY_PORT = (
    # 61410
    None
)
DECODE_DEBUGPY_PORT = (
    # 61420
    None
)

# ========== 共享环境变量(所有进程共用, 后续可在此注入 PYTHONPATH 等) ==========
shared_env = {
    "ASCEND_MF_STORE_URL": f"tcp://127.0.0.1:{STORE_PORT}",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
}
if SGLANG_DEV_PATH:
    shared_env["PYTHONPATH"] = f"{SGLANG_DEV_PATH}:{os.environ.get('PYTHONPATH', '')}"


port_check.check_ports_free(61000, 61999, "PREFILL DECODE STORE ROUTER")

# ========== 进程管理 ==========
_processes = []


def _cleanup():
    print("\n[Manager] 正在关闭所有子进程...")
    for name, p in _processes:
        if p.poll() is None:
            print(f"  终止 {name} (pid={p.pid})")
            p.terminate()
    time.sleep(3)
    for name, p in _processes:
        if p.poll() is None:
            print(f"  强制杀死 {name} (pid={p.pid})")
            p.kill()


atexit.register(_cleanup)


def launch(name: str, cmd: list, extra_env: dict = None, log_file: str = None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    logf = open(log_file, "w") if log_file else None
    p = subprocess.Popen(
        cmd, env=env, stdout=logf or sys.stdout, stderr=subprocess.STDOUT
    )
    _processes.append((name, p))
    print(f"[Manager] 已启动 {name} (pid={p.pid}), 日志: {log_file or 'stdout'}")
    return p


def wait_for_port(host: str, port: int, name: str, timeout: int = 300):
    print(f"[Manager] 等待 {name} 在端口 {port} 上就绪...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"[Manager] {name} 已就绪 ✅")
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"{name} 在 {timeout} 秒内未能就绪,请检查对应日志文件")


def _debugpy_prefix(port: int | None):
    """port 不为 None 时返回 debugpy 命令行片段，否则返回空列表"""
    if port is None:
        return []
    return ["-m", "debugpy", "--listen", f"0.0.0.0:{port}", "--wait-for-client"]


# ========================================
# 1. 启动 Prefill(卡 PREFILL_BASE_GPU ~ PREFILL_BASE_GPU+TP_SIZE-1)
# ========================================
prefill_cmd = [
    "python3",
    *_debugpy_prefix(PREFILL_DEBUGPY_PORT),
    "-m",
    "sglang.launch_server",
    "--model-path",
    MODEL_PATH,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--tp-size",
    str(TP_SIZE),
    "--base-gpu-id",
    str(PREFILL_BASE_GPU),
    "--dist-init-addr",
    f"127.0.0.1:{PREFILL_DIST_INIT_PORT}",
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--disaggregation-bootstrap-port",
    str(PREFILL_BOOTSTRAP_PORT),
    "--port",
    str(PREFILL_PORT),
    "--mem-fraction-static",
    str(MEM_FRACTION_STATIC),
]
prefill_env = {
    **shared_env,
    "HCCL_NPU_SOCKET_PORT_RANGE": PREFILL_HCCL_NPU_SOCKET_PORT_RANGE,
}
launch("Prefill", prefill_cmd, prefill_env, "prefill.log")
if PREFILL_DEBUGPY_PORT is not None:
    print(f"[Manager] Prefill 已启动，等待 debugpy 客户端连接（端口 {PREFILL_DEBUGPY_PORT}，VSCode 里按 F5 attach）...")
    input("连接完成、且已放好断点后，按回车键继续...")
else:
    wait_for_port("127.0.0.1", PREFILL_PORT, "Prefill")

# ========================================
# 2. 启动 Decode(卡 DECODE_BASE_GPU ~ DECODE_BASE_GPU+TP_SIZE-1)
# ========================================
decode_cmd = [
    "python3",
    *_debugpy_prefix(DECODE_DEBUGPY_PORT),
    "-m",
    "sglang.launch_server",
    "--model-path",
    MODEL_PATH,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--tp-size",
    str(TP_SIZE),
    "--base-gpu-id",
    str(DECODE_BASE_GPU),
    "--dist-init-addr",
    f"127.0.0.1:{DECODE_DIST_INIT_PORT}",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--port",
    str(DECODE_PORT),
    "--mem-fraction-static",
    str(MEM_FRACTION_STATIC),
    "--disaggregation-decode-enable-radix-cache",
]
decode_env = {
    **shared_env,
    "HCCL_NPU_SOCKET_PORT_RANGE": DECODE_HCCL_NPU_SOCKET_PORT_RANGE,
    "STREAMS_PER_DEVICE": "32",
}
launch("Decode", decode_cmd, decode_env, "decode.log")
if DECODE_DEBUGPY_PORT is not None:
    print(f"[Manager] Decode 已启动，等待 debugpy 客户端连接（端口 {DECODE_DEBUGPY_PORT}，VSCode 里按 F5 attach）...")
    input("连接完成、且已放好断点后，按回车键继续...")
else:
    wait_for_port("127.0.0.1", DECODE_PORT, "Decode")

# ========================================
# 3. 启动 Router
# ========================================
router_cmd = [
    "python3",
    "-m",
    "sglang_router.launch_router",
    "--pd-disaggregation",
    "--prefill",
    f"http://127.0.0.1:{PREFILL_PORT}",
    str(PREFILL_BOOTSTRAP_PORT),
    "--decode",
    f"http://127.0.0.1:{DECODE_PORT}",
    "--port",
    str(ROUTER_PORT),
    "--disable-health-check",
]
launch("Router", router_cmd, None, "router.log")
wait_for_port("127.0.0.1", ROUTER_PORT, "Router")

print(f"\n[Manager] 🎉 全部就绪!可通过 http://127.0.0.1:{ROUTER_PORT} 访问")
print("[Manager] 按 Ctrl+C 可一键关闭所有进程\n")

# ========================================
# 4. 保持主进程存活,监控子进程状态
# ========================================
try:
    while True:
        for name, p in _processes:
            if p.poll() is not None:
                print(
                    f"[Manager] ⚠️ {name} 已异常退出(exit code={p.returncode}),请查看对应日志"
                )
                sys.exit(1)
        time.sleep(5)
except KeyboardInterrupt:
    print("\n[Manager] 收到 Ctrl+C,准备退出...")
