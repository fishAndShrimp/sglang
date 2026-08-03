"""
port_check.py
职责单一:在启动服务前,预检查指定端口区间是否已被占用。
"""

import subprocess
import re
import sys


def check_ports_free(start_port: int, end_port: int, label: str):
    """
    预检查指定端口区间内是否已有进程在监听。
    一旦发现冲突,直接终止整个脚本(sys.exit(1)),
    避免浪费时间等到 Decode 启动到一半才报错。
    """
    try:
        result = subprocess.run(
            ["netstat", "-tuln"],
            capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        # 部分精简容器镜像没装 net-tools,退化用 ss 命令(iproute2 自带)
        try:
            result = subprocess.run(
                ["ss", "-tuln"],
                capture_output=True, text=True, check=True
            )
        except FileNotFoundError:
            print(f"[WARN] 找不到 netstat 或 ss 命令,跳过 {label} 端口预检查")
            return

    occupied = []
    for line in result.stdout.splitlines():
        for match in re.finditer(r":(\d+)\s", line):
            port = int(match.group(1))
            if start_port <= port <= end_port:
                occupied.append((port, line.strip()))

    if occupied:
        print(f"\n[FATAL] {label} 预检查失败!端口区间 {start_port}-{end_port} 内已有端口被占用:")
        for port, line in sorted(set(occupied)):
            print(f"   - 端口 {port}: {line}")
        sys.exit(1)

    print(f"[OK] {label} 端口区间 {start_port}-{end_port} 空闲,可以启动")


def check_ports_free_by_regex(pattern: str, label: str):
    """按自定义正则表达式检查端口占用(备用方案,适合区间不规则的情况)"""
    result = subprocess.run(["netstat", "-tuln"], capture_output=True, text=True, check=True)
    matched = [line for line in result.stdout.splitlines() if re.search(pattern, line)]
    if matched:
        print(f"[FATAL] {label} 预检查失败,匹配到以下占用记录:")
        for line in matched:
            print(f"   {line.strip()}")
        sys.exit(1)
    print(f"[OK] {label} 未发现端口占用")
