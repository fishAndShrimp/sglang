#!/usr/bin/env bash
# DeepSeek V4 Flash W8A8，TP4 / Attention DP4 / EP4，单请求 MoE offload。
# 内置 HTTP 预热会同时提交多个 DP 输入，违反全局 BS=1，因此加 --skip-server-warmup。
# NPU graph 的 warmup/capture 仍保留；ready 后用 request_bs1.py 发一个请求即可。
# 只支持串行请求；max-running-requests=4 是图容量设置，不代表支持并发。
# 前提：已安装匹配的 CANN、torch_npu、sgl-kernel-npu 和 MemFabric（含 A3 构建源码）。
set -eo pipefail

# ---------- 1. 修改这里：权重路径、四个空闲 die、端口 ----------
MODEL_PATH=${MODEL_PATH:-/path/to/DeepSeek-V4-Flash-W8A8}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
SERVER_PORT=${SERVER_PORT:-30000}
DIST_PORT=${DIST_PORT:-29500}
RUNTIME_DIR=${RUNTIME_DIR:-/tmp/sglang-bs1-runtime}  # 本机编译缓存；换依赖版本后换目录
LOG_DIR=${LOG_DIR:-/tmp/sglang-bs1-$(date +%Y%m%d_%H%M%S)}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
mkdir -p "$RUNTIME_DIR/python" "$RUNTIME_DIR/native_acl" "$LOG_DIR"
exec > >(tee "$LOG_DIR/server.log") 2>&1

# ---------- 2. 昇腾环境：安装位置不同时修改下面四行 ----------
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

# 首次在本机目录构建搬运扩展，不改系统 MemFabric；后续启动复用。
python3 - "$RUNTIME_DIR/python" <<'PY'
import importlib.util, shutil, sys
from pathlib import Path
source = Path(importlib.util.find_spec('memfabric_hybrid').origin).parent
target = Path(sys.argv[1]) / 'memfabric_hybrid'
if not target.exists():
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
PY
export PYTHONPATH="$SGLANG_ROOT/python:$RUNTIME_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$RUNTIME_DIR/python/memfabric_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=4
python3 - "$RUNTIME_DIR/python/memfabric_hybrid/lib" <<'PY'
import sys
from memfabric_hybrid._acc_offload_build import ensure_acc_offload_ext
assert ensure_acc_offload_ext('A3', install_dir=sys.argv[1])
PY
python3 -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$RUNTIME_DIR/native_acl"

# 沿用仓库缓存配置，仅改为本次已验证的容量。host 预算为每 rank 64 GiB。
python3 - "$SGLANG_ROOT/scripts" "$RUNTIME_DIR" "$LOG_DIR" <<'PY'
import json, sys
from pathlib import Path
scripts, runtime, logs = (Path(p).resolve() for p in sys.argv[1:])
config = json.loads((scripts / 'moe_offload_a3.json').read_text())
config.update(cache_ratio=0.28125, execution_slots=8, host_pool_gib=64,
              max_dispatch_rows=8192, batch_log_interval=0,
              offline_stats_path=str(scripts / 'moe_offload_a3_profile.json'),
              native_extension_directory=str(runtime / 'native_acl'))
(logs / 'offload.json').write_text(json.dumps(config, indent=2) + '\n')
PY

# ---------- 3. 固定已验证的算子与 BS=1 快速路径 ----------
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ASCEND_LAUNCH_BLOCKING
unset SGLANG_ZBAL_BOOTSTRAP_URL ZBAL_ENABLE_GRAPH ZBAL_NPU_ALLOC_CONF
unset FORCE_DRAFT_MODEL_NON_QUANT DEEP_NORMAL_MODE_USE_INT8_QUANT
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True SGLANG_SET_CPU_AFFINITY=1
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=300 ACL_DEVICE_SYNC_TIMEOUT=300
export STREAMS_PER_DEVICE=32 INF_NAN_MODE_FORCE_DISABLE=1 TASK_QUEUE_ENABLE=1
export SGLANG_ZBAL_LOCAL_MEM_SIZE=0 SGLANG_PROFILE_V2=0
export SGLANG_DSV4_FP4_EXPERTS=False SGLANG_DSV4_FP4_DEQUANT=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0 SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False SGLANG_OPT_USE_TILELANG_MHC_POST=False
export DEEP_USE_MODE=default MOE_ENABLE_TOPK_NEG_ONE=1 MOE_EXPERT_TOKEN_NUMS_TYPE=1
export DEEPEP_HCCL_BUFFSIZE=1024 DEEPEP_HYBRID_DEPLOYMENT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0
export SGLANG_ENABLE_SPEC_V2=0 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0 SGLANG_NPU_USE_MULTI_STREAM=0
export SGLANG_ENABLE_WAR_BARRIER=1 SGLANG_FORCE_COARSE_WAR_BARRIER=1
export SGLANG_MOE_BS1_COMPACT=1 SGLANG_MOE_GATHER_PACK=1 SGLANG_MOE_FIXED_DEST=1
export SGLANG_MOE_GLOBAL_BS1=1 SGLANG_MOE_BS1_LIGHT=1 SGLANG_MOE_BS1_LL_CAPACITY=1
export SGLANG_MOE_BS1_FAST_META=1 SGLANG_MOE_BS1_DIRECT_HOT=0
export SGLANG_MOE_BS1_COLLECTIVE=0 SGLANG_MOE_BS1_ORDERED_COMBINE=0

# ---------- 4. 启动；ready 后另开终端运行 python3 scripts/moe_bs1/request_bs1.py ----------
exec python3 -u -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name dsv4f-bs1 \
  --device npu --base-gpu-id 0 --host 127.0.0.1 --port "$SERVER_PORT" \
  --dist-init-addr "127.0.0.1:$DIST_PORT" --trust-remote-code \
  --attention-backend dsv4 --page-size 128 \
  --tp-size 4 --attn-dp-size 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode auto \
  --load-format safetensors --quantization modelslim --kv-cache-dtype bfloat16 \
  --enable-dp-lm-head --disable-shared-experts-fusion --disable-overlap-schedule \
  --mem-fraction-static 0.80 --context-length 4096 --max-total-tokens 8192 \
  --chunked-prefill-size 512 --max-prefill-tokens 512 \
  --max-running-requests 4 --prefill-max-requests 4 \
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 1 2 4 --cuda-graph-max-bs-decode 4 \
  --moe-offload-config "$LOG_DIR/offload.json" \
  --disable-radix-cache --watchdog-timeout 600 --log-level info --skip-server-warmup
