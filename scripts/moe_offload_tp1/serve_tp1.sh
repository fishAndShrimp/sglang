#!/usr/bin/env bash
# DeepSeek V4 Flash ModelSlim INT4 W4A8，TP=EP=1，Ascend Ops 本地路由。
# 沿用已验证配置跳过 HTTP 预热；NPU graph 的 warmup/capture 仍保留。
set -eo pipefail

# 修改这里：权重路径、一个空闲逻辑设备、SOC=A3/A5、端口。首轮保持 20% 缓存。
MODEL_PATH=${MODEL_PATH:-/path/to/DeepSeek-V4-Flash-w4a8-mtp}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
SERVER_PORT=${SERVER_PORT:-30000}
DIST_PORT=${DIST_PORT:-29500}
CACHE_RATIO=${CACHE_RATIO:-0.2}
SOC=${SOC:-A5}
case "$SOC" in A3|A5) ;; *) echo "SOC must be A3 or A5" >&2; exit 1 ;; esac
RUNTIME_DIR=${RUNTIME_DIR:-/tmp/sglang-offload-tp1-$SOC-runtime}  # 更换依赖版本后换目录
LOG_DIR=${LOG_DIR:-/tmp/sglang-ascend-offload-$(date +%Y%m%d_%H%M%S)-$$}
[[ -d "$MODEL_PATH" ]] || { echo "Set MODEL_PATH to the checkpoint directory" >&2; exit 1; }
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || { echo "Select one logical NPU device" >&2; exit 1; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
mkdir -p "$RUNTIME_DIR/python" "$RUNTIME_DIR/native_acl" "$LOG_DIR"
exec > >(tee "$LOG_DIR/server.log") 2>&1

# 昇腾环境；安装位置不同则修改下面四行。
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
python3 - "$RUNTIME_DIR/python/memfabric_hybrid/lib" "$SOC" <<'PY'
import sys
from pathlib import Path
from memfabric_hybrid._acc_offload_build import ensure_acc_offload_ext
marker = Path(sys.argv[1]) / '.sglang_offload_soc'
force = not marker.exists() or marker.read_text().strip() != sys.argv[2]
assert ensure_acc_offload_ext(sys.argv[2], install_dir=sys.argv[1], force=force)
marker.write_text(sys.argv[2] + '\n')
PY
python3 -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$RUNTIME_DIR/native_acl"

# 使用同目录的缓存配置，写入本次运行目录；Host 预算为 128 GiB。
python3 - "$SCRIPT_DIR" "$RUNTIME_DIR" "$LOG_DIR" "$CACHE_RATIO" <<'PY_CONFIG'
import json, sys
from pathlib import Path
scripts, runtime, logs = (Path(p).resolve() for p in sys.argv[1:4])
config = json.loads((scripts / 'config.json').read_text())
config.update(cache_ratio=float(sys.argv[4]),
              offline_stats_path=str(scripts / 'initial_profile.json'),
              native_extension_directory=str(runtime / 'native_acl'))
(logs / 'offload.json').write_text(json.dumps(config, indent=2) + '\n')
PY_CONFIG

# 已验证的模型算子配置；offload 优化在代码中默认生效。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ASCEND_LAUNCH_BLOCKING
unset SGLANG_ZBAL_BOOTSTRAP_URL ZBAL_ENABLE_GRAPH ZBAL_NPU_ALLOC_CONF
unset FORCE_DRAFT_MODEL_NON_QUANT DEEP_NORMAL_MODE_USE_INT8_QUANT
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=300 ACL_DEVICE_SYNC_TIMEOUT=300
export STREAMS_PER_DEVICE=32 INF_NAN_MODE_FORCE_DISABLE=1 TASK_QUEUE_ENABLE=1
export SGLANG_ZBAL_LOCAL_MEM_SIZE=0
export SGLANG_DSV4_FP4_EXPERTS=False SGLANG_DSV4_FP4_DEQUANT=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0 SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export DEEP_USE_MODE=default MOE_ENABLE_TOPK_NEG_ONE=1 MOE_EXPERT_TOKEN_NUMS_TYPE=1
export DEEPEP_HCCL_BUFFSIZE=1024 DEEPEP_HYBRID_DEPLOYMENT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0
export SGLANG_ENABLE_SPEC_V2=0 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0 SGLANG_NPU_USE_MULTI_STREAM=1
export SGLANG_ENABLE_WAR_BARRIER=1 SGLANG_FORCE_COARSE_WAR_BARRIER=1

exec python3 -u -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name dsv4f-ascend-offload \
  --device npu --base-gpu-id 0 --host 127.0.0.1 --port "$SERVER_PORT" \
  --dist-init-addr "127.0.0.1:$DIST_PORT" --trust-remote-code \
  --attention-backend dsv4 --page-size 128 \
  --tp-size 1 --attn-dp-size 1 --ep-size 1 \
  --moe-a2a-backend ascend_ops --deepep-mode auto \
  --moe-ascend-ops-config '{"prefill":"allgather","decode":"allgather"}' \
  --load-format safetensors --quantization modelslim --kv-cache-dtype bfloat16 \
  --enable-dp-lm-head --disable-shared-experts-fusion --disable-overlap-schedule \
  --mem-fraction-static 0.95 --context-length 4096 --max-total-tokens 32768 --swa-full-tokens-ratio 0.25 \
  --chunked-prefill-size 128 --max-prefill-tokens 512 \
  --max-running-requests 8 --prefill-max-requests 8 \
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 1 2 4 8 --cuda-graph-max-bs-decode 8 \
  --moe-offload-config "$LOG_DIR/offload.json" \
  --disable-radix-cache --skip-server-warmup --watchdog-timeout 600 --log-level info
