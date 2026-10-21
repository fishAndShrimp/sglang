#!/usr/bin/env bash
# Single-node Ascend A5, native DeepSeek V4 FP4 experts, TP=EP=WORLD_SIZE.
# Source the matching CANN/torch_npu environment before running this script.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/DeepSeek-V4-Flash bash scripts/launch_moe_offload_npu.sh
  MODEL_PATH=/path/to/DeepSeek-V4-Flash bash scripts/launch_moe_offload_npu.sh --dry-run

Defaults: WORLD_SIZE=4, ASCEND_RT_VISIBLE_DEVICES=0,1,2,3, TP=4, EP=4, DP=1.
MODEL_PATH must point to a local DeepSeek V4 native-FP4 safetensors checkpoint.
GLM/modelslim, converted FP8 experts and speculative decoding are not supported.

Overrides:
  SGLANG_ROOT, PYTHON_BIN, WORLD_SIZE, ASCEND_RT_VISIBLE_DEVICES
  MOE_OFFLOAD_CONFIG (defaults to the repository's examples/npu_v4.json)
  SERVER_HOST=127.0.0.1 SERVER_PORT=8818 SERVED_MODEL_NAME=deepseek-v4-moe-offload
  MAX_RUNNING_REQUESTS=8 CHUNKED_PREFILL_SIZE=1024 CONTEXT_LENGTH=8192
  MEM_FRACTION_STATIC=0.8 BUILD_NATIVE=1 RUN_NATIVE_SMOKE=0

--dry-run prints the environment, native build and server commands without
building, loading weights or initializing NPU devices. Budgets in the JSON are
per rank: the default host reservation is 128 GiB/rank, 512 GiB for four ranks.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; die "Unknown argument: $1" ;;
esac
[[ $# -eq 0 ]] || die "Unexpected arguments: $*"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=${SGLANG_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-}
WORLD_SIZE=${WORLD_SIZE:-4}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
SERVER_PORT=${SERVER_PORT:-8818}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4-moe-offload}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-8}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-1024}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
BUILD_NATIVE=${BUILD_NATIVE:-1}
RUN_NATIVE_SMOKE=${RUN_NATIVE_SMOKE:-0}
MOE_OFFLOAD_CONFIG=${MOE_OFFLOAD_CONFIG:-$SGLANG_ROOT/python/sglang/srt/offloading/moe/examples/npu_v4.json}

[[ -n "$MODEL_PATH" ]] || die "Set MODEL_PATH to a DeepSeek V4 native-FP4 checkpoint; see --help."
for name in WORLD_SIZE SERVER_PORT MAX_RUNNING_REQUESTS CHUNKED_PREFILL_SIZE CONTEXT_LENGTH; do
    value=${!name}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer."
done
(( SERVER_PORT <= 65535 )) || die "SERVER_PORT exceeds 65535."
(( MAX_RUNNING_REQUESTS <= 128 )) || die "MAX_RUNNING_REQUESTS must not exceed the DeepEP dispatch limit (128)."
[[ "$BUILD_NATIVE" =~ ^[01]$ && "$RUN_NATIVE_SMOKE" =~ ^[01]$ ]] || die "BUILD_NATIVE and RUN_NATIVE_SMOKE must be 0 or 1."
[[ -d "$SGLANG_ROOT/python/sglang" ]] || die "Invalid SGLANG_ROOT: $SGLANG_ROOT"
[[ -f "$MOE_OFFLOAD_CONFIG" ]] || die "Missing offload JSON: $MOE_OFFLOAD_CONFIG"
command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    ASCEND_RT_VISIBLE_DEVICES=0
    for ((i=1; i<WORLD_SIZE; i++)); do ASCEND_RT_VISIBLE_DEVICES+=",$i"; done
fi
IFS=, read -r -a visible_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
[[ ${#visible_devices[@]} -eq "$WORLD_SIZE" ]] || die "Visible-device count must equal WORLD_SIZE."
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "Use comma-separated numeric device IDs."
for ((i=0; i<WORLD_SIZE; i++)); do
    for ((j=0; j<i; j++)); do
        [[ "${visible_devices[i]}" != "${visible_devices[j]}" ]] || die "Duplicate visible device ID."
    done
done
export ASCEND_RT_VISIBLE_DEVICES
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
unset HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE
unset SGLANG_HICACHE_HOST_MEM_BACKEND
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-300}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2048}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-1}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-0}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-300}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=1
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
export INF_NAN_MODE_FORCE_DISABLE=1
export ENABLE_PROFILING=0
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=0

# These are correctness requirements of the current offload backend.
export DEEP_USE_MODE=default
export MOE_ENABLE_TOPK_NEG_ONE=1
export MOE_EXPERT_TOKEN_NUMS_TYPE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0
export SGLANG_ZBAL_LOCAL_MEM_SIZE=0
export SGLANG_ENABLE_SPEC_V2=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_NPU_USE_MULTI_STREAM=0
export IS_DEEPSEEK_V4=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_DSV4_FP4_DEQUANT=0
export USE_FUSED_HC_PRE_ASCENDC=1
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

NATIVE_BUILD_DIR=$("$PYTHON_BIN" - "$MOE_OFFLOAD_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
directory = config.get("native_extension_directory")
if not isinstance(directory, str) or not Path(directory).is_absolute():
    raise SystemExit("Offload JSON requires an absolute native_extension_directory")
print(directory)
PY
)

GRAPH_BS=()
for ((bs=1; bs<MAX_RUNNING_REQUESTS; bs*=2)); do GRAPH_BS+=("$bs"); done
GRAPH_BS+=("$MAX_RUNNING_REQUESTS")
BUILD_CMD=("$PYTHON_BIN" -m sglang.srt.offloading.moe.native_acl --build-directory "$NATIVE_BUILD_DIR")
SMOKE_CMD=("$PYTHON_BIN" -m sglang.srt.offloading.moe.native_smoke --build-directory "$NATIVE_BUILD_DIR")
SERVER_CMD=(
    "$PYTHON_BIN" -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
    --host "$SERVER_HOST" --port "$SERVER_PORT"
    --tp-size "$WORLD_SIZE" --ep-size "$WORLD_SIZE" --dp-size 1
    --device npu --attention-backend dsv4 --page-size 128 --trust-remote-code
    --load-format safetensors --quantization fp8 --kv-cache-dtype bfloat16
    --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend ascend
    --disable-overlap-schedule --disable-shared-experts-fusion
    --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
    --cuda-graph-bs-decode "${GRAPH_BS[@]}"
    --cuda-graph-max-bs-decode "$MAX_RUNNING_REQUESTS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens "$CHUNKED_PREFILL_SIZE" --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC"
    --watchdog-timeout 9000 --enable-metrics --load-balance-method round_robin
    --moe-offload-config "$MOE_OFFLOAD_CONFIG"
)

printf 'WORLD_SIZE=%s TP=%s EP=%s DP=1 DEVICES=%s\n' "$WORLD_SIZE" "$WORLD_SIZE" "$WORLD_SIZE" "$ASCEND_RT_VISIBLE_DEVICES"
printf 'Offload JSON: %s\nNative build: %s\n' "$MOE_OFFLOAD_CONFIG" "$NATIVE_BUILD_DIR"
if [[ "$BUILD_NATIVE" == 1 ]]; then printf '%q ' "${BUILD_CMD[@]}"; printf '\n'; fi
if [[ "$RUN_NATIVE_SMOKE" == 1 ]]; then printf '%q ' "${SMOKE_CMD[@]}"; printf '\n'; fi
printf '%q ' "${SERVER_CMD[@]}"; printf '\n'
[[ "$DRY_RUN" == 0 ]] || exit 0

"$PYTHON_BIN" - "$MODEL_PATH" "$WORLD_SIZE" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
config = json.loads((model / "config.json").read_text())
if config.get("model_type") != "deepseek_v4":
    raise SystemExit("MODEL_PATH must be DeepSeek V4 native FP4; GLM/modelslim is unsupported")
experts = config.get("n_routed_experts", 0)
if not isinstance(experts, int) or experts <= 0 or experts % int(sys.argv[2]):
    raise SystemExit("n_routed_experts must divide evenly across WORLD_SIZE")
if not any(model.glob("*.safetensors")):
    raise SystemExit("MODEL_PATH contains no local safetensors checkpoint")
PY

if [[ "$BUILD_NATIVE" == 1 ]]; then "${BUILD_CMD[@]}"; fi
if [[ "$RUN_NATIVE_SMOKE" == 1 ]]; then "${SMOKE_CMD[@]}"; fi
exec "${SERVER_CMD[@]}"
