#!/usr/bin/env bash
# DeepSeek V4 Flash W8A8 launcher for a single 16-device Ascend A3 node.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/launch_deepseek_v4_flash_a3.sh --dry-run
  MODE=offload bash scripts/launch_deepseek_v4_flash_a3.sh
  MODE=reference bash scripts/launch_deepseek_v4_flash_a3.sh

MODE=offload (default):
  TP=16, EP=16, MoE DP=1, native V4 expert offload, W8A8, decode full graph.
  EAGLE/MTP is disabled because the current native offload runtime rejects
  speculative extend batches.

MODE=reference:
  The supplied A3 high-throughput recipe with TP=16, DP attention=16,
  DP LM head, EAGLE/MTP and no expert offload.

Common overrides:
  MODEL_PATH, SGLANG_ROOT, PYTHON_BIN, SERVER_HOST, SERVER_PORT
  ASCEND_RT_VISIBLE_DEVICES, WORLD_SIZE, SYSTEM_TUNING, BUILD_NATIVE
  LOG_DIR, LOG_LEVEL, MOE_LOG_LEVEL
  Offload mode: MOE_OFFLOAD_CONFIG, ATTN_DP_SIZE (default WORLD_SIZE),
                MAX_RUNNING_REQUESTS,
                MAX_PREFILL_TOKENS, CHUNKED_PREFILL_SIZE, CONTEXT_LENGTH,
                MEM_FRACTION_STATIC

The default model path is:
  /home/weights/DeepSeek-V4-Flash-0731-w8a8
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; die "Unknown argument: $1" ;;
esac
[[ $# -le 1 ]] || die "Unexpected arguments: ${*:2}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_ROOT=${SGLANG_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-/home/weights/DeepSeek-V4-Flash-0731-w8a8}
MODE=${MODE:-offload}
WORLD_SIZE=${WORLD_SIZE:-16}
ATTN_DP_SIZE=${ATTN_DP_SIZE:-$WORLD_SIZE}
SYSTEM_TUNING=${SYSTEM_TUNING:-1}
BUILD_NATIVE=${BUILD_NATIVE:-1}
SERVER_HOST=${SERVER_HOST:-0.0.0.0}
SERVER_PORT=${SERVER_PORT:-30000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4-flash-a3}
LOG_LEVEL=${LOG_LEVEL:-info}
MOE_LOG_LEVEL=${MOE_LOG_LEVEL:-info}
LOG_DIR=${LOG_DIR:-/home/y00951466/moe-offload/logs/a3-$(date +%Y%m%d-%H%M%S)-$$}
MOE_OFFLOAD_CONFIG=${MOE_OFFLOAD_CONFIG:-$SCRIPT_DIR/moe_offload_a3.json}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-1}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-32768}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-256000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.6}

case "$MODE" in offload|reference) ;; *) die "MODE must be offload or reference" ;; esac
case "$SYSTEM_TUNING" in 0|1) ;; *) die "SYSTEM_TUNING must be 0 or 1" ;; esac
case "$BUILD_NATIVE" in 0|1) ;; *) die "BUILD_NATIVE must be 0 or 1" ;; esac
for name in WORLD_SIZE ATTN_DP_SIZE SERVER_PORT MAX_RUNNING_REQUESTS MAX_PREFILL_TOKENS CHUNKED_PREFILL_SIZE CONTEXT_LENGTH; do
    value=${!name}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer."
done
(( SERVER_PORT <= 65535 )) || die "SERVER_PORT exceeds 65535."
if [[ "$MODE" == offload ]]; then
    (( WORLD_SIZE % ATTN_DP_SIZE == 0 )) || die "ATTN_DP_SIZE must divide WORLD_SIZE."
fi

[[ -d "$SGLANG_ROOT/python/sglang" ]] || die "Invalid SGLANG_ROOT: $SGLANG_ROOT"
[[ -d "$MODEL_PATH" ]] || die "MODEL_PATH does not exist: $MODEL_PATH"
[[ -f "$MODEL_PATH/config.json" ]] || die "Missing model config: $MODEL_PATH/config.json"
[[ -f "$MODEL_PATH/quant_model_description.json" ]] || die "Missing quantization description"
command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    ASCEND_RT_VISIBLE_DEVICES=0
    for ((i=1; i<WORLD_SIZE; i++)); do ASCEND_RT_VISIBLE_DEVICES+=",$i"; done
fi
IFS=, read -r -a visible_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
[[ ${#visible_devices[@]} -eq "$WORLD_SIZE" ]] || die "Visible-device count must equal WORLD_SIZE."
for ((i=0; i<WORLD_SIZE; i++)); do
    [[ "${visible_devices[i]}" =~ ^[0-9]+$ ]] || die "Device IDs must be numeric."
    for ((j=0; j<i; j++)); do
        [[ "${visible_devices[i]}" != "${visible_devices[j]}" ]] || die "Duplicate device ID."
    done
done

if [[ "$DRY_RUN" == 0 && "$SYSTEM_TUNING" == 1 ]]; then
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
fi

for env_script in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash; do
    [[ -f "$env_script" ]] || die "Missing Ascend environment script: $env_script"
    set +u
    # shellcheck disable=SC1090
    source "$env_script"
    set -u
done

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
unset ASCEND_LAUNCH_BLOCKING HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE
unset SGLANG_HICACHE_HOST_MEM_BACKEND

export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export SGLANG_LOG_MS=${SGLANG_LOG_MS:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_SET_CPU_AFFINITY=1
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-300}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2048}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-300}
export TASK_QUEUE_ENABLE=1
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_DSV4_FP4_DEQUANT=0
export IS_DEEPSEEK_V4=1

"$PYTHON_BIN" - "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
quant = json.loads((root / "quant_model_description.json").read_text())
if config.get("model_type") != "deepseek_v4":
    raise SystemExit("MODEL_PATH is not a DeepSeek V4 checkpoint")
if quant.get("model_quant_type") != "W8A8_DYNAMIC":
    raise SystemExit("MODEL_PATH is not ModelSlim W8A8_DYNAMIC")
experts = config.get("n_routed_experts")
if not isinstance(experts, int) or experts <= 0:
    raise SystemExit("Invalid n_routed_experts")
print(f"Model: DeepSeek V4 W8A8, layers={config.get('num_hidden_layers')}, experts={experts}")
PY

export ASCEND_RT_VISIBLE_DEVICES
if [[ "$MODE" == offload ]]; then
    [[ -f "$MOE_OFFLOAD_CONFIG" ]] || die "Missing offload config: $MOE_OFFLOAD_CONFIG"
    NATIVE_BUILD_DIR=$("$PYTHON_BIN" - "$MOE_OFFLOAD_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
if data.get("version") != 1:
    raise SystemExit("Unsupported offload config version")
ratio = data.get("cache_ratio")
if type(ratio) not in (int, float) or not 0 <= ratio <= 1:
    raise SystemExit("cache_ratio must be in [0, 1]")
if ratio > 0:
    profile = data.get("offline_stats_path")
    model_id = data.get("offline_model_id")
    if not isinstance(profile, str) or not profile or not isinstance(model_id, str) or not model_id:
        raise SystemExit("Nonzero cache_ratio requires offline_stats_path and offline_model_id")
    profile_path = Path(profile)
    if not profile_path.is_absolute():
        profile_path = (path.resolve().parent / profile_path).resolve()
    if not profile_path.is_file():
        raise SystemExit(f"Missing offline profile: {profile_path}")
    stats = json.loads(profile_path.read_text())
    if not isinstance(stats, dict) or stats.get("model_id") != model_id:
        raise SystemExit("Offline profile model_id does not match offline_model_id")
    if not isinstance(stats.get("layers"), dict):
        raise SystemExit("Offline profile must contain a layers object")
directory = data.get("native_extension_directory")
if not isinstance(directory, str) or not Path(directory).is_absolute():
    raise SystemExit("native_extension_directory must be absolute")
print(directory)
PY
)
    export DEEP_USE_MODE=default
    export MOE_ENABLE_TOPK_NEG_ONE=1
    export MOE_EXPERT_TOKEN_NUMS_TYPE=1
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0
    export SGLANG_ZBAL_LOCAL_MEM_SIZE=0
    export SGLANG_ENABLE_SPEC_V2=0
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
    export SGLANG_NPU_USE_MULTI_STREAM=0
    unset FORCE_DRAFT_MODEL_NON_QUANT DEEP_NORMAL_MODE_USE_INT8_QUANT
    unset SGLANG_ENABLE_WAR_BARRIER SGLANG_FORCE_COARSE_WAR_BARRIER

    BUILD_CMD=("$PYTHON_BIN" -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$NATIVE_BUILD_DIR")
    # Attention TP can filter out unaligned graph buckets. Include the aligned
    # upper bound even when serving a single request; eager decode is unsupported.
    ATTN_TP_SIZE=$((WORLD_SIZE / ATTN_DP_SIZE))
    GRAPH_MAX_BS=$(((MAX_RUNNING_REQUESTS + ATTN_TP_SIZE - 1) / ATTN_TP_SIZE * ATTN_TP_SIZE))
    GRAPH_BS=()
    for ((bs=1; bs<GRAPH_MAX_BS; bs*=2)); do GRAPH_BS+=("$bs"); done
    GRAPH_BS+=("$GRAPH_MAX_BS")
    SERVER_CMD=(
        "$PYTHON_BIN" -m sglang.launch_server
        --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
        --host "$SERVER_HOST" --port "$SERVER_PORT"
        --tp-size "$WORLD_SIZE" --ep-size "$WORLD_SIZE" --attn-dp-size "$ATTN_DP_SIZE"
        --device npu --attention-backend dsv4 --page-size 128 --trust-remote-code
        --load-format safetensors --quantization modelslim --kv-cache-dtype bfloat16
        --moe-a2a-backend deepep --deepep-mode auto
        --disable-overlap-schedule --disable-shared-experts-fusion
        --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
        --cuda-graph-bs-decode "${GRAPH_BS[@]}"
        --cuda-graph-max-bs-decode "$GRAPH_MAX_BS"
        --max-running-requests "$MAX_RUNNING_REQUESTS"
        --max-prefill-tokens "$MAX_PREFILL_TOKENS"
        --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
        --context-length "$CONTEXT_LENGTH"
        --mem-fraction-static "$MEM_FRACTION_STATIC"
        --watchdog-timeout 9000
        --moe-offload-config "$MOE_OFFLOAD_CONFIG"
        --log-level "$LOG_LEVEL"
    )
else
    export FORCE_DRAFT_MODEL_NON_QUANT=1
    export SGLANG_ENABLE_SPEC_V2=1
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
    export DEEPEP_HCCL_BUFFSIZE=2048
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=60
    export SGLANG_ENABLE_WAR_BARRIER=1
    export SGLANG_FORCE_COARSE_WAR_BARRIER=1
    unset DEEP_USE_MODE MOE_ENABLE_TOPK_NEG_ONE MOE_EXPERT_TOKEN_NUMS_TYPE

    SERVER_CMD=(
        "$PYTHON_BIN" -m sglang.launch_server
        --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME"
        --page-size 128
        --tp-size "$WORLD_SIZE"
        --trust-remote-code
        --device npu
        --attention-backend dsv4
        --watchdog-timeout 9000
        --host "$SERVER_HOST" --port "$SERVER_PORT"
        --mem-fraction-static 0.6
        --prefill-max-requests 160
        --max-prefill-tokens 80000
        --chunked-prefill-size 131072
        --max-running-requests 160
        --dp-size 16 --enable-dp-attention
        --moe-a2a-backend deepep --deepep-mode auto
        --quantization modelslim --enable-dp-lm-head
        --kv-cache-dtype bfloat16
        --speculative-algorithm EAGLE
        --speculative-num-steps 2
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 3
        --cuda-graph-bs 1 2 4
        --log-level "$LOG_LEVEL"
    )
fi

printf 'MODE=%s WORLD_SIZE=%s DEVICES=%s\n' "$MODE" "$WORLD_SIZE" "$ASCEND_RT_VISIBLE_DEVICES"
printf 'Model=%s\nHost=%s Port=%s\n' "$MODEL_PATH" "$SERVER_HOST" "$SERVER_PORT"
if [[ "$MODE" == offload ]]; then
    printf 'ATTN_DP=%s ATTN_TP=%s decode_graph_buckets=%s\n' "$ATTN_DP_SIZE" "$ATTN_TP_SIZE" "${GRAPH_BS[*]}"
    printf 'Offload config=%s\nNative build dir=%s\n' "$MOE_OFFLOAD_CONFIG" "$NATIVE_BUILD_DIR"
    "$PYTHON_BIN" - "$MOE_OFFLOAD_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print(f"cache_ratio={data.get('cache_ratio')} offline_profile={data.get('offline_stats_path')}")
PY
    printf 'max_running=%s max_prefill=%s chunked_prefill=%s context=%s mem_fraction=%s\n' \
        "$MAX_RUNNING_REQUESTS" "$MAX_PREFILL_TOKENS" "$CHUNKED_PREFILL_SIZE" "$CONTEXT_LENGTH" "$MEM_FRACTION_STATIC"
fi
if [[ "$DRY_RUN" == 1 ]]; then
    if [[ "$MODE" == offload && "$BUILD_NATIVE" == 1 ]]; then printf '%q ' "${BUILD_CMD[@]}"; printf '\n'; fi
    printf '%q ' "${SERVER_CMD[@]}"
    printf '\n'
    exit 0
fi

mkdir -p -- "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/server.log") 2>&1
printf 'Log directory: %s\n' "$LOG_DIR"

if [[ "$MODE" == offload && "$BUILD_NATIVE" == 1 ]]; then
    "${BUILD_CMD[@]}"
fi
exec "${SERVER_CMD[@]}"
