#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3.8-Flash-Next checkpoint directory}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select eight available A3 dies}"

# Single-node communication and allocator settings.
unset ASCEND_LAUNCH_BLOCKING ASCEND_CUSTOM_OPP_PATH
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export HCCL_BUFFSIZE=1000 HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_SET_CPU_AFFINITY=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32 ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MULTI_STREAM=0 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 SGLANG_NPU_USE_SCATTER_PA_KV_CACHE=1

# Sparse attention, per-layer embeddings, MoE and graph replay.
export SGLANG_NPU_QSA_LIGHTNING_INDEXER=1 SGLANG_NPU_QSA_REQUEST_LI=1 SGLANG_NPU_QSA_DEVICE_METADATA=1
export SGLANG_NPU_QSA_PREFILL_GBSA=1 SGLANG_NPU_QSA_PAGED_GBSA=1 SGLANG_NPU_QSA_DECODE_GBSA=0
export SGLANG_NPU_QSA_VA_NORM_ROPE=1 SGLANG_NPU_QSA_VA_COMPRESS=1 SGLANG_NPU_QSA_VA_RAW_STORE=1
export SGLANG_NPU_QSA_VA_PREFILL_DISPATCH=1 SGLANG_NPU_QSA_GBSA_SORT_COUNTS=1 SGL_KERNEL_NPU_QSA_DYNAMIC_SCORE=1
export SGLANG_NPU_PLE_VA_GATE=1 SGLANG_NPU_PLE_GROUPED_NORM=1 SGLANG_NPU_PLE_NGRAM_HASH=1
export SGLANG_NPU_PLE_HBM_LOOKUP=1 SGLANG_NPU_PLE_WINDOW_IDS=1 SGLANG_NPU_PLE_VERIFY_STATE=1
export SGLANG_NPU_MOE_SKIP_NOOP_IDS=1 SGLANG_NPU_MOE_VA_TOPK=1 SGLANG_NPU_QWEN_MOE_STAGED=1
export SGLANG_NPU_MOE_PADDED_L1=1 SGLANG_NPU_GDN_FUSED_SIGMOID=1 SGLANG_NPU_PREWARM_FIXED_GRAPHS=1
export SGLANG_NPU_HC_PREFILL_COMBINE_NORM=1 SGLANG_NPU_QSA_SHORT_CUBE=1
export SGLANG_NPU_PLE_SHARD_GATE_ROWS=0 SGLANG_NPU_SPEC_PINNED_LENGTHS=0
export SGLANG_NPU_QSA_MULTISTEP_METADATA=0 SGLANG_NPU_HC_OVERLAP_UP_CAST=0
export SGLANG_NPU_QSA_PREFILL_TP_SHARD=indexer SGLANG_NPU_QSA_REUSE_GBSA_META=1
export SGLANG_NPU_GDN_SHARED_INPUTS=1 SGLANG_NPU_QWEN4_GREEDY_SHARDS=1
export SGLANG_VLM_CACHE_SIZE_MB=0 SGLANG_MM_PRECOMPUTE_HASH=1

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --host 127.0.0.1 --port "${PORT:-30000}" \
  --device npu --attention-backend ascend --nnodes 1 --base-gpu-id 0 --tp-size 8 --ep-size 8 \
  --dtype bfloat16 --mamba-ssm-dtype bfloat16 --page-size 128 \
  --context-length 65536 --chunked-prefill-size 8192 --max-prefill-tokens 98304 \
  --max-running-requests 32 --max-total-tokens 573440 --mem-fraction-static 0.86 \
  --ple-offload-embedding --ple-offload-backend pinned \
  --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --enable-metrics \
  --disable-radix-cache --trust-remote-code --random-seed 674391960 \
  --enable-multimodal --mm-attention-backend ascend_attn --no-mm-enable-dp-encoder \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --moe-a2a-backend ascend_ops --speculative-moe-a2a-backend ascend_ops \
  --moe-ascend-ops-config '{"prefill":"allgather","decode":"low_latency"}' \
  --model-loader-extra-config '{"enable_multithread_load":false}'
