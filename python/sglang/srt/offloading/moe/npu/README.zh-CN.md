# Ascend V4 offload 运行时装配

本目录集中放置 NPU 实现；公共框架见 [上级目录](../README.zh-CN.md)。

| 文件 / 目录 | 职责 |
| --- | --- |
| `config.py` / `backend.py` / `runtime.py` | 启动配置、资源装配与运行时生命周期 |
| `transfer.py` / `experts.py` | ACL/MemFabric 搬运、DeepEP 与原生 FFN |
| `storage.py` / `v4_format.py` | 共享 host master、FP4 有界加载与 bank |
| `native_acl.py` / `native_smoke.py` / `native/` | ACL 扩展构建、真实 smoke 与 C++ 源码 |
| `examples/v4.json` | 每 rank 的 offload 容量配置 |

实现说明：[存储](V4_STORAGE.md)、[EP/FFN](EXPERTS_IMPLEMENTATION.zh-CN.md)、
[ACL](native/README.zh-CN.md)；本地验证见 [验证记录](../LOCAL_VALIDATION_1021.zh-CN.md)。

`NpuMoEBackend` 在模型构造前安装，由 V4 FP4 quant method 的 create_weights 接管
host 参数分配；post-load 后装配共享 host master、hot/execution bank、固定 Tensor
residency、LFU、ACL eager prefill、MemFabric decode 和现有 Ascend EP/FFN。
本项依赖 native ACL、真实存储和 EP/FFN 三项适配一起使用。

`NpuBoundMoERuntime` 在 decoder 层首启动 prefill worker，在 FusedMoE.forward_impl
进入 routed program，外层原有 shared experts 与缩放保持原路径。层间、forward
间复用 A/B 前会把前一个 compute reader frontier 接到 copy stream。激活打包
workspace 在串行层之间共用。Dispatch 后真实本地计数进入 residency；设备
padding 上界由 ForwardBatch 传入 dispatcher，不使用本 rank Router 频次代替。

Capture head/tail 覆盖 warmup 与完整模型；update_enabled 在图外关闭/恢复。
真实 replay 不重复执行 Python 统计或 LFU tail，而在提交后记录 fresh NPU
completion events。Eager decode 会明确拒绝。模式转换先排空，保存元数据，
prefill 借池，decode 按原 slot 从 host 重载后检查错误标记再恢复映射和 LFU。
销毁要求先清理 graph lease，停止 worker、drain 所有 NPU 流、移除 hooks，最后
释放注册。失败上下文在 drain 前保留；硬件 fault 不会清零后继续服务。

## 构建与启动

仅在匹配的 Ascend A5 / CANN / torch_npu 环境执行。需要已安装当前仓库和用户
提供版本的 MemFabric、sgl-kernel-npu DeepEP。此 macOS 开发机不能完成硬件验收。

### 单机四卡启动脚本

从仓库根目录执行，先加载目标机匹配的 CANN 环境，再设置实际的 **DeepSeek V4
原生 FP4 safetensors** 权重目录：

```bash
MODEL_PATH=/path/to/DeepSeek-V4-Flash \
  bash scripts/launch_moe_offload_npu.sh --dry-run

MODEL_PATH=/path/to/DeepSeek-V4-Flash \
  bash scripts/launch_moe_offload_npu.sh
```

默认 `WORLD_SIZE=4`，可见设备 `0,1,2,3`，TP=4、EP=4、DP=1，MoE TP=1；
服务地址 `127.0.0.1:8818`。换用卡 4–7：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=/path/to/DeepSeek-V4-Flash \
  bash scripts/launch_moe_offload_npu.sh
```

脚本从自身位置定位当前源码并设置 `PYTHONPATH`，不依赖固定的 `/home/.../sglang`
目录。默认先构建 ACL 扩展；`BUILD_NATIVE=0` 可复用与当前环境匹配的已构建扩展，
`RUN_NATIVE_SMOKE=1` 可在加载模型前增加 ACL 字节/事件 smoke。扩展目录直接读取
offload JSON 的 `native_extension_directory`，避免构建与运行使用不同目录。

`MOE_OFFLOAD_CONFIG` 可指定自己的 JSON，默认使用
`python/sglang/srt/offloading/moe/npu/examples/v4.json`。默认 host 保留量为
**每 rank 128 GiB，四 rank 合计 512 GiB**，workspace/metadata 限额也是每 rank 的；
启动前应按机器和模型调整 JSON 中的预算。

首启默认最大并发 8、decode buckets 为 `1 2 4 8`、chunked prefill 和最大 prefill
token 数均为 1024、context length 为 8192，`mem_fraction_static=0.8`。
`MAX_RUNNING_REQUESTS`、`CHUNKED_PREFILL_SIZE`、`CONTEXT_LENGTH`、
`MEM_FRACTION_STATIC` 可通过环境变量修改；并发变化时脚本同步生成捕图 buckets。
增大 prefill/concurrency 时还需检查 JSON 的 `max_dispatch_rows` 和 workspace 上界，
容量不足会拒绝运行。`--dry-run` 仅打印命令，不验证真实权重和设备。

此脚本使用 V4 的 `dsv4` attention、DeepEP default、full decode graph 和 eager prefill，
启用本后端要求的 `-1` 路由与逐专家计数。`--quantization fp8` 选择当前仓库的
FP8/V4 量化配置入口，routed experts 由原生 FP4 method 接管；明确关闭 FP4→FP8 解量化。
它不带 EAGLE、DP attention 或 HiCache 参数，关闭 overlap；GLM-5.2/modelslim 权重
不适用于当前后端，实际启动时会在构建扩展前检查 `model_type` 和 EP 可整除性。

### 手动构建与启动

```bash
python -m sglang.srt.offloading.moe.npu.native_acl --build-directory /tmp/sglang-moe-acl
python -m sglang.srt.offloading.moe.npu.native_smoke --build-directory /tmp/sglang-moe-acl
export MOE_ENABLE_TOPK_NEG_ONE=1
export DEEP_USE_MODE=default
export MOE_EXPERT_TOKEN_NUMS_TYPE=1
python -m sglang.launch_server \
  --model-path /path/to/DeepSeek-V4-Flash \
  --device npu --tp-size 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend ascend \
  --disable-overlap-schedule --disable-shared-experts-fusion \
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
  --moe-offload-config python/sglang/srt/offloading/moe/npu/examples/v4.json
```

示例中的 host_pool_gib、workspace/metadata 及 max_dispatch_rows 是明确的容量配置，
须按每 rank 模型大小、batch/EP 接收上界和机器资源调整；不是测得的最优值。
host_pool_gib 必须容纳探测后的原生物理 bundle（含 padding）与 MemFabric 的 GiB
保留粒度。max_dispatch_rows 须覆盖最坏 EP 接收容量，超限拒绝执行，不截断 token。
loader 一次只转换单专家并受 workspace_limit_bytes 限制。capture 从 KV/输入缓冲
分配完后的基线测量 NPU allocator peak，把 graph/临时内存增长保守计入 workspace；
这不能替代实机驱动内存、通信私有 workspace 和真实峰值监测。

当前支持 PP=1、MoE DP=1、MoE TP=1，TP+EP 可用；前三层及 hash 层 routed
专家和 shared experts 常驻。DeepEP default 策略需要 -1 路由屏蔽支持。未接入
FuseEP、DeepEP ops/alltoall、AscendTP、EPLB、elastic EP、LoRA、speculation、torch.compile、并发
forward、其他专家 pager 或在线权重更新；不满足条件在加载或绑定时拒绝。

## 本地验证与目标机验收

CPU 测试验证 capture/warmup 开关、fresh replay 事件、精确恢复、图 lease
销毁、错误保活和 JSON 参数；native/DMA 由显式模拟器替代。

目标机仍需完成：真实扩展链接与 smoke，host 同份存储 ACL/MemFabric 双路径
字节对比，原生 NZ FP4 bank 与常驻 FFN 数值对比，EP padding/空 rank/尾组，
多 bucket 连续 replay 与 prefill/decode 切换，workspace 峰值及 H2D/F 时间线。
`RuntimeCapabilities` 表示实现提供完整捕图路径，不表示已获得这些硬件结果。
