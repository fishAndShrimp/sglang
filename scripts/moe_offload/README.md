# Ascend Ops MoE offload

## 怎么启动

前提：已安装匹配的 CANN、torch_npu、sgl-kernel-npu、DeepSeek recipe 和 MemFabric（含 A3 构建源码）。

修改 [serve_tp4.sh](serve_tp4.sh) 开头的权重路径、四个空闲 die 和端口。`CACHE_RATIO` 默认 `0.28125`，可改为 `0.5`；`ENABLE_MTP=1` 开启 MTP。然后在仓库根目录运行：

```bash
bash scripts/moe_offload/serve_tp4.sh
```

脚本自动准备本地编译缓存并启动服务，日志写入 `LOG_DIR`。更换依赖版本后更换 `RUNTIME_DIR`。所有保留的 offload 优化默认生效。

服务 ready 后，可检查历史算术错题或测短请求性能（结果写入 `/tmp`；端口不同时加 `--port 端口`）：

```bash
python3 scripts/moe_offload/test_requests.py
python3 scripts/moe_offload/test_requests.py --bench
```

这不是 `sglang.bench_serving`：使用 8 条短自然语言提示，固定输出 128 token，并发 1/4/8 各两轮，轮换固定 DP rank；不额外预热、不自动重置缓存，首次与后续轮都记录。BS 指请求并发；TPOT 按首末内容流块间隔除以输出 token 数减一计算，含客户端/流式传输影响，不是设备逐 token 时间。历史错题和原始错误已标在脚本中，文本格式不匹配需人工检查。

## 当前边界

以下“已验证”指合入前的 Ascend Ops 基线；本次整层 prefill 尚待 NPU 验收。

- 已验证 A3、DeepSeek V4 Flash ModelSlim W8A8、TP=EP=Attention DP=4；MoE TP、CP、PP 均为 1。要求专家形状一致、量化 offset 为零。
- 已验证请求并发 1～8、28.125%/50% 缓存及 MTP 开关；MTP 使用 EAGLE steps=2、topk=1、draft tokens=3。基础回归不替代完整精度评测。
- 共享专家、最前面的 `max(3, num_hash_layers)` 层和 MTP draft 权重常驻。Host 预算每 rank 64 GiB，四 rank 合计 256 GiB。
- 每 rank 上限为 256 个本地专家、1280 个 Hot 槽、8192 个专家 key；有效行 mask 默认容纳 128 个本地 padded/验证 token，不等于请求 BS。
- 不支持与 EPLB、冗余专家、batch overlap、模块 CPU offloader 或 MemFabric HiCache 同时使用。

## 保留的实现

- Prefill 适配自 [codex/moe-prefill-opt-20261102](https://github.com/pachi-33/sglang/commit/2dc7903d0bebe6f003db2149f8a612e641a5e190)：在每次全局 prefill forward 的入口、任何模型层计算之前提交前两个卸载层的完整权重，以两个 bank 的整层视图交替执行；第 i 层 GMM2 的 reader event 记录后，向同一 bank 提交第 i+2 个卸载层的预取，覆盖前必须等待 reader 完成。每层仍使用原生一次 dispatch、一次 FFN、一次 combine，不按专家 wave 拆分。dispatch 入口只消费已启动的预取会话，不再延迟启动搬运。
- Prefill 是否启动由本轮 batch 的全局 `is_extend_in_batch` 判断；本地 IDLE / DECODE rank 仍参与其他 rank 的 prefill。不会读取上一轮残留的 RuntimeContext phase。每个 chunk 都重新全量预取；Decode / TARGET_VERIFY / 全局 IDLE 不启动 prefill 搬运，decode 缓存与图路径沿用原实现。
- Prefill 的 ACL 完整 bundle 描述符在装配时准备，保留 W13/W2、scale 和 offset。AllToAllV 直接使用紧凑接收布局；AllGather 保留原生 payload / scale 容量及 post-permute 的无效行屏蔽，不增加逐层 `.sum().item()` 同步。空接收跳过 FFN，但继续处理预取依赖和通信。
- Prefill 用一次 Triton 操作完成计数格式转换、错误屏蔽和访问统计，GMM 按完整本地专家顺序读取 bank。没有额外的专家打包、激活重排或重复量化；FFN 使用实验分支现有的 W13 → 激活 → W2 实现。
- 每 rank 有 E 个本地专家时，A 使用 Execution 前 E 槽；B 优先借用 Hot 前 E 槽。Hot 不足 E 时，需在 offload JSON 显式设置 `execution_slots >= 2*E`，由 Execution 后 E 槽提供 B。默认 `execution_slots=E`，容量不足直接报错，不额外分配权重池。
- 最后一个 prefill FFN 结束后回填借用的 Hot 前缀，异常退出也会在清理前回填并排空流。仅恢复被借用的槽位，保留缓存 owner / generation / LFU 状态，随后才执行缓存维护与 decode。
- Decode 保留固定执行区、活跃专家 W13/W2 和 scale 的稳定打包与清零尾部计数。有效 token mask 排除 padding 和空 rank。
- 融合准备搬运地址、逐专家计数和访问统计，缓存更新后统一发布地址表；跳过已校验为零的 offset，各权重/scale 分别使用相同计数调用 `group_pack_copy`。
- 独立流搬运 W13/W2，W2 与第一段专家计算重叠，第二段计算前等待完成。
- 周期 LFU、固定槽缓存搬运、融合缓存更新及维护图重放。Hot 命中仍需 D2D 到执行区，miss 从 Host 搬入。

## 本次合入验证

CPU 契约测试：`python3 test/registered/unit/offloading/test_ascend_prefill.py`，覆盖 bank 容量、异步读写顺序、跨 forward event、逐专家计数、部分拷贝失败后的 Hot 恢复和 Prefill→Decode 切换。计数内核使用 Triton 的 CPU 解释模式执行生产函数；需要安装 Triton / NumPy。

原适配在隔离 serving / NPU 依赖的 CPU harness 中通过了 21 项 UT。本次补充 forward 入口提前提交、全局 prefill 的 IDLE / DECODE rank、残留 phase 隔离和首次搬运部分失败的恢复回归；测试先在修复前复现入口未预取的问题，再验证修复。使用真实 runner / dispatcher / offloader Python 方法，通信、设备流和 FFN 内核由测试实现模拟；完整 serving 与原生计数内核仍按目标环境验证。

本地主机没有 `torch_npu`，原生 ACL H2D、NZ bank 视图、AllToAllV / AllGather 数值、空 rank、prefill→decode 图重放与 TTFT 仍需在目标 NPU 验收。可继续使用上述请求检查脚本；请同时对比 offload 关闭时的输出与真实设备时间线。
