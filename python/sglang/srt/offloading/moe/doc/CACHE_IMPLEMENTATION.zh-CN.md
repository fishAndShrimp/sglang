# 10 月 21 日 cache draft 实现

> 本文保留 10 月 21 日的实现记录。2026-10-22 原生默认策略已替换为离线 Top-K / 周期
> 层内 LFU，相关 CPU 契约测试已恢复；当前接入及验证见
> [周期缓存说明](../npu/doc/PERIODIC_CACHE_1022.zh-CN.md)。

基线：`59cbb5d226`；目标设计：同目录 `moe-h2d-design.zh-CN.md` v0.4。
本提交实现可组合、可本地验证的缓存规划与生命周期组件。没有自动启用
MoE offload，也没有声明 A5 完整 graph、原生 FP4 或多卡 EP 已通过。

## 已实现

- `TensorResidencyOps`：固定容量设备 Tensor 计划，输入为 Dispatch 后本 rank
  的完整专家执行计数；命中从 HOT_SLOT 读取，miss 从 HOST 读取。临时执行
  不自动准入热池，空需求与尾波不读取无效 lane。
- `prepare_update` / `commit_update`：模型尾部应用 CachePolicy 的目标热集，
  保留幸存者原 slot，先使 victim 失效，再加载完整 bundle，最后发布映射。
  generation、非法 key、重复目标和损坏的反向映射使用共享设备错误标记阻断。
- 所有 planner 持久计划、布尔/索引 scratch 在绑定时分配；`allocations()`
  报告物理 metadata 存储。动态规划不调用 `.item()`、`.tolist()` 或事件查询。
- `ManagedExecutionContext`：线程安全保活，计算/复制/通信完成的汇合，失败后
  drain 成功才释放资源；每次实际提交记录新凭据，capture ticket 不是 replay
  已完成的证据。
- `DecodeStateSnapshot`：预分配 slot / 映射 / LFU 快照，不备份权重；恢复时
  先通过绑定的 reload 回调加载原 slot，再检查共享设备错误并恢复元数据。
  模式转换是 eager 安全边界，允许同步和读取错误标记。
- Graph backend 的 opt-in 生命周期包装：捕图与 warmup 由具体 runtime 提供
  `capture_scope`，graph lease 在图销毁后释放；默认未绑定 scope 会明确拒绝。
  普通常驻与旧 offloader 的默认 graph 绑定保持透传。

## 绑定要求

`BoundExecutionWave` 指明 `(layer_id, ExpertWave, execution_start, bindings)`。
每个 wave 的 metadata 与 bank 地址在捕图前固定。`bindings` 必须关联同一份
`residency.slot_generation` 和 `error_status`，FFN 必须服从 `execution_valid`
及错误状态；不能把复制被屏蔽的 lane 当作有效专家。

每次完整 decode 调用 `begin_step(context)` 一次，随后由现有 program / access
发出 resolve、copy、publish、F、finish；尾部调用 `CacheUpdateCoordinator`。
统计来自本地 Dispatch 接收的计数，不能用本 rank Router 输出代替。
`update_enabled` 在 warmup/capture 期间为假，真实 replay 前由 runtime 更新。

`DecodeStateSnapshot(..., error_status=ops.error_status)` 必须传入与搬运共享的
设备错误状态。reload 回调只恢复物理权重，不提前发布 live 映射。失败后由
owner 的失败路径排空并关闭，不能清除错误后继续使用不完整权重。

## 验收方式

CPU 单测与 runner 已移除。缓存规划、原 slot 恢复、零缓存、空需求、尾波、非法
映射及失败保活，应在目标 NPU 上结合真实搬运、EP/FFN 和 graph replay 验收。

## 仍需硬件适配和验收

具体 `MoEBackend` / `BoundMoERuntime` 的 V4 原生权重 loader、FP4 bank 格式、
EP dispatch/layout/FFN、model hooks 和运行时装配仍需目标硬件实现；本提交
提供其可复用的 cache 组件。必须在目标 CANN / torch_npu / A5 上验证算子覆盖、
跨流捕图、真实 workspace 峰值、多 bucket 和反复 replay 后，才能声明
`RuntimeCapabilities.full_decode_graph=True`。
