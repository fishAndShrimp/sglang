# MoE offload 适配提交说明（2026-10-21）

仓库：`sglang-moe-offload-npu/sglang`，分支：`feat/moe-offload-draft-1021`。
基础提交：`59cbb5d226`。以下两个实现提交已分别落地：

| 提交 | 范围 | 行为与验证 |
| --- | --- | --- |
| `076c6ca020` | cache | 固定 Tensor residency / LFU 热集事务、完整 bundle 发布、原 slot 恢复、graph lease 与失败保活；独立 91 项 CPU 测试通过。 |
| `817da7d9fb` | H2D | 有界 prefill 流水、ACL batch 描述符适配、MemFabric device plan lowering、共享 host capability；组合 132 项 CPU 测试通过。 |

实现细节与边界见 [cache 说明](CACHE_IMPLEMENTATION.zh-CN.md)、
[H2D 说明](H2D_IMPLEMENTATION.zh-CN.md) 和 [本地验证](LOCAL_VALIDATION_1021.zh-CN.md)。
以上验证运行于 macOS CPU；不代表真实 NPU DMA、完整 graph、EP 或 FP4 数值验证。

## 后续四项独立提交

1. **ACL 原生入口**：提供与安装的 CANN 头文件匹配的编译扩展，落实 torch_npu host 队列顺序、stream guard、事件与描述符生命周期。
2. **真实存储与加载**：共享 host master 的分配/注册/DVA/释放、原生 V4 MXFP4 权重与 scale 的有界加载、固定 bank 物理布局。
3. **Cache 运行时接入**：具体 backend/runtime、模型 hooks、每步统计、capture/warmup 控制、fresh replay 事件、精确恢复、销毁及启动配置。
4. **EP / FFN 接入**：现有 dispatcher/runner 的完成事件与本地需求、wave 布局、原生 bank FFN、有效 lane/错误屏蔽、全部 reader 的完成保护。

每项提交均附对应 CPU 可验证的测试；目标环境缺失的编译、算子、跨流 capture、EP 数值和性能验收必须单独标明。默认不开启未经配置的 offload；不以模拟完成事件或抽象接口替代硬件验收结论。
