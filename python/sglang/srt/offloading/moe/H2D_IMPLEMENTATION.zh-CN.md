# 10 月 21 日 H2D draft 实现

本提交在 cache draft 之上实现 eager prefill 流水及两种搬运适配。
MemFabric 语义来自用户指定的本地 `memfabric_hybrid-master`；未修改该目录。

## 已实现

- `HostMemoryCapability`：显式保存 host 可读范围、ACL 异步读取能力、device
  映射地址和 registration 保活。`HostExpertStore` 可接收经认证的外部分配，
  不把 `is_pinned()` 与 MemFabric 注册混为一谈，也不分配第二份 host master。
- `allocations()` 对共享存储只计一次，并计入外部分配器的物理保留容量。
  `host_allocation_limit_bytes` 显式允许页对齐额外容量；默认仍使用原精确预算。
- `AclBatchHostTransferOps`：完整校验 H2D 字节区间、目标设备、注册和重叠写，
  按条目上限拆 batch；提交前保活所有描述符，完整 batch 后才记录 ready，
  部分提交失败仍保留资源并尝试记录 drain 事件。
- `MemFabricDeviceTransferOps`：固定设备计划生成源地址、压紧目标地址和 mask，
  正确适配原生 `src[i] -> dst[j]` 语义；支持任意目标 slot、HOST/HOT_SLOT、
  空/稀疏计划、多权重分量和有界批次。清除 packed metadata 尾部，校验
  generation / 计数 / 索引 / 重复目标，故障通过共享错误状态阻断整个 bundle。
- `BoundedPrefillWeightPipeline`：每趟独立 session、P/Q 有界后台提交、Dispatch
  完成反馈封口 K、首批 A 与随后 B/a 双缓冲；保留所有专家、只做一次 D/C。
  首批微块写不同偏移；下一 wave 和下一层覆盖 bank 前都等待旧 reader。
  停止 worker 后记录 copy fence，失败保活纳入 invocation 生命周期。

## 必须由目标环境提供的部分

ACL 入口必须注入**与目标 CANN 头文件编译匹配**且经过 torch_npu host 队列
顺序验证的 `native_submit(batch, stream)` 薄封装。这里没有猜测 ctypes ABI、
没有假设 PyACL 存在同名 batch 函数，也没有静默改成 Tensor.copy_。未提供并
声明队列顺序验证的封装会拒绝初始化；因此本提交尚不能直接启动实际 ACL DMA。

MemFabric 默认调用本地 Python 包的 `group_pack_copy`，在 caller 的 NPU copy
stream 上执行，事件覆盖全部分量和 batch。库构建/首次加载、映射注册和指针
表绑定必须在 capture 前完成。`allow_cpu=True` 仅用于注入测试 kernel，不能
作为生产 fallback。适配器本身不设置 `full_decode_graph`。

`MemFabricTransferBindings` 显式登记持久地址表和 metadata；Tensor lowering
的临时表达式仍使用 PyTorch allocator，其 capture 峰值必须由具体 runtime
计入 workspace 预算。`name=` 可显式命名多个 layer/bank 的资源登记。

Prefill 绑定必须提供 `make_request`、`combine_bindings`、`worker_scope` 和
`record_copy_completion`。它们分别负责 bank 偏移/物理格式、首批连续视图、
worker device/stream 入口和真实队列完成事件；不允许新增未登记权重存储。

## 本地验证范围

按 `CACHE_IMPLEMENTATION.zh-CN.md` 的命令执行全部 CPU 契约测试。
新增测试校验全部 32 种五 lane 稀疏 mask、非连续目标、多分量字节尾部、空
计划、HOT/HOST 来源、generation 错误、同池写覆盖、metadata 地址复用、
partial failure 保活、动态 K、P/Q 上界和跨层 reader 保护。

另有跨组件测试，把真实 TensorResidencyOps、LFU、搬运 lowering、字节复制
模拟 kernel 和 DecodeStateSnapshot 连起来，连续改变两层路由并恢复原 slot。
模拟器只替代 native DMA/kernel，未把它等同于真实 NPU、完整 graph、EP 或
原生 FP4 数值验证。

仍需 A5 验证共同 host 存储能力、ACL ABI/队列顺序、MemFabric 动态地址、真实
FP4 格式/FFN、EP、完整 decode graph、显存峰值和 H2D/F 重叠时间线。上述
硬件结果未在本机取得，M0–M5 的未验收状态保持不变。
