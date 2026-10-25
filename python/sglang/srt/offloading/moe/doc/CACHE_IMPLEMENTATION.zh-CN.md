# 缓存实现与设备事务契约（2026-10-25）

当前唯一实现是 [residency.py](../residency.py) 的 `TensorResidencyOps`，由
`NpuMoEBackend` 装配。重复 cache backend/runtime/residency 已退休；迁移记录见
[重构记录](REFACTOR_20261025.zh-CN.md)。原生默认 periodic 配置见
[周期缓存说明](../npu/doc/PERIODIC_CACHE_1022.zh-CN.md)。

## 策略输出

所有策略使用同一接口：`observe(state, step_counts, update_enabled, due_out)`、
`select_hot_set(state, key_to_slot, target_keys)`、`finish_step(state)` 和 `copy_state`。
`target_keys[K]` 是唯一逻辑集合契约：有效 key 不重复、空位 -1，位置不指定物理 slot。
`TensorLFUPolicy` 每个有效步输出 due；`PeriodicLFUPolicy` 维护设备周期，按期输出 due。
运行时只消费接口和可选 `graph_maintenance_interval` 声明，不读取策略的私有 config、
catalog 或设备内部计数器。周期策略的在线选入/淘汰允许跨层，不固定离线初始化的各层配额。

`due_out` 是 caller 预分配并计入 metadata 预算的独立 bool 标量，不能与策略内部
`update_due` 别名。Coordinator 完成 observation/selection/finish 后将其交给
`prepare_update(..., update_enabled=due_out)`；planner 与 runtime guard 相与并锁存。
关闭的 gate 不执行有效缓存事务，也不因陈旧无效 targets 产生错误。

周期策略为同一个 bucket 捕获两张完整 decode 图：普通图尾部只 observation，维护图尾部
再执行 selection/finish/prepare/transfer/commit。普通图没有维护排序或规划节点。
host 用现成 batch 元数据统计有效 decode 提交次数，每 `update_interval` 次选维护图；
这不需要回读设备标量。图选择与设备周期都排除 prefill、capture/warmup 和全局空步，
本 rank 无路由但 peer 有 decode 工作时仍推进，跨 bucket 复用同一周期。
新增图共享原有权重和策略存储，其 workspace 由现有 capture peak 计费。
CPU 契约已检查；真实 NPU 双图捕获、切换与内存峰值须实机验证。

## 驻留、统计和快照

Dispatch 后完整本地 counts 驱动 hit D2D / miss H2D，包含零需求和尾波。
临时执行不自动准入热池。完整模型尾部保持幸存者原 slot，失效 victim，搬运完整
bundle 后发布；generation、有效位和共享 sticky error 保留为正确性约束。
有活跃 execution reservation 时禁止热更新；planner 先等待读者，再记录覆盖计算
frontier 的 plan-ready，access 可以据此省去重复 frontier。

`CacheRouteStats` 持有固定地址 accesses/hits/misses，统计有效 assignment，
不是去重专家数或节省字节数。warmup/capture 和全局空步不推进策略；有效 decode
即使本 rank 无路由也推进设备周期。prefill 不改变 decode 热度。

`DecodeStateSnapshot` 仅保存 hot 映射、有效位、generation 和策略状态。恢复先按原
slot 重载所有 bundle，再检查设备错误、恢复持久元数据；execution keys/validity
和 step counts 清零。`update_enabled` 属于当前 invocation，不从快照恢复。

## 所有权与验证

`retain` 只保活，完成事实通过 `register_completion` 显式登记。资源在已有完成证据
后退休；graph capture 资源额外要求成功销毁图。部分提交失败保持保活，直到 worker
停止、全设备流 drain 成功。没有依赖循环 GC 的正常退休流程。

保留的 CPU 测试验证生产装配、周期策略、原 slot 恢复、错误阻断和读者保护。
本次另行新增的六个测试文件已按后续要求删除，删除前覆盖与当前结果见重构记录。
CPU tensor/事件替身不证明 NPU graph 或异步搬运正确，目标机仍需 full graph、跨流、
真实字节/数值、多 rank EP、内存峰值及连续模式切换验收。
