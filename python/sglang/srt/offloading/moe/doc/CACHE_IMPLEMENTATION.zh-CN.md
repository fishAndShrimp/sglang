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
运行时只消费接口，不读取策略的私有 config、catalog 或内部计数器。

`due_out` 是 caller 预分配并计入 metadata 预算的独立 bool 标量，不能与策略内部
`update_due` 别名。Coordinator 完成 observation/selection/finish 后将其交给
`prepare_update(..., update_enabled=due_out)`；planner 与 runtime guard 相与并锁存。
关闭的 gate 不执行有效缓存事务，也不因陈旧无效 targets 产生错误。

Full decode graph 包含策略与更新尾部。当前采用 masked fallback：非 due 的固定选择、
规划节点仍可执行，不等于 kernel 真早退。设备条件执行或融合优化须另行实机验证；
不维护 host 周期镜像，也不通过设备数值回读来选择图。

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

CPU 回归覆盖生产装配、due 锁存、禁用事务、原 slot 恢复、错误阻断、读者保护与
共享 pipeline。旧重复栈测试的有效覆盖迁至这些生产 fixture；具体结果见重构记录。
CPU tensor/事件替身不证明 NPU graph 或异步搬运正确，目标机仍需 full graph、跨流、
真实字节/数值、多 rank EP、内存峰值及连续模式切换验收。
