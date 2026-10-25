# MoE offload 维测日志

默认每个服务 forward 输出一条异步 batch 汇总，各 EP rank 独立记录。日志带有
`seq`、`phase`、`mode`、实际 batch/token 数和 decode bucket；后台输出可能稍有延迟，
应按 rank/seq 关联，不依赖跨进程日志的打印顺序。

```text
MoE offload batch: rank=0 device=npu:0 seq=42 phase=decode mode=DECODE bs=8 tokens=8 bucket=8 assignments=100 hits=75 misses=25 hit_rate=75.00% error=0
```

`assignments/hits/misses` 复用已有 `CacheRouteStats`，统计本 batch、全部 offloaded 层、
dispatch 后本 rank 收到的有效 token-expert 路由。重复使用同一专家逐次计数；
`hit_rate=hits/assignments` 是路由加权命中率，不能解释为去重专家命中率或 H2D 字节节省率。
常驻 routed/shared experts、padding、warmup/capture 不计入服务批次。多个 rank 的全局
命中率需先分别求和分子和分母，不能平均百分比。

prefill 固定读取 host master，三个命中计数字段及命中率显示 N/A；decode 无有效需求时
命中率显示 N/A。非零 `error` 使用 ERROR 级别并将命中率显示为 N/A，不把故障屏蔽后
的计数当成健康命中率。故障仍由原有 runtime/owner 路径处理，日志不会清除错误标记。

JSON 可增加以下可选配置：

```json
{
  "batch_log_interval": 1,
  "batch_log_capacity": 64
}
```

- `batch_log_interval=1`：每个服务 forward 记录一条；N 表示每 N 个 forward 记录一次，
  序号包含 prefill、decode 和 idle；0 关闭 batch 统计回传和输出。
- `batch_log_capacity=64`：最多保留 64 条待完成/待打印快照。固定 pinned host 存储为
  `capacity * 32` 字节，计入 metadata 账本。缓冲饱和会丢弃新诊断记录并报告累计丢弃数；
  不阻塞、不丢弃推理 batch。可增大容量或提高 interval 降低日志压力。

日志只在 Graph replay/eager forward 提交之后读取既有计数，在当前 compute stream 上
向固定 pinned host slot 排入三个 int64 计数和一个 int32 error 的异步拷贝。使用 runtime
原有 fresh `output_ready` completion，后台线程仅 query；不创建 NPU stream/event，不增加
wait、全设备同步或 collective，不变更 D/F/C、权重搬运、缓存更新及 phase 切换顺序。
快照在下一 batch 清零计数之前提交，slot 在完成后才复用。关闭 runtime 时复用已有
device drain，再结束日志线程。分配/回传/query 失败会停用 batch 日志并告警，保留已提交
快照直到原有 drain，避免因诊断失败中断正常推理。

仍有少量异步 D2H 和 CPU 日志开销；以上约束保证不以观测新增同步屏障，不代表零性能开销。
metadata 预算也必须容纳诊断池。

INFO 生命周期日志包括 runtime/bank 容量、初始 cache、内存计账、model hooks、Graph
bucket 捕获及 prefill/decode 切换。DEBUG 明细包括层注册、forward 提交、D/F/C 提交、
wave 专家范围/bank、prefill 首组提交量，以及 ACL batch 字节数。
`forward.begin` 同时记录 EP rank、本地 `mode`、共享 `extend_in_batch`、
`global_work` 和解析后的 `phase`；Attention DP 的空闲 rank 在全局 prefill 时应显示
`mode=IDLE phase=prefill`，本地无请求不代表 dispatch 后没有专家需求。`memfabric.emit` 和
decode wave 的 Python DEBUG 日志发生于 warmup/capture，不会每次 replay 重跑；实际
服务批次的命中率以异步 batch 汇总为准。

`submit/emit` 表示工作已提交或图节点已发出，不表示设备已完成。日志不打印 Tensor、
不读取逐层设备 counts；跨流性能与设备时间仍应通过目标机 NPU trace 验证。

启动脚本已显式配置 batch 日志默认值，默认保存控制台日志与配置快照。可用
`MOE_LOG_LEVEL=debug` 单独打开本包 DEBUG；全局级别仍由 `LOG_LEVEL` 控制。
请求 trace、NPU capture profile 与目录参数参见 [启动脚本说明](README.zh-CN.md)。
