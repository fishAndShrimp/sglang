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

以下“已验证”指合入前的 Ascend Ops 基线或上游测试；整层 prefill 与 Hot 直算的本次兼容合入尚待 NPU 验收。

- 已验证 A3、DeepSeek V4 Flash ModelSlim W8A8、TP=EP=Attention DP=4；MoE TP、CP、PP 均为 1。要求专家形状一致、量化 offset 为零。
- 原固定槽路径已验证请求并发 1～8、28.125%/50% 缓存及 MTP 开关；MTP 使用 EAGLE steps=2、topk=1、draft tokens=3。基础回归不替代完整精度评测。
- 新增 Decode Hot 直算与执行槽复用：四 rank、无 MTP、`execution_slots` 等于本地专家数且 Hot 加执行槽不超过 1024 时自动启用；默认 28.125% 缓存满足条件。其他配置（包括额外 execution 槽）保留本地已有的 packed 搬运路径，无需额外开关。上游快速对照主要改善 BS1/4，BS8 可能略慢；不保证所有 batch 都加速。
- 共享专家、最前面的 `max(3, num_hash_layers)` 层和 MTP draft 权重常驻。Host 预算每 rank 64 GiB，四 rank 合计 256 GiB。
- 每 rank 上限为 256 个本地专家、1280 个 Hot 槽、8192 个专家 key；有效行 mask 默认容纳 128 个本地 padded/验证 token，不等于请求 BS。
- 不支持与 EPLB、冗余专家、batch overlap、模块 CPU offloader 或 MemFabric HiCache 同时使用。

## 保留的实现

- Prefill 适配自 [codex/moe-prefill-opt-20261102](https://github.com/pachi-33/sglang/commit/2dc7903d0bebe6f003db2149f8a612e641a5e190)：每次全局 prefill forward 入口、任何模型层计算之前，提交前两个卸载层的全量专家权重，使用两个 bank 交替执行。第 i 层的回传及最终合并提交后，再提交同一 bank 的第 i+2 个卸载层；拷贝流等待第 i 层 combine 完成后才能覆盖 bank。dispatch 入口只校验层次顺序，不等待未来层权重；GMM1 前等待当前层 W13，GMM2 前等待当前层 W2。这样当前层的 dispatch / FFN / combine 可以与未来层 H2D 并行，W2 仍在传输时也能执行当前层 GMM1 和激活。每层仍使用原生一次 dispatch、一次 FFN、一次 combine，不按专家 wave 拆分。
- Prefill 是否启动由本轮 batch 的全局 `is_extend_in_batch` 判断；本地 IDLE / DECODE rank 仍参与其他 rank 的 prefill。不会读取上一轮残留的 RuntimeContext phase。每个 chunk 都重新全量预取；Decode / TARGET_VERIFY / 全局 IDLE 不启动 prefill 搬运；Hot 直算仅接入 decode 路径，prefill 仍使用整层 bank 和逐专家计数。
- Prefill 的 ACL 描述符在装配时按 W13/W2 分为两个阶段，每阶段包含对应的 weight、scale 和 offset，并独立记录完成事件。只合并同一源注册 owner / 目标 owner 中源、目标地址都连续的已验证范围；不能跨越 checkpoint FP32 scale 到运行时 BF16 scale 留下的源端间隙。对 32 专家、6 组件的典型 W8A8 布局，192 个 region 可合并为 68 个 ACL entry，实际取决于物理布局。所有原始 region 的越界、目标重叠和映射别名检查仍保留。原生 batch 继续通过 `OpCommand::RunOpApi` 有序提交，等待事件、H2D、完成事件依次入队。AllToAllV 使用紧凑接收布局；AllGather 保留原生 payload / scale 容量和无效行屏蔽。空接收跳过 FFN；同一 copy FIFO 仍保证旧 W2 完成后才覆盖 bank。
- Prefill 用一次 Triton 操作完成计数格式转换、错误屏蔽和访问统计，GMM 按完整本地专家顺序读取 bank。没有额外的专家打包、激活重排或重复量化；FFN 使用实验分支现有的 W13 → 激活 → W2 实现。
- 每 rank 有 E 个本地专家时，A 使用 Execution 前 E 槽；B 优先借用 Hot 前 E 槽。Hot 不足 E 时，需在 offload JSON 显式设置 `execution_slots >= 2*E`，由 Execution 后 E 槽提供 B。默认 `execution_slots=E`，容量不足直接报错，不额外分配权重池。
- 最后一个 prefill combine 之后回填借用的 Hot 前缀，异常退出也会在清理前回填并排空流。仅恢复被借用的槽位，保留缓存 owner / generation / LFU 状态，随后才执行缓存维护与 decode。
- Decode 在符合条件时直接读取 Hot；miss 只在执行槽未保留同层同专家权重时从 Host 搬入，使用完整专家 key 防止跨层误复用。其他配置保留固定执行区、活跃专家 W13/W2 和 scale 的稳定打包与清零尾部计数。有效 token mask 排除 padding 和空 rank。
- 每次全局 prefill 入口先使 execution key 失效，再开始整层预取，避免 decode 复用被 prefill 覆盖的权重；本地 IDLE / DECODE rank 同样处理。GMM 的物理槽索引与激活的逻辑专家计数分开，prefill 始终使用自己的计数。
- 融合准备搬运地址、逐专家计数和访问统计，缓存更新后统一发布地址表；跳过已校验为零的 offset，各权重/scale 分别使用相同计数调用 `group_pack_copy`。
- 独立流搬运 W13/W2，W2 与第一段专家计算重叠，第二段计算前等待完成。
- 周期 LFU、固定槽缓存搬运、融合缓存更新及维护图重放。Hot 直算路径跳过命中的 D2D；packed 路径将命中专家 D2D 到执行区，miss 从 Host 搬入。

## 本次合入验证

CPU 契约测试：`python3 test/registered/unit/offloading/test_ascend_prefill.py`，覆盖 bank 容量、异步读写顺序、跨 forward event、逐专家计数、部分拷贝失败后的 Hot 恢复和 Prefill→Decode 切换。计数内核使用 Triton 的 CPU 解释模式执行生产函数；需要安装 Triton / NumPy。

本地隔离 serving / NPU 依赖后通过 31 项 CPU 契约测试；关键回归在旧实现上失败。测试执行真实生产 Python 生命周期与描述符逻辑，设备流 / collective / 内核使用测试边界。覆盖：未来层未就绪时当前层仍能完成计算并提交后续预取、W13 与 W2 各自的读取依赖、空 rank、独立通信流、combine 后 bank 复用、第二阶段提交失败时的 Hot 恢复，以及 decode 路径保持原有依赖。连续范围合并测试检查实际字节结果、源/目标间隙、owner 边界、越界和别名拒绝。CPU 测试不证明 NPU 数值或性能收益。

### 2026-11-04 trace 定位

输入为 `trace_2407_2471ms_analysis.html`，NPU 4、`EXTEND bs=1 toks=2048`、EP=8。时间以下均相对 2407 ms 窗口起点。HTML 的 track 7 实际为物理 stream 31（HCCL），主计算流是 track 3 / stream 32，DMA 是 stream 39。

| 观察 | 实测时间 | 含义 |
| --- | --- | --- |
| 主计算流等待未来权重 | 19.359–36.268 ms，16.909 ms | H2D 完成事件在 36.267 ms，随后 counts AllGather 才能启动 |
| 下一次同类等待 | 47.518–68.141 ms，20.623 ms | 跨出 64 ms 窗口；不能将整段加入窗口内统计 |
| 一层完整 H2D | 8.742–36.267 ms，27.525 ms | 192 个 DMA，计算气泡期间拷贝仍忙 |
| 下一层 H2D 启动 | 40.664 ms | 两批之间约 4.398 ms 的拷贝空档可通过恢复流水减少 |
| 两次回传 AllToAllV | 0.528 / 0.480 ms | 条目缩短，但未来权重等待被提前到了 dispatch 前 |
| 两次 native 队列任务 | 0.987 / 1.063 ms | Python 快速返回仍不能消除共享 launch worker 阻塞 |

按窗口裁剪后，Computing / Communication / Free 分别是 17.756 / 2.161 / 44.083 ms，合计 64 ms。DMA 活跃约 54.972 ms，与这些分类重叠，不能相加；Free 并不表示 DMA 空闲。根据代码布局推算，一层每 rank 约搬运 769.5 MiB（含量化元数据），27.525 ms 对应约 27.3 GiB/s；trace 没有直接给出拷贝字节数，此带宽是推算值。

本次取消 `dispatch(i) -> 等待 W(i+1)` 的过早依赖，将权重截止点恢复到实际使用它的投影。此举允许当前 combine 提前完成，使下一批 H2D 能更早排入 FIFO；同时用分阶段事件重叠 W2 和 GMM1，并合并连续范围减少提交开销。H2D 与通信会重新重叠，HCCL 条目可能变长，应比较稳定态 forward / TTFT 和完整层周期，而非单独优化通信条长度。27.5 ms 的整层传输表明当前也有带宽约束，不能承诺调度修改消除全部气泡。

目标设备验收应同时确认：当前 dispatch 不再等待未来 ticket；GMM1 不读取未完成的 W13、GMM2 不读取未完成的 W2；相邻 H2D 批次的空档是否缩短；native 提交耗时是否降低；两种 prefill 通信模式、空 rank、prefill→decode 数值和 Hot 恢复正确。`serve_tp4.sh` 不需要新增参数。

原生扩展需要重新构建：`python3 -m sglang.srt.offloading.moe.npu.native_acl --build-directory "$RUNTIME_DIR/native_acl"`（先按启动脚本设置环境）。`serve_tp4.sh` 已在每次启动时调用此构建步骤，无须修改启动参数。旧扩展的 `torch_npu_stream_flush` 协议会被拒绝，避免新 Python 静默加载旧提交路径。原生异步任务出错会由 torch_npu 队列在后续检查 / 同步时上报，owner 保留搬运资源直到流排空。

本地主机没有 `torch_npu`，原生 ACL H2D、NZ bank 视图、AllToAllV / AllGather 数值、空 rank、prefill→decode 图重放与 TTFT 仍需在目标 NPU 验收。可继续使用上述请求检查脚本；请同时对比 offload 关闭时的输出与真实设备时间线。

INT8 专家激活遵守模型的 `swiglu_limit`，截断后再量化；该正确性修复与搬运优化分开对照。

### 上游 Hot 直算合入验收

来源为 fishAndShrimp/sglang 的 `f2f42f6c85`（INT8 SwiGLU clipping）和 `67d1f62fdf`（Hot 直算与执行槽复用）。合入保留当前 prefill 事件依赖及本地 packed 回退路径。

本次 CPU 回归额外验证全局 prefill 在提交前清除 execution key（包含本地 IDLE / DECODE）、decode 保留 key，以及激活按实际 GMM 分组格式选择逻辑计数。移除失效操作或恢复错误的计数选择后，对应用例均失败。

目标 NPU 上运行 `test/manual/test_ascend_clipped_int8_activation.py` 和 `test/manual/test_ascend_offload_fixed_slots.py`，并检查 prefill→decode、空 rank、额外 execution 槽、MTP 和大缓存回退。CPU 契约只能验证生命周期和调度选择，不能替代真实权重搬运、GMM 数值或 Graph 验收。

## GLM-5.2 ModelSlim W8A8 / W4A8

GLM-5.2 的 offload 接入已实现，尚未在目标 NPU 上完成整模型数值、Graph 和性能验证。上文 DeepSeek V4 的已验证结果不能直接作为 GLM 验收结果。本节使用独立启动脚本，不修改原 `serve_tp4.sh`，也不继续调整 prefill 调度。

支持 checkpoint 中的 `W8A8_DYNAMIC` 与 `W4A8_DYNAMIC`，均使用 INT8 激活。`--quantization modelslim` 根据 checkpoint 的量化描述选择格式，无须增加 W4/W8 启动开关。W4 支持 per-channel 和 metadata 声明 `group_size` 的 per-group；W13/W2 量化格式、分组和激活配置须一致。不支持 `W4A8_MXFP`、`activation_use_clip=True` 或混合 W4/W8 专家投影。W8 仍要求量化 offset 为零；W4 保留并加载其 scale、offset、`scale_bias` 及 per-group second scale/offset，按 resident 转换生成运行时权重与 scale/bias。

GLM 按实际 routed MoE 层分布建立专家目录，最前面的 routed MoE 层、dense 层及共享专家常驻。W4 decode 使用 packed 执行区路径；W8 在满足原有限制时可使用 Hot 直算。Prefill 沿用当前的双 bank 全层预取和 W13/W2 分阶段等待。

在仓库根目录启动 W8A8：

```bash
MODEL_PATH=/models/GLM-5.2-W8A8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
HOST_POOL_GIB=192 CACHE_RATIO=0.25 \
bash scripts/moe_offload/serve_glm52_tp4.sh
```

W4A8 per-channel 使用同一脚本，并显式调整 Host 预算：

```bash
MODEL_PATH=/models/GLM-5.2-W4A8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
HOST_POOL_GIB=96 CACHE_RATIO=0.25 \
bash scripts/moe_offload/serve_glm52_tp4.sh
```

`HOST_POOL_GIB` 默认 192，单位是 **每 rank 的 GiB**。脚本不从目录名推断量化位数；W4 的 96 GiB 仅为 per-channel 的配置参考。Per-group 还要计入 INT64 second scale 和 offset 等 metadata，例如该模型 group size 128 时估算约需 100 GiB/rank，96 GiB 不够。请按启动日志和实际 schema 核算调大；预算不足时 loader 会明确报告所需 GiB 并拒绝启动。

按本次 GLM-5.2 配置、TP=EP=4 估算，卸载专家的权重本体约为 W8 的 **166.5 GiB/rank**、W4 的 **83.25 GiB/rank**；这些数字尚未包含 scale、bias、offset、转换所需的 Host extent 和分配对齐，也不代表整个服务进程的内存。四个 rank 的 Host 池预算相加分别为 768 GiB（192 × 4）和 384 GiB（96 × 4），此外还要给常驻权重、checkpoint 读取和系统内存留空间。实际配置与 checkpoint 不同时，以 loader 预算核算为准。

默认脚本设置 `--attention-backend ascend --page-size 128`，TP=EP=Attention DP=4，MoE TP/CP/PP=1，服务模型名为 `glm52-ascend-offload`。使用独立的 `RUNTIME_DIR`（默认 `/tmp/sglang-glm52-offload-runtime`）和 `LOG_DIR`；需要与其他服务并行时显式更换 `SERVER_PORT`、`DIST_PORT` 和设备列表。

### GLM-5.2 MTP

`ENABLE_MTP=1` 开启同 checkpoint 内的单层 NextN。沿用 DSV4 offload 的执行方式：主模型的 routed experts 继续卸载；draft 权重全部常驻 NPU，不占用主模型 Host/Hot 池；主模型的 target verify 走 packed 路径，关闭 Hot 直算。Verify 不触发整层 prefill 预取，draft 也不参与主模型的缓存维护。

```bash
MODEL_PATH=/models/GLM-5.2-W8A8-with-MTP \
ENABLE_MTP=1 HOST_POOL_GIB=192 CACHE_RATIO=0.25 \
bash scripts/moe_offload/serve_glm52_tp4.sh
```

W4 主模型使用同一开关，Host 预算仍按上面的实际格式设置。脚本固定 EAGLE `steps=2 / topk=1 / draft_tokens=3`，target 与 draft 均使用 `ascend` attention、同一个 `MODEL_PATH`。默认 `ENABLE_MTP=0`。

默认 `MTP_DRAFT_QUANTIZATION=unquant`，对应仓库 GLM-5.2 NPU 部署示例中的不量化 draft；它不会继承主模型的 W8/W4 量化。必须使用实际包含 MTP 权重的 checkpoint。启动前的 `glm52_mtp_config.py` 会读取配置及 safetensors 的名称/头部信息，在分配主模型和 Host 池前检查 MTP 层是否存在，并拒绝将量化 draft 权重按浮点权重加载。若 draft 本身按 ModelSlim W8A8/W4A8 导出，显式设置 `MTP_DRAFT_QUANTIZATION=modelslim`，由 draft 层自己的量化描述选择 scheme。量化 draft 的 routed experts 支持 `W8A8_DYNAMIC` / `W4A8_DYNAMIC`，其余可量化 linear 支持 `FLOAT` / `W8A8_DYNAMIC`；静态 `W8A8` 未纳入该启动预检范围。暂不支持混合量化且 routed experts 为 FLOAT 的 draft、量化 `eh_proj`，也不支持本脚本中指定另一个 draft checkpoint。

不量化 draft 还需要额外 NPU 显存；`HOST_POOL_GIB` 仅预算卸载专家的 Host 存储，不能代替这部分显存。MTP 接入已做 CPU 契约与启动参数检查，尚未进行 GLM 的 NPU 整模型精度、accept length、Graph replay 或性能验收。实机上分别以 `ENABLE_MTP=0/1` 运行 `test_requests.py` 与 `test_requests.py --bench`，检查短请求、空 DP rank、chunked prefill→verify、缓存维护和并发 1/4/8，再比较相同 checkpoint 的输出与性能。

默认 [initial_profile_glm52.json](initial_profile_glm52.json) 的 `layers` 为空，表示所有专家初始热度为零，按既有确定性规则初始化 Hot；它不是采集得到的访问统计，也不代表推荐的热点分布。可用 `OFFLINE_STATS_PATH=/path/to/glm52-profile.json` 覆盖为该 GLM 模型的实测 profile，不应复用 DeepSeek V4 的 `initial_profile.json`。脚本生成的 `offload.json`、profile 和日志路径都会写入启动输出。

目标 NPU 验收前，先在相同 CANN / torch_npu / MemFabric 环境运行单设备格式与 GMM 对照：

```bash
MOE_NATIVE_ACL_DIR=/tmp/sglang-glm52-offload-runtime/native_acl \
python3 test/manual/test_ascend_glm_offload_formats.py --device 0
```

该脚本使用合成 checkpoint 分片，检查 resident 与 offload Host repack / ACL reload 后的投影结果，覆盖 W8、W4 per-channel 和 per-group；不需要下载模型。它不替代四 rank 的模型加载、长短请求、空 DP rank、prefill→decode、缓存维护和 Graph replay 验证。整模型验收还应与关闭 offload 时的同一量化 checkpoint 对照输出和精度，再评估 TTFT/TPOT。
