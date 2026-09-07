# QSA NPU 优化开发记录

这份文件记录 Qwen3.8-Flash-Next QSA 在 NPU 上的算子优化过程。它不是稳定 API
文档，也不是项目规范；每次继续优化、修复 bug 或改变 dispatch 契约时，都应在这里
新增一节，保留当时的背景、意图、影响、测试遗漏原因和复盘。

需要特别区分：下面并非每一项都是新写的 Triton kernel。decode MQA 是新 Triton
kernel，block expansion 是让既有 Triton kernel 支持 NPU；prefill top-k 第一版只是
批量 Torch 优化，用于验证瓶颈和语义，尚未实现为 Triton。

## 1. 新增 NPU QSA MQA decode Triton kernel

### 【1】背景介绍

QSA indexer 在 decode 阶段，需要让少量 query head 与分页保存的 compressed key
cache 做 MQA 打分。原先 NPU 只能进入 `torch_qsa_mqa_decode`：先根据 page table
收集完整 K，再用多个 Torch 算子计算四个 head 的 dot、ReLU 和求和，最后构造固定
长度 logits。

### 【2】意图分析

目标是减少临时张量和算子 launch：直接在 Triton program 内读取 page table，定位
paged K cache，完成四个 head 的 FP32 dot、ReLU、求和和缩放，并将越过
`context_lens` 的位置写成负无穷。

实现只覆盖已经验证的 Qwen3.8 shape：4 个 query heads、head dim 128、page size
16、BF16/FP16。其他合法 shape 继续走 Torch reference，而不是让 kernel 猜测。

### 【3】代码修改具体带来的影响，为什么能改善原路径

新增：

- `can_run_qsa_mqa_decode`：声明 kernel 能力边界。
- `_triton_qsa_mqa_decode_kernel`：融合 paged gather 和 MQA score。
- `triton_qsa_mqa_decode`：验证输入、分配固定宽度输出并发射 kernel。
- `mqa.qsa_mqa_decode` 的 NPU dispatch：支持时走 Triton，否则走 Torch fallback。

它避免物化完整 gathered K 和中间 `[batch, tokens, heads]` scores，主要影响 decode
延迟/TPOT，不会自动优化 prefill。

### 【4】为什么先前写法存在性能问题，却仍能通过 test

先前 Torch 版本是语义正确的 reference，所以正确性测试能够通过；测试不会因为
实现包含多次 gather、多个临时张量或较多 launch 而失败。只有在真实 context length
和服务循环里测延迟，才能判断它是否值得融合。

新 kernel 的对拍也只能证明数值和 dispatch 正确，不能单独证明端到端 TPOT 必然
提升。后来的 10-request ShareGPT 结果中 TPOT 波动较大，说明还需要更多重复实验和
算子级 profile，不能仅凭一次结果宣布收益。

### 【5】思考：以后如何避免踩坑

- 在命名和汇报中明确阶段：`decode score` 不等于“整个 QSA 已 Triton 化”。
- 能力判断由 kernel 侧集中维护；合法但不支持的 shape 才 fallback。
- 同时验证数值、固定输出 shape、无效位置的负无穷和真实 paged layout。
- 性能结论同时需要 cold/warm kernel 数据和端到端多次 benchmark。

## 2. 在 NPU 上启用既有 block-index expansion Triton kernel

### 【1】背景介绍

top-k 返回 compressed block indices 后，需要将每个 block 按 compression ratio 展开
成 token indices，并追加未形成完整 block 的 causal tail。仓库已有
`triton_expand_qsa_block_indices`，但原 dispatch 只让 CUDA 使用，NPU 落到 Torch
版本；Torch 版本还包含排序和多个中间张量。

### 【2】意图分析

既有 Triton kernel 的计算只依赖普通 load/store、整数索引和 mask，不包含 CUDA
专属 intrinsic。经独立 NPU 对拍后，让 NPU 与 CUDA 共用这一个 Triton 实现，避免
再复制一份同语义 kernel。

### 【3】代码修改具体带来的影响，为什么能改善原路径

dispatch 从 CUDA-only 改为 CUDA/NPU Triton fast path；Torch 实现继续作为 CPU
reference。NPU 因此不再执行 Torch expansion 中的 expand、where、cat、argsort 和
gather 链。

这只优化“top-k block 到 token”的转换，不负责生成 score、做 top-k 或执行真正的
sparse attention，所以无法单独解决 TTFT。

### 【4】为什么之前没有在 NPU 使用，却仍能通过 test

Torch expansion 与 Triton expansion 语义一致，因此原来走 Torch 也能通过正确性
测试。此前缺少的是 NPU 上的性能路径覆盖，而不是结果正确性。

### 【5】思考：以后如何避免踩坑

- 先判断 kernel 使用的 Triton primitive 是否为后端通用，再实机编译和对拍。
- dispatch 改动必须测试实际调用路径，而不只是直接调用 kernel wrapper。
- 明确每个小 kernel 在完整 pipeline 中所占比例，避免把局部优化描述为整体优化。

## 3. Prefill top-k 第一版：将 NPU 逐行 Torch 调用改为批量调用

### 【1】背景介绍

NPU prefill 的 `qsa_fast_topk` 原先在 Python 中遍历每个 query row，并逐行执行 mask、
`torch.topk`、相对索引转换和写回。decode 的 rows 通常接近 batch size，所以 TPOT
看起来正常；prefill 的 rows 接近 prompt token 数，Python 循环会发射成百上千组小
算子，显著放大 TTFT。

微基准中，2048×2048、top-k 512 的逐行路径 warm latency 约 774 ms；一次批量
`torch.topk(..., dim=1)` 约 0.57 ms。这证明主要问题是逐行 launch，而不是
`torch.topk` 算子本身。

### 【2】意图分析

第一版先采用批量 Torch，而不是立即写 Triton：一次构造所有 row 的 valid mask，
一次做二维 top-k，再批量转换为相对索引。这样以最小改动验证 TTFT 瓶颈，同时保留
CUDA/ROCm 原有 fast-topk 路径。

### 【3】代码修改具体带来的影响，为什么能改善原路径

在 `_is_npu` 分支中把 `for row in range(...)` 改为广播 mask 和一次二维 top-k。
微基准修改后，2048×2048 的实际 `qsa_fast_topk` warm latency 从约 774 ms 降到
0.74 ms。

这项修改不是 Triton kernel；它是 NPU prefill dispatch 的批量 Torch 优化。它解决
indexer top-k 的 launch 开销，但尚未解决 `qsa_sparse_attention_reference` 中另一处
逐 query row Python 循环。

### 【4】为什么第一版仍能通过 test，完整服务却失败

第一版直接返回了二维 `torch.topk` 结果，其宽度是：

```text
select_width = min(topk, logits.shape[1])
```

已有 NPU test 使用 `logits.shape[1] > topk`，因此 `select_width == topk`，返回 shape
恰好正确。真实服务 warmup 出现短 prefill：122 个 query rows、只有 30 个 compressed
keys，但配置要求固定 `topk=512`。第一版于是返回 `[122, 30]`，而下游
`expand_qsa_block_indices` 根据稳定接口要求 `[122, 512]`，抛出：

```text
ValueError: expected block indices [M, 512], got (122, 30)
```

测试只检查了“每一行选对哪些元素”，没有覆盖 API 的固定宽度契约，也没有覆盖
`number_of_keys < topk`。因此数值测试通过，但 shape contract 已被破坏。

### 【5】思考：以后如何避免踩坑

- 优化现有函数前，先列出完整契约：dtype、device、shape、padding、排序和无效值。
- 不只测典型/大 shape；至少覆盖空输入、`keys < topk`、`keys == topk`、
  `keys > topk`、不同 row start/length 和 ragged batch。
- reference 对拍应同时断言 shape 和 dtype，不能只比较有效元素。
- 微基准输入要来自真实服务日志或 metadata，而不能只用方便的方阵。
- 完整服务 warmup 是 kernel 单测的补充，不能由单测替代。

## 4. 修复短 prefill 下固定宽度 top-k 契约

### 【1】背景介绍

完整 `serve.sh` 首请求复现了上述 `[122, 30]` 与 `[M, 512]` 不匹配。QSA 下游要求
top-k 始终返回固定宽度；当有效 keys 不足时，剩余位置必须填 `-1`。

### 【2】意图分析

保留批量 top-k 的性能收益，但恢复原函数的稳定输出契约。已有代码已经预分配
`output = full([rows, topk], -1)`，因此批量结果只应写进它的前 `select_width` 列，
而不是直接作为函数结果返回。

### 【3】代码修改具体带来的影响，为什么能修复先前的 BUG

修改为：

```text
output[:, :select_width] = batched_result
return output
```

当 keys 为 30、top-k 为 512 时，前 30 列保存有效选择或行内 padding，其余 482 列
保持 `-1`，最终 shape 恒为 `[rows, 512]`。因此满足 block expansion 的输入契约，
同时仍只有一次批量 top-k。

### 【4】为什么之前的写法没考虑到，修复后怎样补上 test

之前把“top-k 的计算宽度”误当成了“函数输出宽度”，并沿用了只覆盖 keys 多于 top-k
的测试数据。修复后将 NPU 对拍测试改为 `topk=32`、keys=17，明确断言固定宽度输出
和尾部 `-1` padding。修复后的 NPU 定向结果为 5 passed。

完整服务随后成功完成模型加载、graph capture、128-token warmup 和首个请求，并输出
`The server is fired up and ready to roll!`，不再出现 shape mismatch。

### 【5】思考：写 Triton/加速路径时如何不再踩坑

- “支持的计算宽度”和“公开输出 shape”是两个概念，必须分别建模。
- padding 不是异常 fallback，而是这个 QSA fixed-width 接口的组成部分。
- 每次从循环改成批处理时，逐项核对循环原本隐式完成的行为；本例中循环写入预分配
  output，天然保留尾部 `-1`，直接 return 批量结果时丢掉了这一行为。
- 测试矩阵必须从生产契约推导，不应从当前实现推导。
- 后续每次修改 QSA Triton 或相关 dispatch，都继续追加本文件，记录失败输入的真实
  shape 和对应回归测试。
