# 在 A3 上复现 Qwen3.8-Flash-Next 的双 PR 性能

单机 8 个 die，TP8 / EP8。A032 含 16 条纯文本和 16 条图文请求，32 并发，每条输出 4096 token。定版配置的两轮参考结果如下。

| 轮次 | 整批 E2E（s） | 平均 TTFT（s） | 平均 TPOT（ms） | 输出 tok/s | MTP 接受长度 |
|---|---:|---:|---:|---:|---:|
| 1 | 116.30 | 11.70 | 23.65 | 1127.03 | 3.037 |
| 2 | 115.54 | 10.47 | 23.75 | 1134.41 | 3.033 |
| 平均 | **115.92** | **11.08** | **23.70** | **1130.72** | **3.035** |

## 1. 安装两个 PR

使用 CANN 9.2.0 的 SGLang 镜像。本分支包含框架 PR 提交 `75acaa1cd9` 的代码和本目录的复现工具，配套算子 PR 提交是 `af22f597e9`。参考环境使用 Python 3.12、torch / torch-npu 2.13.0 和 Triton 3.6 系列。[框架 PR #52107](https://github.com/sgl-project/sglang/pull/52107) · [算子 PR #1076](https://github.com/sgl-project/sgl-kernel-npu/pull/1076)

在干净的 `sgl-kernel-npu` 仓库根目录编译并安装 wheel。两个构建选项分别提供稀疏注意力和本次使用的 MoE 分组矩阵乘。

```bash
git fetch https://github.com/sgl-project/sgl-kernel-npu.git refs/pull/1076/head
git switch --detach af22f597e9
python3 setup.py --arch-version 2201 --third-party ops_transformer,ops_transformer_gmm93
python3 -m pip install --no-deps --force-reinstall output/sgl_kernel_npu-*.whl
```

回到 SGLang 的 `repro/qwen38-flash-next-a3` 分支根目录，用同一 Python 环境安装框架。

```bash
python3 -m pip install --no-deps -e python
```

## 2. 起 TP8 服务

把权重路径和 8 个设备号换成本机的实际值。以下命令都从 SGLang 仓库根目录执行。

```bash
export MODEL_PATH=/path/to/Qwen3.8-Flash-Next
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash benchmark/qwen38_flash_next/serve.sh
```

脚本使用 BF16、NEXTN 三步和 32 个请求槽。Prefill 按 8192 token 分块，indexer 按 TP 切分，GBSA 分核表在主模型进层前准备。PLE 卸载到 pinned 主存，KV 池为 573440 token。MoE 的 prefill 用 AllGather，decode 用 MC2。Prefill delayer 上限为 30 轮，前缀缓存和图像 embedding 缓存均关闭。

## 3. 测 A032

服务就绪后，在同机第二个终端、同一 Python 环境运行。默认端口是 30000，自定义时在两个终端设置相同的 `PORT`。

```bash
python3 benchmark/qwen38_flash_next/bench_mixed.py
```

脚本校验并解包数据，用 D032 最长的一条请求生成 32 个 token 预热，再调用包内的流式客户端 `bench.py` 连测两轮 A032。每轮 32 条请求应全部成功，输入合计 **343220 token**，输出合计 **131072 token**。客户端逐条核对服务返回的输入和输出长度。终端打印两轮均值和保存逐请求结果的目录。

如果 TTFT 明显高于表中数值，先看服务是否拿到了 32 个槽位和配置的 KV 容量，再看日志中满批 decode 时能否达到 `#running-req: 32`、`#queue-req: 0`。

## 其他负载

- **纯文本**。运行 `bash benchmark/qwen38_flash_next/bench.sh`，使用原来的 32 条文本和 SGLang 官方测速客户端。旧版 TP8 的 127.22 秒对应纯文本和旧服务配置。[历史配置与结果](https://github.com/fishAndShrimp/sglang/blob/baa6fdc8515d02bd3e7c5f38eef204942a979a98/benchmark/qwen38_flash_next/README.md)
- **16 die**。设置好 `MODEL_PATH` 和 16 个空闲设备号后，从本分支直接运行 `bash benchmark/qwen38_flash_next/serve_tp16.sh`。它启动一个 TP16 / EP16 副本，attention DP 为 2。历史纯文本 C32 的 94.47 秒来自当时的补丁版本。本次定版只测了 TP8。如需复核旧的 TP16 成绩，按历史说明使用对应工作树及 `tp16-runtime.patch`。[历史 TP16 步骤](https://github.com/fishAndShrimp/sglang/blob/baa6fdc8515d02bd3e7c5f38eef204942a979a98/benchmark/qwen38_flash_next/README.md#4-单机-16-die-文本配置)
- **其他图文组或官方客户端**。A、B、C、D 四组请求互不重叠，每组可取 32 到 256 条，步长为 32。可用下面的命令导出并测 A032。基准表使用上一节的客户端及预热方式。

```bash
cd benchmark/qwen38_flash_next
python3 unpack_mixed.py
python3 -m sglang.benchmark.serving --backend sglang-oai-chat \
  --host localhost --port "${PORT:-30000}" \
  --dataset-name openai --dataset-path serving_benchmark_v1/openai/A.jsonl \
  --num-prompts 32 --max-concurrency 32 --request-rate inf --warmup-requests 0 \
  --output-details --output-file result-A032.jsonl
```

`serving_benchmark_v1/A032.jsonl` 和 `D032.jsonl` 可直接用 IDE 阅读。测其他并发时，同时调整请求数、并发和服务槽位。比较机器时使用同一组请求。
