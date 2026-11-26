# 在 A3 上复现 Qwen3.8-Flash-Next 的双 PR 性能

## 运行前先准备两个 PR

在 CANN 9.2.0 的 SGLang 镜像里运行这个模型，**两个仓库都要用新 PR**。

**1. `sglang` 框架有两种选法，任选一种。**

- **用框架 PR。** 把 SGLang 仓库切到 PR #52107 的定版提交 `75acaa1cd9`。要照本文测速，把交接分支里的整个 `benchmark/qwen38_flash_next` 目录复制到这个仓库的 `benchmark/` 下。[框架 PR #52107](https://github.com/sgl-project/sglang/pull/52107)
- **用交接分支。** 直接用 `repro/qwen38-flash-next-a3`。从提交 `23f7640164` 起，它就包含了上述 PR 的定版代码，启动脚本、测速脚本和数据集也都在里面。[交接分支](https://github.com/fishAndShrimp/sglang/tree/repro/qwen38-flash-next-a3)

两种选法用的是同一份框架代码，都可以配合第 2 步的算子复现下文的 TP8 性能。启动服务和运行官方测速时，两个终端都进入所选仓库的根目录。下文的 `/path/to/sglang` 请换成这个根目录的绝对路径。

```bash
export PYTHONPATH="/path/to/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
```

**2. `sgl-kernel-npu` 用算子 PR #1076。** 进入这个算子仓库的 `af22f597e9` 版本根目录，用启动服务的同一个 Python 环境重新编译并安装算子。[算子 PR #1076](https://github.com/sgl-project/sgl-kernel-npu/pull/1076) · [对应源码](https://github.com/sgl-project/sgl-kernel-npu/tree/af22f597e9c859b86a2c3a5acf469d98ceb87043)

```bash
python3 setup.py --arch-version 2201 --third-party ops_transformer,ops_transformer_gmm93
python3 -m pip install --no-deps --force-reinstall output/sgl_kernel_npu-*.whl
```

装好后，回到第 1 步选的 SGLang 仓库，按下面的 TP8 或 TP16 命令启动服务。

## 启动 TP8 / EP8（8 个 die）

把权重路径换成本机的实际值，选择同一台 A3 上 4 张空闲物理卡的全部 8 个 die。以下命令都从 SGLang 仓库根目录执行。

```bash
export PYTHONPATH="/path/to/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=/path/to/Qwen3.8-Flash-Next
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash benchmark/qwen38_flash_next/serve.sh
```

脚本使用 BF16、NEXTN 三步和 32 个请求槽。Prefill 按 8192 token 分块，indexer 按 TP 切分，GBSA 分核表在主模型进层前准备。PLE 卸载到 pinned 主存，KV 池为 573440 token。MoE 的 prefill 用 AllGather，decode 用 MC2。Prefill delayer 上限为 30 轮，前缀缓存和图像 embedding 缓存均关闭。

## 启动 TP16 / EP16 / Attention DP2（16 个 die）

装好算子后，在所选的 SGLang 仓库根目录运行。把权重路径换成本机的实际值，选择同一台 A3 上的 16 个空闲 die。

```bash
export PYTHONPATH="/path/to/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=/path/to/Qwen3.8-Flash-Next
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
bash benchmark/qwen38_flash_next/serve_tp16.sh
```

脚本传入 `--tp-size 16 --ep-size 16 --attn-dp-size 2`，启动一个服务。MoE 使用 16 个 die，attention 分成两组，每组 8 个 die。共配置 64 个请求槽，每组 32 个。Prefill 分块参数为 8192，按组折算为 4096。PLE 使用 pinned 主存，MoE prefill 用 AllGather，decode 用 MC2。

等服务就绪，在第二个终端按文末的官方测速命令测 A032 或 A064。两个启动脚本的默认端口都是 30000，自定义时在两个终端设置相同的 `PORT`。TP16 沿用历史配置，prefill delayer 默认关闭，图像 embedding 缓存沿用框架设置。当前定版的整网实测覆盖 TP8。历史 TP16 纯文本 C32 的 94.47 秒对应当时的补丁版本。[历史 TP16 步骤](https://github.com/fishAndShrimp/sglang/blob/baa6fdc8515d02bd3e7c5f38eef204942a979a98/benchmark/qwen38_flash_next/README.md#4-单机-16-die-文本配置)

## 获取、解包和查看数据集

**我们提供了哪些输入？** 数据集分为 **A、B、C、D 四组**，每组 256 条请求，四组之间的请求互不重复。合计 **1024 条**，其中 512 条是纯文本，512 条是文本加图片。

每组选多少条，就取这组的前多少条。下面的数量对 A、B、C、D 每一组都适用。

| 从一组取出的请求数 | 纯文本请求数 | 文本 + 图片请求数 |
|---:|---:|---:|
| 32 | 16 | 16 |
| 64 | 32 | 32 |
| 96 | 48 | 48 |
| 128 | 64 | 64 |
| 160 | 80 | 80 |
| 192 | 96 | 96 |
| 224 | 112 | 112 |
| 256（整组） | 128 | 128 |

- `A032` 是 A 组的前 **32 条请求**，包含 **16 条纯文本**和 **16 条文本 + 图片**。
- `D064` 是 D 组的前 **64 条请求**，包含 **32 条纯文本**和 **32 条文本 + 图片**。

每批平均输入约 1.07 万个 token，最长输入约 3 万个 token，每条请求固定生成 4096 个 token。各批的长度统计见 `summary.json`。

**怎么拿到数据？** 如果已经按第一节准备好了 `benchmark/qwen38_flash_next/`，数据就在里面，直接解包即可。只想单独拿数据，也可以克隆交接分支。[GitHub 数据集目录](https://github.com/fishAndShrimp/sglang/tree/repro/qwen38-flash-next-a3/benchmark/qwen38_flash_next)

```bash
git clone --depth 1 --branch repro/qwen38-flash-next-a3 https://github.com/fishAndShrimp/sglang.git
cd sglang
```

**怎么解包？** 在 SGLang 仓库根目录运行下面一行。脚本会自动检查压缩包是否完整，把文件放到 `benchmark/qwen38_flash_next/serving_benchmark_v1/`。

```bash
python3 benchmark/qwen38_flash_next/unpack_mixed.py
```

**看 A032 或测 A032，分别用哪个文件？**

- **自己看内容**，用 IDE 打开 `A032.jsonl`，每行是一条请求。`prompt` 是发给模型的文字，`images` 列出图片。比如第一行是纯文本，第二行有 12 张图，按文件里的路径就能在 `images/` 里找到原图。
- **用官方客户端测速**，选 `openai/A.jsonl`。文末命令用 `--num-prompts 32 --max-concurrency 32` 跑同样的 32 条。想测 A064，把这两个数都改成 64，就会在原来的 32 条后面再加 32 条。

## 复现 TP8 的 115.92 秒参考结果

参考环境使用 Python 3.12、torch / torch-npu 2.13.0 和 Triton 3.6 系列。数据包内的流式客户端先用 D032 最长的一条请求生成 32 个 token 预热，再在同一 TP8 服务上连续测两轮 A032。要对照这组数字，在第二个终端的仓库根目录运行下面的命令。

```bash
python3 benchmark/qwen38_flash_next/bench_mixed.py
```

| 轮次 | 整批 E2E（s） | 平均 TTFT（s） | 平均 TPOT（ms） | 输出 tok/s | MTP 接受长度 |
|---|---:|---:|---:|---:|---:|
| 1 | 116.30 | 11.70 | 23.65 | 1127.03 | 3.037 |
| 2 | 115.54 | 10.47 | 23.75 | 1134.41 | 3.033 |
| 平均 | **115.92** | **11.08** | **23.70** | **1130.72** | **3.035** |

脚本打印两轮均值和逐请求结果的目录。如果 TTFT 明显高于表中数值，先核对服务的请求槽位和 KV 容量，再看满批 decode 时能否达到 `#running-req: 32`、`#queue-req: 0`。

**纯文本历史基线**。在 SGLang 仓库根目录运行 `PYTHONPATH="/path/to/sglang/python${PYTHONPATH:+:$PYTHONPATH}" bash benchmark/qwen38_flash_next/bench.sh`，使用原来的 32 条文本和 SGLang 官方测速客户端。旧版 TP8 的 127.22 秒对应纯文本和旧服务配置。[历史配置与结果](https://github.com/fishAndShrimp/sglang/blob/baa6fdc8515d02bd3e7c5f38eef204942a979a98/benchmark/qwen38_flash_next/README.md)

## 用 SGLang 官方客户端测速

解包数据集并启动 TP8 或 TP16 服务后，在第二个终端进入 SGLang 仓库根目录。下面的命令对两种配置通用，用 `sglang.benchmark.serving` 发送 A 组前 32 条请求（A032），并发 32，每条生成 4096 个 token。

```bash
export PYTHONPATH="/path/to/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -m sglang.benchmark.serving --backend sglang-oai-chat \
  --host localhost --port "${PORT:-30000}" \
  --dataset-name openai --dataset-path benchmark/qwen38_flash_next/serving_benchmark_v1/openai/A.jsonl \
  --num-prompts 32 --max-concurrency 32 --request-rate inf --warmup-requests 0 \
  --output-details --output-file result-A032.jsonl
```

一轮 A032 应成功完成 32 条。结果明细的 `input_lens` 来自服务端，应与 `A032.jsonl` 的 `input_tokens` 逐条对应，含图片合计 **343220**。`output_lens` 应为 32 个 **4096**，合计 **131072**。终端加载时打印的 `#Input tokens` 和结果摘要的 `total_input_tokens` 都使用客户端估算值，用参考 tokenizer 得到 **48510**。

测 A064 时，把 `--num-prompts` 和 `--max-concurrency` 都改为 64，结果写入 `result-A064.jsonl`。TP16 脚本配置了 64 个请求槽。TP8 脚本只有 32 个，向它发送 A064 时部分请求会排队。换组时把数据文件换成同目录的 `B.jsonl`、`C.jsonl` 或 `D.jsonl`。跨机器比较时，两端使用相同的请求组、并发、缓存设置和预热方式。
