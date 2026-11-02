# Ascend Ops MoE offload 精简评审版

这版将已验证的专家 offload 优化接入主线 Ascend Ops 流程。只需启用 `--moe-offload-config`，保留的优化即全部生效；不需要设置任何试验开关。

保留的实现：

- 每张卡共享一套固定执行区，每个本地专家对应一个位置，跨层复用。
- 使用真实有效 token 的设备侧 mask，避免图 padding 或空 rank 触发多余专家加载。
- 一次融合生成搬运地址、字节数和访问计数；缓存地址表在缓存更新后发布，层内不重新整理。
- 跳过已校验为零的量化 offset，合并搬运描述符。
- 独立流先搬 W13、后搬 W2；W2 与第一段专家计算重叠，第二段计算前等待 W2。
- 周期 LFU、固定槽缓存搬运、融合缓存更新和缓存维护图重放。

dispatch、combine、专家 GMM 和量化计算复用主线。命中 Hot 缓存的专家仍需 D2D 到执行区，miss 从 Host 搬入执行区；这里没有宣称消除全部 D2D。

下一层预取、备用卡负载转交、Hot 缓存直算、执行槽复用、可调搬运核数、decode AllGather 等候选均不包含在此版本。旧试验 JSON 字段会被拒绝；旧试验环境变量没有对应入口。

## 接入现有启动配置

先使用已经能够运行模型的 Ascend 环境。本分支不打包 CANN、模型权重、MemFabric 二进制或个人机器配置。

1. 使用匹配的 CANN、torch_npu、sgl-kernel-npu、DeepSeek recipe 和 MemFabric。MemFabric 必须包含可用于 A3 的 `group_pack_copy` 实现；只有占位动态库的安装不够。首次安装需要按其构建工具编译 A3 搬运扩展。
2. 在仓库根目录编译 ACL 辅助扩展，并确保运行时导入此工作区的 Python 代码：

   ```bash
   export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
   python3 -m sglang.srt.offloading.moe.npu.native_acl \
     --build-directory /tmp/sglang-moe-native
   ```

3. 查看同目录的 `config.json`。缓存比例默认 28.125%，四 die 下也验证 50%；`host_pool_gib` 是**每 rank**的 Host 物理预算。`native_extension_directory` 要与上一步一致。`offline_stats_path` 相对 JSON 所在目录解析；附带的是确定性初始化分布，**不是实测命中率数据**。
4. 在现有模型启动命令中加入下面这个参数：

   ```bash
   --moe-offload-config scripts/moe_offload/config.json
   ```

   原有配置需要满足 `--moe-a2a-backend ascend_ops`、`--disable-overlap-schedule`、`--disable-shared-experts-fusion`，以及 TP = EP = Attention DP、MoE TP = 1、CP = 1、PP = 1、单机。Ascend Ops 默认的 prefill AllToAllV / decode low_latency（MC2）即可。

   实测使用 `SGLANG_NPU_USE_MULTI_STREAM=1`，开启模型共享专家等计算的多流路径；它属于模型现有配置，offload 的 W2 搬运流在本版始终启用。去掉 `--moe-offload-config` 即恢复不启用本特性的原路径。

## 当前边界

- 本轮设备回归使用 A3、TP=EP=Attention DP=4。其他 TP 数量未在本精简版本上做整模型验收。
- 当前格式适配器只支持 DeepSeek V4 的 ModelSlim W8A8、相同专家形状和零量化 offset；不支持 FP8/MXFP4。
- 共享专家和最前面的 `max(3, num_hash_layers)` 层专家仍常驻。其他层的 Hot 缓存跨层共享；28.125%/50% 指这部分专家的缓存容量比例。
- 不与现有模块 CPU offloader、MemFabric HiCache、EPLB、冗余专家或 batch overlap 共同使用。
- 每 rank 最多 256 个本地执行槽、1280 个 Hot 槽、8192 个 offload 专家 key。默认设备有效行 mask 容量为 128，超出会报错；该值是本地 padded/验证 token 行容量，**不是全局请求 BS**。
- MTP 的 draft 权重常驻，target 验证阶段使用同一 offload 路径；实测设置为 EAGLE、steps=2、topk=1、draft tokens=3。MTP 和缓存增大均需要额外 HBM 预算。
- 当前交付是供代码评审的实验分支。短请求和算术回归用于发现明显错误，不替代完整模型精度评测。

## 代码入口

- `python/sglang/srt/offloading/moe/npu/ascend_ops.py`：执行区、权重准备、W2 等待、缓存更新时机。
- `python/sglang/srt/offloading/moe/npu/small_batch_kernels.py`：层内搬运元数据融合。
- `python/sglang/srt/offloading/moe/npu/cache_update.py`、`cache_transfer.py`：缓存维护。
- `python/sglang/srt/layers/moe/moe_runner/ascend.py`：专家计算前准备权重、第二段计算前等待。

2026-11-02 A3 四 die 回归通过：28.125%/50%缓存 × MTP关/开，共四组。每组真实提示词并发1/4/8各两轮，26个请求无错误且输出非空；简单算术并发1–8的73个请求全部通过（4种题重复验证，并非独立精度题集）。

底层检查覆盖固定槽Host/Hot真实字节、空rank和动态路由图重放、2560 keys/1280 Hot槽的24步缓存维护与原实现逐字节对照，以及四rank MC2有效行图重放。Python静态和格式检查通过。
