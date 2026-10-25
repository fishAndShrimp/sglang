# 周期缓存接入原生 Ascend 后端（2026-10-22）

> 2026-10-25 更新：当前唯一运行栈及测试迁移见 [重构记录](../../doc/REFACTOR_20261025.zh-CN.md)。
> 以下保留 10 月 22 日算法/配置来源；旧通用缓存栈已退休。

历史分支 `feat/moe-offload-dev-1022` 从 `818c9b7075` 创建，cherry-pick
`wanerlee/sglang` 的 `a95574ec8a5eed2d8606d9876c30535c7b25b8ec`。

## 缓存行为

原生 JSON 启动入口使用 `PeriodicLFUPolicy`，替换此前每步全局 LFU 的默认策略：

- 离线频率在本 rank 全部可卸载专家中选择 Top-K。resident 层和外 rank 专家排除在外。
- 首次权重通过原生 `ExpertWeightAccess.apply_update` 和 MemFabric 路径装入，完整 bundle
  完成且共享错误状态健康后才结束初始化；初始化不执行虚拟 decode，不注入在线 usage。
- 在线按 post-dispatch 有效 token-expert assignment 累计，命中和 miss 都计入热度。
- 每 `update_interval` 个有效完整 decode forward 做同层冷热配对；最低 usage、加性增益
  和 rank 级 `max_swaps` 限制替换，周期尾再执行衰减。默认参数为 4、2、1.5、96、0.7。
- 每层在线热专家数量由离线初始化固定；零配额层一直走 miss 路径，不跨层抢占。
- prefill 和 capture/warmup 不推进热度或周期；全局空步不推进，有工作但本 rank 无路由仍推进。
- prefill 借池后，按原 slot 从 host 重载完整权重，并恢复 usage、周期和驻留状态。

`TensorLFUPolicy` 与 `PeriodicLFUPolicy` 共用唯一策略接口；原生 `create_npu_offloader`
默认选择周期策略，旧通用 `CacheBackend` 已退休。原生路径继续使用已有的固定设备计划、有效 lane 检查、sticky error、
输入 padding 屏蔽、ACL prefill 流水、原生 FP4 bank 和 Graph lease。生产搬运不增加 CPU fallback。

## 配置和 profile

在原有 JSON 预算字段外配置：

```json
{
  "offline_stats_path": "/path/to/deepseek-v4-expert-profile.json",
  "offline_model_id": "DeepSeek-V4-Flash",
  "update_interval": 4,
  "min_usage": 2.0,
  "replacement_min_gain": 1.5,
  "max_swaps": 96,
  "usage_decay": 0.7
}
```

`offline_stats_path` 的相对路径以 offload JSON 所在目录解析。非零 `cache_ratio`
要求提供路径与 model ID；`cache_ratio=0` 不需要 profile，也不隐式填充缓存。

profile 的专家编号为 **global expert ID**，格式如下。示例数字仅说明格式，不是推荐热度。

```json
{
  "model_id": "DeepSeek-V4-Flash",
  "layers": {
    "3": {"0": 1200, "1": 900},
    "4": {"0": 300, "1": 1800}
  }
}
```

`model_id` 必须与配置匹配；缺失频率按零处理，同频率按 canonical key 排序。
线上 usage 独立从零开始。离线数据会决定每层配额，部署时应提供覆盖实际模型的统计。
已有 `examples/v4.json` 中的 `/path/to/...` 必须替换为真实 profile。

## 命中统计与后端边界

公共 `BoundMoERuntime.cache_stats` 暴露 `CacheRouteStats` 的三个固定地址 int64 标量：
`accesses`、`hits`、`misses`。每次 decode begin 清零，各固定 wave 直接消费真实 counts，
避免把 counts 展开成逐 token 路由，也避免 host 回读。

assignment 命中率为 `hits / accesses`，无访问时应报告 N/A。重复访问同一专家逐次计数；
它不是去重专家命中率，也不是实际 H2D 字节节省率。统计属于本 rank，不能直接平均各 rank
的百分比得到全局命中率。调用方需要在本次设备完成后读取或异步快照；下一批会复用这些标量。
此接入提供设备计数，不在图内打印 Tensor 或为日志逐批增加全设备同步。

原生后端默认异步输出每 batch 路由命中率，使用已有完成事件和有界 pinned host 快照池；
配置、日志含义及缓冲饱和行为见 [维测日志](OBSERVABILITY.zh-CN.md)。

`CacheRouteStats.allocations()` 纳入 metadata 账本。策略快照通过原有 `DecodeStateSnapshot`
持有全部周期字段和持久 hot 状态；execution 元数据/counts 恢复时清零。原生计划保留
int32 count ABI，不再维护第二套 metadata adapter。

公共 `observe(..., due_out)` 写调用方独立的设备 bool 标量；`finish_step` 清内部 due
不影响事务输出。planner 锁存 due 与 runtime guard，不读取策略私有 config 或 host
周期。非 due 屏蔽有效事务；当前 full graph 的固定选择/规划节点仍可执行，未完成
设备条件执行或融合 kernel 真早退验证。

## 本地验证

以下为历史验证记录；本轮完整结果以重构记录为准。2026-10-22 在 macOS / Python 3.9.6 / PyTorch 2.8.0 下运行相关测试集，154 项通过、
无 skip。另通过 Python 3.11 语法解析、Ruff 0.15.1、isort 7.0.0、`git diff --check`
及四卡启动脚本 dry-run。dry-run 只验证命令组装，不加载真实 profile、权重或设备。

在有 PyTorch 的 Python 环境运行：

```bash
python - <<'PY'
import pathlib
import sys
import types
import unittest

root = pathlib.Path.cwd()
sglang = types.ModuleType("sglang")
sglang.__path__ = [str(root / "python/sglang")]
sys.modules["sglang"] = sglang
suite = unittest.defaultTestLoader.discover(
    str(root / "test/registered/unit/offloading"), pattern="test_*.py"
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
```

旧重复缓存栈的部分覆盖迁入既有生产 residency、Graph lease 与原生装配 fixture；
上一版重构新增的六个测试文件曾按要求删除；本轮在既有测试文件中补充覆盖，历史验证及当前结果见[重构记录](../../doc/REFACTOR_20261025.zh-CN.md)。
不再保留第二套测试运行栈。
装配测试中的 CPU 字节复制和事件替身只存在于测试目录，不恢复生产 CPU emulation 开关。
新增用例覆盖离线完整 bundle 初始化、第四步替换/衰减、下一步命中、assignment 计数、
capture gate、空步周期、原 slot/策略恢复、初始化 sticky fault 和配置选择。

本地 CPU 契约不代表真实 NPU Graph、MemFabric kernel、多卡 EP 或 FP4 数值验收；
目标机仍需验证 profile 初始化后的完整 capture、多 bucket replay、prefill/decode 交替和时序。
