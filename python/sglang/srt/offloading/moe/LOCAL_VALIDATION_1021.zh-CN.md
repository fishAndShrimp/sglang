# 2026-10-21 本地验证记录

环境：macOS / Apple Silicon，Python 3.9.6，PyTorch 2.8.0。
本机无 CUDA，未安装 torch_npu；没有执行真实 ACL、MemFabric kernel、NPU graph、
多卡 EP、原生 FP4 或端到端模型性能验收。

## 四项原生适配的组合验证

当前完整套件 **223 项通过**。原有 132 项继续运行，新增部分如下：

| 适配 | 新增 CPU 测试 | 覆盖 |
| --- | ---: | --- |
| ACL 原生提交 | 12 | 生产 C++ 源码分别编译链接显式模拟的 9/10 参数 SDK；队列 flush、stream/device、GIL、失败索引、生命周期和字节复制 |
| 真实 host/FP4 存储 | 23 | 同份内存、注册/映射、初始化失败、逐专家加载、padding、布局漂移和实际 loader hooks |
| Runtime/config | 14 | warmup/capture 开关、fresh replay completion、精确恢复、预算、失败 worker/event drain、直接方法调用和晚更新 padding |
| EP=1 padding 接入 | 6 | 执行真实 enablement 函数；JSON/injection、非 offload 和既有并行模式 |
| 分配前量化检查 | 6 | 真实 native method 类型；错误类型在 create_weights 前拒绝；常驻层和普通 offloader 行为 |
| EP/FFN | 21 | 设备 counts、固定容量 wave gather/scatter、40 轮随机参考差分、空/尾组、padding/error、所有 reader 事件；真实 AscendRunnerCore 配合模拟原生算子 |
| 后端装配 | 9 | 3 层 EP=2、缓存比例 0/.25/.75/1、两块 bank、共享 workspace 单次计账、HOST/HOT 字节复制、原 slot 恢复和失败清理 |

上述测试使用仓库的 CPU contract runner，真实模块正常加载；原生硬件依赖按各测试说明
替换为模拟器。C++ 测试没有使用真实 CANN SDK，不构成目标版本编译结果。
修改的 Python 源码通过语法与 Ruff 检查；另使用 Python 3.13 做语法检查。
本机有 PyTorch 的解释器为 Python 3.9，未安装完整 serving 依赖，不能替代项目要求的
Python >=3.11 / torch_npu 运行环境验收。
尝试 Python 3.13 的 `python -m sglang.launch_server --help`，在顶层导入时因
`ModuleNotFoundError: No module named 'torch'` 停止；未把该项记为 CLI 启动通过。

启动示例与目标机验收步骤见 [NPU runtime](npu/README.zh-CN.md)、
[ACL build/smoke](npu/native/README.zh-CN.md)、[V4 storage](npu/V4_STORAGE.md) 和
[EP/FFN](npu/EXPERTS_IMPLEMENTATION.zh-CN.md)。剩余的是目标机原生编译/链接、
同份 host 双路径搬运、FP4 数值、真实 EP、完整 graph、多 bucket 模式切换、内存峰值及性能验收；
不把 CPU 模拟结果记为硬件通过。

## 目录整理验证

目录整理后再次运行同一套件，223 项通过。NPU 实现、原生扩展及示例统一位于
`moe/npu/`，对应测试导入、ModelRunner factory 和四卡启动脚本均已更新。
另检查了相对导入、文档链接、包及资源发现、ACL build/smoke 命令入口；在阻止
`torch_npu`、MemFabric 和 DeepEP 导入的 CPU 环境中，公共框架、NPU 包及九个
NPU 模块均可导入。四卡脚本的 dry-run 和模拟构建/启动流程通过。

## 前序 cache / H2D 提交记录

| 验证对象 | 结果 |
| --- | --- |
| 原始基线 `59cbb5d226` | 57 项 CPU 契约测试通过 |
| cache 独立改动（不含新 H2D 实现） | 91 项 CPU 测试通过 |
| cache + H2D 组合 | 132 项 CPU 测试通过 |
| 有界 prefill 线程测试 | 17 项 × 50 轮，850 次执行全部通过 |
| 热集事务随机差分 | 320 次，与 CPU 参考结果一致 |
| MemFabric 压紧语义 | 五 lane 的全部 32 种 mask，以及多分量、非连续目标和尾部字节通过 |
| 源码检查 | 修改的 Python 文件编译检查、Ruff 格式及静态检查通过 |
| 提交检查 | 两块分别运行 `git diff --cached --check` |

执行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python test/registered/unit/offloading/run_cpu_contracts.py
```

测试入口建立包命名空间以跳过 serving 顶层可选依赖，加载的 offloading 源码
是真实实现。硬件 API 在对应测试中用字节复制/完成事件模拟器替代，因此不能
据此断言真实 NPU 的队列顺序、capture 算子支持或内存峰值。

验证重点：

- 固定 Tensor 计划、地址保持、动态命中/miss、全部本地需求及尾波覆盖。
- 热集只在模型尾部准入，幸存者原 slot 保持，失效同 key slot 必须重新加载。
- generation/映射/目标冲突触发共享错误状态，禁止发布不完整权重。
- prefill worker 不提前阻塞 compute；跨组、跨层复用 bank 都有 reader 依赖。
- copy/event/FFN/capture/cleanup 失败后的资源保活，图销毁前不释放 lease。
- snapshot 原 slot 重载、LFU 精确恢复；native 成功返回但设备错误时禁止恢复映射。

前序实现说明见 `CACHE_IMPLEMENTATION.zh-CN.md`、`H2D_IMPLEMENTATION.zh-CN.md`。
其后四项具体适配的当前状态与硬件验收边界见本文上半部分。
