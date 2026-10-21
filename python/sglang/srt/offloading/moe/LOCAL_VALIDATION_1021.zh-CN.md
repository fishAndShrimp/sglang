# 2026-10-21 本地验证记录

环境：macOS / Apple Silicon，Python 3.9.6，PyTorch 2.8.0。
本机无 CUDA，未安装 torch_npu；没有执行真实 ACL、MemFabric kernel、NPU graph、
多卡 EP、原生 FP4 或端到端模型性能验收。

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

硬件验收和剩余装配工作分别见 `CACHE_IMPLEMENTATION.zh-CN.md`、
`H2D_IMPLEMENTATION.zh-CN.md`。目前仍不能以这两个 draft 提交启动一个已验收的
完整 V4 Flash offload 服务。
