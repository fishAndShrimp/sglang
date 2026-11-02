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

- 已验证 A3、DeepSeek V4 Flash ModelSlim W8A8、TP=EP=Attention DP=4；MoE TP、CP、PP 均为 1。要求专家形状一致、量化 offset 为零。
- 已验证请求并发 1～8、28.125%/50% 缓存及 MTP 开关；MTP 使用 EAGLE steps=2、topk=1、draft tokens=3。基础回归不替代完整精度评测。
- 共享专家、最前面的 `max(3, num_hash_layers)` 层和 MTP draft 权重常驻。Host 预算每 rank 64 GiB，四 rank 合计 256 GiB。
- 每 rank 上限为 256 个执行槽、1280 个 Hot 槽、8192 个专家 key；有效行 mask 默认容纳 128 个本地 padded/验证 token，不等于请求 BS。
- 不支持与 EPLB、冗余专家、batch overlap、模块 CPU offloader 或 MemFabric HiCache 同时使用。

## 保留的实现

- 复用主线 dispatch、combine 和专家 GMM；固定执行区跨层复用，有效 token mask 排除 padding 和空 rank。
- 融合搬运地址、长度和访问计数，缓存更新后统一发布地址表；跳过已校验为零的 offset，合并搬运。
- 独立流搬运 W13/W2，W2 与第一段专家计算重叠，第二段计算前等待完成。
- 周期 LFU、固定槽缓存搬运、融合缓存更新及维护图重放。Hot 命中仍需 D2D 到执行区，miss 从 Host 搬入。
