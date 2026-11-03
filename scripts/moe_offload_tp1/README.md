# W4A8 单 rank MoE offload 复测

## 启动

在已有 SGLang / CANN / torch_npu / sgl_kernel_npu / MemFabric 环境的容器里，修改 `serve_tp1.sh` 开头的权重路径、逻辑设备号、SOC 和端口，运行：

```bash
SOC=A5 MODEL_PATH=/path/to/DeepSeek-V4-Flash-w4a8-mtp ASCEND_RT_VISIBLE_DEVICES=0 bash scripts/moe_offload_tp1/serve_tp1.sh
```

脚本在本机的 `RUNTIME_DIR` 构建对应 A3/A5 的搬运扩展和 ACL 提交扩展；不复制 A3 编译缓存到 A5。更换 CANN、torch_npu 或 MemFabric 版本后使用新的 `RUNTIME_DIR`。首次编译与模型加载需要数分钟。镜像安装路径不同则修改四行环境初始化。

另一个终端先验证输出，再测短请求：

```bash
python3 scripts/moe_offload_tp1/test_requests.py --port 30000
python3 scripts/moe_offload_tp1/test_requests.py --port 30000 --bench
```

## 当前边界

- DeepSeek V4 Flash **ModelSlim INT4 W4A8_DYNAMIC**；不是 MXFP4。TP=EP=attention-DP=1，MTP 关闭。
- 默认20% routed expert cache、Host池128 GiB、执行区256专家；首3个hash层及共享专家仍驻留。先保持相同设置进行硬件对照。
- 最大并发8，总KV预算32768、滑窗比例0.25（8192槽）；C128池含256个压缩token，可容纳16个物理页，避免8192总预算仅4个C128页导致请求回退。客户端并发不自动等于每一步实际batch。
- 不启用EPLB、冗余专家、双batch重叠或推测预取。A3已完成BS1/4/8运行与profiling，零回退/OOM；A5尚未实测，此为迁移复测入口。
- MemFabric安装包须包含A5 acc_offload构建源码；模型所需NPU算子须已安装。没有把本机依赖或编译产物打包进代码。

## 保留实现

Ascend Ops的本地路由、专家计算和合并；单rank跳过collective。复用设备侧cache管理与MemFabric搬运、W2搬运重叠、decode Graph。W4A8完整搬运权重、scale、offset、scale_bias八组件，并保留模型要求的SwiGLU截断。

严格数字答案仍有失败；全驻留对照也复现，52次算术与15个单请求的输出均与offload一致。允许写步骤后追加的6次复合算式均正确。测试保留原错题并在不匹配时非零退出，不把改写题目视为修复。
