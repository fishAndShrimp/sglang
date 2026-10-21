# ACL batch 原生适配

`../../native_acl.py` 将 `AclBatchHostTransferOps` 接到上级目录的 C++ 扩展。
扩展直接包含目标机的 `acl/acl.h`、torch_npu 的 `NPUStream.h` 和
`NPUGuard.h`；不复制 ACL 结构布局、不使用 ctypes 调用真实运行库，
也不把 Python raw stream handle 当作已经排空的 torch_npu 提交队列。

## 构建和接入

在安装了匹配版本的 PyTorch、torch_npu、CANN SDK 和 C++ 编译器的目标机执行：

```bash
python -m sglang.srt.offloading.moe.npu.native_acl \
  --build-directory /absolute/path/to/moe-acl-build \
  --cann-home /usr/local/Ascend/ascend-toolkit/latest --verbose
```

`--cann-home` 默认读取 `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME`，再使用上例
默认路径。构建检查 `include/acl/acl.h`、`lib64/libascendcl.so`，并链接当前
Python 环境的 torch_npu。此目录绑定该编译环境，不是可跨 CANN/PyTorch
版本复制的通用二进制。源码由项目现有 `srt/**/*` package-data 配置打包。

```python
from sglang.srt.offloading.moe.npu.native_acl import get_native_acl, make_acl_host_ops

native = get_native_acl("/absolute/path/to/moe-acl-build")
host_ops = make_acl_host_ops(
    module=native,
    retain=context.retain,
    source_location="unregistered",  # 外部 MemFabric 分配，再经 ACL 注册锁页
    max_batch_entries=256,
)
```

同一进程按规范化构建目录缓存扩展加载，多个 layer/runtime 可以复用模块。
加载不自动编译；模块不存在时启动失败，不退回 Torch 权重复制。

`source_location="host"` 用于 ACL 分配的 host 内存；`unregistered` 表达
原始分配并非来自 ACL API；`host_numa` 需要匹配分配所在 NUMA ID。
这些位置属性不证明锁页。所有路径仍需 `HostMemoryCapability` 的独立锁页
认证。对 MemFabric 的外部分配，应先成功调用以下注册接口，再建立能力描述。

## SDK 签名和提交顺序

构建会先编译精确函数指针转换，优先检测带 `failIndex` 的十参数 C 接口，
再检测不带该参数的九参数接口。检测包含 SDK 真实头文件，支持 C++ 重载解析；
未知签名、缺失符号或不匹配的库在构建时失败。代码不尝试用猜测的函数指针
调用不同 ABI。实际属性对象为 SDK 定义的 `aclrtMemcpyBatchAttr`，初始化
保留字节为零，单项属性及起始索引零覆盖该批所有 H2D 区间。

提交顺序为：

1. Python 在正确 `torch.npu.stream(copy_stream)` 上下文内创建 C++ `Batch`；
   在原生调用前，将 Batch、原始描述符和所有 storage/registration owner 交给
   runtime 的 `retain`。
2. C++ 设置调用设备，核对当前 `NPUStream.id()` 与调用者的 `stream_id`，
   调用 `stream(true)` 将先前 host task queue 工作下发到设备流。
3. 检查取得的实际 stream 非空，再调用 `aclrtMemcpyBatchAsync`。
4. 返回后，现有 backend 在同一 torch_npu stream 记录 event；只有 event
   完成才是整批可读的凭据。原生失败也保留所有对象，由 owner 排空后再释放。

这里等待的是 host 提交队列，不调用设备或设备流同步。pybind 在原生提交、
注册和初始化复制期间释放 GIL。copy stream 必须遵循已有 pipeline 的单线程
提交权协议；不支持外部 raw stream，也不支持把此 eager 路径用于 graph capture。
`stream(true)` 的实现语义需要随目标 torch_npu 版本核验；头文件可编译本身
不足以证明另一个版本内部的队列行为保持一致。

## 共享 host 主副本和初始化字节复制

模块还提供以下以字节为单位的实际 SDK 调用，供 host owner/格式适配使用：

```python
address = native.malloc_host(nbytes, device_index)
mapped_address = native.register_host(address, nbytes, device_index, mapped=True)
native.unregister_host(address, device_index)
native.free_host(address, device_index)

# 对 MemFabric 自己已经提供 get_dva 的 allocation，仅注册锁页，不建立第二个映射。
native.register_host(memfabric_host_address, physical_bytes, device_index, mapped=False)

native.memcpy_sync(dst, dst_capacity, src, nbytes, "device_to_host", device_index)
```

`register_host` 使用 SDK 的 `ACL_HOST_REG_PINNED`，可选
`ACL_HOST_REG_MAPPED`，映射时通过 `aclrtHostGetDevicePointer` 取设备地址。
当这些宏在安装的 SDK 中缺失时明确拒绝注册，不使用自定义数值代替。
注册范围须非空、起始地址 4 KiB 对齐；外部内存/内核版本的支持条件由 SDK
检查。成功 PINNED 注册后才能将 `acl_async_readable` 标为真。

所有释放均显式执行；先等待完整 `resources_idle`、销毁引用它的图，再取消注册，
最后释放原始 allocation。注册或取消注册失败时不得释放可能仍被 SDK 使用的
内存。对于映射查询失败且取消注册也失败的异常，错误包含两个状态码，owner
必须保留内存，交由失败清理处理。

`memcpy_sync` 只服务初始化和格式重排的物理字节搬运；不会自动等待此前排在
torch_npu host queue 中的算子。调用方应先记录并等待对应算子真实完成事件。
它不是运行时 prefill fallback，也不改变 NZ/ND 布局或量化 dtype。

## 目标机验收

CPU fake-SDK 测试与独立 smoke 脚本已移除。ACL 构建入口保留：SDK 签名探测是
匹配生产 CANN ABI 的必要步骤，不是测试后备路径。

在目标 A5 上验收实际编译/链接、非默认流 wait → batch → event、后台提交与
计算并行、HostRegisterV2 与 MemFabric 同一 allocation 的兼容性、完整 bundle
字节一致性和异常排空。prefill 明确不入图，此模块不会开启 `full_decode_graph`。
这些硬件验收结果仍需由实际部署环境提供。

## 核对依据（2026-10-21）

- [CANN 9.1 ACL batch C 接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1957.html)
- [CANN 9.2 batch C++ 重载](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_2227.html)
- [SDK 属性结构](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1942.html)
- [锁页内存与 HostRegisterV2](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0006.html)
- [注册宏的官方定义](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2229.html)
- [torch_npu NPUStream 源码](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUStream.cpp)：`stream(true)` 委托 `stream()`，后者执行 `MakeSureQueueEmpty()`。
- [torch_npu Stream Python 属性绑定](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/npu/Stream.cpp)：`stream_id` 为真实 managed stream ID；不需要 Python 获取 raw stream。
