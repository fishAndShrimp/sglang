P 端 Mooncake PD 分块与接口计时（version 2）
========================================

基线：83c31fde31c1be37cd7cba60f26ec4f60c14f9af
分支：perf/kv-transfer-timing

这版记录每个发送块的每次出队，并直接测量 engine.batch_transfer_sync()。
支持多块 prefill、前缀提前发送、staging 延迟重排队和线程池并行调用。
它不再要求整个请求能拆成五个互不重叠的串行阶段。
原有 /metrics 的计算和传输策略不变；新增结果通过 P 日志离线汇总。
只改 Python，不需要编译 Mooncake 或 MemFabric，也不需要修改 D。

1. 更新后重新采集
-----------------

version 1 只记录最后一块，旧日志无法还原中间块或 API 的时间。
要得到本版完整细分，需要更新 P 的仓库并重启 P，再跑一轮压测。
D 继续使用原来跑通的版本即可。关闭 L3 的纯 PD 不需要 Store master。

如果 P、D 共用环境文件，就只在 1p.sh 中修改：所有 source 之后、启动 Python
之前设置下面变量，并确保后续代码没有用旧路径覆盖它们。

    export SGLANG_SOURCE=/你的实际路径/sglang-kv-timing
    export SGLANG_KV_TIMING=1
    export PYTHONPATH="$SGLANG_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"

启动并将新一轮日志写到盘上（换成实际路径）：

    bash 1p.sh 2>&1 | tee /你的共享盘/p_v2_round1.log

启动时会出现：
    KV_TIMING_ENABLED version=2 worker=...

随后每块每次尝试一条 event=chunk 日志，每个成功请求一条 event=finish 日志。
日志带 version=2；finish 还带 tp_rank/dp_rank/cp_rank/pp_rank。
完成请求时日志顺序可能交错，脚本按 pid、room、chunk、attempt 匹配。
日志只记录标识、时间、字节数、返回码，不记录内存地址或输入内容。

先用少量请求确认 samples 和 detailed 一致，再跑正常的 100 条压测。
单机/双机对比保持模型、输入长度、并发、TP、缓存和传输协议一致。
不需要为了让脚本接受样本而关闭 chunked prefill 或改动原来的调度配置。

2. 只保存汇总结果
-----------------

在新仓库根目录运行，不加 --metric-only：

    python3 examples/kv_transfer_timing/summarize.py \
      /你的共享盘/p_v2_round1.log \
      > /dev/null 2> /你的共享盘/round1_summary.txt

    cat /你的共享盘/round1_summary.txt

需要更细的调用分类和原指标窗口拆分时加 --details：

    python3 examples/kv_transfer_timing/summarize.py \
      /你的共享盘/p_v2_round1.log --details \
      > /dev/null 2> /你的共享盘/round1_details.txt

stdout 是逐请求 CSV，stderr 是统计文本和异常说明。
需要保留 CSV 时把 /dev/null 改成结果 CSV 路径。
脚本依赖同目录的 chunk_summary.py，只用 Python 标准库。

每个进程的头部类似：
    pid=7355 samples=101 (rank-local observations) detailed=101 tp_rank=0

samples：可还原原指标的完成请求数。
detailed：有完整分块记录、能计算详细耗时的请求数。
每个字段另有 n=，明确该字段使用了多少条样本。缺块时保留 metric_ms，
不会再为了细分而把它从整轮平均值中删除；缺失的细分显示 n=0 或较小的 n。
只有 detailed=samples 且各字段 n 相同时，才是在同一批请求上比较各阶段。
所有耗时单位都是毫秒。统计包括 mean、p50、p95、max、sum 和 n。

3. 默认显示的字段
-----------------

prefill_ms       第一次 forward 开始到最后 prefill 完成，CPU 观察的整段时间。
queue_ms         该请求所有块、所有重排队尝试的队列等待区间并集。
wait_ms          既有设备事件/索引准备等待范围；没有新增设备同步。
send_ms          所有块的 KV、附加状态、aux 发送调用范围并集。
mooncake_ms      直接包住 batch_transfer_sync() 的时间区间并集。
send_other_ms    send_ms 中没有被 Mooncake 接口调用覆盖的时间，例如地址准备、
                 等待线程池、staging 就绪等待、TCP aux 等，不等于纯 CPU 计算。
worker_other_ms  后台处理区间中，未被 wait/send/notify 范围覆盖的部分。
notify_ms        conclude_transfer() 发布/通知发送结果的范围。
detect_ms        所有块的最后一个 send 范围结束，到调度器进入 Success 分支。
                 包含后台收尾、调度器轮询及 TP/CP 等待；notify 是它的部分范围。
cleanup_ms       调度器进入 Success 分支到 completion_time。
metric_ms        原 kv_transfer_latency_ms 的起止点，还原原指标用于对账。

重点：mooncake_ms 才是本次新增的直接接口计时，仍包含 Mooncake/MemFabric
内部的提交、排队、同步等待；它不是纯网络线上搬运时间，也没有拆开库内部。
新版 send_ms 的定义比 version 1 更窄，wait 和 notify 已单独记录。
如果 aux 走 TCP，则 aux_ms 有耗时，但 mooncake_aux_ms 和接口调用数可以为 0。

4. --details 显示的字段
-----------------------

kv_ms/state_ms/aux_ms：三个发送范围分别耗时。
mooncake_kv_ms/mooncake_state_ms/mooncake_aux_ms：相应范围中的接口调用并集。
mooncake_call_sum_ms：每次 batch_transfer_sync 的耗时直接相加。
                    并行调用时可能大于 mooncake_ms，不能当墙钟耗时。
worker_ms：所有块从出队到处理结束的区间并集。
post_send_ms：最后一个 send 范围结束到 completion_time；等于 detect+cleanup。
last_submit_offset_ms：最后一个块首次入队相对 prefill 完成的偏移，可以为负。
send_overlap_prefill_ms：发送范围与 prefill 范围重叠的时间。

多块请求不能直接把这些字段相加：一个块在排队时，另一个块可能正在发送，
早期发送还可能与 prefill 重叠。原 metric_ms 通常只覆盖最后提交之后的窗口。
例如整请求 mooncake_ms 比 metric_ms 大，未必有错：部分接口调用在原指标
开始之前就已执行。

为准确对账，下面五项只计算原指标时间窗口内的区间，并依次去掉重叠：
    metric_mooncake_ms       接口调用覆盖的时间。
    metric_send_other_ms     发送范围里接口调用之外的时间。
    metric_worker_other_ms   后台处理范围里上述两项之外的时间。
    metric_queue_only_ms     有块排队、但没有本请求后台处理覆盖的时间。
    metric_other_ms          窗口里其余时间，包括提交间隙、完成确认和收尾。

这五项之和等于 metric_ms。这是时间覆盖的划分，不意味着这些阶段串行发生。
先看这五项中哪一项大，再结合 wait_ms、notify_ms、detect_ms 等定位。
notify_ms 与 detect_ms 存在包含关系，不能重复相加。

5. 样本对齐与旧日志
------------------

脚本读取整个输入日志，不自动只选最后一轮。每轮使用独立日志/截取，
不要混合多个 P 机器、多个启动轮次或 version 1/version 2 的详细日志。
多个 TP 进程会记录同一请求，所以按 pid 分别统计。用 tp_rank 等标签与
/metrics 对齐，比较同一轮、相同进程/标签的 sum、samples 和 mean。
默认包含 warmup；已知 rid 时用 --exclude-rid 指定排除，可重复指定。

--metric-only 保留，适用于旧日志的全样本指标对账，但它只输出 metric_ms。
它不等于详细分析模式，不会生成未记录过的中间块/接口数据。
旧日志不带 version=2 时，默认仍按旧的单块模式分析并明确标记 mode。

所有时间来自同一 P 进程的 perf_counter，不需要两机校时。
不测 D 收到数据后到开始 decode 的时间，也不对 TP 之间的等待原因做推断。
打点、序列化和日志 I/O 有开销；并行按层调用较多时日志量会增加。
统计开关关闭需 unset SGLANG_KV_TIMING 并重启 P。对比性能时可另跑关闭开关
的一轮，观察诊断开销，不要把开启完整 NPU profiler 的一轮混入基线。

6. 修改位置和检查
-----------------

python/sglang/srt/disaggregation/mooncake/conn.py
    每块入队/出队、重排队、等待、KV/state/aux/notify 计时；两处接口调用
    使用包装器，发送线程池通过上下文传播关联请求和块。
python/sglang/srt/observability/kv_transfer_timing.py
    记录阶段、接口调用时间/字节数/返回码，传播线程上下文并输出每块日志。
python/sglang/srt/disaggregation/prefill.py
    请求级边界、发送块总数、rank 和完成记录。
examples/kv_transfer_timing/chunk_summary.py
    验证块/尝试完整性，区间合并、原指标窗口去重拆分。

    python3 -m unittest discover -s examples/kv_transfer_timing -p 'test_*.py' -v
    git diff --check

测试覆盖真实生产者/后台方法的模拟传输、多块、调用上下文、并行去重、
重排队计算、100 条多块请求全量统计、缺日志保留原指标以及旧格式兼容。
没有在 A5 双机上实测，模拟检查不代替目标环境验证。
