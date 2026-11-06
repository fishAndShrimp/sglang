P 端 Mooncake PD 分段计时
========================

基线：83c31fde31c1be37cd7cba60f26ec4f60c14f9af
分支：perf/kv-transfer-timing

用途：把请求的 P 端过程拆成五段，定位双机 kv_transfer_latency_ms
约 1 秒主要花在后台排队、发送流程，还是调度器确认完成上。
只改 Python，不需要重新编译 Mooncake 或 MemFabric。
原有 /metrics 的计算方式不变；新增数据在 P 日志里，不新增 Prometheus 指标。

1. 启动
-------

先把这个分支放到 P 机器。进入你原来的 deploy 目录，然后：

    source envs/before_serve.env
    export SGLANG_SOURCE=/你的实际路径/sglang
    export SGLANG_KV_TIMING=1
    bash 1p.sh 2>&1 | tee /tmp/p_kv_timing.log

SGLANG_SOURCE 必须指向包含本次修改的仓库根目录。先 source，再设置路径，
以免 before_serve.env 覆盖路径。1p.sh 应通过该变量设置 Python 搜索路径；
可以检查它的 PYTHONPATH 设置，并在同样的环境中核实：

    PYTHONPATH="$SGLANG_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -c 'import sglang; print(sglang.__file__)'

开关在进程启动时读取，所以需要重启 P。D 按原来的方式启动，
不需要打开计时开关。保持 P、D 的基础版本与其他配置一致。
关闭 L3 的纯 PD 测试不需要 Mooncake Store master。

首次对比建议沿用之前跑通的配置：关闭 L3、关闭 overlap、关闭 chunked prefill，
关闭 radix cache。对应已有参数为：
    --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache
不要重复添加已有参数。单机和双机应保持模型、输入长度、TP、并发、传输协议一致。

等待 P、D 就绪后，发送原来的测试请求或 benchmark。P 日志中搜索 KV_TIMING。
普通成功请求每个实际发送的 P 进程输出两条 JSON 记录，顺序可以交错。
等请求结束、两条记录都写入日志后，再汇总。每次启动使用新的日志文件，
不要把不同机器或不同启动轮次的 P 日志拼在一起。

2. 汇总
-------

在包含本次修改的 SGLang 仓库根目录运行：

    python3 examples/kv_transfer_timing/summarize.py /tmp/p_kv_timing.log \
        > /tmp/p_kv_timing.csv

终端显示每个 P 进程的样本数、平均值、p50、p95、最大值，单位都是毫秒。
CSV 保留每条样本的 pid、room、rid 和分段耗时，便于看慢请求。
汇总脚本只用 Python 标准库，也可以复制脚本和日志到另一台机器分析。

五段的含义：

    prefill_ms    开始调度执行 prefill -> 调度器处理到 prefill 完成。
    submit_ms     prefill 完成 -> 构造发送任务并准备放入后台队列。
    queue_ms      准备入队 -> 后台线程从队列取到任务。
    send_ms       后台取到任务 -> KV、额外状态及 aux 的发送流程返回成功，
                  即将调用 conclude_transfer 发布成功状态。
    post_send_ms  上述成功点 -> P 调度器确认完成并记录 completion_time。

第五段再拆成：
    detect_ms     发送流程成功 -> 调度器处理到该请求的 Success 分支。
    cleanup_ms    进入 Success 分支 -> completion_time。

重点看 queue_ms、send_ms、detect_ms 哪个接近那 1 秒。
send_ms 包括已有设备事件等待、索引/地址处理、后端调用、多个目标及附加数据，
不是纯网络线上搬运时间。detect_ms 包括状态发布、后台清理、调度器轮询间隔、
TP/CP 状态一致性等待，以及处理同一轮其他请求的时间。
prefill_ms 是 CPU 观察到的阶段耗时，不是用设备事件测出的纯算子时间。
completion_time 与原指标终点一致，不涵盖后面的所有输出和资源释放工作。
这套计时没有新增设备同步，也没有调整调度、传输或轮询行为。

3. 与原指标对齐
---------------

metric_ms = (completion_time - prefill_transfer_queue_entry_time) * 1000
它使用这个版本 Mooncake 后端原有 kv_transfer_latency_ms 的起止点。

原指标通常在提交任务返回之后才开始计时，因此不能直接把后三段相加。
CSV 的 metric_start_after_enqueue_ms 记录了这两个起点的差值：

    metric_ms = queue_ms + send_ms + post_send_ms
                - metric_start_after_enqueue_ms

如果等待 bootstrap 导致延迟提交，原指标也可能先开始，差值可以是负数；
此时 submit_ms 中也包含提交前的等待。公式仍成立。

日志与 /metrics 对比时，需要选择相同进程/标签、相同请求和相同时间窗口。
多个 TP 进程可能各记录同一请求，不能把进程样本数直接当唯一请求数。
脚本按 pid 分别统计，不会把 TP 样本混成一个请求平均值。

已知 warmup 的 rid 时可以显式排除，例如：
    python3 examples/kv_transfer_timing/summarize.py /tmp/p_kv_timing.log \
        --exclude-rid warmup-request-id > /tmp/p_kv_timing.csv

不指定则包括 warmup；不会自动猜测哪条是 warmup。

4. 范围与限制
-------------

五段串行拆分用于一个请求只有一个发送 chunk、后台只出队一次的成功请求。
分块 prefill、前缀提前发送、多次发送、staging 重排队、失败、缺失/重复日志
以及时间边界重叠的样本会被跳过并说明原因，避免给出错误的串行耗时。
原始日志仍保留，便于后续单独分析这些路径。dummy rank 不一定有发送样本。

所有时刻来自同一个 P 进程的 perf_counter，不需要两机校时。
它不测 D 端收到数据后到开始 decode 的时间。
打点和日志仍有少量开销；后台日志在清理完成后写入，但可能影响下一个任务，
所以可再关闭开关跑一轮比较总吞吐和现有指标。

关闭：unset SGLANG_KV_TIMING，然后重启 P。

5. 本地检查与推送
-----------------

    python3 -m unittest discover -s examples/kv_transfer_timing -p 'test_*.py' -v
    git diff --check

测试覆盖分段计算、原指标还原、TP 进程隔离、warmup 排除、无效样本拒绝，
以及用模拟传输执行真实生产者/后台方法的开关、成功、失败路径。
这些检查不代替 A5 双机测试。

在本次独立 worktree 内，确认修改并创建提交后，可推送到自己的 fork：
    git push -u origin perf/kv-transfer-timing

origin 已指向 fishAndShrimp/sglang。不要使用 force push。
