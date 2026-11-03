"""DSV4 INT8 activation regression: clipping semantics and eager/Graph parity.
Run with one idle NPU visible; no model weights are required.
"""
import json
from types import SimpleNamespace
import torch
import torch_npu
import sglang.srt.layers.quantization
from sglang.srt.hardware_backend.npu.moe.activation import NPUSwigluInt8QuantWithLimit
from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod
from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

torch.npu.set_device(0)
for kernel in (NPUW4A8Int8MoEMethod(is_per_channel_weight=True), NPUW8A8Int8MoEMethod()):
    config = MoeRunnerConfig(layer=SimpleNamespace(w2_kernel=kernel), swiglu_limit=10.0)
    assert isinstance(AscendRunnerCore(config).activation, NPUSwigluInt8QuantWithLimit)
act = NPUSwigluInt8QuantWithLimit(10.0)
results = []
for batch in (1, 4, 8):
    torch.manual_seed(1024 + batch)
    # Explicit outliers force both gate and up clipping; zero rows exercise padding.
    cpu = (torch.randn(batch * 6, 4096) * 20).bfloat16()
    cpu[0].zero_()
    gate, up = cpu.float().chunk(2, -1)
    ref = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16()
    old = (torch.nn.functional.silu(gate) * up).bfloat16()
    assert (old.float() - ref.float()).abs().max() > 100
    x = cpu.npu()
    raw = torch.ops.npu.npu_clipped_swiglu(x, alpha=1, limit=10, bias=0, interleaved=False)
    torch.testing.assert_close(raw.cpu(), ref, rtol=0.008, atol=0.001)
    q, scale = act._apply_activation(x)
    recovered = q.float() * scale[:, None]
    # Per-row quantization error plus BF16 activation rounding allowance.
    err = (recovered.cpu() - ref.float()).abs()
    assert bool((err <= scale.cpu()[:, None] + 0.51).all()), float(err.max())
    assert bool(torch.isfinite(recovered).all())
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3): act._apply_activation(x)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        gq, gs = act._apply_activation(x)
    for multiplier in (1, 0.5, 1.25):
        x.copy_((cpu.float() * multiplier).bfloat16())
        eq, es = act._apply_activation(x)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(gq, eq, rtol=0, atol=0)
        torch.testing.assert_close(gs, es, rtol=0, atol=0)
    results.append(dict(batch=batch, cpu_reference_max_abs=float((raw.cpu().float()-ref.float()).abs().max()), graph_exact=True))
print(json.dumps(results), flush=True)
print("PASS clipped INT8 activation and W4A8/W8A8 selection", flush=True)
