"""Compare resident ModelSlim kernels with offloaded GLM expert bundles on NPU.

Run on an idle Ascend device with the serving torch_npu, CANN and MemFabric:

    MOE_NATIVE_ACL_DIR=/path/to/acl-extension \
      python test/manual/test_ascend_glm_offload_formats.py --device 0

This needs no checkpoint download. It loads deterministic synthetic checkpoint
shards through the real host loader, repacks with the selected format, reloads
with ACL and compares both GMM projections against the resident ModelSlim path.
It does not replace a complete GLM model accuracy / multi-rank serving test.
"""

import argparse
import os

import torch
import torch_npu  # noqa: F401 - register NPU before device construction.

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A8Int8MoEMethod,
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.offloading.moe.config import ExpertLayerSpec
from sglang.srt.offloading.moe.npu.format import ExpertLayerStorage
from sglang.srt.offloading.moe.npu.native_acl import (
    AclHostTransferOps,
    get_native_acl,
)
from sglang.srt.offloading.moe.npu.prefill import _projection_copy_plans
from sglang.srt.offloading.moe.npu.storage import MemFabricHostPool
from sglang.srt.offloading.moe.npu.w4a8_format import NpuW4A8Format
from sglang.srt.offloading.moe.npu.w8a8_format import NpuV4W8A8Format
from sglang.srt.offloading.moe.resources import ResourceKind


def _checkpoint(format, experts, seed):
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for name, component in format.layouts.items():
        shape = (experts,) + component.raw_shape[1:]
        if name.endswith("_weight"):
            value = torch.randint(
                -128, 128, shape, dtype=torch.int8, generator=generator
            )
        elif "offset" in name:
            value = torch.zeros(shape, dtype=torch.float32)
        elif "bias" in name:
            value = torch.randn(shape, generator=generator) * 0.001
        else:
            value = torch.rand(shape, generator=generator) * 0.015 + 0.001
        result[name] = value
    return result


def _load_shards(storage, checkpoint):
    for name, values in checkpoint.items():
        parameter = getattr(storage.layer, name)
        shards = storage.format.layouts[name].shards
        for expert, value in enumerate(values):
            pieces = value.chunk(2, dim=0) if len(shards) == 2 else (value,)
            for shard, piece in zip(shards, pieces, strict=True):
                parameter.weight_loader(
                    parameter, piece, shard_id=shard, expert_id=expert
                )


def _resident(checkpoint, device, kernel):
    layer = torch.nn.Module()
    for name, tensor in checkpoint.items():
        layer.register_parameter(
            name, torch.nn.Parameter(tensor.to(device), requires_grad=False)
        )
    for prefix in ("w13", "w2"):
        kernel.process_weights_after_loading(layer, prefix)
    return layer


def _verify_bank(format, bank, resident):
    for name, component in format.layouts.items():
        if component.runtime_name is None:
            continue
        expected = getattr(resident, component.runtime_name)
        actual = bank.tensors[name]
        assert actual.dtype == expected.dtype, name
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


def _verify_gmms(format, bank, resident, kernel, device, *, seed):
    generator = torch.Generator().manual_seed(seed)
    quant = bank.quant_info()
    # Include empty groups before, between and after active experts. The order
    # must stay identical when the bank has been restored after another copy.
    routes = ([3, 0, 2, 0], [0, 1, 0, 5], [1, 1, 1, 1])
    for counts in routes:
        group_list = torch.tensor(counts, dtype=torch.int64, device=device)
        for prefix, width in (
            ("w13", format.hidden_size),
            ("w2", format.intermediate_size),
        ):
            tokens = sum(counts)
            x = torch.randint(
                -63, 64, (tokens, width), dtype=torch.int8, generator=generator
            ).to(device)
            token_scale = (torch.rand(tokens, generator=generator) * 0.02 + 0.01).to(
                device
            )
            for inputs, scales in ((x, token_scale), (x.to(torch.bfloat16), None)):
                # The offload dispatcher preserves BF16 wire tensors; the
                # kernel must also agree when it performs activation quant.
                outputs = [
                    kernel.apply(
                        info,
                        inputs,
                        group_list,
                        scales,
                        torch.bfloat16,
                        prefix,
                        group_list_type=1,
                    )
                    for info in (resident, quant)
                ]
                torch.testing.assert_close(
                    outputs[1].cpu(), outputs[0].cpu(), rtol=0, atol=0
                )


def _run_case(kind, args, device, native):
    experts = 4
    if kind == "w8a8":
        format = NpuV4W8A8Format(128, 256, device, native_acl=native)
        kernel = NPUW8A8Int8MoEMethod()
    else:
        group_size = 0 if kind == "w4a8" else 128
        format = NpuW4A8Format(
            128,
            256,
            device,
            group_size=group_size,
            tp_size=args.tp_size,
            native_acl=native,
        )
        kernel = NPUW4A8Int8MoEMethod(is_per_channel_weight=group_size == 0)
    checkpoint = _checkpoint(format, experts, seed=103)
    resident = _resident(checkpoint, device, kernel)
    pool = MemFabricHostPool(
        format.host_nbytes(experts),
        device.index,
        physical_bytes=1024**3,
        native_acl=native,
    )
    storage = None
    retained = []
    copies = AclHostTransferOps(native, retained.append)
    try:
        layer = torch.nn.Module()
        storage = ExpertLayerStorage(
            layer, ExpertLayerSpec(1, experts, 1, 0), format, pool, {}
        )
        _load_shards(storage, checkpoint)
        storage.finalize()
        storage.require_sealed()
        assert pool.used_bytes == format.host_nbytes(experts)
        bank = format.allocate_bank(
            experts, ResourceKind.EXECUTION_WEIGHTS, "test.glm.execution"
        )
        plans = _projection_copy_plans(storage, bank, format.layouts)
        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream(device))
        for repeat in range(2):
            # Deliberately corrupt every component between reloads. Testing only
            # an initially zero bank would miss omitted offsets/scales/biases.
            for tensor in bank.tensors.values():
                tensor.zero_()
            stream.wait_stream(torch.npu.current_stream(device))
            for plan in plans:
                copies.prepare(plan, device)
                ready = copies.materialize_into(plan, None, stream)
                ready.completion.wait_on(torch.npu.current_stream(device))
            _verify_bank(format, bank, resident)
            _verify_gmms(format, bank, resident, kernel, device, seed=541 + repeat)
        print(
            f"PASS {kind}: checkpoint shards, sealed host master, ACL reload, "
            "resident tensor parity and both GMM projections; "
            f"expert={format.expert_nbytes} bytes, "
            f"host={format.host_nbytes(experts)} bytes, "
            f"loader workspace={format.workspace_nbytes} bytes",
            flush=True,
        )
    finally:
        # Failed submissions can retain in-flight descriptors; drain before
        # releasing any parameter view, descriptor owner or host registration.
        torch.npu.synchronize(device)
        copies.clear_cache()
        retained.clear()
        if storage is not None:
            storage.close()
        pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--moe-tp-size",
        dest="tp_size",
        type=int,
        choices=(1, 2, 4, 8, 16),
        default=1,
        help="ModelSlim MoE weight/bias shard count; this is independent of EP size",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("w8a8", "w4a8", "w4a8-group128"),
        default=("w8a8", "w4a8", "w4a8-group128"),
    )
    parser.add_argument(
        "--native-acl-dir", default=os.environ.get("MOE_NATIVE_ACL_DIR")
    )
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    native = get_native_acl(args.native_acl_dir)
    for kind in args.formats:
        _run_case(kind, args, device, native)


if __name__ == "__main__":
    main()
