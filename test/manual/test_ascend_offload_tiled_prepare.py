"""Large/non-power-of-two physical-slot preparation -> real INT8 NZ GMM.

Uses small random matrices and an independent CPU integer oracle. This tests
the indexing/graph contract, not full-size model performance or quant accuracy.
"""

import os

import torch
import torch_npu
import triton

from sglang.srt.offloading.moe.npu.small_batch_tiled import (
    prepare_copy_tiles,
    validate_prepare_tiles,
)


def main():
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.set_num_threads(1)
    torch.manual_seed(741)
    checked = 0
    cases = [(e, min(e, 1024 - e)) for e in (72, 128, 288, 896)]
    if os.environ.get("MOE_TEST_ODD_GROUPS") == "1":
        cases = [(18, 212), (24, 270), (32, 663), (72, 71), (896, 127)]
    for e, d in cases:
        h, start = max(3 * e, d + 11), e
        n = max(4 * e, h + e)
        g, k, width, rows = e + d, 128, 128, 512
        wc = torch.randint(-4, 5, (g, k, width), dtype=torch.int8)
        xc = torch.randint(-4, 5, (rows, k), dtype=torch.int8)
        sc = torch.tensor([1 / 128, 1 / 64, 1 / 32, 1 / 16]).repeat(g * width // 4).reshape(g, width).roll(1, 0)
        w = torch_npu.npu_format_cast(wc.to("npu"), 29)
        x, scales = xc.to("npu"), sc.to("npu", torch.bfloat16)
        token_scales = torch.ones(rows, device="npu")
        counts = torch.zeros(e, dtype=torch.int64, device="npu")
        error = torch.zeros((), dtype=torch.int32, device="npu")
        enabled = torch.ones((), dtype=torch.bool, device="npu")
        access, out_counts = torch.zeros_like(counts), torch.empty_like(counts)
        hot = torch.full((n,), -1, dtype=torch.int64, device="npu")
        keys = torch.full((e,), -1, dtype=torch.int64, device="npu")
        groups = torch.zeros((g, 2), dtype=torch.int64, device="npu")
        sources = torch.arange(4 * n, device="npu").reshape(4, n)
        lengths = torch.full((4, e), 128, dtype=torch.int32, device="npu")
        out_sources = torch.empty(4 * e, dtype=torch.int64, device="npu")
        out_lengths = torch.empty(4 * e, dtype=torch.uint32, device="npu")
        tiles = triton.cdiv(e, 64)
        bad = torch.empty(tiles, dtype=torch.int32, device="npu")
        active = torch.empty_like(bad)

        def run(cumulative):
            groups.zero_()
            validate_prepare_tiles[(tiles,)](
                counts, hot, bad, active, E=e, H=h, KEY_START=start,
                CUMULATIVE=cumulative, DIRECT=True, BLOCK=64,
            )
            prepare_copy_tiles[(tiles,)](
                counts, error, enabled, access, sources[:, start:start + e], lengths,
                out_counts, out_sources, out_lengths, hot, groups, keys, bad, active,
                E=e, H=h, KEY_START=start, SOURCE_STRIDE=n, COMPONENT_COUNT=4,
                COMPONENT_MAP=0x3210, CUMULATIVE=cumulative, DIRECT=True,
                DIRECT_CAPACITY=d, HOT_OFFSET=e, TILES=tiles,
                TILE_BLOCK=triton.next_power_of_2(tiles), BLOCK=64,
            )
            return torch_npu.npu_grouped_matmul(
                [x], [w], scale=[scales], per_token_scale=[token_scales],
                group_list=groups, group_list_type=2, group_type=0,
                split_item=2, output_dtype=torch.bfloat16,
            )[0]

        for cumulative in (False, True):
            error.zero_()
            hot.fill_(-1)
            counts.zero_()
            counts[0] = 1
            if cumulative:
                counts.fill_(1)
            run(cumulative)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                y = run(cumulative)
            for step in range(12):
                logical = torch.randperm(e)[: (0, 1, 6, 16, 32, min(e, 128))[step % 6]].sort().values
                cnt = torch.randint(1, 5, (len(logical),))
                raw = torch.zeros(e, dtype=torch.int64)
                raw[logical] = cnt
                slots = torch.full((n,), -1, dtype=torch.int64)
                # Random unique Hot identities, including outside the direct view.
                slots[start:start + e] = torch.randperm(h)[:e]
                slots[start:start + e][torch.arange(e) % 3 == 0] = -1
                hot.copy_(slots)
                counts.copy_(raw.cumsum(0) if cumulative else raw)
                keys.fill_(-1)
                access.zero_()
                error.zero_()
                graph.replay()
                torch.npu.synchronize()
                assert error.item() == 0
                torch.testing.assert_close(access.cpu(), raw)
                expected, offset = [], 0
                for expert in logical.tolist():
                    slot, count = int(slots[start + expert]), int(raw[expert])
                    physical = e + slot if 0 <= slot < d else expert
                    value = xc[offset:offset + count].int() @ wc[physical].int()
                    expected.append((value.float() * sc[physical]).to(torch.bfloat16))
                    offset += count
                if expected:
                    torch.testing.assert_close(y[:offset].cpu(), torch.cat(expected), rtol=0, atol=0)
                checked += 1
            # A fault ONLY in the last tile must suppress work in the first
            # tile too. This catches missing cross-tile fault propagation.
            for invalid_slot in (False, True):
                raw = torch.zeros(e, dtype=torch.int64)
                raw[0], raw[-1] = 1, 1 if invalid_slot else -1
                counts.copy_(raw.cumsum(0) if cumulative else raw)
                hot.fill_(-1)
                if invalid_slot:
                    hot[start + e - 1] = h
                keys.fill_(-1)
                error.zero_()
                access.zero_()
                graph.replay()
                torch.npu.synchronize()
                assert error.item() != 0
                assert not out_lengths.cpu().long().count_nonzero()
                assert (keys.cpu() == -1).all() and not access.cpu().count_nonzero()
            print(f"PASS E={e} groups={g} cumulative={cumulative}", flush=True)
    print(f"PASS {checked} tiled preparation + actual sparse GMM graph cases and {4 * len(cases)} fault checks", flush=True)


if __name__ == "__main__":
    main()
