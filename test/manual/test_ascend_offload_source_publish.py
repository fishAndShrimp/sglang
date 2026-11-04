"""Large source-table graph oracle and original-vs-tiled device timings."""

import json
import os
import statistics
from types import SimpleNamespace

import torch

from sglang.srt.offloading.moe.npu.source_publish import TiledSourcePublisher


def main():
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    results = []
    cases = ((9472, 2368), (18944, 1136), (82432, 23184), (82432, 41216))
    if os.environ.get("MOE_TEST_TAIL_CATALOGS") == "1":
        cases = ((12096, 3402), (10304, 2898))
    for n, h in cases:
        c, stride = 6, n + h + 2
        pointer_base = 0x123456780000
        state = SimpleNamespace(
            key_to_slot=torch.full((n,), -1, dtype=torch.int64, device='npu'),
            slot_to_key=torch.zeros(h, dtype=torch.int64, device='npu'),
            slot_valid=torch.ones(h, dtype=torch.bool, device='npu'),
        )
        o = SimpleNamespace(catalog=SimpleNamespace(num_experts=n), hot_slots=h,
                            device=torch.device('npu:0'), state=state,
                            sources=(torch.arange(c * stride, dtype=torch.int64, device='npu') * 256 + pointer_base).reshape(c, stride),
                            error=torch.zeros((), dtype=torch.int32, device='npu'))
        publisher = TiledSourcePublisher(o)
        def set_valid(step):
            k2s = torch.full((n,), -1, dtype=torch.int64)
            keys = torch.randperm(n)[:h]
            valid = torch.ones(h, dtype=torch.bool)
            if step % 3 == 0:
                valid[::3] = False
            k2s[keys[valid]] = torch.arange(h)[valid]
            state.key_to_slot.copy_(k2s)
            state.slot_to_key.copy_(keys)
            state.slot_valid.copy_(valid)
            o.error.zero_()
            indices = torch.arange(n) + 1
            for slot, key in enumerate(keys.tolist()):
                if valid[slot]:
                    indices[key] = n + 2 + slot
            return torch.stack([(indices + component * stride) * 256 + pointer_base for component in range(c)])
        set_valid(1)
        publisher.run()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            publisher.run()
        for step in range(12):
            expected = set_valid(step)
            graph.replay()
            torch.npu.synchronize()
            assert o.error.item() == 0
            torch.testing.assert_close(publisher.output.cpu(), expected, rtol=0, atol=0)
        for fault in ('negative', 'high_slot', 'reverse', 'invalid', 'latched'):
            set_valid(1)
            if fault == 'negative': state.key_to_slot[-1] = -2
            elif fault == 'high_slot': state.key_to_slot[-1] = h
            elif fault == 'reverse': state.slot_to_key[-1] = n
            elif fault == 'invalid': state.slot_valid[-1] = False
            else: o.error.fill_(4)
            graph.replay()
            torch.npu.synchronize()
            assert o.error.item() == (4 if fault == 'latched' else 1), fault
        set_valid(1)
        old_output = torch.empty_like(publisher.output)
        def old():
            keys = torch.arange(n, device='npu')
            slots = state.key_to_slot
            safe = slots.clamp(0, h - 1)
            hit = slots >= 0
            coherent = state.slot_valid.to(torch.int32).gather(0, safe).bool() & (state.slot_to_key.gather(0, safe) == keys)
            bad = (slots < -1) | (slots >= h) | (hit & ~coherent)
            o.error.bitwise_or_(bad.any().to(torch.int32))
            indices = torch.where(hit, safe + n + 2, keys + 1)
            old_output.copy_(o.sources.gather(1, indices.unsqueeze(0).expand(c, -1)))
        def time_graph(fn):
            fn()
            torch.npu.synchronize()
            g = torch.npu.NPUGraph()
            with torch.npu.graph(g):
                for _ in range(32): fn()
            times = []
            for _ in range(7):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record(); g.replay(); end.record(); end.synchronize()
                times.append(start.elapsed_time(end) * 1000 / 32)
            assert o.error.item() == 0
            return dict(median_us=statistics.median(times), samples_us=times)
        result = dict(n=n, h=h, correct_cases=12, fault_cases=5,
                      old=time_graph(old), tiled=time_graph(publisher.run))
        torch.testing.assert_close(old_output.cpu(), publisher.output.cpu(), rtol=0, atol=0)
        results.append(result)
        print(json.dumps(result), flush=True)
    with open(os.environ['MOE_TIMING_OUTPUT'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f'PASS {12 * len(cases)} graph address cases, {5 * len(cases)} faults and {len(cases)} matched timing shapes', flush=True)


if __name__ == '__main__':
    main()
