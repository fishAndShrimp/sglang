"""Optional MC2 counts for resident layers; no expert-placement changes."""
import json
from pathlib import Path

import numpy as np
import torch


class ResidentDispatchDiagnostics:
    def __init__(self, owner, modules, output, capacity=2048):
        self.owner = owner
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.layers = [m.layer_id for m in modules]
        self.hash_layers = [m.layer_id for m in modules if m.is_hash]
        self.positions = {layer: i for i, layer in enumerate(self.layers)}
        self.live = torch.zeros((len(modules), owner.local_experts),
                                dtype=torch.int64, device=owner.device)
        self.records = torch.empty((capacity, *self.live.shape),
                                   dtype=torch.int64, device=owner.device)
        self.capacity = capacity
        self.frames = []
        self.dropped = 0
        self.dumped = False

    def begin(self):
        self.live.zero_()

    def observe(self, layer_id, counts):
        if counts.shape != (self.owner.local_experts,) or counts.dtype != torch.int64:
            raise ValueError("Resident MC2 observation requires per-expert int64 counts")
        self.live[self.positions[layer_id]].copy_(counts)

    def snapshot(self, batch, step):
        if self.dumped:
            return
        if len(self.frames) == self.capacity:
            self.dropped += 1
            return
        self.records[len(self.frames)].copy_(self.live)
        self.frames.append(dict(step=step, phase=batch.forward_mode.name if batch is not None else "unknown",
                                global_tokens=getattr(batch, "original_global_num_tokens_cpu", None),
                                local_batch_size=batch.batch_size if batch is not None else None))

    def maybe_dump(self):
        if self.dumped or not (self.output / "dump.request").exists():
            return
        from sglang.srt.distributed.parallel_state import get_moe_ep_group
        rank = get_moe_ep_group().rank_in_group
        np.savez_compressed(self.output / f"rank{rank}.npz",
                            counts=self.records[:len(self.frames)].cpu().numpy())
        data = dict(layers=self.layers, hash_layers=self.hash_layers,
                    frames=self.frames, dropped=self.dropped,
                    resident_mask=self.owner.cfg.get("mask_resident_decode_padding", False))
        (self.output / f"rank{rank}.json").write_text(json.dumps(data, indent=2) + "\n")
        self.dumped = True
