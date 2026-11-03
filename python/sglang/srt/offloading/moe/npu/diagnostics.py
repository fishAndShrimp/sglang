"""Bounded, opt-in forward snapshots; export only on an explicit local marker.

The normal path has already collected per-key route counts. Snapshot them before
maintenance changes placement, then retain the actual maintenance copy count.
No per-layer host synchronization or device statistic kernels are added.
"""

import json
from pathlib import Path

import numpy as np
import torch


class ForwardDiagnostics:
    def __init__(self, owner, output, capacity=2048):
        if type(capacity) is not int or not 1 <= capacity <= 16384:
            raise ValueError("diagnostics_capacity must be an integer in 1..16384")
        if not owner.cfg.get("incremental_cache", False):
            raise ValueError("Diagnostics currently require incremental_cache")
        self.owner, self.output, self.capacity = owner, Path(output), capacity
        self.output.mkdir(parents=True, exist_ok=True)
        shape = (capacity, owner.catalog.num_experts)
        self.counts = torch.empty(shape, dtype=torch.int64, device=owner.device)
        self.slots = torch.empty_like(self.counts)
        self.maintenance = torch.zeros(capacity, dtype=torch.int32, device=owner.device)
        self.frames, self.dropped, self.dumped = [], 0, False

    def before_maintenance(self, forward_batch, step, maintenance):
        if self.dumped:
            return None
        i = len(self.frames)
        if i == self.capacity:
            self.dropped += 1
            return None
        with torch.profiler.record_function(f"OFFLOAD_SNAPSHOT::{step}"):
            self.counts[i].copy_(self.owner.state.step_access_count)
            self.slots[i].copy_(self.owner.state.key_to_slot)
        self.frames.append({
            "step": step,
            "phase": forward_batch.forward_mode.name if forward_batch is not None else "unknown",
            "local_batch_size": forward_batch.batch_size if forward_batch is not None else None,
            "global_tokens": getattr(forward_batch, "original_global_num_tokens_cpu", None)
                or getattr(forward_batch, "global_num_tokens_cpu", None),
            "maintenance": maintenance,
        })
        return i

    def after_maintenance(self, index, maintenance):
        if index is not None and maintenance:
            self.maintenance[index].copy_(self.owner.cache_update.plan.count)

    def maybe_dump(self):
        if self.dumped or not (self.output / "dump.request").exists():
            return
        owner = self.owner
        n = len(self.frames)
        # Caller has already synchronized the full forward and checked faults.
        counts, slots = self.counts[:n].cpu().numpy(), self.slots[:n].cpu().numpy()
        copied = self.maintenance[:n].cpu().numpy()
        rank = owner.diagnostics_rank
        np.savez_compressed(self.output / f"rank{rank}.npz", counts=counts,
                            slots=slots, maintenance_experts=copied)
        components = {name: layout.nbytes for name, layout in owner.format.layouts.items()}
        info = {
            "rank": rank, "frames": self.frames, "capacity": self.capacity,
            "dropped_frames": self.dropped, "local_experts": owner.local_experts,
            "layers": [s.layer_id for s in owner.catalog.offloaded_layers],
            "component_nbytes": components,
            "execution_components": [name for name in components
                if not (owner.cfg.get("skip_zero_offsets", False) and name.endswith("_offset"))],
            "maintenance_components": list(components),
            "note": "Counts/slots captured before maintenance; copy count after maintenance. No overwrite. Failed forwards are not exported.",
        }
        (self.output / f"rank{rank}.json").write_text(json.dumps(info, indent=2) + "\n")
        self.dumped = True
