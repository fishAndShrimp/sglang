"""Opt-in, bounded forward snapshots for local offload investigations.

No per-layer CPU reads. Reconstruct direct-copy decisions from the exact
pre-maintenance counts/map and verify the final execution keys. This is a
diagnostic run, not an uninstrumented throughput measurement.
"""
import json
import os
from pathlib import Path


def replay_sources(counts, hot_map, initial_keys, layer_ids, local_experts, direct_slots):
    keys = list(initial_keys)
    rows = []
    for index, layer in enumerate(layer_ids):
        start = index * local_experts
        row = dict(layer=layer, pairs=0, active=0, hot=0, reuse=0, host=0,
                   hot_pairs=0, reuse_pairs=0, host_pairs=0, d2d=0, resident_hits=0)
        for expert in range(local_experts):
            key = start + expert
            count, slot = counts[key], hot_map[key]
            if count < 0:
                raise ValueError("Negative expert count")
            if count == 0:
                continue
            row['pairs'] += count
            row['active'] += 1
            row['resident_hits'] += slot >= 0
            if 0 <= slot < direct_slots:
                source = 'hot'
            elif keys[expert] == key:
                source = 'reuse'
            else:
                source = 'hot' if slot >= 0 else 'host'
                row['d2d'] += slot >= 0
                keys[expert] = key
            row[source] += 1
            row[source + '_pairs'] += count
        rows.append(row)
    return rows, keys


class ForwardSnapshots:
    def __init__(self, owner):
        import torch
        self.owner = owner
        self.torch = torch
        self.root = Path(os.environ['MOE_DIAGNOSTIC_DIR'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.done = set()
        self.active = None
        self.pending = False

    def begin(self, batch):
        self.pending = False
        if batch is None or batch.is_extend_in_batch:
            return
        # Only decode/idle are in the present no-MTP experiment.
        if not (batch.forward_mode.is_decode() or batch.forward_mode.is_idle()):
            return
        control = self.root / 'control.json'
        if not control.exists():
            return
        spec = json.loads(control.read_text())
        run = spec['id']
        if not run.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Invalid diagnostic run name')
        if run in self.done:
            return
        if self.active != run:
            steps = int(spec.get('steps', 32))
            if not 1 <= steps <= 128:
                raise ValueError('Bounded snapshots require 1..128 steps')
            self.active, self.spec, self.index = run, spec, 0
            self.skip = int(spec.get('skip', 32))
            o, t = self.owner, self.torch
            self.counts = t.empty((steps, o.catalog.num_experts), dtype=t.int64, device=o.device)
            self.slots = t.empty_like(self.counts)
            self.before = t.full((steps, o.local_experts), -1, dtype=t.int64, device=o.device)
            self.after = t.empty_like(self.before)
            self.maintenance = t.zeros(steps, dtype=t.int64, device=o.device)
            self.metadata = []
        if self.skip:
            self.skip -= 1
            return
        o = self.owner
        if o.execution_keys is not None:
            self.before[self.index].copy_(o.execution_keys)
        self.metadata.append(dict(step=o.step + 1,
                                  local_tokens=0 if batch.forward_mode.is_idle() else batch.input_ids.numel()))
        self.pending = True

    def before_maintenance(self):
        if not self.pending:
            return
        o, i = self.owner, self.index
        self.counts[i].copy_(o.state.step_access_count)
        self.slots[i].copy_(o.state.key_to_slot)
        if o.execution_keys is not None:
            self.after[i].copy_(o.execution_keys)
        else:
            self.after[i].fill_(-1)

    def after_maintenance(self, due):
        if self.pending and due:
            self.maintenance[self.index].copy_(self.owner.cache_update.plan.count)

    def flush_if_complete(self):
        if not self.pending:
            return
        self.pending = False
        self.index += 1
        if self.index != self.counts.shape[0]:
            return
        o = self.owner
        counts, slots, before, after, maintenance = (
            x.cpu().tolist() for x in
            (self.counts, self.slots, self.before, self.after, self.maintenance)
        )
        layers = [s.layer_id for s in o.catalog.offloaded_layers]
        names = tuple(o.format.layouts)
        components = {names[c]: o.format.layouts[names[c]].nbytes for c in o.fused_components}
        phase_bytes = {prefix: sum(n for name, n in components.items() if name.startswith(prefix))
                       for prefix in ('w13', 'w2')}
        rows = []
        for index, count in enumerate(counts):
            layer_rows, expected = replay_sources(count, slots[index], before[index],
                                                  layers, o.local_experts, o.direct_hot_slots)
            if expected != after[index]:
                raise RuntimeError('Diagnostic source replay disagrees with actual execution keys')
            for row in layer_rows:
                for prefix, size in phase_bytes.items():
                    row[prefix + '_host_bytes'] = row['host'] * size
                    row[prefix + '_d2d_bytes'] = row['d2d'] * size
            rows.append(dict(**self.metadata[index], layers=layer_rows,
                             maintenance_experts=maintenance[index]))
        result = dict(rank=o.catalog.offloaded_layers[0].ep_rank,
                      control=self.spec, direct_hot=o.direct_hot, direct_hot_slots=o.direct_hot_slots,
            execution_reuse=o.capabilities.execution_reuse, optimization_manifest=o.optimization_manifest, hot_slots=o.hot_slots,
                      catalog_experts=o.catalog.num_experts, components=components,
                      interpretation='Exact forward snapshots; source decisions replayed in layer order. '
                      'Direct-path final execution keys verified. Bytes describe selected weight copies, '
                      'not measured bus traffic. Prefill and resident experts excluded.',
                      raw=dict(counts=counts, hot_map=slots, execution_before=before, execution_after=after),
                      steps=rows)
        target = self.root / self.active
        target.mkdir(exist_ok=True)
        path = target / f'rank{result["rank"]}.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result))
        tmp.replace(path)
        self.done.add(self.active)
