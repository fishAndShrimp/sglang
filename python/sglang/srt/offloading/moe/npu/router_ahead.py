"""Opt-in, bounded observation of R_(l+1)(h_l), without changing execution.

Prediction only uses the current MoE input and the next layer's existing router.
No extra collective, CPU readback, cache update or weight copy runs in the hook.
Snapshots are exported after the normal forward completion on an explicit marker.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _store_prediction(Ids, Weights, Active, OutIds, OutWeights,
                      ROWS: tl.constexpr, K: tl.constexpr, CAP: tl.constexpr,
                      BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    row = i // K
    active = tl.load(Active + row, row < ROWS, 0) != 0
    valid = (row < ROWS) & active
    expert = tl.load(Ids + i, valid & (i < CAP*K), -1)
    weight = tl.load(Weights + i, valid & (i < CAP*K), 0)
    tl.store(OutIds + i, expert, i < CAP*K)
    tl.store(OutWeights + i, weight, i < CAP*K)


class RouterAhead:
    def __init__(self, owner, model):
        self.owner = owner
        self.mode = owner.cfg.get('router_ahead_mode', 'off')
        if self.mode != 'shadow':
            raise ValueError('Initial router-ahead experiment only implements shadow mode')
        if owner.decode_active_rows is None or owner.diagnostics is None:
            raise ValueError('Router-ahead observation requires row mask and forward diagnostics')
        self.output = Path(owner.cfg['router_ahead_directory'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.capacity = owner.diagnostics.capacity
        self.row_capacity = 32
        self.layers = [s.layer_id for s in owner.catalog.offloaded_layers]
        self.k = owner.hf_config.num_experts_per_tok
        self.ids = torch.full((len(self.layers), self.row_capacity, self.k), -1,
                              dtype=torch.int32, device=owner.device)
        self.weights = torch.zeros(self.ids.shape, dtype=torch.float32, device=owner.device)
        self.saved_ids = torch.empty((self.capacity, *self.ids.shape), dtype=self.ids.dtype, device=owner.device)
        self.saved_weights = torch.empty_like(self.saved_ids, dtype=torch.float32)
        self.frames = []
        self.dumped = False
        self.hooks = []
        self.async_prediction = owner.cfg.get('router_ahead_async', False)
        self.fused_router = owner.cfg.get('router_ahead_fused_router', False)
        self.fused_record = owner.cfg.get('router_ahead_fused_record', False)
        self.joint_weights = []
        if self.async_prediction and self.fused_router:
            raise ValueError('Joint router and independent prediction stream are separate experiments')
        if self.async_prediction:
            self.stream = torch.npu.Stream()
            self.inputs = torch.empty((len(self.layers), self.row_capacity, owner.hf_config.hidden_size),
                                      dtype=torch.bfloat16, device=owner.device)
            self.input_ready = [torch.npu.Event() for _ in self.layers]
        modules = {m.layer_id: m for m in model.modules()
                   if type(m).__name__ == 'DeepseekV2MoE' and not m.is_nextn}
        for position, target_id in enumerate(self.layers):
            source, target = modules[target_id - 1], modules[target_id]
            if target.is_hash or target.topk.topk_config.scoring_func != 'sqrtsoftplus':
                raise ValueError('Router-ahead prototype expects non-hash DSV4 sqrtsoftplus targets')
            if self.fused_router:
                if source.gate.weight.dtype != target.gate.weight.dtype:
                    raise ValueError('Joint router requires equal weight dtypes')
                with torch.no_grad():
                    joint = torch.cat((source.gate.weight, target.gate.weight), dim=0)
                self.joint_weights.append(joint)
                original = source.gate.forward
                source.gate.forward = self._joint_gate(position, target, joint, original)
            else:
                self.hooks.append(source.register_forward_pre_hook(self._hook(position, target)))
        if self.async_prediction:
            # One end-of-model join, never a per-layer wait for predictions.
            self.hooks.append(modules[self.layers[-1]].register_forward_hook(self._join))

    def _joint_gate(self, position, target, joint, original):
        def run(hidden, gemm_output_zero_allocator=None, forward_batch=None):
            if (forward_batch is None or forward_batch.forward_mode.name not in
                    ('DECODE', 'TARGET_VERIFY', 'IDLE') or hidden.shape[0] == 0
                    or hidden.shape[0] > self.row_capacity):
                return original(hidden, gemm_output_zero_allocator, forward_batch)
            x = hidden.float() if joint.dtype == torch.float32 else hidden
            logits = F.linear(x, joint)
            experts = target.gate.weight.shape[0]
            self._predict_logits(position, target, hidden, logits[:, experts:])
            return logits[:, :experts]
        return run

    def _join(self, module, args, output):
        if args[1].forward_mode.name in ('DECODE', 'TARGET_VERIFY', 'IDLE'):
            torch.npu.current_stream().wait_stream(self.stream)

    def _hook(self, position, target):
        def predict(module, args):
            hidden, batch = args[0], args[1]
            if batch.forward_mode.name not in ('DECODE', 'TARGET_VERIFY', 'IDLE'):
                return
            rows = hidden.shape[0]
            if rows == 0 or rows > self.row_capacity:
                return  # cleared snapshot denotes unsupported shape, never a stale prediction
            if self.async_prediction:
                # Dedicated storage avoids in-place changes or allocator reuse of
                # the main path's MoE input while the speculative stream reads it.
                self.inputs[position, :rows].copy_(hidden)
                self.input_ready[position].record(torch.npu.current_stream())
                with torch.npu.stream(self.stream):
                    self.stream.wait_event(self.input_ready[position])
                    self._predict(position, target, self.inputs[position, :rows], batch)
            else:
                self._predict(position, target, hidden, batch)
        return predict

    def _predict(self, position, target, hidden, batch):
        logits = target.gate(hidden, forward_batch=batch)
        self._predict_logits(position, target, hidden, logits)

    def _predict_logits(self, position, target, hidden, logits):
        rows = hidden.shape[0]
        cfg = target.topk.topk_config
        # Same low-level router op as the real NPU top-k, but do not call
        # TopK's recorder/EPLB/LFU hooks for speculative selections.
        weights, ids, _ = torch.ops.custom.npu_moe_gating_top_k(
            x=logits.to(torch.float32), k=self.k,
            bias=cfg.correction_bias.to(torch.float32) if cfg.correction_bias is not None else None,
            input_ids=None, tid2eid=None, routed_scaling_factor=1.0, norm_type=2)
        active = self.owner.decode_row_mask(rows)
        if self.fused_record:
            _store_prediction[(1,)](ids, weights, active, self.ids[position], self.weights[position],
                ROWS=rows, K=self.k, CAP=self.row_capacity,
                BLOCK=triton.next_power_of_2(self.row_capacity*self.k))
        else:
            self.ids[position, :rows].copy_(ids.to(torch.int32).masked_fill(~active[:, None], -1))
            self.weights[position, :rows].copy_(weights.float().masked_fill(~active[:, None], 0))

    def begin(self):
        self.ids.fill_(-1)
        self.weights.zero_()

    def snapshot(self, index):
        if index is None or self.dumped:
            return
        self.saved_ids[index].copy_(self.ids)
        self.saved_weights[index].copy_(self.weights)
        self.frames.append(index)

    def maybe_dump(self):
        if self.dumped or not (self.output/'dump.request').exists():
            return
        n = len(self.frames)
        assert self.frames == list(range(n)), 'prediction frames must align with diagnostics'
        rank = self.owner.diagnostics_rank
        np.savez_compressed(self.output/f'rank{rank}.npz',
                            predicted_ids=self.saved_ids[:n].cpu().numpy(),
                            predicted_weights=self.saved_weights[:n].cpu().numpy())
        (self.output/f'rank{rank}.json').write_text(json.dumps(dict(
            mode=self.mode, rank=rank, layers=self.layers, frames=n,
            row_capacity=self.row_capacity, topk=self.k,
            prediction='next-layer original router on current-layer normalized MoE input',
            no_extra_collectives=True, no_prefetch_yet=True, async_prediction=self.async_prediction, fused_router=self.fused_router, fused_record=self.fused_record,
            note='Align frame indices with diagnostics; unsupported/extend rows are -1. Not a prediction model trained on future routing.'), indent=2)+'\n')
        self.dumped = True
