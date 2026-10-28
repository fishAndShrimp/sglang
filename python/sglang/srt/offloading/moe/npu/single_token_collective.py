"""Global-BS1 fixed packet all-gather and weighted all-reduce experiment.

One BF16 token and six FP32 routing weights are packed as FP32 (losslessly for
BF16 and expert IDs). Each rank contributes a fixed packet; padding ranks send
zeros. No DeepEP receive layout, variable token counts, or handles are needed.
"""
from types import SimpleNamespace

import torch

from ..config import Phase
from ..resources import tensor_allocations
from .experts import NPUDispatch, NPUOffloadDispatcher, _Combined, _compute_scope


def local_topk_plan(ids, weights, rank, local_experts):
    """Six selected experts, local active prefix; inputs are shared by all."""
    start = rank * local_experts
    local = (ids >= start) & (ids < start + local_experts)
    order = torch.argsort(torch.where(local, ids, ids.numel() + local_experts * 4 + torch.arange(
        ids.numel(), device=ids.device)).float())
    selected = ids.gather(0, order)
    needed = (selected >= start) & (selected < start + local_experts)
    local_ids = (selected - start).clamp(0, local_experts - 1)
    row_weights = torch.where(needed, weights.gather(0, order), 0)
    demand = (torch.arange(local_experts, device=ids.device).unsqueeze(1) + start == ids).sum(1)
    return local_ids, needed, row_weights, demand


class SingleTokenCollectiveDispatcher(NPUOffloadDispatcher):
    def __init__(self, experts, *, error_status, state):
        super().__init__(experts, error_status=error_status)
        from sglang.srt.distributed import get_moe_ep_group

        self.group = get_moe_ep_group()
        self.state = state
        self.hidden = experts.hidden_size
        self.topk = 6
        if (self.group.world_size != 4 or experts.moe_ep_size != 4
            or experts.moe_tp_size != 1 or experts.num_fused_shared_experts != 0
            or experts.num_local_experts != 64 or experts.reduce_results):
            raise ValueError("Initial BS1 collective requires EP4/DP4, 64 local experts, separate shared FFN")
        self.packet_width = self.hidden + 2 * self.topk + 1
        self.gathered = torch.empty(self.group.world_size * self.packet_width,
                                   dtype=torch.float32, device=error_status.device)

    def allocations(self):
        return tensor_allocations(f"bs1.collective.{self.experts.layer_id}.",
                                  [("gathered", self.gathered)])

    def dispatch(self, hidden_states, routing, context):
        if (context.phase is not Phase.DECODE or hidden_states.shape != (1, self.hidden)
            or hidden_states.dtype != torch.bfloat16 or routing.topk_ids.shape != (1, self.topk)):
            raise ValueError("Collective decode supports exactly one fixed BF16 input slot")
        with _compute_scope(context, hidden_states.device):
            routing, count = self._mask_routing(hidden_states, routing, context)
            active = (count == 1) & (self.error_status == 0)
            invalid_ids = (((routing.topk_ids < 0)
                            | (routing.topk_ids >= self.experts.num_experts)) & active).any()
            self.error_status.bitwise_or_(invalid_ids.to(torch.int32) * 4096)
            active = active & (self.error_status == 0)
            packet = torch.cat((
                torch.where(active, hidden_states.flatten().float(), 0),
                torch.where(active, routing.topk_ids.flatten().float() + 1, 0),
                torch.where(active, routing.topk_weights.flatten().float(), 0),
                active.reshape(1).float()))
            self.group.all_gather_into_tensor(self.gathered, packet)
            total = self.gathered.view(self.group.world_size, self.packet_width).sum(0)
            one = total[-1] == 1
            self.error_status.bitwise_or_((~one & self.state.update_enabled).to(torch.int32) * 2048)
            good = one & (self.error_status == 0)
            ids = torch.where(good, total[self.hidden:self.hidden+self.topk].long() - 1, -1)
            weights = torch.where(good, total[self.hidden+self.topk:-1], 0)
            local_ids, needed, row_weights, demand = local_topk_plan(
                ids, weights, self.experts.moe_ep_rank, self.experts.num_local_experts)
            x = torch.where(good, total[:self.hidden], 0).to(torch.bfloat16)
            # All selected experts consume exactly this token; fixed GMM input
            # has six rows, with unused rows masked by the compact executor.
            raw = SimpleNamespace(hidden_states=x.expand(self.topk, -1).contiguous(),
                hidden_states_scale=None, masked_m=demand, single_token_ids=local_ids,
                single_token_needed=needed, single_token_weights=row_weights,
                topk_ids=ids, topk_weights=weights, format="bs1_collective")
            result = NPUDispatch(raw, context.record_compute_completion(), self.hidden,
                                 None, count, self.topk)
            context.retain((self, packet, total, raw, result))
            return result

    def combine(self, layout, context):
        with _compute_scope(context, layout.output.device):
            weights = layout.dispatched.raw.single_token_weights
            local = (layout.output.float() * weights.unsqueeze(1)).sum(0, keepdim=True)
            reduced = self.group.all_reduce(local)
            result = _Combined(reduced.to(torch.bfloat16), self.hidden,
                               layout.dispatched.valid_token_count)
            context.retain((local, reduced, result))
            return result
