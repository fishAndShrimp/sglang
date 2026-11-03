"""BS1 local prefetch using the token already delivered by MC2.

No new decode collective, physical-ID remap, SHARED Host pool or LFU rewrite.
The owner can prefetch only if it received this layer's token. A startup-only
global cold mask selects one deterministic candidate; current Hot always wins.
"""
import json
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.distributed.parallel_state import get_moe_ep_group
from ..resources import ResourceKind
from .transfer import _group_pack_copy
from .replica_flag import ReplicaFlag


@triton.jit
def _snapshot_input(X, Counts, Enabled, Input, HasInput, Scale,
                    HIDDEN:tl.constexpr, EXPERTS:tl.constexpr,
                    EB:tl.constexpr, BLOCK:tl.constexpr, INT8:tl.constexpr=False):
    e=tl.arange(0,EB)
    has=tl.sum(tl.load(Counts+e,e<EXPERTS,0),0)>0
    has=has & (tl.load(Enabled)!=0)
    h=tl.arange(0,BLOCK)
    x=tl.load(X+h,(h<HIDDEN)&has,0)
    if INT8:
        x=x.to(tl.float32)*tl.load(Scale,has,0)
    tl.store(Input+h,x,h<HIDDEN)
    tl.store(HasInput,has.to(tl.int32))


@triton.jit
def _plan(Ids, Weights, Cold, Slots, HasInput, Sources, Sizes,
          OutSources, OutLengths, Completion, Stats,
          K:tl.constexpr, LOCAL:tl.constexpr, GLOBAL:tl.constexpr,
          RANK:tl.constexpr, STRIDE:tl.constexpr, C:tl.constexpr, BLOCK:tl.constexpr,
          EP:tl.constexpr=4, BUSIEST:tl.constexpr=False,W13_ONLY:tl.constexpr=False,COPY:tl.constexpr=True):
    i=tl.arange(0,BLOCK)
    e=tl.load(Ids+i,i<K,-1).to(tl.int32)
    w=tl.load(Weights+i,i<K,0)
    valid=(i<K)&(e>=0)&(e<GLOBAL)
    if BUSIEST:
        busy=tl.full((),0,tl.int32)
        maximum=tl.full((),-1,tl.int32)
        for owner in tl.static_range(EP):
            load=tl.sum((valid&(e//LOCAL==owner)).to(tl.int32),0)
            busy=tl.where(load>maximum,owner,busy)
            maximum=tl.maximum(maximum,load)
        own=valid&(e//LOCAL==RANK)
        cold=(tl.load(Slots+tl.maximum(e,0)%LOCAL,own,0)<0)&own&(busy==RANK)
    else:
        cold=tl.load(Cold+tl.maximum(e,0),valid,0)!=0
    score=tl.where(valid&cold,w,float('-inf'))
    best=tl.max(score,0)
    index=tl.min(tl.where(valid&cold&(score==best),i,BLOCK),0)
    chosen=tl.sum(tl.where(i==index,e+1,0),0)-1
    local=tl.maximum(chosen,0)%LOCAL
    slot=tl.load(Slots+local)
    has=tl.load(HasInput)!=0
    copy=has&(chosen>=0)&(chosen//LOCAL==RANK)&(slot<0)
    for c in tl.static_range(C):
        address=tl.load(Sources+c*STRIDE+local)
        size=tl.load(Sizes+c)
        tl.store(OutSources+c,address)
        tl.store(OutLengths+c,tl.where(copy & ((c<3) | (not W13_ONLY)),size,0))
    tl.store(Completion,tl.where(copy & COPY,local+1,0))
    tl.store(Stats,tl.load(Stats)+has.to(tl.int64))
    tl.store(Stats+32,tl.load(Stats+32)+copy.to(tl.int64))
    tl.store(Stats+64,tl.load(Stats+64)+(copy & COPY).to(tl.int64))


@triton.jit
def _prepare(Counts, Error, Enabled, Access, Sources, Lengths,
             Replica, Observed, Slots, Stats, OutCounts, OutSources, OutLengths,
             EXPERTS:tl.constexpr, SOURCE_STRIDE:tl.constexpr,
             COMPONENT_COUNT:tl.constexpr, SKIP_OFFSETS:tl.constexpr,
             CUMULATIVE:tl.constexpr, USE:tl.constexpr, BLOCK:tl.constexpr,
             PATCH:tl.constexpr=False,W13_ONLY:tl.constexpr=False):
    e=tl.arange(0,BLOCK)
    raw=tl.load(Counts+e,e<EXPERTS,0)
    count=raw
    if CUMULATIVE:
        previous=tl.load(Counts+e-1,(e>0)&(e<EXPERTS),0)
        count=raw-previous
    bad=tl.sum(((count<0)&(e<EXPERTS)).to(tl.int32),0)>0
    error=tl.load(Error)|bad.to(tl.int32)
    tl.store(Error,error)
    tl.store(OutCounts+e,tl.where(error==0,raw,0),e<EXPERTS)
    enabled=tl.load(Enabled)!=0
    accesses=tl.load(Access+e,e<EXPERTS,0)
    tl.store(Access+e,accesses+tl.where(enabled,count,0),e<EXPERTS)
    needed=(count>0)&(error==0)
    ready=tl.load(Observed)-1
    if PATCH:
        in_range=(ready>=0)&(ready<EXPERTS)
        safe=tl.maximum(ready,0)
        slot=tl.load(Slots+safe,in_range,0)
        selected=tl.load(Counts+safe,in_range,0)
        if CUMULATIVE:
            selected-=tl.load(Counts+safe-1,in_range&(ready>0),0)
        hit=in_range&(slot<0)&(selected>0)&(error==0)
        n=hit.to(tl.int64)
    else:
        cold=tl.load(Slots+e,e<EXPERTS,0)<0
        hit=(ready>=0)&(e==ready)&cold&needed&(e<EXPERTS)
        n=tl.sum(hit.to(tl.int64),0)
    for position in tl.static_range(COMPONENT_COUNT):
        c=(position//2)*3+position%2 if SKIP_OFFSETS else position
        base=tl.load(Sources+c*SOURCE_STRIDE+e,e<EXPERTS,0)
        source=base
        if not PATCH and ((not W13_ONLY) or c<3):
            replica=tl.load(Replica+c)
            source=tl.where(hit&USE,replica,base)
        size=tl.load(Lengths+c*EXPERTS+e,e<EXPERTS,0)
        tl.store(OutSources+position*EXPERTS+e,source,e<EXPERTS)
        tl.store(OutLengths+position*EXPERTS+e,tl.where(needed,size,0),e<EXPERTS)
    if PATCH and USE:
        # The sparse override aliases one element of each preceding full-row
        # store. Finish those stores before publishing the scalar replacement.
        tl.debug_barrier()
        for position in tl.static_range(COMPONENT_COUNT):
            c=(position//2)*3+position%2 if SKIP_OFFSETS else position
            if (not W13_ONLY) or c<3:
                replica=tl.load(Replica+c)
                tl.store(OutSources+position*EXPERTS+ready,replica,hit)
    tl.store(Stats+96,tl.load(Stats+96)+n)
    tl.store(Stats+128,tl.load(Stats+128)+tl.where(USE,n,0))


class DispatchPrefetch:
    def __init__(self,owner,model):
        self.owner=owner
        self.mode=owner.cfg['dispatch_prefetch_mode']
        if self.mode not in ('unused','use') or not owner.cfg.get('fused_copy_metadata'):
            raise ValueError('Dispatch prefetch requires unused/use and fused metadata')
        self.layers=[s.layer_id for s in owner.catalog.offloaded_layers]
        self.positions={layer:i for i,layer in enumerate(self.layers)}
        self.local=owner.local_experts
        ep=get_moe_ep_group();self.rank=ep.rank_in_group;self.ep=ep.world_size
        self.n=owner.catalog.num_experts
        self.k=owner.hf_config.num_experts_per_tok
        self.busiest=owner.cfg.get('dispatch_prefetch_selection')=='busiest'
        self.w13_only=owner.cfg.get('dispatch_prefetch_w13_only',False)
        self.stage=owner.cfg.get('dispatch_prefetch_stage','full')
        self.consume=owner.cfg.get('dispatch_prefetch_consume',True)
        self.copy_cores=int(owner.cfg.get('dispatch_prefetch_copy_cores',0))
        if self.stage not in ('snapshot','predict','full'):
            raise ValueError('Unknown prefetch stage')
        if (self.consume and self.stage!='full') or (self.mode=='use' and not self.consume):
            raise ValueError('Consumption requires full prefetch; use requires consumption')
        if not 0 <= self.copy_cores <= 32:
            raise ValueError('A3 experimental copy budget must be 0..32')
        self.budget_copy=None
        if self.copy_cores:
            from .copy_budget import CopyBudgetKernel
            self.budget_copy=CopyBudgetKernel(owner.cfg['dispatch_prefetch_copy_library'])
        if self.busiest:
            # Dummy pointer for an unused kernel argument; no cache-map exchange.
            self.initial_cold=torch.zeros((len(self.layers),1),dtype=torch.bool,device=owner.device)
        else:
            cold=(owner.state.key_to_slot<0).reshape(len(self.layers),self.local).cpu().tolist()
            cold_by_rank=[None]*self.ep
            dist.all_gather_object(cold_by_rank,cold,group=ep.cpu_group)
            self.initial_cold=torch.tensor(cold_by_rank,dtype=torch.bool,device=owner.device).permute(1,0,2).reshape(len(self.layers),-1).contiguous()
        self.names=tuple(owner.format.layouts);self.c=len(self.names)
        self.bank=owner.format.allocate_bank(len(self.layers),ResourceKind.EXECUTION_WEIGHTS,'dispatch_prefetch')
        self.addresses=torch.tensor([[self.bank.component_regions[n][i].address for n in self.names]
                                    for i in range(len(self.layers))],dtype=torch.int64,device=owner.device)
        self.sources=torch.zeros_like(self.addresses)
        self.lengths=torch.zeros_like(self.addresses,dtype=torch.int32)
        self.sizes=torch.tensor([owner.format.layouts[n].nbytes for n in self.names],dtype=torch.int32,device=owner.device)
        self.host_sources=owner.sources[:,1:self.n+1]
        self.groups=torch.ones(self.c,dtype=torch.int64,device=owner.device)
        self.scan=torch.tensor(self.c,dtype=torch.uint32,device=owner.device)
        self.packed=torch.zeros_like(self.addresses)
        self.inputs=torch.zeros((len(self.layers),1,owner.hf_config.hidden_size),dtype=torch.bfloat16,device=owner.device)
        # Different layers/streams never share a scalar-write cache line.
        self.has_input=torch.zeros((len(self.layers),32),dtype=torch.int32,device=owner.device)
        self.enabled=torch.zeros(1,dtype=torch.int32,device=owner.device)
        self.ready=torch.zeros_like(self.has_input)
        self.observed=torch.zeros_like(self.has_input)
        self.completion=torch.zeros_like(self.has_input)
        self.stats=torch.zeros((len(self.layers),160),dtype=torch.int64,device=owner.device)
        self.stream=torch.npu.Stream()
        self.pending=False
        self.input_format=None
        self.input_ready=[torch.npu.Event() for _ in self.layers]
        self.flag=ReplicaFlag(owner.cfg['dispatch_prefetch_flag_library'])
        self.output=Path(owner.cfg['dispatch_prefetch_directory']);self.output.mkdir(parents=True,exist_ok=True)
        self.dumped=False;self.host_stats=dict(forwards=0,eligible=0)
        modules={m.layer_id:m for m in model.modules() if type(m).__name__=='DeepseekV2MoE' and not m.is_nextn}
        self.targets=[]
        for target_id in self.layers:
            target=modules[target_id]
            if target.is_hash or target.topk.topk_config.scoring_func!='sqrtsoftplus':
                raise ValueError('Initial local predictor expects non-hash sqrtsoftplus targets')
            self.targets.append(target)
            modules[target_id-1].experts._dispatch_prefetch=self
        self.hook=modules[self.layers[-1]].register_forward_hook(self._join)

    def _join(self,module,args,output):
        self.drain()

    def drain(self):
        # A prefill/no-producer graph must not wait on an unregistered stream.
        # During capture this join becomes a graph task, so replay needs no
        # Python pending flag. All per-layer work is joined once at model end.
        if self.pending:
            torch.npu.current_stream().wait_stream(self.stream)
            self.pending=False

    def begin(self,batch):
        counts=getattr(batch,'original_global_num_tokens_cpu',None)
        eligible=(batch is not None and batch.forward_mode.name in ('DECODE','IDLE')
                  and counts is not None and sum(counts)==1 and getattr(batch,'spec_info',None) is None)
        self.ready.zero_();self.observed.zero_();self.enabled.fill_(int(eligible))
        self.host_stats['forwards']+=1;self.host_stats['eligible']+=int(eligible)

    def on_dispatch(self,source_layer,x,counts,scales):
        target_id=source_layer+1
        if target_id not in self.positions or x.shape[0]==0:
            return
        if x.dtype not in (torch.bfloat16,torch.int8):
            raise ValueError(f'Unsupported speculative dispatch input: {x.dtype}')
        quantized=x.dtype==torch.int8
        if quantized and (scales is None or scales.dtype!=torch.float32):
            raise ValueError('INT8 prediction input requires per-token FP32 scale')
        if self.input_format is None:
            self.input_format=str(x.dtype)
            print(f'DispatchPrefetch rank={self.rank} input={x.dtype} shape={tuple(x.shape)} scale={None if scales is None else tuple(scales.shape)}',flush=True)
        p=self.positions[target_id];o=self.owner
        _snapshot_input[(1,)](x,counts,self.enabled,self.inputs[p],self.has_input[p,:1],scales if quantized else x,
            HIDDEN=x.shape[1],EXPERTS=counts.numel(),EB=triton.next_power_of_2(counts.numel()),
            BLOCK=triton.next_power_of_2(x.shape[1]),INT8=quantized)
        self.input_ready[p].record(torch.npu.current_stream())
        self.pending=True
        with torch.npu.stream(self.stream):
            self.stream.wait_event(self.input_ready[p])
            if self.stage=='snapshot':return
            target=self.targets[p]
            x=self.inputs[p]
            logits=F.linear(x.float() if target.gate.weight.dtype==torch.float32 else x,target.gate.weight)
            cfg=target.topk.topk_config
            weights,ids,_=torch.ops.custom.npu_moe_gating_top_k(x=logits.float(),k=self.k,
                bias=cfg.correction_bias.float() if cfg.correction_bias is not None else None,
                input_ids=None,tid2eid=None,routed_scaling_factor=1.0,norm_type=2)
            start=o.catalog.key(target_id,0)
            _plan[(1,)](ids,weights,self.initial_cold[p],o.state.key_to_slot[start:start+self.local],
                self.has_input[p,:1],self.host_sources[:,start:start+self.local],self.sizes,
                self.sources[p],self.lengths[p],self.completion[p,:1],self.stats[p],
                K=self.k,LOCAL=self.local,GLOBAL=self.local*self.ep,RANK=self.rank,
                STRIDE=self.host_sources.stride(0),C=self.c,BLOCK=triton.next_power_of_2(self.k),
                EP=self.ep,BUSIEST=self.busiest,W13_ONLY=self.w13_only,COPY=self.stage=='full')
            if self.stage!='full':return
            args=(self.sources[p],self.addresses[p],self.lengths[p].view(torch.uint32),
                  self.scan,self.groups,self.packed[p],o.device)
            status=(self.budget_copy(*args,self.copy_cores) if self.budget_copy is not None
                    else _group_pack_copy(*args))
            if status not in (None,0):raise RuntimeError(f'Local prefetch submission failed: {status}')
            if self.consume:self.flag(self.completion[p,:1],self.ready[p,:1])

    def prepare(self,layer_id,runner_input):
        o=self.owner;p=self.positions[layer_id];counts=runner_input.expert_tokens
        if counts.shape!=(self.local,) or counts.dtype!=torch.int64 or runner_input.group_list_type not in (0,1):
            raise ValueError('Local prefetch expects original per-local-expert int64 counts')
        start=o.catalog.key(layer_id,0)
        self.flag(self.ready[p,:1],self.observed[p,:1])
        _prepare[(1,)](counts,o.error,o.state.update_enabled,o.state.step_access_count[start:start+self.local],
            o.published_sources[:,start:start+self.local],o.lengths,self.addresses[p],self.observed[p,:1],
            o.state.key_to_slot[start:start+self.local],self.stats[p],o.fused_counts,o.fused_sources,o.fused_lengths,
            EXPERTS=self.local,SOURCE_STRIDE=self.n,COMPONENT_COUNT=len(o.fused_components),
            SKIP_OFFSETS=o.cfg.get('skip_zero_offsets',False),CUMULATIVE=runner_input.group_list_type==0,
            USE=self.mode=='use',BLOCK=triton.next_power_of_2(self.local),
            PATCH=o.cfg.get('dispatch_prefetch_patch',False),W13_ONLY=self.w13_only)
        runner_input.expert_tokens=o.fused_counts
        o._launch_copies(layer_id,o.fused_descriptors)
        return o.quant_info

    def maybe_dump(self):
        if self.dumped or not (self.output/'dump.request').exists():return
        rows=self.stats[:,::32].cpu().tolist()
        (self.output/f'rank{self.rank}.json').write_text(json.dumps(dict(mode=self.mode,
            layers=self.layers,columns=['received_token','owned_cold_candidate','copied','ready_needed','used'],
            counts=rows,totals=[sum(row[i] for row in rows) for i in range(5)],host_stats=self.host_stats,
            replica_bytes=len(self.layers)*self.owner.format.expert_nbytes,
            input_format=self.input_format,
            sparse_patch=self.owner.cfg.get('dispatch_prefetch_patch',False),
            w13_only=self.w13_only,stage=self.stage,consume=self.consume,copy_cores=self.copy_cores,
            prefetched_components=list(self.names[:3] if self.w13_only else self.names),
            no_decode_collectives=True,no_route_remap=True,initial_cold_mask_only=not self.busiest,
            selection='predicted_busiest_then_local_cold' if self.busiest else 'initial_cold_mask'),indent=2)+'\n')
        self.dumped=True
