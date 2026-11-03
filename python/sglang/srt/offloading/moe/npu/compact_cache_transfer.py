"""Bounded maintenance descriptors; original cache policy/transaction unchanged.

Initialization uses the original transfer. This adapter is installed afterwards,
with a policy-proven upper bound on swaps. Exceeding that bound latches error
and suppresses all writes. Only certified constant-zero components may be skipped.
"""
import torch
import triton
import triton.language as tl
from sglang.srt.offloading.moe.npu.transfer import (
    _group_pack_copy, _submission_failure, _ticket, _npu_completion,
)

@triton.jit
def compact_valid(Kind, Index, Destination, Valid, Count, Error, Keys, Slots, Live,
                  CAPACITY:tl.constexpr, HOST_COUNT:tl.constexpr,
                  LIMIT:tl.constexpr, BLOCK:tl.constexpr):
    slot=tl.arange(0,BLOCK); inside=slot<CAPACITY
    valid=tl.load(Valid+slot,inside,0)!=0
    kind=tl.load(Kind+slot,inside,0)
    index=tl.load(Index+slot,inside,0)
    dst=tl.load(Destination+slot,inside,0)
    bad_source=valid&((kind!=1)|(index<0)|(index>=HOST_COUNT))
    bad=tl.sum((inside&(bad_source|(dst!=slot))).to(tl.int32),0)>0
    count=tl.sum((inside&valid).to(tl.int32),0)
    bad=bad|(tl.load(Count)!=count)|(count>LIMIT)
    error=tl.load(Error)|bad.to(tl.int32);tl.store(Error,error)
    selected=inside&valid&(error==0)
    position=tl.cumsum(selected.to(tl.int32),0)-1
    tl.store(Keys+position,index,selected)
    tl.store(Slots+position,slot,selected)
    tl.store(Live,tl.where(error==0,count,0))

@triton.jit
def build_compact(Keys, Slots, Live, Hosts, Targets, Sizes, Sources, Destinations, Lengths,
                  HOST_COUNT:tl.constexpr,HOT_COUNT:tl.constexpr,ARENA:tl.constexpr):
    component=tl.program_id(0);j=tl.arange(0,ARENA)
    selected=j<tl.load(Live)
    key=tl.load(Keys+j,selected,0)
    slot=tl.load(Slots+j,selected,0)
    source=tl.load(Hosts+component*(HOST_COUNT+1)+tl.where(selected,key+1,0))
    target=tl.load(Targets+component*(HOT_COUNT+1)+tl.where(selected,slot+1,0))
    size=tl.load(Sizes+component)
    tl.store(Sources+component*ARENA+j,source)
    tl.store(Destinations+component*ARENA+j,target)
    tl.store(Lengths+component*ARENA+j,tl.where(selected,size,0))

class CompactCacheTransfer:
    def __init__(self,bindings,*,retain,max_updates=96,skip_zero_components=()):
        if bindings.capacity!=bindings.destination_count or not bindings.hot_targets_alias:
            raise ValueError('Requires fixed full Hot destination bank')
        if not 0<max_updates<=bindings.capacity or max_updates>256:
            raise ValueError('Bounded trial requires 1..256 updates, no more than Hot slots')
        names=tuple(bindings.address_catalog.component_nbytes)
        if any(name not in names or not name.endswith('_offset') for name in skip_zero_components):
            raise ValueError('Only certified zero offsets may be skipped')
        selected=[i for i,name in enumerate(names) if name not in skip_zero_components]
        self.bindings,self.retain,self.limit=bindings,retain,max_updates
        self.arena=triton.next_power_of_2(max_updates)
        d=bindings.device;self.components=len(selected)
        self.hosts=torch.stack([bindings.tables[c][0] for c in selected])
        self.targets=torch.stack([bindings.tables[c][2] for c in selected])
        self.sizes=torch.tensor([bindings.address_catalog.component_nbytes[names[c]] for c in selected],dtype=torch.uint32,device=d)
        self.keys=torch.empty(self.arena,dtype=torch.int64,device=d)
        self.slots=torch.empty(self.arena,dtype=torch.int64,device=d)
        self.live=torch.zeros((),dtype=torch.int32,device=d)
        total=self.components*self.arena
        self.sources=torch.empty(total,dtype=torch.int64,device=d)
        self.destinations=torch.empty_like(self.sources)
        self.lengths=torch.empty(total,dtype=torch.uint32,device=d)
        self.descriptors=[]
        for begin in range(0,total,bindings.max_batch_entries):
            end=min(total,begin+bindings.max_batch_entries);n=end-begin
            self.descriptors.append((self.sources[begin:end],self.destinations[begin:end],self.lengths[begin:end],
                torch.tensor(n,dtype=torch.uint32,device=d),torch.ones(n,dtype=torch.int64,device=d),
                torch.zeros(n,dtype=torch.int64,device=d)))
    def materialize_into(self,plan,bindings,stream):
        if bindings is not self.bindings:raise ValueError('Immutable binding required')
        if torch.device(stream.device)!=bindings.device:raise ValueError('Wrong device stream')
        bindings.validate_plan(plan)
        resources=(self,plan,bindings);self.retain(resources)
        with torch.npu.stream(stream):
            try:
                compact_valid[(1,)](plan.source_kind,plan.source_index,plan.destination_slot,plan.valid,
                    plan.count,bindings.error_status,self.keys,self.slots,self.live,
                    CAPACITY=bindings.capacity,HOST_COUNT=bindings.host_count,LIMIT=self.limit,
                    BLOCK=triton.next_power_of_2(bindings.capacity))
                build_compact[(self.components,)](self.keys,self.slots,self.live,self.hosts,self.targets,
                    self.sizes,self.sources,self.destinations,self.lengths,
                    HOST_COUNT=bindings.host_count,HOT_COUNT=bindings.destination_count,ARENA=self.arena)
                for desc in self.descriptors:
                    status=_group_pack_copy(*desc,bindings.device)
                    if status is not None and status!=0:raise RuntimeError(f'copy status {status}')
                return _ticket(_npu_completion,stream,resources,self.retain)
            except Exception as error:
                raise _submission_failure(error,_npu_completion,stream,resources,self.retain) from error
