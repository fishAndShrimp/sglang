"""Initialization-loaded fixed replicas, isolated from prediction and async flags."""
import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import triton

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from ..resources import ResourceKind
from .transfer import _group_pack_copy
from .static_replica_kernels import route_static, prepare_static

logger=logging.getLogger(__name__)


class StaticReplica:
    def __init__(self, owner):
        self.owner=owner
        self.debug=owner.cfg.get('static_replica_debug',False)
        self.debug_steps=0
        self.use=owner.static_replica_mode=='balance'
        self.physical_lfu=owner.cfg.get('static_replica_physical_lfu',False)
        if type(self.physical_lfu) is not bool:raise ValueError('static_replica_physical_lfu must be boolean')
        ep=get_moe_ep_group()
        self.ep,self.rank=ep.world_size,ep.rank_in_group
        self.local,self.physical=owner.local_experts,owner.copy_capacity
        self.layers=[s.layer_id for s in owner.catalog.offloaded_layers]
        self.positions={l:i for i,l in enumerate(self.layers)}
        self.k=owner.hf_config.num_experts_per_tok
        if self.ep*self.local != triton.next_power_of_2(self.ep*self.local):
            raise ValueError('Static replica histogram requires power-of-two expert count')
        data=json.loads(Path(owner.cfg['static_replica_selection']).read_text())
        selections=[data['layers'][str(l)] for l in self.layers]
        for row in selections:
            if len(row)!=self.ep or any(type(e) is not int or not r*self.local<=e<(r+1)*self.local for r,e in enumerate(row)):
                raise ValueError('Fixed replica IDs must contain one expert from each logical owner')
        self.candidates=torch.tensor(selections,dtype=torch.int32,device=owner.device)
        self.eligible=torch.zeros(1,dtype=torch.int32,device=owner.device)
        self.logical_counts=torch.zeros((len(self.layers),self.ep*self.local),dtype=torch.int32,device=owner.device)
        self.stats=torch.zeros((len(self.layers),3),dtype=torch.int64,device=owner.device)
        self.output=Path(owner.cfg['static_replica_directory'])
        self.output.mkdir(parents=True,exist_ok=True);self.dumped=False
        self.bank=owner.format.allocate_bank(len(self.layers),ResourceKind.EXECUTION_WEIGHTS,'static.replicas')
        names=tuple(owner.format.layouts)
        self.addresses=torch.tensor([[self.bank.component_regions[n][i].address for n in names] for i in range(len(self.layers))],dtype=torch.int64,device=owner.device)
        if owner.cfg.get('static_replica_load','shared')=='gloo_init':
            self._load_gloo(selections,names,ep)
        else:
            nkeys=owner.catalog.num_experts
            sources=owner.sources[:,1:nkeys+1].cpu().tolist()
            all_sources=[None]*self.ep
            dist.all_gather_object(all_sources,sources,group=ep.cpu_group)
            donor=(self.rank-1)%self.ep
            selected_sources=[]
            for i,l in enumerate(self.layers):
                key=owner.catalog.key(l,selections[i][donor]%self.local)
                selected_sources.append([all_sources[donor][c][key] for c in range(len(names))])
            self.initial_sources=torch.tensor(selected_sources,dtype=torch.int64,device=owner.device)
            lengths=torch.tensor([owner.format.layouts[n].nbytes for n in names],dtype=torch.uint32,device=owner.device)
            scan=torch.tensor(len(names),dtype=torch.uint32,device=owner.device)
            groups=torch.ones(len(names),dtype=torch.int64,device=owner.device)
            packed=torch.zeros_like(self.addresses)
            for i in range(len(self.layers)):
                status=_group_pack_copy(self.initial_sources[i],self.addresses[i],lengths,scan,groups,packed[i],owner.device)
                if status not in (None,0):raise RuntimeError(f'Static replica initialization failed: {status}')
        # All reads finish before serving/capture; no producer stream or ready flag.
        torch.npu.synchronize()
        dist.barrier(group=ep.cpu_group)
        logger.info('Static replicas ready: mode=%s per-rank=%d extra=%.3f GiB physical=%d',
                    owner.static_replica_mode,len(self.layers),len(self.layers)*owner.format.expert_nbytes/2**30,self.physical)

    def _load_gloo(self,selections,names,ep):
        """Startup only: CPU ring exchange, then local MemFabric Host -> replica."""
        import ctypes
        o=self.owner
        staging=o.host_pool.allocate(o.format.expert_nbytes)
        offsets=[];offset=0
        for name in names:
            offsets.append(offset);offset+=o.format.layouts[name].nbytes
        assert offset==staging.tensor.numel()
        addresses=[staging.capability.device_pointer(staging.tensor.data_ptr()+off,o.format.layouts[n].nbytes)
                   for n,off in zip(names,offsets)]
        self.initial_sources=torch.tensor([addresses]*len(self.layers),dtype=torch.int64,device=o.device)
        lengths=torch.tensor([o.format.layouts[n].nbytes for n in names],dtype=torch.uint32,device=o.device)
        scan=torch.tensor(len(names),dtype=torch.uint32,device=o.device)
        groups=torch.ones(len(names),dtype=torch.int64,device=o.device)
        packed=torch.zeros_like(self.addresses)
        for i,layer in enumerate(self.layers):
            expert=selections[i][self.rank]%self.local
            sends=[];operations=[]
            for name,off in zip(names,offsets):
                cap=o.storages[layer].component_capabilities[name][expert]
                buf=(ctypes.c_ubyte*cap.nbytes).from_address(cap.host_address)
                source=torch.frombuffer(buf,dtype=torch.uint8)
                assert source.numel()==o.format.layouts[name].nbytes
                sends.append(source)
                recv=staging.tensor.narrow(0,off,source.numel())
                operations.extend([dist.P2POp(dist.irecv,recv,ep.ranks[(self.rank-1)%self.ep],group=ep.cpu_group),
                                   dist.P2POp(dist.isend,source,ep.ranks[(self.rank+1)%self.ep],group=ep.cpu_group)])
            for work in dist.batch_isend_irecv(operations):work.wait()
            status=_group_pack_copy(self.initial_sources[i],self.addresses[i],lengths,scan,groups,packed[i],o.device)
            if status not in (None,0):raise RuntimeError(f'Gloo replica initialization failed: {status}')
            torch.npu.synchronize()  # before reusing the sole startup scratch expert
        self.startup_staging=staging

    def begin(self,batch):
        self.debug_steps+=1
        if self.debug and self.debug_steps<=4:
            logger.warning('STATIC_BEGIN rank=%s step=%s phase=%s global_mode=%s counts=%s extend=%s',self.rank,self.debug_steps,batch.forward_mode,getattr(batch,'global_forward_mode',None),getattr(batch,'original_global_num_tokens_cpu',None),getattr(batch,'is_extend_in_batch',None))
        if not self.physical_lfu:self.logical_counts.zero_()
        counts=getattr(batch,'original_global_num_tokens_cpu',None)
        ordinary=batch is not None and batch.forward_mode.name in ('DECODE','IDLE')
        self.eligible.fill_(int(ordinary and counts is not None and sum(counts)==1))

    def route(self,layer_id,ids,active):
        if active is None or active.numel()<ids.shape[0]:raise ValueError('Static replicas require effective row mask')
        if ids.ndim!=2 or ids.shape[1]!=self.k or not ids.is_contiguous():raise ValueError('Unexpected static route shape')
        p=self.positions[layer_id];out=torch.empty_like(ids)
        route_static[(1,)](ids,active,self.eligible,self.candidates[p],out,self.logical_counts[p],self.stats[p],
                          ROWS=ids.shape[0],K=self.k,EP=self.ep,LOCAL=self.local,PHYSICAL=self.physical,
                          USE=self.use,BLOCK=triton.next_power_of_2(max(1,ids.numel())),PHYSICAL_LFU=self.physical_lfu)
        return out

    def prepare_weights(self,layer_id,runner_input):
        o=self.owner;counts=runner_input.expert_tokens
        if counts.shape not in ((self.local,),(self.physical,)) or counts.dtype!=torch.int64 or not counts.is_contiguous():
            raise ValueError('Unexpected static replica count vector')
        if runner_input.group_list_type not in (0,1):raise ValueError('Unexpected group list type')
        start=o.catalog.key(layer_id,0);p=self.positions[layer_id]
        prepare_static[(1,)](counts,o.error,o.state.update_enabled,o.state.step_access_count[start:start+self.local],
                            o.published_sources[:,start:start+self.local],self.addresses[p],o.lengths,
                            o.fused_counts,o.fused_sources,o.fused_lengths,LOCAL=self.local,PHYSICAL=self.physical,
                            INPUT_COUNT=counts.numel(),SOURCE_STRIDE=o.catalog.num_experts,
                            COMPONENT_COUNT=len(o.fused_components),SKIP_OFFSETS=o.cfg.get('skip_zero_offsets',False),
                            CUMULATIVE=runner_input.group_list_type==0,BLOCK=triton.next_power_of_2(self.physical),PHYSICAL_LFU=self.physical_lfu)
        runner_input.expert_tokens=o.fused_counts
        o._launch_copies(layer_id,o.fused_descriptors)
        return o.quant_info

    def finish(self):
        if self.physical_lfu:return
        if self.debug and self.debug_steps<=4:logger.warning('STATIC_FINISH rank=%s step=%s',self.rank,self.debug_steps)
        # Preserve logical-owner LFU even if physical execution moved.
        dist.all_reduce(self.logical_counts,group=get_moe_ep_group().device_group)
        counts=self.logical_counts[:,self.rank*self.local:(self.rank+1)*self.local].reshape(-1)
        self.owner.state.step_access_count.add_(counts)

    def maybe_dump(self):
        if self.dumped or not (self.output/'dump.request').exists():return
        values=self.stats.cpu().tolist()
        data=dict(rank=self.rank,mode=self.owner.static_replica_mode,layers=self.layers,counts=values,
                  columns=['moved_routes','sum_max_tasks_before','sum_max_tasks_after'],
                  totals=[sum(x[i] for x in values) for i in range(3)],
                  replica_bytes=len(self.layers)*self.owner.format.expert_nbytes,
                  extra_execution_bytes=(self.physical-self.local)*self.owner.format.expert_nbytes,
                  candidates=self.candidates.cpu().tolist(),
                  lfu_counting='local physical execution' if self.physical_lfu else 'original logical routing',
                  note='Stats include all forwards; no prediction/copies during serving other than normal replica D2D.')
        (self.output/f'rank{self.rank}.json').write_text(json.dumps(data,indent=2)+'\n');self.dumped=True
