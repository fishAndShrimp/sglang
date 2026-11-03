"""Fixed ready replicas; real routing only, at most one global BS1 move."""
import triton
import triton.language as tl


@triton.jit
def route_static(Ids, Active, Eligible, Candidates, Out, LogicalCounts, Stats,
                 ROWS: tl.constexpr, K: tl.constexpr, EP: tl.constexpr,
                 LOCAL: tl.constexpr, PHYSICAL: tl.constexpr,
                 USE: tl.constexpr, BLOCK: tl.constexpr, PHYSICAL_LFU: tl.constexpr=False):
    i = tl.arange(0, BLOCK)
    e = tl.load(Ids+i, i < ROWS*K, -1).to(tl.int32)
    active = tl.load(Active+i//K, i < ROWS*K, 0) != 0
    valid = (i < ROWS*K) & active & (e >= 0) & (e < EP*LOCAL)
    if not PHYSICAL_LFU:
        histogram = tl.histogram(tl.where(valid, e, EP*LOCAL), EP*LOCAL)
        tl.store(LogicalCounts+tl.arange(0,EP*LOCAL), histogram)
    largest = 0
    for r in tl.static_range(EP):
        n = tl.sum((valid & (e//LOCAL==r)).to(tl.int32),0)
        largest = tl.maximum(largest,n)
    best = largest
    chosen = -1
    for r in tl.static_range(EP):
        candidate = tl.load(Candidates+r)
        present = tl.sum((valid & (e==candidate)).to(tl.int32),0)==1
        after = 0
        for q in tl.static_range(EP):
            n = tl.sum((valid & (e//LOCAL==q)).to(tl.int32),0)
            shifted = n - (q==r) + (q==(r+1)%EP)
            after = tl.maximum(after,shifted)
        take = present & (after < best)
        chosen = tl.where(take,candidate,chosen)
        best = tl.where(take,after,best)
    eligible = tl.load(Eligible)!=0
    move = USE & eligible & (chosen>=0)
    physical = (e//LOCAL)*PHYSICAL + e%LOCAL
    backup = ((tl.maximum(chosen,0)//LOCAL+1)%EP)*PHYSICAL+LOCAL
    physical = tl.where(move & (e==chosen),backup,physical)
    tl.store(Out+i,tl.where(valid,physical,-1),i<ROWS*K)
    # Only the rank with a live global-BS1 token contributes to these counters.
    tl.store(Stats,tl.load(Stats)+move.to(tl.int64))
    tl.store(Stats+1,tl.load(Stats+1)+tl.where(eligible,largest,0).to(tl.int64))
    tl.store(Stats+2,tl.load(Stats+2)+tl.where(eligible,tl.where(move,best,largest),0).to(tl.int64))


@triton.jit
def prepare_static(Counts, Error, Enabled, Access, Sources, ReplicaSources,
                   Lengths, OutCounts, OutSources, OutLengths,
                   LOCAL:tl.constexpr, PHYSICAL:tl.constexpr, INPUT_COUNT:tl.constexpr,
                   SOURCE_STRIDE:tl.constexpr, COMPONENT_COUNT:tl.constexpr,
                   SKIP_OFFSETS:tl.constexpr, CUMULATIVE:tl.constexpr, BLOCK:tl.constexpr, PHYSICAL_LFU:tl.constexpr=False):
    e=tl.arange(0,BLOCK)
    raw=tl.load(Counts+e,e<INPUT_COUNT,0)
    count=raw
    if CUMULATIVE:
        previous=tl.load(Counts+e-1,(e>0)&(e<INPUT_COUNT),0)
        count=raw-previous
        last=tl.load(Counts+INPUT_COUNT-1)
        raw=tl.where(e<INPUT_COUNT,raw,last)
    bad=tl.sum(((count<0)&(e<INPUT_COUNT)).to(tl.int32),0)>0
    bad=bad | (tl.sum(((e>LOCAL)&(e<PHYSICAL)&(count>0)).to(tl.int32),0)>0)
    error=tl.load(Error)|bad.to(tl.int32)
    tl.store(Error,error)
    tl.store(OutCounts+e,tl.where(error==0,raw,0),e<PHYSICAL)
    if INPUT_COUNT==LOCAL or PHYSICAL_LFU:
        access=tl.load(Access+e,e<LOCAL,0)
        enabled=tl.load(Enabled)!=0
        tl.store(Access+e,access+tl.where(enabled,count,0),e<LOCAL)
    needed=(count>0)&(error==0)&(e<INPUT_COUNT)
    for pos in tl.static_range(COMPONENT_COUNT):
        c=(pos//2)*3+pos%2 if SKIP_OFFSETS else pos
        source=tl.load(Sources+c*SOURCE_STRIDE+e,e<LOCAL,0)
        replica=tl.load(ReplicaSources+c)
        source=tl.where(e==LOCAL,replica,source)
        length=tl.load(Lengths+c*PHYSICAL+e,e<PHYSICAL,0)
        tl.store(OutSources+pos*PHYSICAL+e,source,e<PHYSICAL)
        tl.store(OutLengths+pos*PHYSICAL+e,tl.where(needed,length,0),e<PHYSICAL)
