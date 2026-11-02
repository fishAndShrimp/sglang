"""Four-rank hardware test of changing active rows in captured MC2 dispatch/combine."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_buffer_size": 1024}
    dist.init_process_group("hccl", timeout=timedelta(seconds=180), pg_options=options)
    d = torch.device("npu", rank)
    name = dist.group.WORLD._get_backend(d).get_hccl_comm_name(rank)
    x = torch.randn((128, 4096), dtype=torch.bfloat16, device=d)
    ids = torch.tensor([0, 65, 130, 195, 4, 69], dtype=torch.int32, device=d).repeat(
        128, 1
    )
    weights = torch.full((128, 6), 1 / 6, dtype=torch.float32, device=d)
    mask = torch.zeros(128, dtype=torch.bool, device=d)

    def run():
        values, scale, assist, counts, recv, *_ = (
            torch_npu.npu_moe_distribute_dispatch_v2(
                x,
                ids,
                name,
                4,
                rank,
                256,
                x_active_mask=mask,
                quant_mode=0,
                global_bs=0,
                expert_token_nums_type=1,
                comm_alg="fullmesh_v1",
            )
        )
        out = torch_npu.npu_moe_distribute_combine_v2(
            values,
            ids,
            assist,
            recv,
            weights,
            name,
            4,
            rank,
            256,
            x_active_mask=mask,
            global_bs=0,
            comm_quant_mode=0,
            comm_alg="",
        )
        return out, counts

    mask[0] = True
    for _ in range(2):
        run()
    torch.npu.synchronize()
    dist.barrier()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out, counts = run()
    cases = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [2, 1, 0, 2],
    ]
    for rows in cases:
        mask.fill_(False)
        mask[: rows[rank]].fill_(True)
        graph.replay()
        torch.npu.synchronize()
        total = counts.sum()
        dist.all_reduce(total)
        assert total.item() == sum(rows) * 6, (rank, rows, total)
        if rows[rank]:
            torch.testing.assert_close(
                out[: rows[rank]], x[: rows[rank]], rtol=0.02, atol=0.02
            )
        if rank == 0:
            print(
                "PASS active rows", rows, "route assignments", total.item(), flush=True
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
