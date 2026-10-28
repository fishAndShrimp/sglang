import unittest
import torch
from sglang.srt.offloading.moe.npu.single_token_collective import local_topk_plan


class TestSingleTokenCollective(unittest.TestCase):
    def test_every_owner_and_empty_ranks(self):
        for ids in ([0, 1, 2, 3, 4, 5], [255, 200, 128, 127, 64, 63],
                    [190, 188, 130, 185, 129, 154], [-1]*6):
            ids = torch.tensor(ids)
            weights = torch.arange(1, 7, dtype=torch.float32) / 10
            total = torch.zeros(4)
            reference = torch.zeros(4)
            for i, w in zip(ids, weights):
                if i >= 0:
                    reference += (torch.arange(4).float() + int(i)) * w
            all_active = 0
            for rank in range(4):
                local, needed, rw, demand = local_topk_plan(ids, weights, rank, 64)
                n = int(needed.sum())
                all_active += n
                self.assertEqual(needed.tolist(), [True]*n + [False]*(6-n))
                self.assertEqual(local[:n].tolist(), sorted((ids[(ids//64)==rank] % 64).tolist()))
                self.assertEqual(int(demand.sum()), n)
                values = torch.arange(4).float().unsqueeze(0) + (local + rank*64).unsqueeze(1)
                total += (values * rw.unsqueeze(1)).sum(0)
            self.assertEqual(all_active, int((ids >= 0).sum()))
            self.assertTrue(torch.allclose(total, reference))

    def test_packet_gather_handles_each_source_and_nan_padding(self):
        h = torch.tensor([1.5, -2.0, 3.125, 0.0], dtype=torch.bfloat16)
        ids = torch.tensor([255, 200, 128, 127, 64, 63])
        weights = torch.arange(1, 7).float() / 10
        for source in range(4):
            packets = []
            for rank in range(4):
                active = torch.tensor(rank == source)
                x = h if rank == source else torch.full_like(h, float('nan'))
                packets.append(torch.cat((torch.where(active, x.float(), 0),
                    torch.where(active, ids.float()+1, 0), torch.where(active, weights, 0), active.reshape(1).float())))
            combined = torch.stack(packets).sum(0)
            self.assertTrue(torch.equal(combined[:4].to(torch.bfloat16), h))
            self.assertTrue(torch.equal(combined[4:10].long()-1, ids))
            self.assertTrue(torch.equal(combined[10:16], weights))
            self.assertEqual(combined[-1], 1)


if __name__ == '__main__':
    unittest.main()
