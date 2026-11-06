import copy
import unittest

import test_summary
from chunk_summary import make_chunk_row


def events(room=20, pid=10):
    finish = dict(
        event="finish",
        version=2,
        pid=pid,
        room=room,
        rid=f"request-{room}",
        tp_rank=0,
        forward=10.0,
        prefill=12.0,
        transfer=12.5,
        poll=15.5,
        done=15.6,
        chunks=2,
        chunked=True,
    )
    first = dict(
        event="chunk",
        version=2,
        pid=pid,
        room=room,
        chunk=1,
        attempt=1,
        last=False,
        outcome="complete",
        enqueue=10.2,
        dequeue=10.4,
        work_end=11.5,
        cleanup_end=11.6,
        phases=[["wait", 10.4, 10.5], ["kv", 10.5, 11.4]],
        calls=[["kv", 10.7, 11.2, 16, 0], ["kv", 10.9, 11.3, 16, 0]],
    )
    last = dict(
        event="chunk",
        version=2,
        pid=pid,
        room=room,
        chunk=2,
        attempt=1,
        last=True,
        outcome="complete",
        enqueue=12.1,
        dequeue=12.3,
        work_end=15.2,
        cleanup_end=15.3,
        phases=[
            ["wait", 12.3, 12.4],
            ["kv", 12.4, 14.8],
            ["state", 14.8, 14.9],
            ["aux", 14.9, 15.0],
            ["notify", 15.0, 15.15],
        ],
        calls=[
            ["kv", 12.6, 14.3, 16, 0],
            ["kv", 13.0, 14.6, 16, 0],
            ["state", 14.82, 14.88, 8, 0],
            ["aux", 14.92, 14.98, 8, 0],
        ],
    )
    return first, last, finish


class ChunkSummaryTests(unittest.TestCase):
    def test_overlapping_calls_and_early_chunks_are_not_double_counted(self):
        first, last, finish = events()
        row = make_chunk_row([last, first], finish)
        expected = dict(
            prefill_ms=2000,
            queue_ms=400,
            wait_ms=200,
            send_ms=3500,
            mooncake_ms=2720,
            mooncake_call_sum_ms=4320,
            send_other_ms=780,
            worker_other_ms=150,
            notify_ms=150,
            detect_ms=500,
            cleanup_ms=100,
            post_send_ms=600,
            worker_ms=4000,
            send_overlap_prefill_ms=900,
            last_submit_offset_ms=100,
            metric_ms=3100,
            metric_mooncake_ms=2120,
            metric_send_other_ms=380,
            metric_worker_other_ms=200,
            metric_queue_only_ms=0,
            metric_other_ms=400,
        )
        for name, value in expected.items():
            self.assertAlmostEqual(row[name], value, msg=name)
        self.assertEqual(row["mooncake_calls"], 6)
        self.assertAlmostEqual(
            sum(
                row[name]
                for name in (
                    "metric_mooncake_ms",
                    "metric_send_other_ms",
                    "metric_worker_other_ms",
                    "metric_queue_only_ms",
                    "metric_other_ms",
                )
            ),
            row["metric_ms"],
        )

    def test_staging_defer_and_retry_are_included(self):
        first, last, finish = events()
        deferred = copy.deepcopy(first)
        deferred.update(
            enqueue=10.1,
            dequeue=10.2,
            work_end=10.3,
            cleanup_end=10.31,
            outcome="deferred",
            phases=[["kv", 10.2, 10.29]],
            calls=[],
        )
        first.update(attempt=2, enqueue=10.29)
        row = make_chunk_row([deferred, last, first], finish)
        self.assertEqual((row["chunks"], row["attempts"]), (2, 3))
        self.assertAlmostEqual(row["metric_ms"], 3100)
        self.assertAlmostEqual(row["send_ms"], 3590)

    def test_missing_and_repeated_attempts_reject_details(self):
        first, last, finish = events()
        for records in ([first], [first, first, last]):
            with self.assertRaises(ValueError):
                make_chunk_row(records, finish)
        first["attempt"] = 2
        with self.assertRaises(ValueError):
            make_chunk_row([first, last], finish)

    def test_hundred_multi_chunk_requests_keep_full_statistics(self):
        records = [event for i in range(100) for event in events(room=i)]
        rows = test_summary.TimingSummaryTests().parse(records)
        self.assertEqual(len(rows), 100)
        self.assertTrue(all(row["detail_status"] == "complete" for row in rows))
        self.assertTrue(all(row["mooncake_calls"] == 6 for row in rows))
        self.assertAlmostEqual(sum(row["metric_ms"] for row in rows), 310000)

    def test_missing_chunk_does_not_remove_original_metric(self):
        first, _, finish = events()
        rows = test_summary.TimingSummaryTests().parse([finish, first])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["metric_ms"], 3100)
        self.assertNotIn("send_ms", rows[0])
        self.assertIn("incomplete chunks", rows[0]["detail_status"])

    def test_rank_matching_and_log_order(self):
        a, b = events(pid=10), events(pid=11)
        rows = test_summary.TimingSummaryTests().parse(
            [b[2], a[1], b[0], a[2], a[0], b[1]]
        )
        self.assertEqual({row["pid"] for row in rows}, {10, 11})
        self.assertTrue(all(row["detail_status"] == "complete" for row in rows))

    def test_bad_api_phase_or_clock_preserves_metric_but_not_details(self):
        first, last, finish = events()
        first["calls"][0][0] = "other"
        rows = test_summary.TimingSummaryTests().parse([first, last, finish])
        self.assertNotIn("mooncake_ms", rows[0])
        self.assertIn("metric_ms", rows[0])


if __name__ == "__main__":
    unittest.main()
