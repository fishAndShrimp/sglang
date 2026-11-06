import concurrent.futures
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from chunk_summary import duration
from test_instrumentation import kv_timing


class TimingContextTests(unittest.TestCase):
    def new_timing(self):
        now = time.perf_counter()
        chunk = SimpleNamespace(
            room=1,
            _kv_timing_id=1,
            _kv_timing_attempts=1,
            is_last_chunk=True,
            _kv_timing_enqueued=now,
        )
        return kv_timing.ChunkTiming(chunk, time.perf_counter())

    def test_executor_context_tracks_parallel_calls_and_does_not_leak(self):
        barrier = threading.Barrier(2)

        def transfer(*args):
            barrier.wait(timeout=5)
            return 0

        engine = SimpleNamespace(batch_transfer_sync=Mock(side_effect=transfer))
        timing = self.new_timing()
        token = timing.activate()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                with kv_timing.phase("kv"):
                    futures = [
                        kv_timing.submit(
                            pool,
                            kv_timing.batch_transfer_sync,
                            engine,
                            "peer",
                            [1],
                            [2],
                            [16],
                        )
                        for _ in range(2)
                    ]
                    self.assertEqual(
                        [future.result(timeout=5) for future in futures], [0, 0]
                    )
                # A reused executor thread outside the propagated context must
                # not append another call to the previous request.
                engine.batch_transfer_sync.side_effect = None
                engine.batch_transfer_sync.return_value = 0
                pool.submit(
                    kv_timing.batch_transfer_sync, engine, "peer", [1], [2], [8]
                ).result(timeout=5)
        finally:
            timing.finish(token, Mock())
        calls = timing.event["calls"]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[0] == "kv" and call[3] == 16 for call in calls))
        intervals = [(call[1], call[2]) for call in calls]
        self.assertLess(
            duration(intervals), sum(end - start for start, end in intervals)
        )

    def test_engine_arguments_results_and_exceptions_are_preserved(self):
        sources, destinations, lengths = [1], [2], [32]
        engine = SimpleNamespace(batch_transfer_sync=Mock(return_value=-7))
        timing = self.new_timing()
        token = timing.activate()
        try:
            with kv_timing.phase("aux"):
                self.assertEqual(
                    kv_timing.batch_transfer_sync(
                        engine,
                        "peer",
                        sources,
                        destinations,
                        lengths,
                    ),
                    -7,
                )
                engine.batch_transfer_sync.assert_called_once_with(
                    "peer", sources, destinations, lengths
                )
                engine.batch_transfer_sync.side_effect = RuntimeError(
                    "transport failed"
                )
                with self.assertRaisesRegex(RuntimeError, "transport failed"):
                    kv_timing.batch_transfer_sync(
                        engine, "peer", sources, destinations, lengths
                    )
        finally:
            timing.finish(token, Mock())
        self.assertEqual([call[4] for call in timing.event["calls"]], [-7, None])
        self.assertIsNone(kv_timing._current.get())


if __name__ == "__main__":
    unittest.main()
