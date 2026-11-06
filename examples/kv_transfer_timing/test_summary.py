import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from summarize import make_row, read_rows


class TimingSummaryTests(unittest.TestCase):
    def events(self, pid=10, room=20):
        worker = dict(event="worker", pid=pid, room=room,
                      enqueue=10.71, dequeue=10.73, ready=10.91, attempts=1)
        finish = dict(event="finish", pid=pid, room=room, rid="request-a",
                      forward=10.0, prefill=10.70, transfer=10.72,
                      poll=11.70, done=11.71, chunked=False, chunks=1)
        return worker, finish

    def parse(self, events, exclude=()):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.log"
            path.write_text("".join("[TP rank] KV_TIMING " + json.dumps(event) + "\n"
                                    for event in events))
            with contextlib.redirect_stderr(io.StringIO()):
                return read_rows([path], set(exclude))

    def test_split_and_existing_metric_reconcile(self):
        row = make_row(*self.events())
        for key, expected in dict(prefill_ms=700, submit_ms=10, queue_ms=20,
                                  send_ms=180, post_send_ms=800, detect_ms=790,
                                  cleanup_ms=10, metric_ms=990).items():
            self.assertAlmostEqual(row[key], expected)
        self.assertAlmostEqual(row["queue_ms"] + row["send_ms"] + row["post_send_ms"]
                               - row["metric_start_after_enqueue_ms"], row["metric_ms"])

    def test_interleaved_ranks_do_not_mix(self):
        a, b = self.events(pid=10), self.events(pid=11)
        rows = self.parse([b[1], a[0], a[1], b[0]])
        self.assertEqual({row["pid"] for row in rows}, {10, 11})

    def test_missing_or_repeated_events_are_not_averaged(self):
        worker, finish = self.events()
        self.assertEqual(self.parse([worker]), [])
        self.assertEqual(self.parse([worker, worker, finish]), [])

    def test_chunked_requests_are_not_treated_as_whole_request_split(self):
        worker, finish = self.events()
        finish["chunked"] = True
        self.assertEqual(self.parse([worker, finish]), [])

    def test_warmup_exclusion_is_explicit(self):
        self.assertEqual(self.parse(self.events(), ["request-a"]), [])

    def test_multiple_sends_and_staging_retries_are_not_misattributed(self):
        worker, finish = self.events()
        finish["chunks"] = 2
        self.assertEqual(self.parse([worker, finish]), [])
        finish["chunks"] = 1
        worker["attempts"] = 2
        self.assertEqual(self.parse([worker, finish]), [])

    def test_delayed_bootstrap_submission_reconciles(self):
        worker, finish = self.events()
        finish["transfer"] = 10.705  # The metric starts before deferred submission.
        row = make_row(worker, finish)
        self.assertLess(row["metric_start_after_enqueue_ms"], 0)
        self.assertAlmostEqual(
            row["queue_ms"] + row["send_ms"] + row["post_send_ms"]
            - row["metric_start_after_enqueue_ms"], row["metric_ms"]
        )

    def test_invalid_log_lines_do_not_discard_valid_samples(self):
        worker, finish = self.events()
        self.assertEqual(len(self.parse([None, {}, worker, finish])), 1)

    def test_wrong_clock_or_invalid_timestamp_is_rejected(self):
        worker, finish = self.events()
        worker["ready"] = finish["done"] + 1
        self.assertEqual(self.parse([worker, finish]), [])
        worker["ready"] = float("nan")
        self.assertEqual(self.parse([worker, finish]), [])


if __name__ == "__main__":
    unittest.main()
