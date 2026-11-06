import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from summarize import make_row, read_rows


class TimingSummaryTests(unittest.TestCase):
    def events(self, pid=10, room=20):
        worker = dict(
            event="worker",
            pid=pid,
            room=room,
            enqueue=10.71,
            dequeue=10.73,
            ready=10.91,
            attempts=1,
        )
        finish = dict(
            event="finish",
            pid=pid,
            room=room,
            rid="request-a",
            forward=10.0,
            prefill=10.70,
            transfer=10.72,
            poll=11.70,
            done=11.71,
            chunked=False,
            chunks=1,
        )
        return worker, finish

    def parse(self, events, exclude=(), metric_only=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.log"
            path.write_text(
                "".join(
                    "[TP rank] KV_TIMING " + json.dumps(event) + "\n"
                    for event in events
                )
            )
            with contextlib.redirect_stderr(io.StringIO()):
                return read_rows([path], set(exclude), metric_only=metric_only)

    def test_split_and_existing_metric_reconcile(self):
        row = make_row(*self.events())
        for key, expected in dict(
            prefill_ms=700,
            submit_ms=10,
            queue_ms=20,
            send_ms=180,
            post_send_ms=800,
            detect_ms=790,
            cleanup_ms=10,
            metric_ms=990,
        ).items():
            self.assertAlmostEqual(row[key], expected)
        self.assertAlmostEqual(
            row["queue_ms"]
            + row["send_ms"]
            + row["post_send_ms"]
            - row["metric_start_after_enqueue_ms"],
            row["metric_ms"],
        )

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
            row["queue_ms"]
            + row["send_ms"]
            + row["post_send_ms"]
            - row["metric_start_after_enqueue_ms"],
            row["metric_ms"],
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

    def test_metric_only_keeps_full_population_when_five_stage_keeps_three(self):
        events = []
        expected_sum = 0.0
        for i in range(100):
            worker, finish = self.events(room=i)
            # A slower single-chunk subset must not stand in for all requests.
            latency_ms = 2955.0 if i < 3 else 1043.0
            finish["done"] = finish["transfer"] + latency_ms / 1000
            finish["chunks"] = 1 if i < 3 else 2
            finish["chunked"] = i >= 3
            expected_sum += latency_ms
            events.extend([worker, finish])
        self.assertEqual(len(self.parse(events)), 3)
        rows = self.parse(events, metric_only=True)
        self.assertEqual(len(rows), 100)
        self.assertAlmostEqual(sum(row["metric_ms"] for row in rows), expected_sum)

    def test_metric_only_needs_no_worker_or_prefill_timestamps(self):
        _, finish = self.events()
        finish = {
            k: finish[k] for k in ("event", "pid", "room", "rid", "done", "transfer")
        }
        rows = self.parse([finish], metric_only=True)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["metric_ms"], 990)
        self.assertNotIn("send_ms", rows[0])

    def test_metric_only_still_rejects_duplicate_finish_and_bad_interval(self):
        worker, finish = self.events()
        self.assertEqual(self.parse([finish, finish], metric_only=True), [])
        self.assertEqual(self.parse([worker], metric_only=True), [])
        for done in (None, float("nan"), finish["transfer"], finish["transfer"] - 1):
            finish["done"] = done
            self.assertEqual(self.parse([finish], metric_only=True), [])

    def test_metric_only_preserves_rank_isolation_and_warmup_exclusion(self):
        a, b = self.events(pid=10)[1], self.events(pid=11)[1]
        rows = self.parse([a, b], metric_only=True)
        self.assertEqual({row["pid"] for row in rows}, {10, 11})
        self.assertEqual(
            self.parse([a, b], exclude=["request-a"], metric_only=True), []
        )


if __name__ == "__main__":
    unittest.main()
