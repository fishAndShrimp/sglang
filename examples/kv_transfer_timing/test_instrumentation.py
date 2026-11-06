"""Exercise the actual producer/worker methods with a fake transport, without NPU imports."""

import ast
import collections
import importlib.util
import json
import queue
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/disaggregation/mooncake/conn.py"
)


spec = importlib.util.spec_from_file_location(
    "kv_timing_test_helper",
    SOURCE.parents[2] / "observability/kv_transfer_timing.py",
)
kv_timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv_timing)


class InstrumentationTests(unittest.TestCase):
    def run_transfer(
        self, enabled, result=0, chunk_count=1, with_state=False, retry=False
    ):
        # Compile the production methods unchanged, with postponed annotations
        # and fake dependencies. This exercises queueing, send completion and
        # cleanup rather than importing the hardware-dependent server package.
        tree = ast.parse(SOURCE.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MooncakeKVManager"
        )
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("add_transfer_request", "transfer_worker", "_transfer_data")
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *methods,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        ticks = iter(range(1, 1000))
        statuses = SimpleNamespace(Success=1, Failed=2, Transferring=3)
        logs = []
        namespace = {
            "_KV_TIMING": enabled,
            "kv_timing": kv_timing,
            "time": SimpleNamespace(perf_counter=lambda: float(next(ticks))),
            "os": SimpleNamespace(getpid=lambda: 42),
            "json": json,
            "KVPoll": statuses,
            "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
            "TransferKVChunk": lambda **kw: SimpleNamespace(
                staging_counted=False, **kw
            ),
            "TraceNullContext": SimpleNamespace,
            "torch": SimpleNamespace(Tensor=type("FakeTensor", (), {})),
            "NetworkAddress": lambda *a: SimpleNamespace(
                to_host_port_str=lambda: "peer"
            ),
        }
        exec(compile(module, str(SOURCE), "exec"), namespace)
        work_queue = queue.SimpleQueue()
        room, session = 7, "peer:1234"
        req = SimpleNamespace(
            room=room,
            mooncake_session_id=session,
            is_dummy=False,
            dst_device_kv_indices=None,
            dst_kv_indices=list(range(chunk_count)),
            required_dst_info_num=1,
            endpoint="peer",
            dst_port=1234,
        )
        peer = SimpleNamespace(
            requires_dcp_relayout=False,
            dst_kv_ptrs=[100],
            dst_kv_layer_ids=None,
            dst_kv_item_len=16,
            dst_attn_tp_size=1,
            dst_aux_ptrs=[200],
        )
        mgr = SimpleNamespace(
            disaggregation_mode="prefill",
            request_status={room: statuses.Transferring},
            transfer_infos={room: {session: req}},
            transfer_queues=[work_queue],
            enable_trace=False,
            enable_staging=False,
            enable_deferred_decode_kv_release=False,
            _staging_outstanding=collections.defaultdict(int),
            _prefill_unique_rank=lambda: 0,
            session_lock=threading.Lock(),
            state_layout_rejections={},
            failed_sessions=set(),
            session_failures=collections.defaultdict(int),
            decode_kv_args_table={session: peer},
            _get_dsa_cache_transfer_skip_flags=lambda info: (False, False),
            kv_args=SimpleNamespace(kv_data_ptrs=[100]),
            is_mla_backend=True,
            engine=SimpleNamespace(batch_transfer_sync=Mock(return_value=result)),
            req_to_decode_prefix_len={},
            _poison_unsettled_send=Mock(),
            poison_deferred_ack_room=Mock(),
            bootstrap_port=32100,
        )
        mgr.check_status = lambda r: mgr.request_status[r]
        mgr.conclude_transfer = lambda **kw: mgr.request_status.update(
            {kw["bootstrap_room"]: kw["status"]}
        )
        mgr.conclude_failure = lambda **kw: mgr.request_status.update(
            {kw["bootstrap_room"]: statuses.Failed}
        )

        def send_kv(*args, **kwargs):
            return namespace["_transfer_data"](mgr, session, [(1, 2, 16)])

        def send_aux(*args, **kwargs):
            return kv_timing.batch_transfer_sync(mgr.engine, session, [3], [4], [8])

        mgr.send_kvcache = Mock(side_effect=send_kv)
        mgr.send_aux = Mock(side_effect=send_aux)
        mgr.maybe_send_extra = Mock(side_effect=send_aux)
        if retry:
            mgr.enable_staging = True
            mgr.is_mla_backend = False
            mgr.is_hybrid_mla_backend = False
            mgr.attn_tp_size = 2
            peer.staging_base_ptr, peer.staging_total_size = 100, 100
            mgr._try_create_staging_strategy = Mock(return_value=object())
            mgr._staging_ctx = SimpleNamespace(
                prefetch_requested=set(), prefetched_rooms=set()
            )
            tries = 0

            def staging_transfer(
                strategy, item, req, info, indices, executor, queue, rank
            ):
                nonlocal tries
                tries += 1
                if tries == 1:
                    if enabled:
                        item._kv_timing_enqueued = namespace["time"].perf_counter()
                    queue.put(item)
                    return -1, True
                queue.put(None)
                return mgr.send_kvcache(), False

            mgr._do_staging_transfer = staging_transfer

        def log_info(fmt, payload):
            if fmt != "KV_TIMING %s":  # Startup banner.
                return
            event = json.loads(payload)
            if event["outcome"] == "complete" and event["last"]:
                self.assertEqual(mgr.request_status[room], statuses.Success)
                self.assertNotIn(room, mgr._staging_outstanding)
            logs.append(event)

        namespace["logger"] = SimpleNamespace(info=log_info, error=Mock(), debug=Mock())
        wait_event, indices_event = Mock(), Mock()
        with patch.object(kv_timing, "time", namespace["time"]):
            for i in range(chunk_count):
                namespace["add_transfer_request"](
                    mgr,
                    room,
                    [i],
                    slice(i, i + 1),
                    i == chunk_count - 1,
                    aux_index=9,
                    state_indices=[1] if with_state and i == chunk_count - 1 else None,
                    wait_event=wait_event if i == 0 else None,
                    indices_event=indices_event if i == 0 else None,
                    timing_chunk_id=i + 1,
                )
            chunk = work_queue.get_nowait()
            self.assertIs(chunk.wait_event, wait_event)
            self.assertIs(chunk.indices_event, indices_event)
            # Preserve the order when putting the inspected first chunk back.
            remaining = [work_queue.get_nowait() for _ in range(chunk_count - 1)]
            for item in [chunk, *remaining] + ([] if retry else [None]):
                work_queue.put(item)
            namespace["transfer_worker"](mgr, work_queue, None, staging_buffer=object())
        wait_event.synchronize.assert_called_once_with()
        indices_event.synchronize.assert_called_once_with()
        self.assertEqual(mgr.send_kvcache.call_count, chunk_count)
        return chunk, logs, mgr

    def test_enabled_records_ordered_timestamps_after_cleanup(self):
        _, logs, _ = self.run_transfer(True)
        self.assertEqual(len(logs), 1)
        event = logs[0]
        self.assertLess(event["enqueue"], event["dequeue"])
        self.assertLess(event["dequeue"], event["work_end"])
        self.assertEqual((event["room"], event["attempt"], event["chunk"]), (7, 1, 1))

    def test_disabled_preserves_transfer_without_timing_logs(self):
        chunk, logs, mgr = self.run_transfer(False)
        self.assertEqual(logs, [])
        self.assertFalse(hasattr(chunk, "_kv_timing_enqueued"))
        mgr.send_aux.assert_called_once()

    def test_failed_send_does_not_emit_a_success_sample(self):
        _, logs, mgr = self.run_transfer(True, result=-1)
        self.assertEqual([e["outcome"] for e in logs], ["failed"])
        mgr.send_aux.assert_not_called()
        mgr.poison_deferred_ack_room.assert_not_called()

    def test_each_chunk_and_engine_call_is_logged(self):
        _, logs, mgr = self.run_transfer(True, chunk_count=3)
        self.assertEqual([e["chunk"] for e in logs], [1, 2, 3])
        self.assertEqual([e["last"] for e in logs], [False, False, True])
        self.assertEqual([e["outcome"] for e in logs], ["complete"] * 3)
        self.assertEqual([len(e["calls"]) for e in logs], [1, 1, 2])
        self.assertEqual(
            [c[0] for e in logs for c in e["calls"]], ["kv", "kv", "kv", "aux"]
        )
        self.assertEqual(mgr.engine.batch_transfer_sync.call_count, 4)

    def test_state_and_aux_calls_are_separate(self):
        _, logs, _ = self.run_transfer(True, with_state=True)
        self.assertEqual([call[0] for call in logs[0]["calls"]], ["kv", "state", "aux"])

    def test_worker_records_each_deferred_attempt(self):
        _, logs, _ = self.run_transfer(True, retry=True)
        self.assertEqual([event["outcome"] for event in logs], ["deferred", "complete"])
        self.assertEqual([event["attempt"] for event in logs], [1, 2])
        self.assertLess(logs[0]["enqueue"], logs[1]["enqueue"])
        self.assertLess(logs[0]["work_end"], logs[1]["dequeue"])
        self.assertEqual(sum(len(event["calls"]) for event in logs), 2)


if __name__ == "__main__":
    unittest.main()
