"""Exercise the actual producer/worker methods with a fake transport, without NPU imports."""

import ast
import collections
import json
import queue
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/disaggregation/mooncake/conn.py"
)


class InstrumentationTests(unittest.TestCase):
    def run_transfer(self, enabled, result=0):
        # Compile the production methods unchanged, with postponed annotations
        # and fake dependencies. This exercises queueing, send completion and
        # cleanup rather than importing the hardware-dependent server package.
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "MooncakeKVManager")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("add_transfer_request", "transfer_worker")]
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[
                ast.alias(name="annotations")], level=0), *methods],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        ticks = iter(range(1, 100))
        statuses = SimpleNamespace(Success=1, Failed=2, Transferring=3)
        logs = []
        namespace = {
            "_KV_TIMING": enabled,
            "time": SimpleNamespace(perf_counter=lambda: float(next(ticks))),
            "os": SimpleNamespace(getpid=lambda: 42),
            "json": json,
            "KVPoll": statuses,
            "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
            "TransferKVChunk": lambda **kw: SimpleNamespace(staging_counted=False, **kw),
            "TraceNullContext": SimpleNamespace,
            "torch": SimpleNamespace(Tensor=type("FakeTensor", (), {})),
            "NetworkAddress": lambda *a: SimpleNamespace(to_host_port_str=lambda: "peer"),
        }
        exec(compile(module, str(SOURCE), "exec"), namespace)
        work_queue = queue.SimpleQueue()
        room, session = 7, "peer:1234"
        req = SimpleNamespace(
            room=room, mooncake_session_id=session, is_dummy=False,
            dst_device_kv_indices=None, dst_kv_indices=[2],
            required_dst_info_num=1, endpoint="peer", dst_port=1234,
        )
        peer = SimpleNamespace(
            requires_dcp_relayout=False, dst_kv_ptrs=[100],
            dst_kv_layer_ids=None, dst_kv_item_len=16,
            dst_attn_tp_size=1, dst_aux_ptrs=[200],
        )
        mgr = SimpleNamespace(
            disaggregation_mode="prefill", request_status={room: statuses.Transferring},
            transfer_infos={room: {session: req}}, transfer_queues=[work_queue],
            enable_trace=False, enable_staging=False,
            enable_deferred_decode_kv_release=False,
            _staging_outstanding=collections.defaultdict(int),
            _prefill_unique_rank=lambda: 0,
            session_lock=threading.Lock(), state_layout_rejections={},
            failed_sessions=set(), session_failures=collections.defaultdict(int),
            decode_kv_args_table={session: peer},
            _get_dsa_cache_transfer_skip_flags=lambda info: (False, False),
            kv_args=SimpleNamespace(kv_data_ptrs=[100]), is_mla_backend=True,
            send_kvcache=Mock(return_value=result), send_aux=Mock(return_value=0),
            req_to_decode_prefix_len={}, _poison_unsettled_send=Mock(),
            poison_deferred_ack_room=Mock(), bootstrap_port=32100,
        )
        mgr.check_status = lambda r: mgr.request_status[r]
        mgr.conclude_transfer = lambda **kw: mgr.request_status.update(
            {kw["bootstrap_room"]: kw["status"]}
        )
        mgr.conclude_failure = lambda **kw: mgr.request_status.update(
            {kw["bootstrap_room"]: statuses.Failed}
        )

        def log_info(fmt, payload):
            self.assertEqual(mgr.request_status[room], statuses.Success)
            self.assertNotIn(room, mgr._staging_outstanding)
            logs.append(json.loads(payload))

        namespace["logger"] = SimpleNamespace(info=log_info, error=Mock(), debug=Mock())
        wait_event, indices_event = Mock(), Mock()
        namespace["add_transfer_request"](
            mgr, room, [1], slice(0, 1), True, aux_index=9,
            wait_event=wait_event, indices_event=indices_event,
        )
        chunk = work_queue.get_nowait()
        self.assertIs(chunk.wait_event, wait_event)
        self.assertIs(chunk.indices_event, indices_event)
        work_queue.put(chunk)
        work_queue.put(None)
        namespace["transfer_worker"](mgr, work_queue, None)
        wait_event.synchronize.assert_called_once_with()
        indices_event.synchronize.assert_called_once_with()
        mgr.send_kvcache.assert_called_once()
        return chunk, logs, mgr

    def test_enabled_records_ordered_timestamps_after_cleanup(self):
        _, logs, _ = self.run_transfer(True)
        self.assertEqual(len(logs), 1)
        event = logs[0]
        self.assertLess(event["enqueue"], event["dequeue"])
        self.assertLess(event["dequeue"], event["ready"])
        self.assertEqual((event["pid"], event["room"], event["attempts"]), (42, 7, 1))

    def test_disabled_preserves_transfer_without_timing_logs(self):
        chunk, logs, mgr = self.run_transfer(False)
        self.assertEqual(logs, [])
        self.assertFalse(hasattr(chunk, "_kv_timing_enqueued"))
        mgr.send_aux.assert_called_once()

    def test_failed_send_does_not_emit_a_success_sample(self):
        _, logs, mgr = self.run_transfer(True, result=-1)
        self.assertEqual(logs, [])
        mgr.send_aux.assert_not_called()
        mgr.poison_deferred_ack_room.assert_not_called()


if __name__ == "__main__":
    unittest.main()
