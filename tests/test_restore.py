import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app
from app import (
    WalCorruptionError,
    WalPendingError,
    WalStore,
)


class RestoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "store.wal"

    def tearDown(self):
        self.dir.cleanup()

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def build(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", {"n": [1]})
        s.commit()  # seq 1: {"a": 1, "b": {"n": [1]}}
        s.delete("a")
        s.set("c", 3)
        s.commit()  # seq 2: {"b": {"n": [1]}, "c": 3}
        s.set("d", True)
        s.commit()  # seq 3: {"b": ..., "c": 3, "d": True}
        return s

    def test_restore_to_past_seq(self):
        s = self.build()
        seq = s.restore(1)
        self.assertEqual(seq, 4)
        self.assertEqual(s.state, {"a": 1, "b": {"n": [1]}})
        self.assertEqual(s.commit_seq, 4)
        self.assertEqual(s.recover()["pending_count"], 0)
        # snapshot/history/reopen all agree on the new state and seq
        self.assertEqual(s.snapshot()["state"], {"a": 1, "b": {"n": [1]}})
        self.assertEqual(s.snapshot(4)["commit_seq"], 4)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq),
                         ({"a": 1, "b": {"n": [1]}}, 4))
        self.assertEqual(s2.recover()["pending_count"], 0)
        # old commits were never rewritten
        self.assertEqual(
            s2.snapshot(2)["state"], {"b": {"n": [1]}, "c": 3})
        self.assertEqual([e["commit_seq"] for e in s2.history()], [1, 2, 3, 4])

    def test_restore_latest_writes_empty_batch_and_advances_once(self):
        s = self.build()
        before = self.path.read_bytes()
        seq = s.restore(3)
        self.assertEqual(seq, 4)
        self.assertEqual(s.commit_seq, 4)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3, "d": True})
        self.assertEqual(s.recover()["pending_count"], 0)
        # exactly one empty commit record was appended: one new line
        new = self.path.read_bytes()
        self.assertTrue(new.startswith(before))
        added = new[len(before):]
        self.assertEqual(added.count(b"\n"), 1)
        self.assertEqual(json.loads(added), {"op": "commit", "seq": 4})
        # history records an empty changes batch for seq 4
        self.assertEqual(s.history()[-1], {"commit_seq": 4, "changes": []})

    def test_restore_zero_on_empty_log_makes_seq_one(self):
        s = WalStore(self.path)
        self.assertFalse(self.path.exists())
        seq = s.restore(0)
        self.assertEqual(seq, 1)
        self.assertEqual((s.state, s.commit_seq), ({}, 1))
        self.assertEqual(s.recover()["pending_count"], 0)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({}, 1))
        self.assertEqual(s.history(), [{"commit_seq": 1, "changes": []}])
        # restore(0) again: another empty batch, seq 2
        self.assertEqual(s.restore(0), 2)
        self.assertEqual((s.state, s.commit_seq), ({}, 2))

    def test_restore_zero_after_history(self):
        s = self.build()
        seq = s.restore(0)
        self.assertEqual(seq, 4)
        self.assertEqual((s.state, s.commit_seq), ({}, 4))
        self.assertEqual(s.recover()["pending_count"], 0)
        self.assertEqual(WalStore(self.path).state, {})

    def test_changes_written_in_unicode_key_order(self):
        s = self.build()
        s.restore(1)  # drop c, d; re-add a
        # inspect the restore batch (seq 4) via raw log lines
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        batch = [r for r in rows if r["seq"] == 4 and r["op"] != "commit"]
        keys = [r["key"] for r in batch]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(
            [(r["op"], r["key"]) for r in batch],
            [("set", "a"), ("delete", "c"), ("delete", "d")],
        )

    def test_unicode_sort_order(self):
        s = WalStore(self.path)
        s.set("é", 1)
        s.set("中", 2)
        s.set("a", 3)
        s.commit()  # 1
        s.delete("a")
        s.set("b", 9)
        s.commit()  # 2
        s.restore(1)  # seq 3: add a, delete b; 中/é unchanged
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        keys = [r["key"] for r in rows if r["seq"] == 3 and r["op"] != "commit"]
        self.assertEqual(keys, ["a", "b"])
        self.assertEqual(
            s.state, {"a": 3, "é": 1, "中": 2})

    def test_value_comparison_distinguishes_json_types(self):
        # current has 1 (int); target history has true (bool) -> must set
        s = WalStore(self.path)
        s.set("k", True)
        s.commit()  # seq 1: true
        s.set("k", 1)
        s.commit()  # seq 2: 1
        s.restore(1)  # seq 3 -> must write set k=true
        self.assertIs(s.state["k"], True)
        self.assertNotIn(False, [s.state["k"]])
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        change = [r for r in rows if r["seq"] == 3 and r["op"] == "set"]
        self.assertEqual(len(change), 1)
        self.assertIs(change[0]["value"], True)
        # reverse: current true, target 1
        s2 = WalStore(Path(self.dir.name) / "other.wal")
        s2.set("k", 1)
        s2.commit()
        s2.set("k", False)
        s2.commit()
        s2.restore(1)
        self.assertEqual(s2.state["k"], 1)
        self.assertNotIsInstance(s2.state["k"], bool)

    def test_nested_type_difference_forces_set(self):
        s = WalStore(self.path)
        s.set("k", [1])
        s.commit()  # 1: [1]
        s.set("k", [True])
        s.commit()  # 2: [true]
        s.restore(1)
        self.assertEqual(s.state, {"k": [1]})
        self.assertNotIsInstance(s.state["k"][0], bool)

    def test_float_vs_int_distinct(self):
        s = WalStore(self.path)
        s.set("k", 1)
        s.commit()
        s.set("k", 1.0)
        s.commit()
        s.restore(1)
        self.assertEqual(s.state["k"], 1)
        self.assertIsInstance(s.state["k"], int)
        self.assertNotIsInstance(s.state["k"], bool)

    def test_state_is_independent_deep_copy(self):
        s = self.build()
        s.restore(1)
        s.state["b"]["n"].append(99)
        s.state["z"] = 1
        # the durable state / snapshot / reopen are unaffected
        self.assertEqual(WalStore(self.path).state,
                         {"a": 1, "b": {"n": [1]}})
        self.assertEqual(s.snapshot(1)["state"], {"a": 1, "b": {"n": [1]}})

    def test_invalid_target_seq_raises_valueerror(self):
        s = self.build()
        for bad in (True, False, -1, 1.0, "1", b"1", [1], {"s": 1}, None):
            with self.assertRaises(ValueError, msg=bad):
                s.restore(bad)
            # state/file/seq untouched after each rejection
            self.assertEqual(s.commit_seq, 3, msg=bad)
            self.assertEqual(s.state,
                             {"b": {"n": [1]}, "c": 3, "d": True}, msg=bad)
        with self.assertRaises(ValueError):
            s.restore(4)
        with self.assertRaises(ValueError):
            s.restore(10**9)
        # empty log: only 0 is valid
        empty = WalStore(Path(self.dir.name) / "e.wal")
        with self.assertRaises(ValueError):
            empty.restore(1)

    def test_pending_records_raise_walpendingerror(self):
        s = self.build()
        s.set("tail", 9)
        s.delete("c")
        before = self.path.read_bytes()
        with self.assertRaises(WalPendingError):
            s.restore(1)
        # file, state, commit_seq all unchanged
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(s.commit_seq, 3)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3, "d": True})
        self.assertEqual(s.recover()["pending_count"], 2)
        # restore(0) also blocked; latest target doesn't help
        with self.assertRaises(WalPendingError):
            s.restore(3)

    def test_pending_before_any_commit_blocks_restore_zero(self):
        s = WalStore(self.path)
        s.set("x", 1)
        with self.assertRaises(WalPendingError):
            s.restore(0)
        self.assertEqual((s.state, s.commit_seq), ({}, 0))

    def test_tail_fragment_raises_walpendingerror(self):
        s = self.build()
        before = self.path.read_bytes()
        for frag in (
            '{"op": "set", "key": "x"',
            "{",
            json.dumps({"op": "set", "key": "x", "value": 9, "seq": 4}),
            '{"op": "set", "key": "hé'.encode("utf-8")[:-1],
        ):
            raw = frag if isinstance(frag, bytes) else frag.encode("utf-8")
            self.path.write_bytes(before)
            self.append_bytes(raw)
            size = self.path.stat().st_size
            with self.assertRaises(WalPendingError, msg=frag):
                s.restore(1)
            # nothing truncated or appended before/by the rejected call
            self.assertEqual(self.path.stat().st_size, size, msg=frag)
            self.assertEqual(self.path.read_bytes(), before + raw, msg=frag)
            self.assertEqual(s.commit_seq, 3, msg=frag)
            self.assertEqual(s.state,
                             {"b": {"n": [1]}, "c": 3, "d": True}, msg=frag)

    def test_fragment_only_log_blocks_restore_zero(self):
        self.path.write_bytes(b'{"op": "set", "key": "x"')
        s = WalStore(self.path)
        before = self.path.read_bytes()
        with self.assertRaises(WalPendingError):
            s.restore(0)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({}, 0))

    def test_corruption_raises_walcorruptionerror_not_pending(self):
        s = self.build()
        prefix = self.path.read_bytes()
        bad_tails = [
            b'{"op": "set", "key": "x"}\n',  # invalid record, terminated
            b"\xff",
            b"not json\n",
        ]
        for tail in bad_tails:
            self.path.write_bytes(prefix + tail)
            with self.assertRaises(WalCorruptionError, msg=tail):
                s.restore(1)
            self.assertEqual(self.path.read_bytes(), prefix + tail, msg=tail)
            self.assertEqual(s.commit_seq, 3, msg=tail)
            self.assertEqual(s.state,
                             {"b": {"n": [1]}, "c": 3, "d": True}, msg=tail)

    def test_no_bytes_written_on_rejection(self):
        s = self.build()
        s.set("tail", 9)
        size = self.path.stat().st_size
        for bad in (True, -1, 1.0, 4):
            with self.assertRaises(ValueError):
                s.restore(bad)
        with self.assertRaises(WalPendingError):
            s.restore(1)
        self.assertEqual(self.path.stat().st_size, size)

    def test_oserror_on_first_change_write_propagates_and_keeps_state(self):
        s = self.build()
        real_fsync = os.fsync

        def boom(fd):
            raise OSError("disk on fire")

        with mock.patch("app.os.fsync", side_effect=boom):
            with self.assertRaises(OSError):
                s.restore(1)
        # old committed state/seq unchanged
        self.assertEqual(s.commit_seq, 3)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3, "d": True})
        # a reopen sees the same committed state; seq chain intact
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq),
                         ({"b": {"n": [1]}, "c": 3, "d": True}, 3))
        self.assertEqual(s2.recover()["pending_count"], 0)
        # retry succeeds with the next seq
        self.assertEqual(s2.restore(1), 4)
        self.assertEqual(s2.state, {"a": 1, "b": {"n": [1]}})

    def test_oserror_mid_batch_leaves_pending_for_rollback(self):
        # Fail on fsync of the 2nd restore record (delete c): the 1st
        # record (set a) is already durable and survives as a pending
        # record that rollback clears under its usual rules.
        s = self.build()
        real_fsync = os.fsync
        n = {"i": 0}

        def flaky(fd):
            n["i"] += 1
            if n["i"] == 2:
                raise OSError("disk on fire")
            return real_fsync(fd)

        with mock.patch("app.os.fsync", side_effect=flaky):
            with self.assertRaises(OSError):
                s.restore(1)
        self.assertEqual(s.commit_seq, 3)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3, "d": True})
        # the one durable change record is pending and rollback clears it
        pc = s.pending_changes()
        self.assertEqual(pc.pending_count, 1)
        self.assertEqual(pc.changes, [{"op": "set", "key": "a", "value": 1}])
        self.assertEqual(s.rollback(), 1)
        self.assertEqual(s.pending_changes().pending_count, 0)
        self.assertEqual((s.state, s.commit_seq),
                         ({"b": {"n": [1]}, "c": 3, "d": True}, 3))
        # and the restore can run cleanly afterwards
        self.assertEqual(s.restore(1), 4)

    def test_oserror_on_commit_record_keeps_batch_pending(self):
        s = self.build()
        real_fsync = os.fsync
        n = {"i": 0}

        def flaky(fd):
            n["i"] += 1
            if n["i"] == 4:  # set a(1), del c(2), del d(3), commit(4)
                raise OSError("disk on fire")
            return real_fsync(fd)

        with mock.patch("app.os.fsync", side_effect=flaky):
            with self.assertRaises(OSError):
                s.restore(1)
        self.assertEqual((s.commit_seq, s.state),
                         (3, {"b": {"n": [1]}, "c": 3, "d": True}))
        pc = s.pending_changes()
        self.assertEqual(pc.pending_count, 3)
        self.assertEqual(
            pc.changes,
            [
                {"op": "set", "key": "a", "value": 1},
                {"op": "delete", "key": "c"},
                {"op": "delete", "key": "d"},
            ],
        )
        self.assertEqual(s.rollback(), 3)
        self.assertEqual(s.pending_changes().pending_count, 0)

    def test_restore_under_exclusive_lease(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        self.assertEqual(s.restore(0), 2)
        self.assertEqual((s.state, s.commit_seq), ({}, 2))
        s.close()
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({}, 2))

    def test_closed_store_raises(self):
        s = WalStore(self.path)
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.restore(0)

    def test_repeated_restores(self):
        s = self.build()
        self.assertEqual(s.restore(0), 4)       # {}
        self.assertEqual(s.restore(2), 5)       # {b, c}
        self.assertEqual(s.restore(1), 6)       # {a, b}
        self.assertEqual(s.state, {"a": 1, "b": {"n": [1]}})
        self.assertEqual(s.commit_seq, 6)
        self.assertEqual(s.recover()["pending_count"], 0)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq),
                         ({"a": 1, "b": {"n": [1]}}, 6))
        # full history intact: 1..6
        self.assertEqual([e["commit_seq"] for e in s2.history()],
                         [1, 2, 3, 4, 5, 6])

    def test_restore_uses_original_log_structure(self):
        s = self.build()
        s.restore(1)
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        for r in rows:
            op = r["op"]
            if op == "set":
                self.assertEqual(set(r), {"op", "key", "value", "seq"})
            elif op == "delete":
                self.assertEqual(set(r), {"op", "key", "seq"})
            else:
                self.assertEqual(set(r), {"op", "seq"})
            self.assertEqual(r["seq"], r["seq"])
        # seq continuity: restore batch carries the next seq only
        seqs = [(r["op"], r["seq"]) for r in rows if r["op"] == "commit"]
        self.assertEqual(seqs, [("commit", 1), ("commit", 2),
                                ("commit", 3), ("commit", 4)])


if __name__ == "__main__":
    unittest.main()
