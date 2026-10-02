import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import app
from app import WalCorruptionError, WalStore


class RecoverTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "store.wal"

    def tearDown(self):
        self.dir.cleanup()

    def write_lines(self, *rows):
        with self.path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(row if isinstance(row, str) else json.dumps(row))
                f.write("\n")

    def test_empty_log(self):
        s = WalStore(self.path)
        r = s.recover()
        self.assertEqual(r["state"], {})
        self.assertEqual(r["commit_seq"], 0)
        self.assertEqual(r["pending_count"], 0)
        self.assertEqual((s.state, s.commit_seq), ({}, 0))

    def test_committed_state_and_pending(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", 2)
        s.commit()
        s.set("c", 3)
        s.delete("a")
        r = s.recover()
        self.assertEqual(r["state"], {"a": 1, "b": 2})
        self.assertEqual(r["commit_seq"], 1)
        self.assertEqual(r["pending_count"], 2)
        # pending tail is ignored after reopening
        s2 = WalStore(self.path)
        self.assertEqual(s2.state, {"a": 1, "b": 2})
        self.assertEqual(s2.commit_seq, 1)

    def test_order_and_delete_semantics(self):
        self.write_lines(
            {"op": "set", "key": "k", "value": 1, "seq": 1},
            {"op": "set", "key": "k", "value": 2, "seq": 1},
            {"op": "delete", "key": "k", "seq": 1},
            {"op": "set", "key": "k", "value": 3, "seq": 1},
            {"op": "commit", "seq": 1},
        )
        s = WalStore(self.path)
        self.assertEqual(s.state, {"k": 3})

    def test_empty_commit_advances_seq_only(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.commit()
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 2)

    def test_result_is_consistent_and_isolated(self):
        s = WalStore(self.path)
        s.set("a", {"nested": [1]})
        s.commit()
        r1 = s.recover()
        r2 = s.recover()
        self.assertEqual(r1, r2)
        self.assertEqual(r1["state"], s.state)
        self.assertEqual(r1["commit_seq"], s.commit_seq)
        r1["state"]["a"]["nested"].append(2)
        r1["state"]["b"] = 9
        self.assertEqual(s.state, {"a": {"nested": [1]}})
        self.assertEqual(s.recover()["state"], {"a": {"nested": [1]}})

    def test_last_line_without_newline(self):
        self.write_lines({"op": "set", "key": "a", "value": 1, "seq": 1})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"op": "commit", "seq": 1}))
        s = WalStore(self.path)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))

    def test_corruption_raises_and_preserves_memory(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "x"')  # truncated
        with self.assertRaises(WalCorruptionError):
            s.recover()
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)

    def test_corruption_cases(self):
        bad_logs = [
            ["not json"],
            ['["set"]'],  # not an object
            [json.dumps({"op": "bogus", "seq": 1})],
            [json.dumps({"op": "set", "key": "a", "seq": 1})],  # missing value
            [json.dumps({"op": "commit", "seq": 1, "extra": 1})],  # extra field
            [json.dumps({"op": "set", "key": "a", "value": 1, "seq": 0})],  # non-positive
            [json.dumps({"op": "set", "key": "a", "value": 1, "seq": "1"})],  # not int
            [json.dumps({"op": "set", "key": "a", "value": 1, "seq": True})],  # bool
            [json.dumps({"op": "set", "key": "a", "value": 1, "seq": 2})],  # mod mismatch
            [json.dumps({"op": "commit", "seq": 2})],  # commit jump
            [json.dumps({"op": "commit", "seq": 1}), json.dumps({"op": "commit", "seq": 1})],  # dup
        ]
        for lines in bad_logs:
            self.write_lines(*lines)
            with self.assertRaises(WalCorruptionError, msg=lines):
                WalStore(self.path)

    def test_no_partial_result_on_corruption(self):
        self.write_lines(
            {"op": "set", "key": "a", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1},
            {"op": "set", "key": "b", "value": 2, "seq": 2},
            {"op": "commit", "seq": 5},  # jump after valid prefix
        )
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)


class DurabilityTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "store.wal"

    def tearDown(self):
        self.dir.cleanup()

    def reopen(self):
        return WalStore(self.path)

    def test_committed_seq_stable_and_strictly_increasing_across_reopens(self):
        s = WalStore(self.path)
        s.set("a", 1)
        self.assertEqual(s.commit(), 1)
        s = self.reopen()
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))

        s.set("b", 2)
        s.delete("a")
        self.assertEqual(s.commit(), 2)
        s = self.reopen()
        self.assertEqual((s.state, s.commit_seq), ({"b": 2}, 2))

        s.set("c", 3)
        self.assertEqual(s.commit(), 3)
        s = self.reopen()
        self.assertEqual((s.state, s.commit_seq), ({"b": 2, "c": 3}, 3))

    def test_empty_commits_advance_seq_on_disk(self):
        s = WalStore(self.path)
        self.assertEqual(s.commit(), 1)
        self.assertEqual(s.commit(), 2)
        s = self.reopen()
        self.assertEqual(s.state, {})
        self.assertEqual(s.commit_seq, 2)
        self.assertEqual(s.commit(), 3)
        s = self.reopen()
        self.assertEqual(s.commit_seq, 3)

    def test_successive_uncommitted_batches_counted_and_ignored_after_reopen(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        # first uncommitted batch
        s.set("b", 2)
        s.set("c", 3)
        # second uncommitted batch (seq stays 2; no commit in between)
        s.delete("a")
        s.set("d", 4)
        r = s.recover()
        self.assertEqual(r["state"], {"a": 1})
        self.assertEqual(r["commit_seq"], 1)
        self.assertEqual(r["pending_count"], 4)
        s2 = self.reopen()
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        r2 = s2.recover()
        self.assertEqual(r2["pending_count"], 4)
        self.assertEqual(r2.state, {"a": 1})

    def test_kill_after_commit_keeps_state_kill_before_commit_drops_it(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()  # simulate process exit right after commit returned
        s = self.reopen()
        self.assertEqual(s.state, {"a": 1})

        s.set("b", 2)  # simulate process exit without commit
        s = self.reopen()
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))

        s.set("b", 20)
        s.set("a", 10)
        s.commit()
        s = self.reopen()
        self.assertEqual((s.state, s.commit_seq), ({"a": 10, "b": 20}, 2))

    def test_delete_nonexistent_key_is_not_an_error(self):
        s = WalStore(self.path)
        s.delete("ghost")
        s.commit()
        s = self.reopen()
        self.assertEqual((s.state, s.commit_seq), ({}, 1))

    def test_recovery_deterministic_under_same_prefix(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", [1, 2])
        s.commit()
        s.set("c", 3)
        s.delete("a")
        raw = self.path.read_bytes()

        results = []
        for _ in range(3):
            other_dir = tempfile.TemporaryDirectory()
            try:
                p = Path(other_dir.name) / "copy.wal"
                p.write_bytes(raw)
                w = WalStore(p)
                r = w.recover()
                results.append((dict(r.state), r.commit_seq, r.pending_count))
            finally:
                other_dir.cleanup()
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(results[0], ({"a": 1, "b": [1, 2]}, 1, 2))

    def test_each_append_is_fsynced(self):
        calls = []
        real_fsync = os.fsync

        def tracking_fsync(fd):
            calls.append(fd)
            return real_fsync(fd)

        s = WalStore(self.path)
        with unittest.mock.patch("app.os.fsync", side_effect=tracking_fsync):
            s.set("a", 1)
            s.commit()
        # one fsync for the data record, one for the commit record
        self.assertGreaterEqual(len(calls), 2)

    def test_failed_commit_raises_oserror_without_advancing_memory(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()

        def boom(fd):
            raise OSError("disk on fire")

        with unittest.mock.patch("app.os.fsync", side_effect=boom):
            with self.assertRaises(OSError):
                s.set("b", 2)
            with self.assertRaises(OSError):
                s.commit()

        # No fabricated committed state and no consumed sequence number:
        # commit_seq is advanced only after the durable record succeeds.
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))

    def test_when_writes_cannot_complete_nothing_is_persisted(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()

        def denied_open(*args, **kwargs):
            raise OSError("cannot write")

        with unittest.mock.patch.object(Path, "open", side_effect=denied_open):
            with self.assertRaises(OSError):
                s.set("b", 2)
            with self.assertRaises(OSError):
                s.commit()

        # The store still reports the old boundary in memory...
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        # ...nothing partial reached the log, and the seq was not burned:
        # the next successful commit is exactly 2.
        s2 = self.reopen()
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        self.assertEqual(s2.commit(), 2)
        s3 = self.reopen()
        self.assertEqual((s3.state, s3.commit_seq), ({"a": 1}, 2))

    def test_reopen_after_corruption_only_reports_corruption(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "commit", "seq": 3}\n')
        with self.assertRaises(WalCorruptionError):
            self.reopen()

    def test_corruption_does_not_partially_update_live_state(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "b", "value": 2, "seq": 2}\n')
            f.write("garbage-line\n")
        with self.assertRaises(WalCorruptionError):
            s.recover()
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))

    def test_returned_state_mutation_cannot_affect_store(self):
        s = WalStore(self.path)
        s.set("a", {"x": [1]})
        s.commit()
        r = s.recover()
        r["state"]["a"]["x"].append(2)
        r.state["a"]["y"] = 9
        self.assertEqual(r.commit_seq, 1)
        self.assertEqual(s.state, {"a": {"x": [1]}})
        self.assertEqual(self.reopen().state, {"a": {"x": [1]}})


if __name__ == "__main__":
    unittest.main()
