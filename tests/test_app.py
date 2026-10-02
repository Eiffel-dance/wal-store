import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
            f.write('{"op": "set", "key": "x"}\n')  # parseable but invalid record
        with self.assertRaises(WalCorruptionError):
            s.recover()
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)

    def test_interrupted_write_tail_is_discarded(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix = self.path.read_bytes()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "x"')  # interrupted write fragment
        r = s.recover()
        self.assertEqual(r["state"], {"a": 1})
        self.assertEqual(r["commit_seq"], 1)
        self.assertEqual(r["pending_count"], 0)
        # reopening sees the same committed state, never the fragment
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        # the seq chain continues from the last commit...
        s2.set("b", 2)
        self.assertEqual(s2.commit(), 2)
        s3 = WalStore(self.path)
        self.assertEqual((s3.state, s3.commit_seq), ({"a": 1, "b": 2}, 2))
        # ...and the legal prefix bytes were never rewritten
        self.assertTrue(self.path.read_bytes().startswith(prefix))
        self.assertNotIn(b'"key": "x"', self.path.read_bytes())

    def test_fragment_only_log_recovers_empty(self):
        with self.path.open("w", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "x"')
        for _ in range(2):
            s = WalStore(self.path)
            r = s.recover()
            self.assertEqual(r["state"], {})
            self.assertEqual(r["commit_seq"], 0)
            self.assertEqual(r["pending_count"], 0)
            self.assertEqual((s.state, s.commit_seq), ({}, 0))
        # a fresh seq chain starts cleanly on top of the discarded fragment
        s.set("a", 1)
        self.assertEqual(s.commit(), 1)
        self.assertEqual(WalStore(self.path).state, {"a": 1})

    def test_fragment_of_commit_record_does_not_commit(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)  # complete pending record, seq 2
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "commit", "seq": 2')  # interrupted commit record
        r = s.recover()
        self.assertEqual(r["state"], {"a": 1})  # b must not leak into state
        self.assertEqual(r["commit_seq"], 1)
        self.assertEqual(r["pending_count"], 1)  # only the complete record

    def test_interrupted_write_fragment_shapes(self):
        fragments = [
            "{",
            '{"op": "set"',
            '{"op": "set", "key": "a", "value": 1.',  # inside a number
            '{"op": "set", "key": "a", "value": tru',  # inside a literal
            '{"op": "set", "key": "a", "value": "x\\',  # inside an escape
            '{"op": "set", "key": "a", "value": "\\u12',  # inside a \uXXXX escape
        ]
        for frag in fragments:
            with self.path.open("w", encoding="utf-8") as f:
                f.write(frag)
            s = WalStore(self.path)
            self.assertEqual((s.state, s.commit_seq), ({}, 0), msg=frag)

    def test_truncated_utf8_tail_is_discarded(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.path.open("ab") as f:
            f.write('{"op": "set", "key": "hé'.encode("utf-8")[:-1])
        r = s.recover()
        self.assertEqual((r["state"], r["commit_seq"], r["pending_count"]), ({"a": 1}, 1, 0))
        # a log holding only truncated UTF-8 bytes recovers empty
        with self.path.open("wb") as f:
            f.write('{"op": "set", "key": "hé'.encode("utf-8")[:-1])
        s2 = WalStore(self.path)
        r2 = s2.recover()
        self.assertEqual((r2["state"], r2["commit_seq"], r2["pending_count"]), ({}, 0, 0))

    def test_invalid_utf8_beyond_tail_fragment_is_corruption(self):
        bad = [
            b"\xff",  # never a valid UTF-8 byte
            b"\xe4\x28",  # invalid continuation, not a truncation
            b'{"op": "commit", "seq": 1}\n\xff',  # invalid byte after a record
            b'\xff{"op": "commit", "seq": 1}\n',  # invalid byte before a record
        ]
        for data in bad:
            with self.path.open("wb") as f:
                f.write(data)
            with self.assertRaises(WalCorruptionError, msg=data):
                WalStore(self.path)

    def test_invalid_json_at_tail_is_corruption(self):
        bad_tails = [
            "not json",
            '{"op": "set", "key": "x",}',  # trailing comma
            '{"op": "set", "key": "x"}extra',  # trailing data
            "[1, 2]",  # not an object
            '{"op": "set", "key": "\\q"}',  # bad escape
        ]
        for tail in bad_tails:
            with self.path.open("w", encoding="utf-8") as f:
                f.write(tail)
            with self.assertRaises(WalCorruptionError, msg=tail):
                WalStore(self.path)

    def test_trailing_whitespace_and_empty_records_are_corruption(self):
        self.write_lines({"op": "commit", "seq": 1})
        with self.path.open("a", encoding="utf-8") as f:
            f.write("   ")  # trailing whitespace, no newline
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)
        self.write_lines({"op": "commit", "seq": 1}, "")  # blank record line
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    def test_duplicate_field_is_corruption(self):
        with self.path.open("w", encoding="utf-8") as f:
            f.write('{"op": "set", "op": "set", "key": "a", "value": 1, "seq": 1}\n')
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

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

    def write_lines(self, *rows):
        with self.path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(row if isinstance(row, str) else json.dumps(row))
                f.write("\n")

    def test_seq_monotonic_across_reopens(self):
        s = WalStore(self.path)
        self.assertEqual(s.set("a", 1), None)
        self.assertEqual(s.commit(), 1)
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        self.assertEqual(s.commit(), 3)  # empty commit advances seq

        s2 = WalStore(self.path)
        self.assertEqual(s2.commit_seq, 3)
        self.assertEqual(s2.state, {"a": 1, "b": 2})
        self.assertEqual(s2.commit(), 4)  # empty commit after reopen

        s3 = WalStore(self.path)
        self.assertEqual(s3.commit_seq, 4)
        self.assertEqual(s3.state, {"a": 1, "b": 2})
        # committed seq stays strictly increasing
        seqs = [s3.commit() for _ in range(3)]
        self.assertEqual(seqs, [5, 6, 7])

    def test_state_only_changes_at_commit(self):
        s = WalStore(self.path)
        s.set("a", 1)
        self.assertEqual(s.state, {})  # deferred until commit
        s.commit()
        self.assertEqual(s.state, {"a": 1})
        s.delete("a")
        self.assertEqual(s.state, {"a": 1})  # delete deferred too
        s.delete("missing")  # deleting absent key is not an error
        s.commit()
        self.assertEqual(s.state, {})

    def test_successive_uncommitted_batches_after_reopen(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        # first uncommitted batch (abandoned)
        s.set("b", 2)
        s.delete("a")
        # reopen simulates termination before commit: pending tail ignored/counted
        s2 = WalStore(self.path)
        r = s2.recover()
        self.assertEqual(r["state"], {"a": 1})
        self.assertEqual(r["commit_seq"], 1)
        self.assertEqual(r["pending_count"], 2)
        # attribute access on the result keeps working
        self.assertEqual((r.state, r.commit_seq, r.pending_count), ({"a": 1}, 1, 2))

        # second uncommitted batch
        s2.set("c", 3)
        s3 = WalStore(self.path)
        r = s3.recover()
        self.assertEqual(r["state"], {"a": 1})
        self.assertEqual(r["commit_seq"], 1)
        self.assertEqual(r["pending_count"], 3)  # old tail + new record

    def test_termination_after_commit_is_durable(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", [1, 2])
        seq = s.commit()
        self.assertEqual(seq, 1)
        s2 = WalStore(self.path)  # "process restarted"
        self.assertEqual(s2.state, {"a": 1, "b": [1, 2]})
        self.assertEqual(s2.commit_seq, 1)

    def test_termination_before_commit_recovers_earlier_state(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("a", 2)
        s.set("c", 3)
        s.delete("a")
        # no commit; reopen must show only the earlier committed state
        s2 = WalStore(self.path)
        self.assertEqual(s2.state, {"a": 1})
        self.assertEqual(s2.commit_seq, 1)
        r = s2.recover()
        self.assertEqual(r["pending_count"], 3)

    def test_commit_record_fsyncs_before_returning(self):
        # The commit boundary must be flushed and fsynced before commit()
        # returns, otherwise an abrupt termination could lose it.
        s = WalStore(self.path)
        s.set("a", 1)
        with mock.patch("app.os.fsync", autospec=True) as fsync:
            seq = s.commit()
            self.assertEqual(seq, 1)
            self.assertGreaterEqual(fsync.call_count, 1)

    def test_set_record_fsyncs_before_returning(self):
        s = WalStore(self.path)
        with mock.patch("app.os.fsync", autospec=True) as fsync:
            s.set("a", 1)
            self.assertGreaterEqual(fsync.call_count, 1)
            # durability on reopen is observable via a real fresh instance
        self.assertEqual(WalStore(self.path).recover()["pending_count"], 1)

    def test_write_failure_on_commit_raises_oserror_and_keeps_state(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()

        def boom(fd):
            raise OSError("disk on fire")

        with mock.patch("app.os.fsync", side_effect=boom):
            with self.assertRaises(OSError):
                s.commit()
        # seq not consumed, no faked committed state
        self.assertEqual(s.commit_seq, 1)
        self.assertEqual(s.state, {"a": 1})

        # the failed commit boundary must not appear in the log on reopen
        s2 = WalStore(self.path)
        self.assertEqual(s2.commit_seq, 1)
        self.assertEqual(s2.state, {"a": 1})

    def test_write_failure_on_set_raises_oserror_and_no_pending_record(self):
        s = WalStore(self.path)

        def boom(fd):
            raise OSError("disk on fire")

        with mock.patch("app.os.fsync", side_effect=boom):
            with self.assertRaises(OSError):
                s.set("a", 1)
        self.assertEqual(s.state, {})
        self.assertEqual(s.commit_seq, 0)
        # retry after the transient failure works with the same seq chain
        s.set("a", 1)
        self.assertEqual(s.commit(), 1)
        self.assertEqual(WalStore(self.path).state, {"a": 1})

    def test_failed_commit_does_not_block_next_seq(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()  # seq 1

        with mock.patch("app.os.fsync", side_effect=OSError("transient")):
            with self.assertRaises(OSError):
                s.commit()
        self.assertEqual(s.commit_seq, 1)
        # next successful commit is exactly seq 2, never skipped
        self.assertEqual(s.commit(), 2)
        self.assertEqual(WalStore(self.path).commit_seq, 2)

    def test_corruption_after_reopen_reports_only_corruption_error(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "x"}\n')  # invalid record, terminated
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    def test_interrupted_write_then_reopen_continues_seq_chain(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "b", "value": 2, "seq": 2')  # interrupted
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        self.assertEqual(s2.recover()["pending_count"], 0)
        # the next records form a continuous chain from the last commit
        s2.set("b", 2)
        self.assertEqual(s2.commit(), 2)
        s3 = WalStore(self.path)
        self.assertEqual((s3.state, s3.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(s3.recover()["pending_count"], 0)

    def test_corruption_in_committed_region_on_reopen(self):
        self.write_lines(
            {"op": "set", "key": "a", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1},
            {"op": "set", "key": "b", "value": 2, "seq": 1},  # stale seq chain
        )
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    def test_non_string_op_is_corruption_not_typeerror(self):
        self.write_lines(json.dumps({"op": 123, "seq": 1}))
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)
        self.write_lines(json.dumps({"op": None, "seq": 1}))
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    def test_result_is_independent_copy_of_storage(self):
        s = WalStore(self.path)
        s.set("a", {"n": [1]})
        s.commit()
        r = s.recover()
        r["state"]["a"]["n"].append(99)
        r["state"]["b"] = 2
        self.assertEqual(s.state, {"a": {"n": [1]}})
        self.assertEqual(WalStore(self.path).state, {"a": {"n": [1]}})

    def test_recovery_is_deterministic_under_same_prefix(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", 2)
        s.delete("a")
        s.set("a", 3)
        s.commit()
        s.set("z", 9)

        results = []
        for _ in range(3):
            rs = WalStore(self.path)
            results.append((dict(rs.recover()), rs.state, rs.commit_seq))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        r0 = results[0][0]
        self.assertEqual(r0["state"], {"a": 3, "b": 2})  # original batch order
        self.assertEqual(r0["commit_seq"], 1)
        self.assertEqual(r0["pending_count"], 1)


class QueryTest(unittest.TestCase):
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

    def log_size(self):
        return self.path.stat().st_size if self.path.exists() else 0

    def test_empty_log(self):
        s = WalStore(self.path)
        with self.assertRaises(KeyError):
            s.get("x")
        self.assertIs(s.contains("x"), False)

    def test_get_returns_independent_deep_copy(self):
        s = WalStore(self.path)
        s.set("a", {"nested": [1, {"k": 2}]})
        s.set("n", None)
        s.commit()
        got = s.get("a")
        got["nested"].append(99)
        got["nested"][1]["k"] = 7
        # state, subsequent queries, and a reopened store all stay unchanged
        self.assertEqual(s.state, {"a": {"nested": [1, {"k": 2}]}, "n": None})
        self.assertEqual(s.get("a"), {"nested": [1, {"k": 2}]})
        self.assertEqual(WalStore(self.path).get("a"), {"nested": [1, {"k": 2}]})

    def test_default_is_independent_copy(self):
        s = WalStore(self.path)
        default = {"d": [1]}
        returned = s.get("missing", default)
        returned["d"].append(2)
        self.assertEqual(default, {"d": [1]})
        self.assertEqual(s.get("missing", default), {"d": [1]})
        self.assertIsNone(s.get("missing", None))
        self.assertEqual(s.get("missing", 5), 5)
        self.assertEqual(s.get("missing", default=9), 9)

    def test_missing_key_without_default_raises_keyerror(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.assertRaises(KeyError):
            s.get("nope")

    def test_stored_none_is_present_not_missing(self):
        s = WalStore(self.path)
        s.set("n", None)
        s.commit()
        self.assertIsNone(s.get("n"))
        self.assertIs(s.contains("n"), True)
        self.assertIs(s.contains("missing"), False)
        with self.assertRaises(KeyError):
            s.get("missing")

    def test_committed_delete_removes_key(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.delete("a")
        s.commit()
        with self.assertRaises(KeyError):
            s.get("a")
        self.assertIs(s.contains("a"), False)
        self.assertEqual(s.get("a", "d"), "d")

    def test_uncommitted_changes_are_invisible(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)
        s.delete("a")
        s.set("a", 99)
        # only the last commit is observable, including for an uncommitted delete
        self.assertEqual(s.get("a"), 1)
        self.assertIs(s.contains("a"), True)
        self.assertIs(s.contains("b"), False)
        with self.assertRaises(KeyError):
            s.get("b")
        self.assertEqual(s.get("b", "def"), "def")
        s.commit()
        self.assertEqual(s.get("a"), 99)
        self.assertIs(s.contains("b"), True)
        self.assertEqual(s.get("b"), 2)

    def test_non_string_key_raises_valueerror(self):
        s = WalStore(self.path)
        for bad in (1, 1.5, None, b"x", ["x"], {"x": 1}):
            with self.assertRaises(ValueError, msg=bad):
                s.get(bad)
            with self.assertRaises(ValueError, msg=bad):
                s.get(bad, "d")
            with self.assertRaises(ValueError, msg=bad):
                s.contains(bad)

    def test_invalid_default_raises_valueerror(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        for bad in (float("nan"), float("inf"), {1: 2}, object(), {"x": object()}):
            with self.assertRaises(ValueError, msg=bad):
                s.get("missing", bad)
        # rejected default leaves state and the stored value intact
        self.assertEqual(s.get("a"), 1)

    def test_queries_do_not_touch_log_seq_or_pending(self):
        s = WalStore(self.path)
        s.set("a", {"v": [1]})
        s.commit()
        s.set("tail", 2)
        s.delete("a")
        size = self.log_size()
        for _ in range(3):
            self.assertEqual(s.get("a"), {"v": [1]})
            self.assertIs(s.contains("a"), True)
            s.get("missing", {"x": [1]})
            s.contains("missing")
        self.assertEqual(self.log_size(), size)  # no log append
        self.assertEqual(s.commit_seq, 1)
        self.assertEqual(s.recover()["pending_count"], 2)
        s2 = WalStore(self.path)
        self.assertEqual(s2.commit_seq, 1)
        self.assertEqual(s2.recover()["pending_count"], 2)

    def test_corruption_during_query_raises_and_keeps_memory(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "x"}\n')  # invalid record, terminated
        with self.assertRaises(WalCorruptionError):
            s.get("a")
        with self.assertRaises(WalCorruptionError):
            s.contains("a")
        # no partial replay replaced the committed in-memory state
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)

    def test_tail_fragment_is_invisible_to_queries(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "x"')  # interrupted write fragment
        size = self.log_size()
        self.assertEqual(s.get("a"), 1)
        self.assertIs(s.contains("a"), True)
        self.assertIs(s.contains("x"), False)
        with self.assertRaises(KeyError):
            s.get("x")
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        # queries never rewrite the log, fragment included
        self.assertEqual(self.log_size(), size)

    def test_non_standard_json_constant_in_log_is_corruption(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "b", "value": NaN, "seq": 2}\n')
        with self.assertRaises(WalCorruptionError):
            s.get("a")
        self.assertEqual(s.state, {"a": 1})

    def test_all_json_value_shapes_round_trip(self):
        s = WalStore(self.path)
        values = {
            "i": 1, "f": 1.5, "s": "hi", "b": True, "n": None,
            "arr": [1, "two", False, None, {"x": []}],
            "obj": {"k": [1, 2]},
        }
        for k, v in values.items():
            s.set(k, v)
        s.commit()
        for k, v in values.items():
            self.assertEqual(s.get(k), v)
            self.assertIs(s.contains(k), True)

    def test_legacy_log_serves_queries_without_migration(self):
        self.write_lines(
            {"op": "set", "key": "legacy", "value": [1, 2], "seq": 1},
            {"op": "commit", "seq": 1},
        )
        s = WalStore(self.path)
        self.assertEqual(s.get("legacy"), [1, 2])
        self.assertIs(s.contains("legacy"), True)
        self.assertIs(s.contains("other"), False)


if __name__ == "__main__":
    unittest.main()
