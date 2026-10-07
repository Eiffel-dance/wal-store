import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app
from app import WalCapacityError, WalCorruptionError, WalPendingError, WalStore


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
        # The terminator is the adoption boundary: a final segment without
        # one is an unfinished write and is discarded whole, even when its
        # bytes parse as a complete, fully valid record.
        self.write_lines({"op": "set", "key": "a", "value": 1, "seq": 1})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"op": "commit", "seq": 1}))
        s = WalStore(self.path)
        self.assertEqual((s.state, s.commit_seq), ({}, 0))
        r = s.recover()
        self.assertEqual(r["state"], {})
        self.assertEqual(r["commit_seq"], 0)
        # only the terminated set counts as pending; the fragment does not
        self.assertEqual(r["pending_count"], 1)

    def test_unterminated_complete_record_fragment_is_discarded(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix = self.path.read_bytes()
        # a complete, valid set record whose terminator never became durable
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"op": "set", "key": "x", "value": 9, "seq": 2}))
        for _ in range(2):
            r = s.recover()
            self.assertEqual(r["state"], {"a": 1})
            self.assertEqual(r["commit_seq"], 1)
            self.assertEqual(r["pending_count"], 0)  # fragment is not pending
            s2 = WalStore(self.path)
            self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        # the fragment is invisible to read-only queries
        self.assertIs(s.contains("x"), False)
        with self.assertRaises(KeyError):
            s.get("x")
        # the next append drops the fragment and continues the seq chain
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        s3 = WalStore(self.path)
        self.assertEqual((s3.state, s3.commit_seq), ({"a": 1, "b": 2}, 2))
        data = self.path.read_bytes()
        self.assertTrue(data.startswith(prefix))
        self.assertNotIn(b'"key": "x"', data)  # never spliced into new JSON

    def test_unterminated_complete_commit_fragment_does_not_commit(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)  # complete pending record, seq 2
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"op": "commit", "seq": 2}))  # no terminator
        r = s.recover()
        self.assertEqual(r["state"], {"a": 1})  # b must not leak into state
        self.assertEqual(r["commit_seq"], 1)
        self.assertEqual(r["pending_count"], 1)  # only the terminated record

    def test_unterminated_invalid_record_is_corruption(self):
        bad_tails = [
            json.dumps({"op": "set", "key": "a", "value": 1, "seq": 5}),  # seq jump
            json.dumps({"op": "set", "key": "a", "seq": 1}),  # missing value
            json.dumps({"op": "commit", "seq": 1, "extra": 1}),  # extra field
            json.dumps({"op": "bogus", "seq": 1}),  # unknown op
            json.dumps({"op": "set", "key": "a", "value": 1, "seq": "1"}),  # bad seq
            json.dumps([1, 2]),  # not an object
            '{"op": "set", "key": "a", "value": 1, "seq": 1}extra',  # trailing data
        ]
        for tail in bad_tails:
            with self.path.open("w", encoding="utf-8") as f:
                f.write(tail)  # no terminator
            with self.assertRaises(WalCorruptionError, msg=tail):
                WalStore(self.path)

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


class SnapshotTest(unittest.TestCase):
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

    def build(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", {"n": [1]})
        s.commit()  # seq 1: {"a": 1, "b": {"n": [1]}}
        s.delete("a")
        s.set("c", 3)
        s.commit()  # seq 2: {"b": {"n": [1]}, "c": 3}
        s.commit()  # seq 3: empty commit, same state
        s.set("tail", 9)  # uncommitted
        return s

    def test_default_targets_latest_commit(self):
        s = self.build()
        snap = s.snapshot()
        self.assertEqual(set(snap), {"state", "commit_seq"})
        self.assertEqual(snap["commit_seq"], 3)
        self.assertEqual(snap["state"], {"b": {"n": [1]}, "c": 3})

    def test_empty_log_snapshots(self):
        s = WalStore(self.path)
        for snap in (s.snapshot(), s.snapshot(0)):
            self.assertEqual(snap["state"], {})
            self.assertEqual(snap["commit_seq"], 0)

    def test_historical_commits(self):
        s = self.build()
        self.assertEqual(s.snapshot(0), {"state": {}, "commit_seq": 0})
        self.assertEqual(
            s.snapshot(1), {"state": {"a": 1, "b": {"n": [1]}}, "commit_seq": 1}
        )
        self.assertEqual(
            s.snapshot(2), {"state": {"b": {"n": [1]}, "c": 3}, "commit_seq": 2}
        )
        self.assertEqual(
            s.snapshot(3), {"state": {"b": {"n": [1]}, "c": 3}, "commit_seq": 3}
        )

    def test_uncommitted_tail_never_visible(self):
        s = self.build()
        for snap in (s.snapshot(), s.snapshot(3), s.snapshot(2), s.snapshot(1)):
            self.assertNotIn("tail", snap["state"])

    def test_invalid_target_seq_raises_valueerror(self):
        s = self.build()
        for bad in (True, False, -1, 1.0, "1", b"1", [1], {"s": 1}):
            with self.assertRaises(ValueError, msg=bad):
                s.snapshot(bad)
        with self.assertRaises(ValueError):
            s.snapshot(4)  # beyond latest commit
        with self.assertRaises(ValueError):
            s.snapshot(10**9)

    def test_result_is_independent_deep_copy(self):
        s = self.build()
        snap = s.snapshot(1)
        snap["state"]["b"]["n"].append(99)
        snap["state"]["x"] = 1
        self.assertEqual(s.snapshot(1)["state"], {"a": 1, "b": {"n": [1]}})
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        self.assertEqual(WalStore(self.path).snapshot(1)["state"], {"a": 1, "b": {"n": [1]}})

    def test_deterministic_across_calls_and_reopens(self):
        s = self.build()
        first = s.snapshot(2)
        for _ in range(3):
            self.assertEqual(s.snapshot(2), first)
            self.assertEqual(WalStore(self.path).snapshot(2), first)

    def test_snapshot_is_read_only(self):
        s = self.build()
        size = self.log_size()
        for target in (None, 0, 1, 2, 3):
            s.snapshot() if target is None else s.snapshot(target)
        self.assertEqual(self.log_size(), size)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        self.assertEqual(s.commit_seq, 3)
        self.assertEqual(s.recover()["pending_count"], 1)

    def test_corruption_anywhere_raises_no_partial_snapshot(self):
        s = self.build()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "x"}\n')  # invalid record, terminated
        for target in (None, 0, 1, 2, 3):
            with self.assertRaises(WalCorruptionError, msg=target):
                s.snapshot() if target is None else s.snapshot(target)
        # in-memory state untouched
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        self.assertEqual(s.commit_seq, 3)

    def test_tail_fragment_does_not_affect_snapshot(self):
        s = self.build()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "frag"')  # interrupted write
        size = self.log_size()
        snap = s.snapshot()
        self.assertEqual(snap["state"], {"b": {"n": [1]}, "c": 3})
        self.assertEqual(snap["commit_seq"], 3)
        self.assertEqual(self.log_size(), size)  # fragment left in place

    def test_legacy_log_serves_snapshots(self):
        self.write_lines(
            {"op": "set", "key": "k", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1},
            {"op": "set", "key": "k", "value": 2, "seq": 2},
            {"op": "commit", "seq": 2},
        )
        s = WalStore(self.path)
        self.assertEqual(s.snapshot(1)["state"], {"k": 1})
        self.assertEqual(s.snapshot()["state"], {"k": 2})


class AuditTest(unittest.TestCase):
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

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def line_end(self, n):
        # Byte offset just past the nth newline-terminated line of the file.
        data = self.path.read_bytes()
        pos = -1
        for _ in range(n):
            pos = data.index(b"\n", pos + 1)
        return pos + 1

    def test_nonexistent_and_empty_log(self):
        s = WalStore(self.path)  # path does not exist
        r = s.audit()
        self.assertEqual(
            set(r),
            {"state", "commit_seq", "pending_count",
             "valid_bytes", "committed_bytes", "tail_bytes"},
        )
        self.assertEqual(
            (r.state, r.commit_seq, r.pending_count), ({}, 0, 0)
        )
        self.assertEqual(
            (r.valid_bytes, r.committed_bytes, r.tail_bytes), (0, 0, 0)
        )
        self.path.write_bytes(b"")
        self.assertEqual(dict(s.audit()), dict(r))

    def test_nonexistent_log_never_created_by_audit(self):
        s = WalStore(self.path)
        for _ in range(2):
            s.audit()
        self.assertFalse(self.path.exists())

    def test_empty_file_created_on_disk(self):
        self.path.write_bytes(b"")
        s = WalStore(self.path)
        r = s.audit()
        self.assertEqual(r["state"], {})
        self.assertEqual(
            (r.commit_seq, r.pending_count,
             r.valid_bytes, r.committed_bytes, r.tail_bytes),
            (0, 0, 0, 0, 0),
        )

    def test_committed_prefix_and_uncommitted_records(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", 2)
        s.commit()
        s.set("c", 3)
        s.delete("a")
        size = self.path.stat().st_size
        r = s.audit()
        # recover triple is exactly what recover() reports
        rec = s.recover()
        self.assertEqual(
            (r["state"], r["commit_seq"], r["pending_count"]),
            (rec["state"], rec["commit_seq"], rec["pending_count"]),
        )
        self.assertEqual(r["state"], {"a": 1, "b": 2})
        self.assertEqual((r["commit_seq"], r["pending_count"]), (1, 2))
        # every record is terminated and accepted: valid prefix is the file
        self.assertEqual(r["valid_bytes"], size)
        self.assertEqual(r["tail_bytes"], 0)
        # committed prefix ends after the commit record (2 set lines + commit)
        self.assertEqual(r["committed_bytes"], self.line_end(3))
        self.assertLessEqual(r["committed_bytes"], r["valid_bytes"])

    def test_no_commit_means_committed_bytes_zero(self):
        self.write_lines({"op": "set", "key": "a", "value": 1, "seq": 1})
        s = WalStore(self.path)
        r = s.audit()
        self.assertEqual(r["state"], {})
        self.assertEqual((r["commit_seq"], r["pending_count"]), (0, 1))
        self.assertEqual(r["committed_bytes"], 0)
        self.assertEqual(r["valid_bytes"], self.path.stat().st_size)
        self.assertEqual(r["tail_bytes"], 0)

    def test_empty_commit_committed_bytes_advances(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.commit()
        r = s.audit()
        self.assertEqual((r["commit_seq"], r["pending_count"]), (2, 0))
        self.assertEqual(r["committed_bytes"], self.path.stat().st_size)
        self.assertEqual(
            (r["valid_bytes"], r["tail_bytes"]),
            (self.path.stat().st_size, 0),
        )

    def test_tail_fragment_json_prefix_counts_only_as_tail(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        fragment = '{"op": "set", "key": "x"'
        self.append_bytes(fragment)
        prefix_end = self.line_end(2)
        size = self.path.stat().st_size
        r = s.audit()
        self.assertEqual(r["state"], {"a": 1})
        self.assertEqual((r["commit_seq"], r["pending_count"]), (1, 0))
        self.assertEqual(r["valid_bytes"], prefix_end)
        self.assertEqual(r["committed_bytes"], prefix_end)
        self.assertEqual(r["tail_bytes"], len(fragment.encode("utf-8")))
        self.assertEqual(r["valid_bytes"] + r["tail_bytes"], size)

    def test_tail_fragment_shapes(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix_end = self.line_end(2)
        fragments = [
            "{",
            '{"op": "set"',
            '{"op": "set", "key": "a", "value": 1.',
            '{"op": "set", "key": "a", "value": tru',
            '{"op": "set", "key": "a", "value": "x\\',
            '{"op": "set", "key": "a", "value": "\\u12',
            '{"op": "commit", "seq": 2',
        ]
        for frag in fragments:
            self.path.write_bytes(self.path.read_bytes()[:prefix_end])
            self.append_bytes(frag)
            size = self.path.stat().st_size
            r = s.audit()
            self.assertEqual(r["state"], {"a": 1}, msg=frag)
            self.assertEqual(r["commit_seq"], 1, msg=frag)
            self.assertEqual(r["pending_count"], 0, msg=frag)
            self.assertEqual(r["valid_bytes"], prefix_end, msg=frag)
            self.assertEqual(
                r["tail_bytes"], len(frag.encode("utf-8")), msg=frag
            )
            self.assertEqual(r["valid_bytes"] + r["tail_bytes"], size, msg=frag)

    def test_unterminated_complete_valid_record_is_tail(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)  # terminated pending record
        frag = json.dumps({"op": "set", "key": "x", "value": 9, "seq": 2})
        self.append_bytes(frag)
        r = s.audit()
        self.assertEqual(r["state"], {"a": 1})
        self.assertEqual(r["commit_seq"], 1)
        self.assertEqual(r["pending_count"], 1)  # terminated record only
        self.assertEqual(r["valid_bytes"], self.line_end(3))
        self.assertEqual(r["committed_bytes"], self.line_end(2))
        self.assertEqual(r["tail_bytes"], len(frag.encode("utf-8")))

    def test_unterminated_complete_commit_fragment_is_tail(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)
        frag = json.dumps({"op": "commit", "seq": 2})
        self.append_bytes(frag)
        r = s.audit()
        self.assertEqual(r["state"], {"a": 1})  # pending batch not committed
        self.assertEqual(r["commit_seq"], 1)
        self.assertEqual(r["pending_count"], 1)
        self.assertEqual(r["valid_bytes"], self.line_end(3))
        self.assertEqual(r["tail_bytes"], len(frag.encode("utf-8")))

    def test_truncated_utf8_tail_counts_only_as_tail(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        frag = '{"op": "set", "key": "hé'.encode("utf-8")[:-1]
        self.append_bytes(frag)
        prefix_end = self.line_end(2)
        r = s.audit()
        self.assertEqual((r["state"], r["commit_seq"], r["pending_count"]),
                         ({"a": 1}, 1, 0))
        self.assertEqual(r["valid_bytes"], prefix_end)
        self.assertEqual(r["committed_bytes"], prefix_end)
        self.assertEqual(r["tail_bytes"], len(frag))
        # a log holding only truncated UTF-8 bytes audits as fully tail
        self.path.write_bytes(frag)
        r2 = WalStore(self.path).audit()
        self.assertEqual((r2["state"], r2["commit_seq"], r2["pending_count"]),
                         ({}, 0, 0))
        self.assertEqual(
            (r2["valid_bytes"], r2["committed_bytes"], r2["tail_bytes"]),
            (0, 0, len(frag)),
        )

    def test_corruption_after_prefix_raises_without_partial_result(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix = self.path.read_bytes()
        bad_tails = [
            b"   ",  # trailing whitespace, no terminator
            b"\n",  # blank line
            b"  \n",  # blank record
            b"\xff",  # illegal UTF-8 at end
            b'{"op": "set", "key": "b", "value": NaN, "seq": 2}\n',
            b'{"op": "set", "op": "set", "key": "b", "value": 2, "seq": 2}\n',
            b'{"op": "set", "key": "b", "value": 2, "seq": 5}',  # seq jump
            b'{"op": "set", "key": "b", "seq": 2}',  # missing field
            b'{"op": "bogus", "seq": 2}',  # unknown op
            b"not json\n",
            b'[1, 2]\n',  # not an object
            b'{"op": "set", "key": "b", "value": 2, "seq": 2}extra',
        ]
        for tail in bad_tails:
            self.path.write_bytes(prefix)
            self.append_bytes(tail)
            with self.assertRaises(WalCorruptionError, msg=tail):
                s.audit()
            # memory state untouched after the failed audit
            self.assertEqual(s.state, {"a": 1}, msg=tail)
            self.assertEqual(s.commit_seq, 1, msg=tail)

    def test_corruption_anywhere_in_fragment_only_log(self):
        bad = [
            b"\xff",
            b"not json",
            b'{"op": "set", "key": "a", "value": 1, "seq": 5}',  # seq jump
            b'{"op": "set", "key": "a", "seq": 1}',
            b'{"op": "set", "key": "a", "value": 1, "seq": "1"}',
        ]
        for data in bad:
            self.path.write_bytes(data)
            with self.assertRaises(WalCorruptionError, msg=data):
                WalStore(self.path).audit()

    def test_audit_does_not_touch_log(self):
        s = WalStore(self.path)
        s.set("a", {"n": [1]})
        s.commit()
        s.set("tail", 2)
        fragment = '{"op": "set", "key": "frag"'
        self.append_bytes(fragment)
        before = self.path.read_bytes()
        size = len(before)
        results = []
        for _ in range(4):
            r = s.audit()
            results.append(dict(r))
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(self.path.stat().st_size, size)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        self.assertEqual(results[2], results[3])

    def test_audit_does_not_change_memory_state(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)  # pending, invisible in state
        self.append_bytes('{"op": "set", "key": "x"')
        for _ in range(3):
            s.audit()
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)
        # recover and queries behave exactly as without audit
        self.assertEqual(s.recover()["pending_count"], 1)
        self.assertEqual(s.get("a"), 1)
        self.assertIs(s.contains("b"), False)

    def test_audit_state_is_independent_deep_copy(self):
        s = WalStore(self.path)
        s.set("a", {"nested": [1]})
        s.commit()
        r = s.audit()
        r["state"]["a"]["nested"].append(2)
        r["state"]["z"] = 9
        self.assertEqual(s.state, {"a": {"nested": [1]}})
        again = s.audit()
        self.assertEqual(again["state"], {"a": {"nested": [1]}})

    def test_audit_deterministic_across_reopens(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", 2)
        s.commit()
        s.delete("a")
        self.append_bytes('{"op": "set", "key": "frag", "value":')
        expected = dict(s.audit())
        for _ in range(3):
            reopened = WalStore(self.path)
            self.assertEqual(dict(reopened.audit()), expected)
        # attribute access works on the six fields
        r = s.audit()
        self.assertEqual(
            (r.state, r.commit_seq, r.pending_count,
             r.valid_bytes, r.committed_bytes, r.tail_bytes),
            (expected["state"], expected["commit_seq"], expected["pending_count"],
             expected["valid_bytes"], expected["committed_bytes"],
             expected["tail_bytes"]),
        )

    def test_audit_then_append_still_drops_fragment(self):
        # audit must not disturb the cached accepted-prefix boundary that a
        # later append relies on to discard the tail fragment
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix = self.path.read_bytes()
        self.append_bytes('{"op": "set", "key": "frag"')
        s.audit()
        s.audit()
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        data = self.path.read_bytes()
        self.assertTrue(data.startswith(prefix))
        self.assertNotIn(b'"frag"', data)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))
        r = s2.audit()
        self.assertEqual(r["tail_bytes"], 0)
        self.assertEqual(r["valid_bytes"], len(data))
        self.assertEqual(r["committed_bytes"], len(data))

    def test_audit_after_clean_appends_reports_full_prefix(self):
        s = WalStore(self.path)
        s.audit()  # on a nonexistent log
        s.set("a", 1)
        r = s.audit()
        self.assertEqual(
            (r["valid_bytes"], r["committed_bytes"], r["tail_bytes"]),
            (self.path.stat().st_size, 0, 0),
        )
        self.assertEqual((r["commit_seq"], r["pending_count"]), (0, 1))
        s.commit()
        r = s.audit()
        self.assertEqual((r["commit_seq"], r["pending_count"]), (1, 0))
        self.assertEqual(
            (r["valid_bytes"], r["committed_bytes"], r["tail_bytes"]),
            (self.path.stat().st_size, self.path.stat().st_size, 0),
        )

    def test_audit_matches_recover_byte_boundaries_on_legacy_log(self):
        self.write_lines(
            {"op": "set", "key": "k", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1},
            {"op": "set", "key": "k", "value": 2, "seq": 2},
            {"op": "commit", "seq": 2},
        )
        s = WalStore(self.path)
        r = s.audit()
        rec = s.recover()
        self.assertEqual(r["state"], rec["state"])
        self.assertEqual(r["commit_seq"], rec["commit_seq"])
        self.assertEqual(r["pending_count"], rec["pending_count"])
        self.assertEqual(r["valid_bytes"], self.path.stat().st_size)
        self.assertEqual(r["committed_bytes"], self.path.stat().st_size)
        self.assertEqual(r["tail_bytes"], 0)


class RepairTailTest(unittest.TestCase):
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

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def line_end(self, n):
        # Byte offset just past the nth newline-terminated line of the file.
        data = self.path.read_bytes()
        pos = -1
        for _ in range(n):
            pos = data.index(b"\n", pos + 1)
        return pos + 1

    def test_nonexistent_log_returns_zero_and_creates_nothing(self):
        s = WalStore(self.path)  # path does not exist
        for _ in range(2):
            r = s.repair_tail()
            self.assertEqual(set(r), {"state", "commit_seq",
                                      "pending_count", "removed_bytes"})
            self.assertEqual(
                (r.state, r.commit_seq, r.pending_count, r.removed_bytes),
                ({}, 0, 0, 0),
            )
            self.assertFalse(self.path.exists())

    def test_empty_log_returns_zero(self):
        self.path.write_bytes(b"")
        s = WalStore(self.path)
        r = s.repair_tail()
        self.assertEqual(
            (r.state, r.commit_seq, r.pending_count, r.removed_bytes),
            ({}, 0, 0, 0),
        )
        self.assertEqual(self.path.read_bytes(), b"")

    def test_clean_log_needs_no_repair(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", 2)
        s.commit()
        s.set("c", 3)
        before = self.path.read_bytes()
        r = s.repair_tail()
        self.assertEqual(r.removed_bytes, 0)
        self.assertEqual((r.state, r.commit_seq, r.pending_count), ({"a": 1, "b": 2}, 1, 1))
        self.assertEqual(self.path.read_bytes(), before)
        # idempotent: nothing left to remove on a second call
        self.assertEqual(s.repair_tail().removed_bytes, 0)
        self.assertEqual(self.path.read_bytes(), before)

    def test_removes_fragment_after_committed_prefix(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix = self.path.read_bytes()
        fragment = '{"op": "set", "key": "x", "value": 9, "seq": 2'
        self.append_bytes(fragment)
        # the recover triple before repair is the acceptable-prefix view
        before = s.recover()
        r = s.repair_tail()
        self.assertEqual((r.state, r.commit_seq, r.pending_count),
                         (before.state, before.commit_seq, before.pending_count))
        self.assertEqual(r.removed_bytes, len(fragment.encode("utf-8")))
        # only the fragment is gone; the prefix bytes are byte-for-byte intact
        self.assertEqual(self.path.read_bytes(), prefix)
        # a reopen recovers exactly the acceptable prefix, fragment gone
        s2 = WalStore(self.path)
        rec = s2.recover()
        self.assertEqual((rec.state, rec.commit_seq, rec.pending_count),
                         (before.state, before.commit_seq, before.pending_count))
        self.assertEqual(s2.audit().tail_bytes, 0)
        # and a second repair is a no-op
        self.assertEqual(s2.repair_tail().removed_bytes, 0)

    def test_removes_fragment_but_keeps_complete_pending_records(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)
        s.delete("a")  # two complete, terminated, uncommitted records
        prefix_end = self.line_end(4)
        fragment = '{"op": "set", "key": "x"'
        self.append_bytes(fragment)
        size_before = self.path.stat().st_size
        r = s.repair_tail()
        self.assertEqual(r.state, {"a": 1})  # pending never visible in state
        self.assertEqual((r.commit_seq, r.pending_count), (1, 2))
        self.assertEqual(r.removed_bytes, len(fragment.encode("utf-8")))
        self.assertEqual(self.path.stat().st_size, prefix_end)
        self.assertEqual(size_before - prefix_end, r.removed_bytes)
        # complete pending records survive the repair and stay pending ...
        s2 = WalStore(self.path)
        rec = s2.recover()
        self.assertEqual((rec.state, rec.commit_seq, rec.pending_count),
                         ({"a": 1}, 1, 2))
        # ... and keep following rollback's semantics
        self.assertEqual(s2.rollback(), 2)
        self.assertEqual(self.path.stat().st_size, self.line_end(2))
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        self.assertEqual(WalStore(self.path).recover()["pending_count"], 0)

    def test_unterminated_complete_commit_fragment_removed_without_committing(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)  # terminated pending record, seq 2
        frag = json.dumps({"op": "commit", "seq": 2})
        self.append_bytes(frag)
        r = s.repair_tail()
        self.assertEqual(r.state, {"a": 1})  # batch must not leak in
        self.assertEqual((r.commit_seq, r.pending_count), (1, 1))
        self.assertEqual(r.removed_bytes, len(frag.encode("utf-8")))
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        self.assertEqual(s2.recover()["pending_count"], 1)

    def test_fragment_only_log_is_fully_removed(self):
        fragment = '{"op": "set", "key": "x"'
        self.path.write_bytes(fragment.encode("utf-8"))
        s = WalStore(self.path)
        r = s.repair_tail()
        self.assertEqual(
            (r.state, r.commit_seq, r.pending_count), ({}, 0, 0)
        )
        self.assertEqual(r.removed_bytes, len(fragment.encode("utf-8")))
        self.assertEqual(self.path.read_bytes(), b"")
        # a fresh seq chain starts cleanly on the repaired empty log
        s.set("a", 1)
        self.assertEqual(s.commit(), 1)
        self.assertEqual(WalStore(self.path).state, {"a": 1})

    def test_truncated_utf8_tail_is_removed(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix = self.path.read_bytes()
        frag = '{"op": "set", "key": "hé'.encode("utf-8")[:-1]
        self.append_bytes(frag)
        r = s.repair_tail()
        self.assertEqual((r.state, r.commit_seq, r.pending_count), ({"a": 1}, 1, 0))
        self.assertEqual(r.removed_bytes, len(frag))
        self.assertEqual(self.path.read_bytes(), prefix)

    def test_all_recognised_fragment_shapes_are_repairable(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix_end = self.line_end(2)
        fragments = [
            "{",
            '{"op": "set"',
            '{"op": "set", "key": "a", "value": 1.',
            '{"op": "set", "key": "a", "value": tru',
            '{"op": "set", "key": "a", "value": "x\\',
            '{"op": "set", "key": "a", "value": "\\u12',
            '{"op": "commit", "seq": 2',
            json.dumps({"op": "set", "key": "x", "value": 9, "seq": 2}),
        ]
        for frag in fragments:
            raw = frag.encode("utf-8")
            self.path.write_bytes(self.path.read_bytes()[:prefix_end])
            self.append_bytes(raw)
            r = s.repair_tail()
            self.assertEqual(r.state, {"a": 1}, msg=frag)
            self.assertEqual(r.commit_seq, 1, msg=frag)
            self.assertEqual(r.removed_bytes, len(raw), msg=frag)
            self.assertEqual(self.path.stat().st_size, prefix_end, msg=frag)
            self.assertEqual(s.audit().tail_bytes, 0, msg=frag)

    def test_corruption_after_prefix_raises_and_changes_nothing(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix = self.path.read_bytes()
        bad_tails = [
            b"   ",  # trailing whitespace, no terminator
            b"\n",  # blank line
            b"  \n",  # blank record
            b"\xff",  # illegal UTF-8 at end
            b'{"op": "set", "key": "b", "value": NaN, "seq": 2}\n',
            b'{"op": "set", "op": "set", "key": "b", "value": 2, "seq": 2}\n',
            b'{"op": "set", "key": "b", "value": 2, "seq": 5}',  # seq jump
            b'{"op": "set", "key": "b", "seq": 2}',  # missing field
            b'{"op": "bogus", "seq": 2}',  # unknown op
            b"not json\n",
            b'[1, 2]\n',  # not an object
            b'{"op": "set", "key": "b", "value": 2, "seq": 2}extra',
        ]
        for tail in bad_tails:
            self.path.write_bytes(prefix)
            self.append_bytes(tail)
            size = self.path.stat().st_size
            with self.assertRaises(WalCorruptionError, msg=tail):
                s.repair_tail()
            # file bytes untouched ...
            self.assertEqual(self.path.read_bytes(), prefix + tail, msg=tail)
            self.assertEqual(self.path.stat().st_size, size, msg=tail)
            # ... and in-memory committed state untouched
            self.assertEqual(s.state, {"a": 1}, msg=tail)
            self.assertEqual(s.commit_seq, 1, msg=tail)

    def test_corruption_in_fragment_only_log_raises_and_changes_nothing(self):
        bad = [
            b"\xff",
            b"not json",
            b'{"op": "set", "key": "a", "value": 1, "seq": 5}',  # seq jump
            b'{"op": "set", "key": "a", "seq": 1}',
            b'{"op": "set", "key": "a", "value": 1, "seq": "1"}',
        ]
        for data in bad:
            if self.path.exists():
                self.path.unlink()
            s = WalStore(self.path)  # constructed while the log is absent
            self.path.write_bytes(data)  # corruption appears before repair
            with self.assertRaises(WalCorruptionError, msg=data):
                s.repair_tail()
            self.assertEqual(self.path.read_bytes(), data, msg=data)
            self.assertEqual((s.state, s.commit_seq), ({}, 0), msg=data)

    def test_corruption_in_committed_region_raises_without_truncation(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix = self.path.read_bytes()
        bad = '{"op": "set", "key": "x"}\n'  # parseable but invalid, terminated
        self.append_bytes(bad)
        with self.assertRaises(WalCorruptionError):
            s.repair_tail()
        self.assertEqual(self.path.read_bytes(), prefix + bad.encode("utf-8"))
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))

    def test_persistence_failure_propagates_oserror_and_keeps_memory(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        self.append_bytes('{"op": "set", "key": "x"')

        def boom(fd):
            raise OSError("disk on fire")

        with mock.patch("app.os.fsync", side_effect=boom):
            with self.assertRaises(OSError):
                s.repair_tail()
        # in-memory commit state is not adopted from a failed repair
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)
        # the store stays consistent and the seq chain continues once the
        # transient fault clears (a reopen discards the fragment regardless
        # of whether the truncate reached the page cache)
        self.assertEqual(
            (WalStore(self.path).state, WalStore(self.path).commit_seq),
            ({"a": 1}, 1),
        )
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(s2.audit().tail_bytes, 0)

    def test_unwritable_path_propagates_oserror_and_keeps_memory(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        before = self.path.read_bytes()
        self.append_bytes('{"op": "set", "key": "x"')
        size = self.path.stat().st_size
        real_open = Path.open

        def deny_write(self_path, mode="r", *args, **kwargs):
            if any(c in mode for c in ("w", "a", "+")):
                raise OSError("denied")
            return real_open(self_path, mode, *args, **kwargs)

        with mock.patch.object(Path, "open", deny_write):
            with self.assertRaises(OSError):
                s.repair_tail()
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)
        # validation ran (reads allowed) but nothing was removed from the file
        self.assertEqual(self.path.stat().st_size, size)
        self.assertEqual(self.path.read_bytes()[: len(before)], before)

    def test_result_state_is_independent_deep_copy(self):
        s = WalStore(self.path)
        s.set("a", {"nested": [1]})
        s.commit()
        self.append_bytes('{"op": "set", "key": "x"')
        r = s.repair_tail()
        r["state"]["a"]["nested"].append(2)
        r["state"]["b"] = 9
        self.assertEqual(s.state, {"a": {"nested": [1]}})
        self.assertEqual(s.repair_tail()["state"], {"a": {"nested": [1]}})
        self.assertEqual(WalStore(self.path).state, {"a": {"nested": [1]}})

    def test_repair_result_matches_audit_boundary(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)
        fragment = '{"op": "set", "key": "x", "value":'
        self.append_bytes(fragment)
        audit_before = s.audit()
        r = s.repair_tail()
        self.assertEqual(r.removed_bytes, audit_before.tail_bytes)
        self.assertEqual((r.state, r.commit_seq, r.pending_count),
                         (audit_before.state, audit_before.commit_seq,
                          audit_before.pending_count))
        self.assertEqual(self.path.stat().st_size, audit_before.valid_bytes)
        audit_after = s.audit()
        self.assertEqual(
            (audit_after.valid_bytes, audit_after.tail_bytes),
            (audit_before.valid_bytes, 0),
        )

    def test_repair_then_append_continues_seq_chain(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix = self.path.read_bytes()
        self.append_bytes('{"op": "set", "key": "b", "value": 2, "seq": 2')
        r = s.repair_tail()
        self.assertGreater(r.removed_bytes, 0)
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        data = self.path.read_bytes()
        self.assertTrue(data.startswith(prefix))
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(s2.audit().tail_bytes, 0)

    def test_repair_preserves_legacy_committed_log(self):
        self.write_lines(
            {"op": "set", "key": "k", "value": [1, 2], "seq": 1},
            {"op": "commit", "seq": 1},
        )
        before = self.path.read_bytes()
        s = WalStore(self.path)
        r = s.repair_tail()
        self.assertEqual(r.removed_bytes, 0)
        self.assertEqual(r.state, {"k": [1, 2]})
        self.assertEqual(self.path.read_bytes(), before)


class PendingChangesTest(unittest.TestCase):
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

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def log_size(self):
        return self.path.stat().st_size if self.path.exists() else 0

    def test_nonexistent_and_empty_log(self):
        s = WalStore(self.path)  # path does not exist
        r = s.pending_changes()
        self.assertEqual(set(r), {"commit_seq", "pending_count", "changes"})
        self.assertEqual(
            (r.commit_seq, r.pending_count, r.changes), (0, 0, [])
        )
        # attribute and mapping access agree
        self.assertEqual(r["commit_seq"], 0)
        self.assertEqual(r["pending_count"], 0)
        self.assertEqual(r["changes"], [])
        self.assertFalse(self.path.exists())  # never created
        self.path.write_bytes(b"")
        self.assertEqual(dict(s.pending_changes()), dict(r))

    def test_no_pending_after_commit_returns_current_seq(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.commit()  # empty commit
        r = s.pending_changes()
        self.assertEqual(
            (r.commit_seq, r.pending_count, r.changes), (2, 0, [])
        )

    def test_pending_set_and_delete_in_log_order_with_exact_fields(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", {"nested": [1, 2]})
        s.commit()
        s.set("c", 3)
        s.delete("a")
        s.set("a", 99)
        r = s.pending_changes()
        self.assertEqual(r.commit_seq, 1)
        self.assertEqual(r.pending_count, 3)
        self.assertEqual(len(r.changes), 3)
        self.assertEqual(
            r.changes,
            [
                {"op": "set", "key": "c", "value": 3},
                {"op": "delete", "key": "a"},
                {"op": "set", "key": "a", "value": 99},
            ],
        )
        # seq is not exposed; only the specified field sets are
        self.assertEqual(set(r.changes[0]), {"op", "key", "value"})
        self.assertEqual(set(r.changes[1]), {"op", "key"})
        # the committed view is unaffected by the pending batch
        self.assertEqual(s.state, {"a": 1, "b": {"nested": [1, 2]}})
        self.assertEqual(s.get("a"), 1)
        self.assertIs(s.contains("c"), False)

    def test_records_before_any_commit_are_pending(self):
        self.write_lines(
            {"op": "set", "key": "k", "value": 1, "seq": 1},
            {"op": "delete", "key": "k", "seq": 1},
        )
        s = WalStore(self.path)
        r = s.pending_changes()
        self.assertEqual(
            (r.commit_seq, r.pending_count), (0, 2)
        )
        self.assertEqual(
            r.changes,
            [
                {"op": "set", "key": "k", "value": 1},
                {"op": "delete", "key": "k"},
            ],
        )

    def test_tail_fragments_are_discarded_not_returned(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)  # complete, terminated, pending
        prefix_end = self.log_size()
        fragments = [
            '{"op": "set", "key": "x"',  # unfinished JSON prefix
            "{",
            '{"op": "commit", "seq": 2',
            json.dumps({"op": "set", "key": "x", "value": 9, "seq": 2}),
            '{"op": "set", "key": "hé'.encode("utf-8")[:-1],  # truncated UTF-8
        ]
        for frag in fragments:
            raw = frag if isinstance(frag, bytes) else frag.encode("utf-8")
            self.path.write_bytes(self.path.read_bytes()[:prefix_end])
            self.append_bytes(raw)
            size = self.log_size()
            r = s.pending_changes()
            self.assertEqual(r.commit_seq, 1, msg=frag)
            self.assertEqual(r.pending_count, 1, msg=frag)
            self.assertEqual(
                r.changes, [{"op": "set", "key": "b", "value": 2}], msg=frag
            )
            # the fragment is left exactly in place
            self.assertEqual(self.log_size(), size, msg=frag)

    def test_fragment_only_log_returns_empty(self):
        for frag in ('{"op": "set", "key": "x"', "{"):
            self.path.write_bytes(frag.encode("utf-8"))
            s = WalStore(self.path)
            r = s.pending_changes()
            self.assertEqual(
                (r.commit_seq, r.pending_count, r.changes), (0, 0, []),
                msg=frag,
            )
        self.path.write_bytes('{"op": "set", "key": "hé'.encode("utf-8")[:-1])
        s = WalStore(self.path)
        r = s.pending_changes()
        self.assertEqual((r.commit_seq, r.pending_count, r.changes), (0, 0, []))

    def test_changes_are_independent_deep_copies(self):
        s = WalStore(self.path)
        s.set("a", {"nested": [1, {"k": 2}]})
        s.commit()
        s.set("b", {"v": [3]})
        s.delete("a")
        r1 = s.pending_changes()
        r1.changes[0]["value"]["v"].append(99)
        r1.changes[0]["value"]["v"][0] = 0
        r1.changes.append({"op": "set", "key": "z", "value": 1})
        r1.changes[1]["key"] = "mutated"
        # store and repeated calls unaffected
        r2 = s.pending_changes()
        self.assertEqual(
            r2.changes,
            [
                {"op": "set", "key": "b", "value": {"v": [3]}},
                {"op": "delete", "key": "a"},
            ],
        )
        self.assertEqual(s.state, {"a": {"nested": [1, {"k": 2}]}})
        # a reopened instance parses the same log independently
        reopened = WalStore(self.path).pending_changes()
        self.assertEqual(reopened.changes, r2.changes)

    def test_mutating_result_does_not_change_later_commit_or_rollback(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", {"n": [1]})
        s.delete("a")
        r = s.pending_changes()
        r.changes[0]["value"]["n"].append(99)
        r.changes.clear()
        self.assertEqual(s.pending_changes().pending_count, 2)
        # commit adopts the real pending records, not the mutated result
        self.assertEqual(s.commit(), 2)
        self.assertEqual(s.state, {"b": {"n": [1]}})
        self.assertEqual(
            (s.pending_changes().commit_seq, s.pending_changes().pending_count),
            (2, 0),
        )
        s2 = WalStore(self.path)
        self.assertEqual(s2.state, {"b": {"n": [1]}})
        self.assertEqual(s2.commit_seq, 2)

        # same for rollback on a fresh pending batch
        s.set("c", 3)
        s.delete("b")
        r = s.pending_changes()
        r.changes[0]["key"] = "mutated"
        self.assertEqual(s.rollback(), 2)
        self.assertEqual(s.state, {"b": {"n": [1]}})
        self.assertEqual(s.pending_changes().changes, [])
        self.assertEqual(WalStore(self.path).state, {"b": {"n": [1]}})

    def test_read_only_does_not_touch_log_seq_or_state(self):
        s = WalStore(self.path)
        s.set("a", {"v": [1]})
        s.commit()
        s.set("tail", 2)
        s.delete("a")
        self.append_bytes('{"op": "set", "key": "frag"')
        before = self.path.read_bytes()
        results = []
        for _ in range(4):
            r = s.pending_changes()
            results.append(dict(r))
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(self.log_size(), len(before))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        self.assertEqual(results[2], results[3])
        self.assertEqual(s.commit_seq, 1)
        self.assertEqual(s.state, {"a": {"v": [1]}})
        self.assertEqual(s.recover()["pending_count"], 2)
        # state stays the committed view
        self.assertEqual(s.get("a"), {"v": [1]})
        self.assertIs(s.contains("tail"), False)

    def test_deterministic_across_reopens(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", 2)
        s.commit()
        s.delete("a")
        s.set("c", [1, 2])
        expected = [
            {"op": "delete", "key": "a"},
            {"op": "set", "key": "c", "value": [1, 2]},
        ]
        for _ in range(3):
            store = WalStore(self.path)
            r = store.pending_changes()
            self.assertEqual((r.commit_seq, r.pending_count), (1, 2))
            self.assertEqual(r.changes, expected)

    def test_pending_changes_then_append_still_drops_fragment(self):
        # like audit, the query must not disturb the cached accepted-prefix
        # boundary a later append relies on
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        prefix = self.path.read_bytes()
        self.append_bytes('{"op": "set", "key": "frag"')
        s.pending_changes()
        s.pending_changes()
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        data = self.path.read_bytes()
        self.assertTrue(data.startswith(prefix))
        self.assertNotIn(b'"frag"', data)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))

    def test_corruption_raises_without_partial_result_or_side_effects(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)
        prefix = self.path.read_bytes()
        bad_tails = [
            b"   ",  # trailing whitespace, no terminator
            b"\n",  # blank line
            b"  \n",  # blank record
            b"\xff",  # illegal UTF-8 beyond a discardable tail
            b'{"op": "set", "key": "x", "value": NaN, "seq": 2}\n',
            b'{"op": "set", "op": "set", "key": "x", "value": 1, "seq": 2}\n',
            b'{"op": "set", "key": "x", "value": 1, "seq": 5}\n',  # seq jump
            b'{"op": "set", "key": "x", "seq": 2}\n',  # missing field
            b'{"op": "commit", "seq": 2, "extra": 1}\n',  # bad field set
            b'{"op": "bogus", "seq": 2}\n',  # unknown op
            b'{"op": "set", "key": "x", "value": 1, "seq": "2"}\n',  # bad seq
            b"not json\n",
            b"[1, 2]\n",  # not an object
            b'{"op": "set", "key": "x", "value": 1, "seq": 2}extra\n',
        ]
        for tail in bad_tails:
            self.path.write_bytes(prefix)
            self.append_bytes(tail)
            size = self.log_size()
            with self.assertRaises(WalCorruptionError, msg=tail):
                s.pending_changes()
            # no partial result leaked into memory ...
            self.assertEqual(s.state, {"a": 1}, msg=tail)
            self.assertEqual(s.commit_seq, 1, msg=tail)
            # ... and the file was neither truncated nor appended to
            self.assertEqual(self.path.read_bytes(), prefix + tail, msg=tail)
            self.assertEqual(self.log_size(), size, msg=tail)
            # the still-valid prefix remains independently queryable only
            # after the corrupt tail is gone
            self.path.write_bytes(prefix)
            self.assertEqual(s.pending_changes().pending_count, 1)

    def test_corruption_in_fragment_only_log_raises(self):
        bad = [
            b"\xff",
            b"not json",
            b'{"op": "set", "key": "a", "value": 1, "seq": 5}',
            b'{"op": "set", "key": "a", "seq": 1}',
            b'{"op": "set", "key": "a", "value": 1, "seq": "1"}',
            b'{"op": "set", "key": "a", "value": 1, "seq": 1}extra',
        ]
        for data in bad:
            self.path.write_bytes(data)
            with self.assertRaises(WalCorruptionError, msg=data):
                WalStore(self.path).pending_changes()

    def test_all_json_value_shapes_round_trip(self):
        s = WalStore(self.path)
        s.set("committed", 0)
        s.commit()
        values = {
            "i": 1, "f": 1.5, "s": "hi", "b": True, "n": None,
            "arr": [1, "two", False, None, {"x": []}],
            "obj": {"k": [1, 2]},
        }
        for k, v in values.items():
            s.set(k, v)
        changes = s.pending_changes().changes
        self.assertEqual(
            changes,
            [{"op": "set", "key": k, "value": v} for k, v in values.items()],
        )

    def test_legacy_log_serves_pending_changes_without_migration(self):
        self.write_lines(
            {"op": "set", "key": "legacy", "value": [1, 2], "seq": 1},
            {"op": "commit", "seq": 1},
            {"op": "set", "key": "tail", "value": {"x": 1}, "seq": 2},
            {"op": "delete", "key": "legacy", "seq": 2},
        )
        s = WalStore(self.path)
        r = s.pending_changes()
        self.assertEqual((r.commit_seq, r.pending_count), (1, 2))
        self.assertEqual(
            r.changes,
            [
                {"op": "set", "key": "tail", "value": {"x": 1}},
                {"op": "delete", "key": "legacy"},
            ],
        )
        size = self.log_size()
        s.pending_changes()
        self.assertEqual(self.log_size(), size)


class HistoryTest(unittest.TestCase):
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

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def log_size(self):
        return self.path.stat().st_size if self.path.exists() else 0

    def build(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", {"n": [1]})
        s.commit()  # seq 1
        s.delete("a")
        s.set("c", 3)
        s.commit()  # seq 2
        s.commit()  # seq 3: empty commit
        s.set("tail", 9)  # uncommitted
        return s

    def test_default_exports_all_commits(self):
        s = self.build()
        h = s.history()
        self.assertEqual(
            h,
            [
                {
                    "commit_seq": 1,
                    "changes": [
                        {"op": "set", "key": "a", "value": 1},
                        {"op": "set", "key": "b", "value": {"n": [1]}},
                    ],
                },
                {
                    "commit_seq": 2,
                    "changes": [
                        {"op": "delete", "key": "a"},
                        {"op": "set", "key": "c", "value": 3},
                    ],
                },
                {"commit_seq": 3, "changes": []},
            ],
        )

    def test_empty_and_nonexistent_log_return_empty(self):
        s = WalStore(self.path)  # path does not exist
        self.assertEqual(s.history(), [])
        self.assertEqual(s.history(0), [])
        self.assertEqual(s.history(0, 0), [])
        self.assertFalse(self.path.exists())
        self.path.write_bytes(b"")
        self.assertEqual(s.history(), [])

    def test_range_filters_commits(self):
        s = self.build()
        self.assertEqual([e["commit_seq"] for e in s.history(0, 2)], [1, 2])
        self.assertEqual([e["commit_seq"] for e in s.history(1)], [2, 3])
        self.assertEqual([e["commit_seq"] for e in s.history(1, 2)], [2])
        self.assertEqual([e["commit_seq"] for e in s.history(2, 2)], [])
        self.assertEqual([e["commit_seq"] for e in s.history(3)], [])
        self.assertEqual([e["commit_seq"] for e in s.history(0, 3)], [1, 2, 3])
        # until_seq=None is the latest committed seq
        self.assertEqual(s.history(0, None), s.history())
        self.assertEqual(s.history(1, None), s.history(1))

    def test_uncommitted_tail_never_exported(self):
        s = self.build()
        for entry in s.history():
            for change in entry["changes"]:
                self.assertNotEqual(change.get("key"), "tail")
        # a log with only uncommitted records has no history
        self.write_lines({"op": "set", "key": "p", "value": 1, "seq": 1})
        self.assertEqual(WalStore(self.path).history(), [])

    def test_invalid_arguments_raise_valueerror(self):
        s = self.build()
        for bad in (True, False, -1, 1.0, "1", b"1", [1], {"s": 1}):
            with self.assertRaises(ValueError, msg=bad):
                s.history(bad)
            with self.assertRaises(ValueError, msg=bad):
                s.history(0, bad)
        with self.assertRaises(ValueError):
            s.history(2, 1)  # inverted range
        with self.assertRaises(ValueError):
            s.history(3, 0)
        # beyond the latest committed seq (latest is 3)
        with self.assertRaises(ValueError):
            s.history(4)
        with self.assertRaises(ValueError):
            s.history(0, 4)
        with self.assertRaises(ValueError):
            s.history(10**9)
        # on an empty log any positive bound exceeds the latest seq
        empty = WalStore(Path(self.dir.name) / "other.wal")
        with self.assertRaises(ValueError):
            empty.history(1)
        with self.assertRaises(ValueError):
            empty.history(0, 1)

    def test_argument_errors_precede_log_validation(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "x"}\n')  # corrupt record
        with self.assertRaises(ValueError):
            s.history(-1)
        with self.assertRaises(ValueError):
            s.history(3, 1)
        with self.assertRaises(WalCorruptionError):
            s.history()

    def test_result_is_independent_deep_copy(self):
        s = self.build()
        h = s.history()
        h[0]["changes"][1]["value"]["n"].append(99)
        h[0]["changes"].append({"op": "set", "key": "x", "value": 1})
        h.append({"commit_seq": 99, "changes": []})
        again = s.history()
        self.assertEqual(len(again), 3)
        self.assertEqual(again[0]["changes"][1]["value"], {"n": [1]})
        self.assertEqual(len(again[0]["changes"]), 2)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        self.assertEqual(WalStore(self.path).history(), again)

    def test_history_is_read_only(self):
        s = self.build()
        size = self.log_size()
        for args in ((), (0,), (1, 2), (0, None)):
            s.history(*args)
        self.assertEqual(self.log_size(), size)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        self.assertEqual(s.commit_seq, 3)
        self.assertEqual(s.recover()["pending_count"], 1)

    def test_deterministic_across_calls_and_reopens(self):
        s = self.build()
        first = s.history()
        for _ in range(3):
            self.assertEqual(s.history(), first)
            self.assertEqual(WalStore(self.path).history(), first)

    def test_corruption_anywhere_raises_no_partial_result(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "x"}\n')  # invalid, terminated
        for args in ((), (0,), (1, 2), (0, 3)):
            with self.assertRaises(WalCorruptionError, msg=args):
                s.history(*args)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        self.assertEqual(s.commit_seq, 3)

    def test_tail_fragment_does_not_affect_history(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "frag"')  # interrupted write
        size = self.log_size()
        h = s.history()
        self.assertEqual([e["commit_seq"] for e in h], [1, 2, 3])
        self.assertEqual(self.log_size(), size)  # fragment left in place

    def test_legacy_log_serves_history(self):
        self.write_lines(
            {"op": "set", "key": "k", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1},
            {"op": "set", "key": "k", "value": 2, "seq": 2},
            {"op": "delete", "key": "k", "seq": 2},
            {"op": "commit", "seq": 2},
        )
        s = WalStore(self.path)
        self.assertEqual(
            s.history(),
            [
                {
                    "commit_seq": 1,
                    "changes": [{"op": "set", "key": "k", "value": 1}],
                },
                {
                    "commit_seq": 2,
                    "changes": [
                        {"op": "set", "key": "k", "value": 2},
                        {"op": "delete", "key": "k"},
                    ],
                },
            ],
        )


class DiffTest(unittest.TestCase):
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

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def log_size(self):
        return self.path.stat().st_size if self.path.exists() else 0

    def build(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", {"n": [1]})
        s.commit()  # seq 1: {a: 1, b: {n: [1]}}
        s.delete("a")
        s.set("c", 3)
        s.commit()  # seq 2: {b: {n: [1]}, c: 3}
        s.commit()  # seq 3: empty commit, same state as seq 2
        s.set("tail", 9)  # uncommitted
        return s

    def test_basic_diff_between_commits(self):
        s = self.build()
        self.assertEqual(
            s.diff(1, 2),
            {
                "from_seq": 1,
                "to_seq": 2,
                "changes": [
                    {"op": "delete", "key": "a"},
                    {"op": "set", "key": "c", "value": 3},
                ],
            },
        )
        # the reverse direction of time is not expressible (from <= to),
        # but any earlier baseline works, including the empty object
        self.assertEqual(
            s.diff(0, 1),
            {
                "from_seq": 0,
                "to_seq": 1,
                "changes": [
                    {"op": "set", "key": "a", "value": 1},
                    {"op": "set", "key": "b", "value": {"n": [1]}},
                ],
            },
        )
        self.assertEqual(
            s.diff(0, 2)["changes"],
            [
                {"op": "set", "key": "b", "value": {"n": [1]}},
                {"op": "set", "key": "c", "value": 3},
            ],
        )

    def test_identical_states_yield_empty_changes(self):
        s = self.build()
        for args in ((0, 0), (1, 1), (2, 2), (2, 3), (3, 3)):
            result = s.diff(*args)
            self.assertEqual(result["changes"], [], msg=args)
            self.assertEqual(result["from_seq"], args[0])
            self.assertEqual(result["to_seq"], args[1])

    def test_empty_and_nonexistent_log(self):
        s = WalStore(self.path)  # path does not exist
        self.assertEqual(
            s.diff(0, 0), {"from_seq": 0, "to_seq": 0, "changes": []}
        )
        self.assertFalse(self.path.exists())  # no file is created
        self.path.write_bytes(b"")
        self.assertEqual(s.diff(0, 0)["changes"], [])
        # any positive seq exceeds the latest committed seq of 0
        with self.assertRaises(ValueError):
            s.diff(0, 1)
        with self.assertRaises(ValueError):
            s.diff(1, 1)

    def test_invalid_arguments_raise_valueerror(self):
        s = self.build()
        for bad in (True, False, -1, 1.0, "1", b"1", [1], {"s": 1}, None):
            with self.assertRaises(ValueError, msg=bad):
                s.diff(bad, 0)
            with self.assertRaises(ValueError, msg=bad):
                s.diff(0, bad)
        with self.assertRaises(ValueError):
            s.diff(2, 1)  # inverted range
        with self.assertRaises(ValueError):
            s.diff(1, 0)

    def test_argument_errors_precede_log_validation(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "x"}\n')  # corrupt record
        with self.assertRaises(ValueError):
            s.diff(-1, 0)
        with self.assertRaises(ValueError):
            s.diff(True, 1)
        with self.assertRaises(ValueError):
            s.diff(3, 1)
        with self.assertRaises(WalCorruptionError):
            s.diff(0, 1)

    def test_out_of_range_seqs_raise_valueerror(self):
        s = self.build()  # latest committed seq is 3
        with self.assertRaises(ValueError):
            s.diff(4, 4)
        with self.assertRaises(ValueError):
            s.diff(0, 4)
        with self.assertRaises(ValueError):
            s.diff(0, 10**9)

    def test_closed_instance_raises_closed_error_first(self):
        s = self.build()
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.diff(0, 1)
        with self.assertRaises(app.WalClosedError):
            s.diff(-1, 0)  # closed beats the argument form check
        with self.assertRaises(app.WalClosedError):
            s.diff(0, 0)

    def test_uncommitted_records_never_enter_diff(self):
        s = self.build()
        for args in ((0, 3), (1, 2), (2, 3)):
            for change in s.diff(*args)["changes"]:
                self.assertNotEqual(change.get("key"), "tail")
        # a log with only uncommitted records diffs like an empty log
        self.write_lines({"op": "set", "key": "p", "value": 1, "seq": 1})
        s2 = WalStore(self.path)
        self.assertEqual(s2.diff(0, 0)["changes"], [])
        with self.assertRaises(ValueError):
            s2.diff(0, 1)

    def test_corruption_anywhere_raises_no_partial_result(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "x"}\n')  # invalid, terminated
        for args in ((0, 0), (0, 1), (1, 2), (0, 3)):
            with self.assertRaises(WalCorruptionError, msg=args):
                s.diff(*args)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        self.assertEqual(s.commit_seq, 3)

    def test_tail_fragment_does_not_affect_diff(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "frag"')  # interrupted write
        size = self.log_size()
        self.assertEqual(
            s.diff(1, 2)["changes"],
            [{"op": "delete", "key": "a"}, {"op": "set", "key": "c", "value": 3}],
        )
        self.assertEqual(self.log_size(), size)  # fragment left in place

    def test_keys_sorted_by_code_point_and_one_entry_each(self):
        s = WalStore(self.path)
        s.set("b", 1)
        s.set("A", 1)
        s.set("é", 1)
        s.set("a", 1)
        s.commit()  # seq 1
        s.set("b", 2)
        s.delete("A")
        s.set("é", 1)  # unchanged value, no diff entry
        s.set("z", 5)
        s.commit()  # seq 2
        changes = s.diff(1, 2)["changes"]
        self.assertEqual(
            changes,
            [
                {"op": "delete", "key": "A"},
                {"op": "set", "key": "b", "value": 2},
                {"op": "set", "key": "z", "value": 5},
            ],
        )
        self.assertEqual([c["key"] for c in changes], sorted(c["key"] for c in changes))

    def test_json_type_distinction_forces_set(self):
        s = WalStore(self.path)
        s.set("n", 1)
        s.set("deep", {"x": [1, 2]})
        s.commit()  # seq 1
        s.set("n", True)  # 1 -> true: different JSON types
        s.set("deep", {"x": [1, True]})  # boolean nested in a list
        s.commit()  # seq 2
        self.assertEqual(
            s.diff(1, 2)["changes"],
            [
                {"op": "set", "key": "deep", "value": {"x": [1, True]}},
                {"op": "set", "key": "n", "value": True},
            ],
        )
        # numeric family: 1 and 1.0 are the same JSON number, no entry
        s2 = WalStore(Path(self.dir.name) / "other.wal")
        s2.set("n", 1)
        s2.commit()
        s2.set("n", 1.0)
        s2.commit()
        self.assertEqual(s2.diff(1, 2)["changes"], [])

    def test_result_is_independent_deep_copy(self):
        s = self.build()
        result = s.diff(0, 1)
        result["changes"][1]["value"]["n"].append(99)
        result["changes"].append({"op": "delete", "key": "a"})
        result["from_seq"] = 99
        again = s.diff(0, 1)
        self.assertEqual(again["from_seq"], 0)
        self.assertEqual(len(again["changes"]), 2)
        self.assertEqual(again["changes"][1]["value"], {"n": [1]})
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        self.assertEqual(WalStore(self.path).diff(0, 1), again)

    def test_diff_is_read_only(self):
        s = self.build()
        size = self.log_size()
        for args in ((0, 0), (0, 1), (1, 2), (2, 3), (0, 3)):
            s.diff(*args)
        self.assertEqual(self.log_size(), size)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        self.assertEqual(s.commit_seq, 3)
        self.assertEqual(s.recover()["pending_count"], 1)
        # the pending record is still appended after, not re-sequenced
        s.commit()
        self.assertEqual(s.commit_seq, 4)
        self.assertEqual(s.state["tail"], 9)

    def test_deterministic_across_calls_and_reopens(self):
        s = self.build()
        first = s.diff(0, 2)
        for _ in range(3):
            self.assertEqual(s.diff(0, 2), first)
            self.assertEqual(WalStore(self.path).diff(0, 2), first)

    def test_readonly_and_exclusive_instances_can_diff(self):
        s = self.build()
        expected = s.diff(1, 2)
        ro = WalStore(self.path, readonly=True)
        self.assertEqual(ro.diff(1, 2), expected)
        ex = WalStore(Path(self.dir.name) / "ex.wal", exclusive=True)
        ex.set("k", 1)
        ex.commit()
        ex.set("k", 2)
        ex.commit()
        self.assertEqual(
            ex.diff(1, 2)["changes"], [{"op": "set", "key": "k", "value": 2}]
        )
        # the exclusive lease and writability survive the diff
        ex.set("k", 3)
        ex.commit()
        self.assertEqual(ex.commit_seq, 3)
        ex.close()
        ro.close()

    def test_legacy_log_serves_diff(self):
        self.write_lines(
            {"op": "set", "key": "k", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1},
            {"op": "set", "key": "k", "value": 2, "seq": 2},
            {"op": "delete", "key": "k", "seq": 2},
            {"op": "commit", "seq": 2},
        )
        s = WalStore(self.path)
        self.assertEqual(
            s.diff(1, 2)["changes"], [{"op": "delete", "key": "k"}]
        )
        self.assertEqual(
            s.diff(0, 1)["changes"], [{"op": "set", "key": "k", "value": 1}]
        )


class ExclusiveLeaseTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "store.wal"

    def tearDown(self):
        self.dir.cleanup()

    def test_exclusive_param_must_be_bool(self):
        for bad in (0, 1, "yes", None, object()):
            with self.assertRaises(ValueError):
                WalStore(self.path, exclusive=bad)
        self.assertFalse(self.path.exists())

    def test_exclusive_recovers_and_writes_normally(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)
        s.set("b", 2)
        self.assertEqual(s.rollback(), 1)
        self.assertEqual(s.pending_changes()["pending_count"], 0)
        s.close()
        # Log format is untouched: a plain open replays the same records.
        s2 = WalStore(self.path)
        self.assertEqual(s2.state, {"a": 1})
        self.assertEqual(s2.commit_seq, 1)

    def test_busy_raises_before_reading_or_touching_log(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        size = self.path.stat().st_size
        # Conflict is repeatable and leaves the log and the holder untouched.
        for _ in range(2):
            with self.assertRaises(app.WalBusyError):
                WalStore(self.path, exclusive=True)
        self.assertEqual(self.path.stat().st_size, size)
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)
        s.set("b", 2)
        s.commit()
        self.assertEqual(s.commit_seq, 2)
        s.close()

    def test_busy_does_not_create_log(self):
        s = WalStore(self.path, exclusive=True)
        self.assertFalse(self.path.exists())
        with self.assertRaises(app.WalBusyError):
            WalStore(self.path, exclusive=True)
        self.assertFalse(self.path.exists())
        s.close()

    def test_normalized_paths_share_lease(self):
        s = WalStore(self.path, exclusive=True)
        dotted = self.path.parent / "sub" / ".." / "store.wal"
        with self.assertRaises(app.WalBusyError):
            WalStore(dotted, exclusive=True)
        s.close()

    def test_non_exclusive_open_ignores_lease(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        s2 = WalStore(self.path)
        self.assertEqual(s2.state, {"a": 1})
        s2.close()
        s.close()

    def test_close_releases_lease_and_is_idempotent(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        s.close()
        s.close()
        s2 = WalStore(self.path, exclusive=True)
        self.assertEqual(s2.state, {"a": 1})
        self.assertEqual(s2.commit_seq, 1)
        s2.close()

    def test_context_manager_releases_lease(self):
        with WalStore(self.path, exclusive=True) as s:
            s.set("a", 1)
            s.commit()
        with WalStore(self.path, exclusive=True) as s2:
            self.assertEqual(s2.state, {"a": 1})

    def test_public_methods_raise_walclosederror_after_close(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        s.close()
        calls = [
            lambda: s.set("b", 2),
            lambda: s.delete("a"),
            lambda: s.commit(),
            lambda: s.rollback(),
            lambda: s.restore(0),
            lambda: s.recover(),
            lambda: s.get("a"),
            lambda: s.contains("a"),
            lambda: s.snapshot(),
            lambda: s.history(),
            lambda: s.pending_changes(),
            lambda: s.audit(),
            lambda: s.scan(),
            lambda: s.repair_tail(),
        ]
        for call in calls:
            with self.assertRaises(app.WalClosedError):
                call()
        # close itself stays callable and the log is untouched.
        s.close()
        s2 = WalStore(self.path)
        self.assertEqual(s2.state, {"a": 1})
        self.assertEqual(s2.commit_seq, 1)

    def test_closed_non_exclusive_store_also_raises(self):
        s = WalStore(self.path)
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.get("a")
        s.close()

    def test_reacquire_after_close_preserves_all_results(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)
        s.commit()
        s.set("c", 3)  # pending
        before = (
            s.state,
            s.commit_seq,
            s.snapshot(),
            s.history(),
            s.audit(),
            s.pending_changes(),
        )
        s.close()
        s2 = WalStore(self.path, exclusive=True)
        after = (
            s2.state,
            s2.commit_seq,
            s2.snapshot(),
            s2.history(),
            s2.audit(),
            s2.pending_changes(),
        )
        self.assertEqual(before, after)
        s2.close()

    def test_lease_survives_holder_crash(self):
        import subprocess, sys

        code = (
            "import sys; sys.path.insert(0, %r);"
            "from app import WalStore;"
            "s = WalStore(%r, exclusive=True);"
            "s.set('a', 1); s.commit();"
            "print('ready', flush=True);"
            "import time; time.sleep(60)"
            % (str(Path(app.__file__).parent), str(self.path))
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.assertEqual(proc.stdout.readline().strip(), b"ready")
            # While the child holds the lease, an exclusive open fails.
            with self.assertRaises(app.WalBusyError):
                WalStore(self.path, exclusive=True)
        finally:
            proc.kill()
            proc.wait()
        # Abnormal termination leaves no stale lease: recovery proceeds.
        s = WalStore(self.path, exclusive=True)
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)
        s.close()

    def test_write_failure_semantics_unchanged_under_lease(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()

        def boom(fd):
            raise OSError("fsync failed")

        with mock.patch.object(app.os, "fsync", side_effect=boom):
            with self.assertRaises(OSError):
                s.commit()
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)
        s.close()


class AliasLeaseTest(unittest.TestCase):
    """Single-writer lease must follow the real log object, not the path.

    Symlink chains, hard links, and other path aliases that name one actual
    log file all compete for the same exclusive lease; aliases of a log that
    does not exist yet are serialized so concurrent first opens still elect
    exactly one writer.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "store.wal"

    def tearDown(self):
        self.dir.cleanup()

    def build_committed(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        return s

    # ----- existing log: every alias of one inode shares the lease -----

    def test_symlink_alias_is_busy_in_both_directions(self):
        s = self.build_committed()
        link = self.path.parent / "by-link.wal"
        os.symlink(self.path, link)
        # holder reached via the real path, contender via the symlink
        with self.assertRaises(app.WalBusyError):
            WalStore(link, exclusive=True)
        s.close()
        # holder reached via the symlink, contender via the real path
        s2 = WalStore(link, exclusive=True)
        with self.assertRaises(app.WalBusyError):
            WalStore(self.path, exclusive=True)
        s2.close()

    def test_hardlink_alias_is_busy(self):
        s = self.build_committed()
        hard = self.path.parent / "by-hardlink.wal"
        os.link(self.path, hard)
        self.assertEqual(os.stat(hard).st_ino, os.stat(self.path).st_ino)
        with self.assertRaises(app.WalBusyError):
            WalStore(hard, exclusive=True)
        s.close()
        # after release the hard link leases the same object and recovers it
        s2 = WalStore(hard, exclusive=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        s2.close()

    def test_alias_through_linked_directory_is_busy(self):
        real_dir = self.path.parent / "realdata"
        real_dir.mkdir()
        inside = real_dir / "store.wal"
        s = WalStore(inside, exclusive=True)
        s.set("a", 1)
        s.commit()
        link_dir = self.path.parent / "linkdata"
        os.symlink(real_dir, link_dir)
        alias = link_dir / "store.wal"
        with self.assertRaises(app.WalBusyError):
            WalStore(alias, exclusive=True)
        s.close()
        s2 = WalStore(alias, exclusive=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        s2.close()

    def test_normalized_dotted_alias_of_existing_log_is_busy(self):
        s = self.build_committed()
        dotted = self.path.parent / "sub" / ".." / "store.wal"
        with self.assertRaises(app.WalBusyError):
            WalStore(dotted, exclusive=True)
        s.close()

    def test_distinct_logs_do_not_share_a_lease(self):
        other = self.path.parent / "other.wal"
        s1 = WalStore(self.path, exclusive=True)
        s2 = WalStore(other, exclusive=True)
        s1.set("a", 1)
        s2.set("b", 2)
        self.assertEqual(s1.commit(), 1)
        self.assertEqual(s2.commit(), 1)
        s1.close()
        s2.close()

    def test_symlinks_to_distinct_targets_stay_independent(self):
        other = self.path.parent / "other.wal"
        link_a = self.path.parent / "link-a.wal"
        link_b = self.path.parent / "link-b.wal"
        os.symlink(self.path, link_a)
        os.symlink(other, link_b)
        s1 = WalStore(self.path, exclusive=True)
        s2 = WalStore(link_b, exclusive=True)  # different real object
        s1.set("a", 1)
        s2.set("b", 2)
        self.assertEqual(s1.commit(), 1)
        self.assertEqual(s2.commit(), 1)
        # each alias conflicts only with its own object
        with self.assertRaises(app.WalBusyError):
            WalStore(link_a, exclusive=True)
        s1.close()
        s2.close()

    def test_busy_via_alias_never_reads_or_touches_the_log(self):
        s = self.build_committed()
        link = self.path.parent / "by-link.wal"
        os.symlink(self.path, link)
        # append corruption the loser would reject *if it read the log*;
        # the lease conflict must win before recovery runs
        with self.path.open("a", encoding="utf-8") as f:
            f.write("not-a-record\n")
        size = self.path.stat().st_size
        for _ in range(2):
            with self.assertRaises(app.WalBusyError):
                WalStore(link, exclusive=True)
        self.assertEqual(self.path.stat().st_size, size)
        # holder keeps its own lease, state, seq, and file bytes. The
        # out-of-band corruption is a terminated invalid record: a write now
        # revalidates the log from the head and refuses with
        # WalCorruptionError rather than overwriting the unknown bytes; the
        # lease (and read-only/busy) priority is unchanged -- the loser above
        # still got WalBusyError without ever reading the log.
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)
        with self.assertRaises(app.WalCorruptionError):
            s.set("b", 2)
        with self.assertRaises(app.WalCorruptionError):
            s.delete("a")
        with self.assertRaises(app.WalCorruptionError):
            s.commit()
        # nothing was appended or truncated, and no state/seq was adopted
        self.assertEqual(self.path.stat().st_size, size)
        self.assertTrue(self.path.read_bytes().endswith(b"not-a-record\n"))
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)
        s.close()

    def test_alias_close_and_context_manager_release_identity_lease(self):
        self.build_committed().close()
        link = self.path.parent / "by-link.wal"
        os.symlink(self.path, link)
        with WalStore(link, exclusive=True) as s:
            self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
            with self.assertRaises(app.WalBusyError):
                WalStore(self.path, exclusive=True)
        # released: the real path leases the same object again
        s2 = WalStore(self.path, exclusive=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        s2.close()

    # ----- missing log: aliases converge on one real target -----

    def test_dangling_symlink_to_missing_log_elects_one_writer(self):
        link = self.path.parent / "future-link.wal"
        os.symlink(self.path, link)  # target does not exist yet
        winner = WalStore(self.path, exclusive=True)
        # the dangling alias names the same not-yet-created target
        with self.assertRaises(app.WalBusyError):
            WalStore(link, exclusive=True)
        # winner goes on to create the log; loser still cannot enter
        winner.set("x", 9)
        self.assertEqual(winner.commit(), 1)
        with self.assertRaises(app.WalBusyError):
            WalStore(link, exclusive=True)
        # the winner's create resolved the formerly dangling symlink target
        self.assertTrue(self.path.exists())
        winner.close()
        # once released the alias reaches the object the winner created
        s2 = WalStore(link, exclusive=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"x": 9}, 1))
        s2.close()

    def test_reverse_dangling_symlink_winner(self):
        link = self.path.parent / "future-link.wal"
        os.symlink(self.path, link)
        # winner enters through the symlink spelling; contender uses the
        # concrete path -- still exactly one lease for the future object
        winner = WalStore(link, exclusive=True)
        with self.assertRaises(app.WalBusyError):
            WalStore(self.path, exclusive=True)
        winner.set("k", 7)
        self.assertEqual(winner.commit(), 1)
        winner.close()
        s2 = WalStore(self.path, exclusive=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"k": 7}, 1))
        s2.close()

    def test_concurrent_first_open_threads_leave_single_writer(self):
        import threading

        link = self.path.parent / "future-link.wal"
        os.symlink(self.path, link)
        results = {}

        def attempt(name, target):
            try:
                results[name] = ("open", WalStore(target, exclusive=True))
            except app.WalBusyError:
                results[name] = ("busy", None)
            except Exception as exc:  # pragma: no cover - surfaced as failure
                results[name] = ("other", exc)

        t1 = threading.Thread(target=attempt, args=("one", self.path))
        t2 = threading.Thread(target=attempt, args=("two", link))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        statuses = sorted(name for name, (kind, _) in results.items())
        kinds = [results[n][0] for n in statuses]
        self.assertEqual(sorted(kinds), ["busy", "open"])
        winner_name = [n for n in statuses if results[n][0] == "open"][0]
        winner = results[winner_name][1]
        winner.set("v", 1)
        self.assertEqual(winner.commit(), 1)
        # while the winner is live, opening the *other* alias still fails
        other_target = link if winner_name == "one" else self.path
        with self.assertRaises(app.WalBusyError):
            WalStore(other_target, exclusive=True)
        winner.close()

    def test_winner_that_never_creates_releases_gate_on_close(self):
        link = self.path.parent / "future-link.wal"
        os.symlink(self.path, link)
        first = WalStore(self.path, exclusive=True)
        with self.assertRaises(app.WalBusyError):
            WalStore(link, exclusive=True)
        first.close()  # never wrote: no log was created, gate must be freed
        self.assertFalse(self.path.exists())
        second = WalStore(link, exclusive=True)
        second.set("y", 3)
        self.assertEqual(second.commit(), 1)
        second.close()
        self.assertEqual(WalStore(self.path).state, {"y": 3})

    def test_dotted_alias_through_existing_directory_converges(self):
        # a lexical "sub/.." spelling only names the same file once sub
        # physically exists; with it present both spellings are one target
        real = self.path.parent / "data"
        (real / "sub").mkdir(parents=True)
        a = real / "store.wal"
        b = real / "sub" / ".." / "store.wal"
        first = WalStore(a, exclusive=True)
        with self.assertRaises(app.WalBusyError):
            WalStore(b, exclusive=True)
        first.set("d", 1)
        self.assertEqual(first.commit(), 1)
        first.close()
        second = WalStore(b, exclusive=True)
        self.assertEqual((second.state, second.commit_seq), ({"d": 1}, 1))
        second.close()

    # ----- identity resolution / IO failure error semantics -----

    def test_unresolvable_symlink_loop_raises_oserror_not_busy(self):
        loop = self.path.parent / "loop.wal"
        os.symlink(loop, loop)
        with self.assertRaises(OSError):
            WalStore(loop, exclusive=True)
        # and it must not be misreported as a lease conflict
        with self.assertRaises(OSError) as cm:
            WalStore(loop, exclusive=True)
        self.assertNotIsInstance(cm.exception, app.WalBusyError)

    def test_identity_swap_after_unlink_aborts_write_with_oserror(self):
        s = self.build_committed()
        s.close()
        holder = WalStore(self.path, exclusive=True)
        leased = (holder._leased_dev, holder._leased_ino)
        # replace the path with a brand-new object behind the holder's back
        os.unlink(self.path)
        with self.path.open("w", encoding="utf-8") as f:
            f.write("")
        self.assertNotEqual(
            (os.stat(self.path).st_dev, os.stat(self.path).st_ino), leased
        )
        with self.assertRaises(OSError):
            holder.set("rogue", 1)
        holder.close()

    def test_non_exclusive_and_readonly_aliases_unaffected(self):
        s = self.build_committed()
        link = self.path.parent / "by-link.wal"
        os.symlink(self.path, link)
        plain = WalStore(link)  # no lease: reads alongside the writer
        reader = WalStore(link, readonly=True)
        self.assertEqual((plain.state, plain.commit_seq), ({"a": 1}, 1))
        self.assertEqual((reader.state, reader.commit_seq), ({"a": 1}, 1))
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        self.assertEqual(plain.get("b"), 2)
        self.assertEqual(reader.get("b"), 2)
        plain.close()
        reader.close()
        s.close()

    def test_alias_lease_survives_holder_crash_cross_process(self):
        import subprocess
        import sys

        self.build_committed().close()
        link = self.path.parent / "by-link.wal"
        os.symlink(self.path, link)
        code = (
            "import sys; sys.path.insert(0, %r);"
            "from app import WalStore;"
            "s = WalStore(%r, exclusive=True);"
            "s.set('c', 3); s.commit();"
            "print('ready', flush=True);"
            "import time; time.sleep(60)"
            % (str(Path(app.__file__).parent), str(link))
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.assertEqual(proc.stdout.readline().strip(), b"ready")
            # the live holder reached the object through the symlink; a
            # contender using the concrete path must still be refused
            with self.assertRaises(app.WalBusyError):
                WalStore(self.path, exclusive=True)
        finally:
            proc.kill()
            proc.wait()
        # abnormal termination releases the identity lease kernel-side
        s2 = WalStore(self.path, exclusive=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "c": 3}, 2))
        s2.close()


class ReadOnlyModeTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "store.wal"

    def tearDown(self):
        self.dir.cleanup()

    def lock_path(self, path=None):
        import os
        return Path(os.path.abspath(str(path or self.path)) + ".lock")

    def build_committed(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        s.close()

    def build_with_pending(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)
        return s

    def test_readonly_param_must_be_bool(self):
        for bad in (0, 1, "yes", None, object()):
            with self.assertRaises(ValueError):
                WalStore(self.path, readonly=bad)
        # Validation fails before any path is read or created.
        self.assertFalse(self.path.exists())
        self.assertFalse(self.lock_path().exists())

    def test_readonly_and_exclusive_together_raise_valueerror(self):
        with self.assertRaises(ValueError):
            WalStore(self.path, exclusive=True, readonly=True)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.lock_path().exists())

    def test_validation_failure_does_not_touch_existing_log(self):
        w = self.build_with_pending()
        raw = self.path.read_bytes()
        size = self.path.stat().st_size
        for kwargs in (
            {"readonly": "x"},
            {"exclusive": 1, "readonly": True},
            {"exclusive": True, "readonly": True},
        ):
            with self.assertRaises(ValueError):
                WalStore(self.path, **kwargs)
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertEqual(self.path.stat().st_size, size)
        self.assertEqual(w.state, {"a": 1})
        self.assertEqual(w.commit_seq, 1)
        w.close()

    def test_missing_log_recovers_empty_and_creates_nothing(self):
        r = WalStore(self.path, readonly=True)
        self.assertEqual(r.state, {})
        self.assertEqual(r.commit_seq, 0)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.lock_path().exists())
        self.assertEqual(r.recover()["pending_count"], 0)
        self.assertEqual(r.snapshot(0)["state"], {})
        self.assertEqual(r.history(), [])
        self.assertEqual(r.pending_changes()["changes"], [])
        audit = r.audit()
        self.assertEqual(
            (audit["valid_bytes"], audit["committed_bytes"], audit["tail_bytes"]),
            (0, 0, 0),
        )
        r.close()
        self.assertFalse(self.path.exists())
        self.assertFalse(self.lock_path().exists())

    def test_missing_log_does_not_create_parent_dir(self):
        deep = self.path.parent / "sub" / "deep" / "x.wal"
        r = WalStore(deep, readonly=True)
        self.assertFalse(deep.exists())
        self.assertFalse(deep.parent.exists())
        r.close()

    def test_empty_log_recovers_empty(self):
        self.path.write_bytes(b"")
        r = WalStore(self.path, readonly=True)
        self.assertEqual(r.state, {})
        self.assertEqual(r.commit_seq, 0)
        self.assertEqual(r.recover()["pending_count"], 0)
        self.assertEqual(self.path.read_bytes(), b"")
        r.close()

    def test_queries_match_writable_instance(self):
        w = self.build_with_pending()
        w_ref = WalStore(self.path)
        r = WalStore(self.path, readonly=True)
        self.assertEqual(r.state, w_ref.state)
        self.assertEqual(r.state, {"a": 1})
        self.assertEqual(r.commit_seq, 1)
        self.assertEqual(r.recover(), w_ref.recover())
        self.assertEqual(r.snapshot(), w_ref.snapshot())
        self.assertEqual(r.snapshot(0), w_ref.snapshot(0))
        self.assertEqual(r.history(), w_ref.history())
        self.assertEqual(r.pending_changes(), w_ref.pending_changes())
        self.assertEqual(r.audit(), w_ref.audit())
        self.assertEqual(r.get("a"), w_ref.get("a"))
        self.assertEqual(r.get("z", 7), w_ref.get("z", 7))
        self.assertEqual(r.contains("a"), w_ref.contains("a"))
        # The pending record is observable, not committed.
        pc = r.pending_changes()
        self.assertEqual(pc["pending_count"], 1)
        self.assertEqual(
            pc["changes"], [{"op": "set", "key": "b", "value": 2}]
        )
        r.close()
        w_ref.close()
        w.close()

    def test_tail_fragment_uses_existing_discard_and_audit_rules(self):
        self.build_committed()
        fragment = b'{"op": "set", "key": "x", "val'
        self.path.write_bytes(self.path.read_bytes() + fragment)
        raw = self.path.read_bytes()
        r = WalStore(self.path, readonly=True)
        self.assertEqual(r.state, {"a": 1})
        self.assertEqual(r.commit_seq, 1)
        self.assertEqual(r.pending_changes()["pending_count"], 0)
        audit = r.audit()
        self.assertEqual(audit["tail_bytes"], len(fragment))
        self.assertEqual(audit["pending_count"], 0)
        # Bytes survive: read-only repair cannot remove the fragment.
        self.assertEqual(self.path.read_bytes(), raw)
        r.close()

    def test_unterminated_complete_record_kept_as_tail(self):
        self.build_committed()
        rec = json.dumps(
            {"op": "set", "key": "b", "value": 2, "seq": 2}, sort_keys=True
        )
        self.path.write_bytes(self.path.read_bytes() + rec.encode("utf-8"))
        raw = self.path.read_bytes()
        r = WalStore(self.path, readonly=True)
        self.assertEqual(r.state, {"a": 1})
        self.assertEqual(r.commit_seq, 1)
        self.assertEqual(r.audit()["tail_bytes"], len(rec.encode("utf-8")))
        self.assertEqual(r.pending_changes()["changes"], [])
        self.assertEqual(self.path.read_bytes(), raw)
        r.close()

    def test_corruption_still_raises_on_readonly_open(self):
        self.build_committed()
        self.path.write_bytes(self.path.read_bytes() + b"not-a-record\n")
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path, readonly=True)

    def test_corruption_in_query_raises_and_changes_nothing(self):
        self.build_committed()
        r = WalStore(self.path, readonly=True)
        self.path.write_bytes(self.path.read_bytes() + b"garbage\n")
        raw = self.path.read_bytes()
        with self.assertRaises(WalCorruptionError):
            r.snapshot()
        with self.assertRaises(WalCorruptionError):
            r.audit()
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertEqual(r.state, {"a": 1})
        self.assertEqual(r.commit_seq, 1)
        r.close()

    def test_mutators_raise_readonly_without_any_change(self):
        w = self.build_with_pending()
        raw = self.path.read_bytes()
        size = self.path.stat().st_size
        r = WalStore(self.path, readonly=True)
        calls = [
            lambda: r.set("c", 3),
            lambda: r.delete("a"),
            lambda: r.commit(),
            lambda: r.rollback(),
            lambda: r.restore(0),
            lambda: r.repair_tail(),
        ]
        for call in calls:
            with self.assertRaises(app.WalReadOnlyError):
                call()
        # No truncation, append, seq advance, or in-memory change.
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertEqual(self.path.stat().st_size, size)
        self.assertEqual(r.state, {"a": 1})
        self.assertEqual(r.commit_seq, 1)
        self.assertEqual(r.pending_changes()["pending_count"], 1)
        r.close()
        w.close()

    def test_mutators_raise_before_argument_validation_side_effects(self):
        # Even with a bad key/value/target, read-only refusal comes first and
        # leaves the log untouched; the read-only error type is stable.
        self.build_committed()
        raw = self.path.read_bytes()
        r = WalStore(self.path, readonly=True)
        with self.assertRaises(app.WalReadOnlyError):
            r.set(123, 1)
        with self.assertRaises(app.WalReadOnlyError):
            r.restore("not-an-int")
        self.assertEqual(self.path.read_bytes(), raw)
        r.close()

    def test_complete_uncommitted_record_bytes_preserved(self):
        w = self.build_with_pending()
        raw = self.path.read_bytes()
        r = WalStore(self.path, readonly=True)
        for call in (r.commit, r.rollback, r.repair_tail, lambda: r.restore(0)):
            with self.assertRaises(app.WalReadOnlyError):
                call()
        self.assertEqual(self.path.read_bytes(), raw)
        # A writer can still commit those preserved bytes afterwards.
        self.assertEqual(w.commit(), 2)
        self.assertEqual(r.get("b"), 2)
        r.close()
        w.close()

    def test_opens_alongside_live_exclusive_writer(self):
        w = self.build_with_pending()  # still open, lease held
        r = WalStore(self.path, readonly=True)
        self.assertEqual(r.state, {"a": 1})
        r2 = WalStore(self.path, readonly=True)
        self.assertEqual(r2.state, {"a": 1})
        # Reader observes the writer's subsequent durable commits.
        self.assertEqual(w.commit(), 2)
        self.assertEqual(r.snapshot()["state"], {"a": 1, "b": 2})
        self.assertEqual(r2.get("b"), 2)
        r2.close()
        r.close()
        w.close()

    def test_readonly_never_acquires_lease(self):
        self.build_committed()
        lock = self.lock_path()
        w = WalStore(self.path, exclusive=True)
        self.assertTrue(lock.exists())
        mtime = lock.stat().st_mtime_ns
        r = WalStore(self.path, readonly=True)
        r.audit()
        r.close()
        # The reader neither recreated nor altered the lock file.
        self.assertEqual(lock.stat().st_mtime_ns, mtime)
        w.close()

    def test_readonly_does_not_create_lock_for_missing_log(self):
        r = WalStore(self.path, readonly=True)
        r.close()
        self.assertFalse(self.lock_path().exists())

    def test_closed_readonly_prefers_walclosederror(self):
        self.build_committed()
        r = WalStore(self.path, readonly=True)
        r.close()
        r.close()  # idempotent
        calls = [
            lambda: r.set("c", 3),
            lambda: r.delete("a"),
            lambda: r.commit(),
            lambda: r.rollback(),
            lambda: r.restore(0),
            lambda: r.repair_tail(),
            lambda: r.recover(),
            lambda: r.get("a"),
            lambda: r.contains("a"),
            lambda: r.snapshot(),
            lambda: r.history(),
            lambda: r.pending_changes(),
            lambda: r.audit(),
            lambda: r.scan(),
        ]
        for call in calls:
            with self.assertRaises(app.WalClosedError):
                call()

    def test_context_manager_is_idempotent(self):
        self.build_committed()
        with WalStore(self.path, readonly=True) as r:
            self.assertEqual(r.get("a"), 1)
        with self.assertRaises(app.WalClosedError):
            r.get("a")
        r.close()

    def test_deep_copy_conventions_preserved(self):
        s = WalStore(self.path)
        s.set("nested", {"x": [1, 2]})
        s.commit()
        s.close()
        r = WalStore(self.path, readonly=True)
        value = r.get("nested")
        value["x"].append(3)
        self.assertEqual(r.get("nested"), {"x": [1, 2]})
        snap = r.snapshot()["state"]
        snap["nested"]["x"].append(9)
        self.assertEqual(r.snapshot()["state"]["nested"], {"x": [1, 2]})
        r.close()

    def test_legacy_log_reads_without_migration(self):
        self.build_committed()
        raw_before = self.path.read_bytes()
        r = WalStore(self.path, readonly=True)
        self.assertEqual(r.state, {"a": 1})
        r.close()
        self.assertEqual(self.path.read_bytes(), raw_before)

    def test_omitted_readonly_and_false_remain_writable(self):
        s = WalStore(self.path)
        s.set("x", 1)
        self.assertEqual(s.commit(), 1)
        s.close()
        s2 = WalStore(self.path, exclusive=False, readonly=False)
        s2.set("y", 2)
        self.assertEqual(s2.commit(), 2)
        self.assertEqual(s2.state, {"x": 1, "y": 2})
        s2.rollback()
        s2.close()


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
        s.commit()  # 1: {a:1, b:{n:[1]}}
        s.delete("a")
        s.set("c", 3)
        s.commit()  # 2: {b:{n:[1]}, c:3}
        s.commit()  # 3: empty commit, same state
        return s

    def test_restore_to_earlier_commit(self):
        s = self.build()
        new_seq = s.restore(1)
        self.assertEqual(new_seq, 4)
        self.assertEqual(s.commit_seq, 4)
        self.assertEqual(s.state, {"a": 1, "b": {"n": [1]}})
        self.assertEqual(s.recover()["pending_count"], 0)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq),
                         ({"a": 1, "b": {"n": [1]}}, 4))
        self.assertEqual(s.snapshot(),
                         {"state": {"a": 1, "b": {"n": [1]}}, "commit_seq": 4})

    def test_restore_keeps_old_history_and_snapshots(self):
        s = self.build()
        s.restore(1)
        self.assertEqual(s.snapshot(1)["state"], {"a": 1, "b": {"n": [1]}})
        self.assertEqual(s.snapshot(2)["state"], {"b": {"n": [1]}, "c": 3})
        self.assertEqual(s.snapshot(3)["state"], {"b": {"n": [1]}, "c": 3})
        self.assertEqual([e["commit_seq"] for e in s.history()], [1, 2, 3, 4])
        # batch 4 sets a and deletes c, in Unicode key order (a before c)
        self.assertEqual(s.history()[-1]["changes"], [
            {"op": "set", "key": "a", "value": 1},
            {"op": "delete", "key": "c"},
        ])

    def test_restore_to_latest_writes_empty_batch_and_one_seq(self):
        s = self.build()
        size_before = self.path.stat().st_size
        new_seq = s.restore(3)
        self.assertEqual(new_seq, 4)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        added = self.path.read_bytes()[size_before:]
        self.assertEqual(
            added.decode(),
            json.dumps({"op": "commit", "seq": 4}, sort_keys=True) + "\n",
        )
        self.assertEqual(s.history()[-1], {"commit_seq": 4, "changes": []})

    def test_restore_zero_empties_state(self):
        s = self.build()
        new_seq = s.restore(0)
        self.assertEqual(new_seq, 4)
        self.assertEqual(s.state, {})
        self.assertEqual(s.get("b", "missing"), "missing")
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({}, 4))
        self.assertEqual(s.snapshot(2)["state"], {"b": {"n": [1]}, "c": 3})
        self.assertEqual(s.history()[-1]["changes"],
                         [{"op": "delete", "key": "b"},
                          {"op": "delete", "key": "c"}])

    def test_restore_zero_on_empty_log_starts_seq_chain_at_one(self):
        s = WalStore(self.path)
        self.assertFalse(self.path.exists())
        new_seq = s.restore(0)
        self.assertEqual(new_seq, 1)
        self.assertEqual((s.state, s.commit_seq), ({}, 1))
        self.assertEqual(s.recover()["pending_count"], 0)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({}, 1))
        self.assertEqual(
            self.path.read_text(),
            json.dumps({"op": "commit", "seq": 1}, sort_keys=True) + "\n",
        )

    def test_chained_restores(self):
        s = self.build()
        self.assertEqual(s.restore(1), 4)
        self.assertEqual(s.state, {"a": 1, "b": {"n": [1]}})
        self.assertEqual(s.restore(0), 5)
        self.assertEqual(s.state, {})
        self.assertEqual(s.restore(2), 6)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq),
                         ({"b": {"n": [1]}, "c": 3}, 6))
        self.assertEqual([e["commit_seq"] for e in s2.history()],
                         [1, 2, 3, 4, 5, 6])

    def test_invalid_target_seq_raises_valueerror(self):
        s = self.build()
        for bad in (True, False, -1, -10**9, 1.0, 1.5, "1", b"1", [1],
                    None, {"s": 1}, object()):
            with self.assertRaises(ValueError, msg=bad):
                s.restore(bad)
        with self.assertRaises(ValueError):
            s.restore(4)  # beyond latest (3)
        with self.assertRaises(ValueError):
            s.restore(10**9)
        self.assertEqual((s.state, s.commit_seq),
                         ({"b": {"n": [1]}, "c": 3}, 3))

    def test_value_error_on_empty_log_for_positive_target(self):
        s = WalStore(self.path)
        with self.assertRaises(ValueError):
            s.restore(1)
        self.assertFalse(self.path.exists())

    def test_pending_records_raise_pending_error_without_change(self):
        s = self.build()
        s.set("z", 9)
        s.delete("c")
        before = self.path.read_bytes()
        with self.assertRaises(WalPendingError):
            s.restore(1)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq),
                         ({"b": {"n": [1]}, "c": 3}, 3))
        self.assertEqual(s.pending_changes()["pending_count"], 2)
        self.assertEqual(s.rollback(), 2)
        self.assertEqual(s.restore(1), 4)

    def test_pending_records_before_any_commit_raise_pending_error(self):
        s = WalStore(self.path)
        s.set("a", 1)
        before = self.path.read_bytes()
        with self.assertRaises(WalPendingError):
            s.restore(0)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({}, 0))

    def test_tail_fragment_raises_pending_error(self):
        s = self.build()
        before = self.path.read_bytes()
        fragments = [
            '{"op": "set", "key": "x"',
            "{",
            '{"op": "commit", "seq": 4',
            json.dumps({"op": "set", "key": "x", "value": 9, "seq": 4}),
            '{"op": "set", "key": "hé'.encode("utf-8")[:-1],
        ]
        for frag in fragments:
            raw = frag if isinstance(frag, bytes) else frag.encode("utf-8")
            self.path.write_bytes(before)
            self.append_bytes(raw)
            size = self.path.stat().st_size
            with self.assertRaises(WalPendingError, msg=frag):
                s.restore(1)
            self.assertEqual(self.path.stat().st_size, size, msg=frag)
            self.assertEqual((s.state, s.commit_seq),
                             ({"b": {"n": [1]}, "c": 3}, 3), msg=frag)
        self.path.write_bytes(before)
        self.append_bytes(fragments[0])
        s.repair_tail()
        self.assertEqual(s.restore(1), 4)

    def test_corruption_raises_corruption_error_not_pending(self):
        s = self.build()
        prefix = self.path.read_bytes()
        bad = b'{"op": "set", "key": "x"}\n'  # terminated but invalid
        self.append_bytes(bad)
        with self.assertRaises(WalCorruptionError):
            s.restore(1)
        self.assertEqual(self.path.read_bytes(), prefix + bad)
        self.assertEqual((s.state, s.commit_seq),
                         ({"b": {"n": [1]}, "c": 3}, 3))

    def test_json_type_distinction_one_vs_true(self):
        s = WalStore(self.path)
        s.set("k", 1)
        s.commit()  # 1: {k: 1}
        s.set("k", True)
        s.commit()  # 2: {k: true}
        s.restore(1)
        self.assertEqual(s.state, {"k": 1})
        self.assertIs(s.get("k"), 1)
        self.assertEqual(s.history()[-1]["changes"],
                         [{"op": "set", "key": "k", "value": 1}])
        s.restore(2)
        self.assertIs(s.get("k"), True)
        # nested distinction inside objects/arrays
        s2 = WalStore(self.path.parent / "nested.wal")
        s2.set("o", {"x": [1]})
        s2.commit()
        s2.set("o", {"x": [True]})
        s2.commit()
        s2.restore(1)
        self.assertEqual(s2.get("o"), {"x": [1]})
        self.assertEqual(s2.history()[-1]["changes"],
                         [{"op": "set", "key": "o", "value": {"x": [1]}}])
        # numerically equal int/float values compare as equal: empty batch
        s3 = WalStore(self.path.parent / "nums.wal")
        s3.set("n", 1)
        s3.commit()
        s3.set("n", 1.0)
        s3.commit()
        s3.restore(1)
        self.assertEqual(s3.history()[-1]["changes"], [])

    def test_unicode_key_order(self):
        s = WalStore(self.path)
        for k in ("z", "a", "中", "B", "é", "aa"):
            s.set(k, 1)
        s.commit()  # 1
        s.restore(0)  # 2: deletes all, sorted by code point
        keys = [c["key"] for c in s.history()[-1]["changes"]]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(keys, ["B", "a", "aa", "z", "é", "中"])

    def test_state_is_independent_deep_copy(self):
        s = self.build()
        s.restore(1)
        s.state["b"]["n"].append(99)
        s.state["x"] = 5
        s2 = WalStore(self.path)
        self.assertEqual(s2.state, {"a": 1, "b": {"n": [1]}})
        self.assertEqual(s.snapshot(4)["state"], {"a": 1, "b": {"n": [1]}})
        s3 = WalStore(self.path.parent / "iso.wal")
        s3.set("a", {"v": [1]})
        s3.commit()
        target = s3.snapshot(1)
        s3.commit()  # seq 2, same state
        s3.restore(1)
        target["state"]["a"]["v"].append(2)
        self.assertEqual(s3.state, {"a": {"v": [1]}})

    def test_oserror_mid_batch_propagates_and_keeps_committed_state(self):
        s = self.build()  # seq 3
        real_fsync = app.os.fsync
        calls = {"n": 0}

        def fail_after_first(fd):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk on fire")
            return real_fsync(fd)

        # restore(1) writes set a, delete c, then commit; the second
        # record's fsync fails, so set a is left pending and retryable.
        with mock.patch("app.os.fsync", side_effect=fail_after_first):
            with self.assertRaises(OSError):
                s.restore(1)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3})
        self.assertEqual(s.commit_seq, 3)
        pc = s.pending_changes()
        self.assertEqual(pc.commit_seq, 3)
        self.assertGreaterEqual(pc.pending_count, 1)
        self.assertEqual(pc.changes[0],
                         {"op": "set", "key": "a", "value": 1})
        self.assertEqual(s.rollback(), pc.pending_count)
        self.assertEqual(s.pending_changes()["pending_count"], 0)
        self.assertEqual(s.restore(1), 4)
        self.assertEqual(s.state, {"a": 1, "b": {"n": [1]}})

    def test_oserror_on_commit_keeps_all_changes_pending(self):
        s = self.build()  # restore(1) -> set a, delete c, commit
        real_fsync = app.os.fsync
        state = {"n": 0}

        def third_fails(fd):
            state["n"] += 1
            if state["n"] >= 3:
                raise OSError("disk on fire")
            return real_fsync(fd)

        with mock.patch("app.os.fsync", side_effect=third_fails):
            with self.assertRaises(OSError):
                s.restore(1)
        self.assertEqual((s.state, s.commit_seq),
                         ({"b": {"n": [1]}, "c": 3}, 3))
        pc = s.pending_changes()
        self.assertEqual(pc.pending_count, 2)
        self.assertEqual(pc.changes, [
            {"op": "set", "key": "a", "value": 1},
            {"op": "delete", "key": "c"},
        ])
        # the surviving pending batch still reaches the target on commit
        self.assertEqual(s.commit(), 4)
        self.assertEqual(s.state, {"a": 1, "b": {"n": [1]}})

    def test_exclusive_lease_covers_restore(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        with self.assertRaises(app.WalBusyError):
            WalStore(self.path, exclusive=True)
        self.assertEqual(s.restore(0), 2)
        self.assertEqual((s.state, s.commit_seq), ({}, 2))
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({}, 2))
        s.close()

    def test_restored_value_is_an_independent_copy(self):
        s = self.build()
        s.restore(2)
        got = s.get("b")
        got["n"].append(2)
        self.assertEqual(s.get("b"), {"n": [1]})


class ApplyBatchTest(unittest.TestCase):
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

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def log_size(self):
        return self.path.stat().st_size if self.path.exists() else 0

    def build(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", {"n": [1]})
        s.commit()  # 1: {a:1, b:{n:[1]}}
        s.delete("a")
        s.set("c", 3)
        s.commit()  # 2: {b:{n:[1]}, c:3}
        return s

    def test_basic_batch_commits_atomically_and_survives_reopen(self):
        s = self.build()
        new_seq = s.apply_batch(
            2,
            [
                {"op": "set", "key": "d", "value": [1, 2]},
                {"op": "delete", "key": "c"},
                {"op": "set", "key": "b", "value": None},
            ],
        )
        self.assertEqual(new_seq, 3)
        self.assertEqual(s.commit_seq, 3)
        self.assertEqual(s.state, {"b": None, "d": [1, 2]})
        self.assertEqual(s.recover()["pending_count"], 0)
        # reopening recovers exactly the batch's committed view
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"b": None, "d": [1, 2]}, 3))
        self.assertEqual(s2.snapshot(), {"state": {"b": None, "d": [1, 2]}, "commit_seq": 3})
        # older commits stay queryable under their old seqs
        self.assertEqual(s2.snapshot(1)["state"], {"a": 1, "b": {"n": [1]}})
        self.assertEqual([e["commit_seq"] for e in s2.history()], [1, 2, 3])
        self.assertEqual(
            s2.history()[-1]["changes"],
            [
                {"op": "set", "key": "d", "value": [1, 2]},
                {"op": "delete", "key": "c"},
                {"op": "set", "key": "b", "value": None},
            ],
        )

    def test_log_is_byte_identical_to_item_by_item_replay(self):
        changes = [
            {"op": "set", "key": "x", "value": {"n": [1, True]}},
            {"op": "delete", "key": "a"},
            {"op": "set", "key": "a", "value": 9},
        ]
        s1 = WalStore(self.path)
        s1.set("seed", 0)
        s1.commit()
        self.assertEqual(s1.apply_batch(1, changes), 2)

        other = Path(self.dir.name) / "manual.wal"
        s2 = WalStore(other)
        s2.set("seed", 0)
        s2.commit()
        for change in changes:
            if change["op"] == "set":
                s2.set(change["key"], change["value"])
            else:
                s2.delete(change["key"])
        s2.commit()
        # same records, same order, same commit: the durable bytes match
        self.assertEqual(self.path.read_bytes(), other.read_bytes())
        self.assertEqual(s1.state, s2.state)
        self.assertEqual(s1.history(), s2.history())

    def test_empty_batch_advances_seq_only(self):
        s = self.build()
        size_before = self.log_size()
        self.assertEqual(s.apply_batch(2, []), 3)
        self.assertEqual((s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 3))
        added = self.path.read_bytes()[size_before:]
        self.assertEqual(
            added.decode(),
            json.dumps({"op": "commit", "seq": 3}, sort_keys=True) + "\n",
        )
        self.assertEqual(s.history()[-1], {"commit_seq": 3, "changes": []})
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"b": {"n": [1]}, "c": 3}, 3))

    def test_empty_batch_on_empty_log_starts_seq_chain_at_one(self):
        s = WalStore(self.path)
        self.assertFalse(self.path.exists())
        self.assertEqual(s.apply_batch(0, []), 1)
        self.assertEqual((s.state, s.commit_seq), ({}, 1))
        self.assertEqual(
            self.path.read_text(),
            json.dumps({"op": "commit", "seq": 1}, sort_keys=True) + "\n",
        )
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({}, 1))

    def test_tuple_changes_collection_is_accepted(self):
        s = WalStore(self.path)
        seq = s.apply_batch(0, ({"op": "set", "key": "k", "value": 1},))
        self.assertEqual(seq, 1)
        self.assertEqual(s.state, {"k": 1})

    def test_duplicate_keys_are_applied_in_given_order(self):
        s = WalStore(self.path)
        seq = s.apply_batch(
            0,
            [
                {"op": "set", "key": "k", "value": 1},
                {"op": "set", "key": "k", "value": 2},
                {"op": "delete", "key": "k"},
                {"op": "set", "key": "k", "value": 3},
                {"op": "delete", "key": "absent"},  # deleting a missing key is fine
            ],
        )
        self.assertEqual(seq, 1)
        self.assertEqual(s.state, {"k": 3})
        # the batch's full order is preserved in history and on reopen
        expected = [
            {"op": "set", "key": "k", "value": 1},
            {"op": "set", "key": "k", "value": 2},
            {"op": "delete", "key": "k"},
            {"op": "set", "key": "k", "value": 3},
            {"op": "delete", "key": "absent"},
        ]
        self.assertEqual(s.history()[-1]["changes"], expected)
        s2 = WalStore(self.path)
        self.assertEqual(s2.state, {"k": 3})
        self.assertEqual(s2.history()[-1]["changes"], expected)

    def test_successive_batches_across_reopens_keep_seq_monotonic(self):
        s = WalStore(self.path)
        self.assertEqual(s.apply_batch(0, [{"op": "set", "key": "a", "value": 1}]), 1)
        s2 = WalStore(self.path)
        self.assertEqual(
            s2.apply_batch(1, [{"op": "set", "key": "b", "value": 2}]), 2
        )
        s3 = WalStore(self.path)
        self.assertEqual(s3.commit_seq, 2)
        self.assertEqual(s3.apply_batch(2, []), 3)
        self.assertEqual(
            s3.apply_batch(3, [{"op": "delete", "key": "a"}]), 4
        )
        s4 = WalStore(self.path)
        self.assertEqual((s4.state, s4.commit_seq), ({"b": 2}, 4))
        self.assertEqual([e["commit_seq"] for e in s4.history()], [1, 2, 3, 4])

    def test_invalid_base_seq_raises_valueerror_without_touching_anything(self):
        s = self.build()
        before = self.path.read_bytes()
        for bad in (True, False, -1, -10**9, 1.0, 1.5, "1", b"1", [1],
                    None, {"s": 1}, object()):
            with self.assertRaises(ValueError, msg=bad):
                s.apply_batch(bad, [{"op": "set", "key": "z", "value": 1}])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))
        self.assertEqual(s.recover()["pending_count"], 0)

    def test_invalid_changes_raise_valueerror_without_touching_anything(self):
        s = self.build()
        before = self.path.read_bytes()
        cyclic = {}
        cyclic["self"] = cyclic
        bad_changes = [
            "not-a-list",
            {"op": "set", "key": "k", "value": 1},  # the collection itself
            42,
            None,
            [1],
            ["x"],
            [[{"op": "set", "key": "k", "value": 1}]],
            [{"op": "commit", "key": "k"}],  # only set/delete may be batched
            [{"op": "bogus", "key": "k"}],
            [{"op": 1, "key": "k"}],
            [{"key": "k", "value": 1}],  # missing op
            [{"op": "set", "key": "k"}],  # missing value
            [{"op": "set", "key": "k", "value": 1, "seq": 3}],  # extra field
            [{"op": "set", "value": 1}],  # missing key
            [{"op": "delete", "key": "k", "value": 1}],  # delete carries no value
            [{"op": "delete"}],  # missing key
            [{"op": "delete", "key": "k", "seq": 3}],
            [{"op": "set", "key": 1, "value": 1}],  # non-string key
            [{"op": "delete", "key": None}],
            [{"op": "set", "key": "k", "value": float("nan")}],
            [{"op": "set", "key": "k", "value": float("inf")}],
            [{"op": "set", "key": "k", "value": object()}],
            [{"op": "set", "key": "k", "value": {1: 2}}],
            [{"op": "set", "key": "k", "value": cyclic}],
            # a valid prefix does not rescue a later invalid item
            [{"op": "set", "key": "k", "value": 1}, {"op": "delete"}],
        ]
        for bad in bad_changes:
            with self.assertRaises(ValueError, msg=bad):
                s.apply_batch(2, bad)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))
        self.assertEqual(s.recover()["pending_count"], 0)

    def test_invalid_arguments_do_not_create_missing_log(self):
        s = WalStore(self.path)
        with self.assertRaises(ValueError):
            s.apply_batch(-1, [])
        with self.assertRaises(ValueError):
            s.apply_batch(0, [{"op": "set", "key": "k"}])
        self.assertFalse(self.path.exists())

    def test_argument_errors_precede_log_validation(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "x"}\n')  # corrupt record
        with self.assertRaises(ValueError):
            s.apply_batch(-1, [])
        with self.assertRaises(ValueError):
            s.apply_batch(2, [{"op": "bogus"}])
        with self.assertRaises(WalCorruptionError):
            s.apply_batch(2, [])

    def test_conflict_raises_unique_walconflicterror_without_change(self):
        s = self.build()  # committed seq is 2
        before = self.path.read_bytes()
        for bad_base in (0, 1, 3, 10**9):
            with self.assertRaises(app.WalConflictError, msg=bad_base):
                s.apply_batch(bad_base, [{"op": "set", "key": "z", "value": 1}])
            with self.assertRaises(app.WalConflictError, msg=bad_base):
                s.apply_batch(bad_base, [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))
        self.assertEqual(s.recover()["pending_count"], 0)
        # the correct base still goes through afterwards
        self.assertEqual(s.apply_batch(2, [{"op": "set", "key": "z", "value": 1}]), 3)

    def test_conflict_on_empty_log_for_nonzero_base(self):
        s = WalStore(self.path)
        with self.assertRaises(app.WalConflictError):
            s.apply_batch(1, [{"op": "set", "key": "a", "value": 1}])
        self.assertFalse(self.path.exists())
        self.assertEqual((s.state, s.commit_seq), ({}, 0))

    def test_pending_records_raise_pending_error_without_change(self):
        s = self.build()
        s.set("z", 9)
        s.delete("c")
        before = self.path.read_bytes()
        with self.assertRaises(WalPendingError):
            s.apply_batch(2, [{"op": "set", "key": "y", "value": 1}])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))
        self.assertEqual(s.pending_changes()["pending_count"], 2)
        # settling the log by committing or rolling back unblocks the batch
        self.assertEqual(s.rollback(), 2)
        self.assertEqual(s.apply_batch(2, [{"op": "set", "key": "y", "value": 1}]), 3)
        self.assertEqual(s.state, {"b": {"n": [1]}, "c": 3, "y": 1})

    def test_pending_error_takes_precedence_over_conflict(self):
        # an unsettled log is refused before the base seq is even compared
        s = self.build()
        s.set("z", 9)
        with self.assertRaises(WalPendingError):
            s.apply_batch(999, [{"op": "set", "key": "y", "value": 1}])
        self.assertEqual((s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))

    def test_tail_fragment_raises_pending_error_without_change(self):
        s = self.build()
        before = self.path.read_bytes()
        fragments = [
            '{"op": "set", "key": "x"',
            "{",
            '{"op": "commit", "seq": 3',
            json.dumps({"op": "set", "key": "x", "value": 9, "seq": 3}),
            '{"op": "set", "key": "hé'.encode("utf-8")[:-1],
        ]
        for frag in fragments:
            raw = frag if isinstance(frag, bytes) else frag.encode("utf-8")
            self.path.write_bytes(before)
            self.append_bytes(raw)
            size = self.log_size()
            with self.assertRaises(WalPendingError, msg=frag):
                s.apply_batch(2, [{"op": "set", "key": "y", "value": 1}])
            self.assertEqual(self.log_size(), size, msg=frag)
            self.assertEqual(
                (s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2), msg=frag
            )
        self.path.write_bytes(before)
        self.append_bytes(fragments[0])
        s.repair_tail()
        self.assertEqual(s.apply_batch(2, [{"op": "set", "key": "y", "value": 1}]), 3)

    def test_corruption_raises_corruption_error_without_change(self):
        s = self.build()
        before = self.path.read_bytes()
        bad = b'{"op": "set", "key": "x"}\n'  # terminated but invalid
        self.append_bytes(bad)
        with self.assertRaises(WalCorruptionError):
            s.apply_batch(2, [{"op": "set", "key": "y", "value": 1}])
        self.assertEqual(self.path.read_bytes(), before + bad)
        self.assertEqual((s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))

    def test_readonly_raises_readonly_error_before_any_validation(self):
        self.build().close()
        before = self.path.read_bytes()
        r = WalStore(self.path, readonly=True)
        with self.assertRaises(app.WalReadOnlyError):
            r.apply_batch(2, [{"op": "set", "key": "y", "value": 1}])
        # read-only refusal wins even over argument validation
        with self.assertRaises(app.WalReadOnlyError):
            r.apply_batch("not-an-int", "not-a-list")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((r.state, r.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))
        r.close()

    def test_closed_instance_raises_closed_error(self):
        s = self.build()
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.apply_batch(2, [{"op": "set", "key": "y", "value": 1}])
        r = WalStore(self.path, readonly=True)
        r.close()
        with self.assertRaises(app.WalClosedError):
            r.apply_batch(2, [])

    def test_oserror_mid_batch_propagates_and_leaves_pending_prefix(self):
        s = self.build()  # seq 2
        changes = [
            {"op": "set", "key": "d", "value": 4},
            {"op": "delete", "key": "c"},
            {"op": "set", "key": "e", "value": 5},
        ]
        real_fsync = app.os.fsync
        calls = {"n": 0}

        def fail_after_first(fd):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk on fire")
            return real_fsync(fd)

        # the second record's fsync fails: only the first change is durable
        with mock.patch("app.os.fsync", side_effect=fail_after_first):
            with self.assertRaises(OSError):
                s.apply_batch(2, changes)
        # neither memory state nor commit_seq advanced
        self.assertEqual((s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))
        pc = s.pending_changes()
        self.assertEqual(pc.commit_seq, 2)
        self.assertEqual(pc.changes, [{"op": "set", "key": "d", "value": 4}])
        # a reopen recovers the last complete commit, never the unsealed part
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))
        self.assertEqual(s2.pending_changes().changes, pc.changes)
        # rollback clears the prefix and the batch can be retried
        self.assertEqual(s.rollback(), 1)
        self.assertEqual(s.pending_changes()["pending_count"], 0)
        self.assertEqual(s.apply_batch(2, changes), 3)
        self.assertEqual(s.state, {"b": {"n": [1]}, "d": 4, "e": 5})
        s3 = WalStore(self.path)
        self.assertEqual((s3.state, s3.commit_seq), ({"b": {"n": [1]}, "d": 4, "e": 5}, 3))

    def test_oserror_on_sealing_commit_keeps_whole_batch_pending(self):
        s = self.build()  # seq 2
        changes = [
            {"op": "set", "key": "d", "value": 4},
            {"op": "delete", "key": "c"},
        ]
        real_fsync = app.os.fsync
        calls = {"n": 0}

        def third_fails(fd):
            calls["n"] += 1
            if calls["n"] >= 3:
                raise OSError("disk on fire")
            return real_fsync(fd)

        with mock.patch("app.os.fsync", side_effect=third_fails):
            with self.assertRaises(OSError):
                s.apply_batch(2, changes)
        self.assertEqual((s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))
        pc = s.pending_changes()
        self.assertEqual(pc.pending_count, 2)
        self.assertEqual(
            pc.changes,
            [
                {"op": "set", "key": "d", "value": 4},
                {"op": "delete", "key": "c"},
            ],
        )
        # the surviving pending batch still seals under the same seq chain
        self.assertEqual(s.commit(), 3)
        self.assertEqual(s.state, {"b": {"n": [1]}, "d": 4})
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"b": {"n": [1]}, "d": 4}, 3))

    def test_oserror_on_first_record_leaves_no_pending_and_same_seq(self):
        s = self.build()

        def boom(fd):
            raise OSError("disk on fire")

        with mock.patch("app.os.fsync", side_effect=boom):
            with self.assertRaises(OSError):
                s.apply_batch(2, [{"op": "set", "key": "d", "value": 4}])
        self.assertEqual((s.state, s.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))
        self.assertEqual(s.pending_changes()["pending_count"], 0)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"b": {"n": [1]}, "c": 3}, 2))
        # the failed batch consumed no seq: the retry is exactly seq 3
        self.assertEqual(
            s.apply_batch(2, [{"op": "set", "key": "d", "value": 4}]), 3
        )

    def test_inputs_do_not_share_mutable_objects_with_store(self):
        s = WalStore(self.path)
        nested = {"n": [1]}
        changes = [
            {"op": "set", "key": "a", "value": nested},
            {"op": "set", "key": "b", "value": [1, {"k": 2}]},
        ]
        self.assertEqual(s.apply_batch(0, changes), 1)
        # mutating the caller's collection and values afterwards must not
        # reach the store, later queries, or a reopened instance
        nested["n"].append(99)
        changes[1]["value"].append(5)
        changes.append({"op": "delete", "key": "a"})
        self.assertEqual(s.state, {"a": {"n": [1]}, "b": [1, {"k": 2}]})
        self.assertEqual(s.get("a"), {"n": [1]})
        self.assertEqual(s.history()[-1]["changes"][0]["value"], {"n": [1]})
        s2 = WalStore(self.path)
        self.assertEqual(s2.state, {"a": {"n": [1]}, "b": [1, {"k": 2}]})

    def test_exclusive_lease_covers_apply_batch(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        with self.assertRaises(app.WalBusyError):
            WalStore(self.path, exclusive=True)
        self.assertEqual(
            s.apply_batch(1, [{"op": "set", "key": "b", "value": 2}]), 2
        )
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))
        s.close()

    def test_old_entries_and_legacy_logs_unaffected_by_batch_commits(self):
        # a legacy log written before apply_batch existed stays readable and
        # writable, and batches mix freely with the old entries afterwards
        self.write_lines(
            {"op": "set", "key": "k", "value": [1, 2], "seq": 1},
            {"op": "commit", "seq": 1},
        )
        s = WalStore(self.path)
        self.assertEqual(s.state, {"k": [1, 2]})
        self.assertEqual(s.apply_batch(1, [{"op": "delete", "key": "k"}]), 2)
        self.assertEqual(s.state, {})
        # the old entry points keep working on top of batch commits
        s.set("x", 1)
        self.assertEqual(s.rollback(), 1)
        s.set("x", 1)
        self.assertEqual(s.commit(), 3)
        self.assertEqual(s.state, {"x": 1})
        self.assertEqual(s.snapshot(1)["state"], {"k": [1, 2]})
        self.assertEqual([e["commit_seq"] for e in s.history()], [1, 2, 3])
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"x": 1}, 3))


class ScanTest(unittest.TestCase):
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

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def log_size(self):
        return self.path.stat().st_size if self.path.exists() else 0

    def build(self):
        s = WalStore(self.path)
        s.set("z", {"n": [1]})
        s.set("a", 1)
        s.set("m", [True, None])
        s.set("B", 2)
        s.set("aa", 3)
        s.set("中", 4)
        s.set("é", 5)
        s.commit()  # seq 1
        s.set("tail", 9)
        s.delete("a")
        return s

    def test_nonexistent_and_empty_log_return_empty_list(self):
        s = WalStore(self.path)  # path does not exist
        for _ in range(3):
            self.assertEqual(s.scan(), [])
            self.assertEqual(s.scan("a"), [])
            self.assertEqual(s.scan("a", "z"), [])
            self.assertEqual(s.scan(limit=10), [])
        self.assertFalse(self.path.exists())  # never creates the file
        self.path.write_bytes(b"")
        self.assertEqual(s.scan(), [])
        self.assertTrue(self.path.exists())  # an empty file is still readable

    def test_missing_log_bad_arguments_do_not_create_file(self):
        s = WalStore(self.path)
        for bad in (1, True, b"a", ["a"]):
            with self.assertRaises(ValueError, msg=bad):
                s.scan(bad)
            with self.assertRaises(ValueError, msg=bad):
                s.scan(None, bad)
        with self.assertRaises(ValueError):
            s.scan("b", "a")
        for bad in (True, False, -1, 1.5, "2", [1]):
            with self.assertRaises(ValueError, msg=bad):
                s.scan(limit=bad)
        self.assertFalse(self.path.exists())

    def test_full_scan_sorted_by_unicode_code_point(self):
        s = self.build()
        rows = s.scan()
        keys = [row["key"] for row in rows]
        self.assertEqual(keys, ["B", "a", "aa", "m", "z", "é", "中"])
        self.assertEqual(keys, sorted(keys))
        # exactly key and value per row; seq is never exposed
        for row in rows:
            self.assertEqual(set(row), {"key", "value"})
        by_key = {row["key"]: row["value"] for row in rows}
        self.assertEqual(
            by_key,
            {
                "B": 2,
                "a": 1,
                "aa": 3,
                "m": [True, None],
                "z": {"n": [1]},
                "é": 5,
                "中": 4,
            },
        )

    def test_rows_carry_exact_fields_even_for_none_values(self):
        s = WalStore(self.path)
        s.set("n", None)
        s.set("x", {"k": [1, {"j": 2}]})
        s.commit()
        self.assertEqual(
            s.scan(),
            [
                {"key": "n", "value": None},
                {"key": "x", "value": {"k": [1, {"j": 2}]}},
            ],
        )

    def test_start_bound_is_inclusive(self):
        s = self.build()
        self.assertEqual(
            [r["key"] for r in s.scan("aa")],
            ["aa", "m", "z", "é", "中"],
        )
        # an exact-match start key is included
        self.assertEqual([r["key"] for r in s.scan("a")], ["a", "aa", "m", "z", "é", "中"])
        self.assertEqual([r["key"] for r in s.scan("zzz")], ["é", "中"])
        self.assertEqual(s.scan("\U0001f600"), [])

    def test_end_bound_is_exclusive(self):
        s = self.build()
        self.assertEqual(
            [r["key"] for r in s.scan(None, "m")],
            ["B", "a", "aa"],
        )
        # an exact-match end key is excluded
        self.assertEqual(
            [r["key"] for r in s.scan(None, "z")],
            ["B", "a", "aa", "m"],
        )
        self.assertEqual([r["key"] for r in s.scan(None, "B")], [])
        self.assertEqual([r["key"] for r in s.scan(None, "aaa")], ["B", "a", "aa"])

    def test_half_open_window_with_both_bounds(self):
        s = self.build()
        self.assertEqual(
            [r["key"] for r in s.scan("a", "z")],
            ["a", "aa", "m"],
        )
        # equal bounds form an empty (but legal) half-open window
        self.assertEqual(s.scan("a", "a"), [])
        self.assertEqual(s.scan("m", "m"), [])

    def test_zero_limit_returns_empty_even_with_matches(self):
        s = self.build()
        self.assertEqual(s.scan(limit=0), [])
        self.assertEqual(s.scan("a", "z", 0), [])
        # zero limit still validates the log but returns nothing
        s.set("later", 1)
        self.assertEqual(s.scan(limit=0), [])

    def test_limit_caps_after_sorting(self):
        s = self.build()
        full = [r["key"] for r in s.scan()]
        for n in (1, 2, 3, 7, 100):
            got = s.scan(limit=n)
            self.assertEqual([r["key"] for r in got], full[: min(n, len(full))])
        # limit interacts with the window: ordering first, window second,
        # cap last -- i.e. the first `limit` ordered keys within the window
        self.assertEqual(
            [r["key"] for r in s.scan("a", "中", 2)],
            ["a", "aa"],
        )
        self.assertEqual(
            [r["key"] for r in s.scan(None, None, 3)],
            ["B", "a", "aa"],
        )

    def test_no_matching_keys_returns_empty_list(self):
        s = self.build()
        self.assertEqual(s.scan("0", "9"), [])
        self.assertEqual(s.scan("\0", "\t"), [])
        self.assertIsInstance(s.scan("0", "9"), list)

    def test_committed_delete_removes_key_from_scan(self):
        s = self.build()
        s.commit()  # commits the pending delete of "a" and set of "tail" (seq 2)
        keys = [r["key"] for r in s.scan()]
        self.assertNotIn("a", keys)
        self.assertIn("tail", keys)
        s.delete("tail")
        s.delete("missing")
        s.commit()  # seq 3
        keys = [r["key"] for r in s.scan()]
        self.assertNotIn("a", keys)
        self.assertNotIn("tail", keys)

    def test_uncommitted_changes_and_fragments_are_invisible(self):
        s = self.build()  # pending set "tail", pending delete "a"
        committed = ["B", "a", "aa", "m", "z", "é", "中"]
        self.assertEqual([r["key"] for r in s.scan()], committed)
        # every fragment shape recover can ignore stays invisible too
        fragments = [
            '{"op": "set", "key": "x"',
            "{",
            '{"op": "commit", "seq": 2',
            json.dumps({"op": "set", "key": "x", "value": 9, "seq": 2}),
            '{"op": "set", "key": "hé'.encode("utf-8")[:-1],
        ]
        prefix_size = self.log_size()
        for frag in fragments:
            raw = frag if isinstance(frag, bytes) else frag.encode("utf-8")
            self.path.write_bytes(self.path.read_bytes()[:prefix_size])
            self.append_bytes(raw)
            self.assertEqual(
                [r["key"] for r in s.scan()], committed, msg=frag
            )
            self.assertEqual(s.scan("t", "u"), [], msg=frag)

    def test_values_are_independent_deep_copies(self):
        s = WalStore(self.path)
        s.set("a", {"nested": [1, {"k": 2}]})
        s.set("b", [3])
        s.commit()
        rows = s.scan()
        rows[0]["value"]["nested"].append(99)
        rows[0]["value"]["nested"][1]["k"] = 7
        rows.append({"key": "z", "value": "injected"})
        # store, repeated scans, and a reopen all stay unchanged
        again = s.scan()
        self.assertEqual(
            again,
            [
                {"key": "a", "value": {"nested": [1, {"k": 2}]}},
                {"key": "b", "value": [3]},
            ],
        )
        self.assertEqual(s.state, {"a": {"nested": [1, {"k": 2}]}, "b": [3]})
        self.assertEqual(
            WalStore(self.path).scan(),
            [
                {"key": "a", "value": {"nested": [1, {"k": 2}]}},
                {"key": "b", "value": [3]},
            ],
        )
        # independent even across two calls and across range windows
        r1 = s.scan("a", "b")[0]["value"]
        r2 = s.scan("a", "b")[0]["value"]
        self.assertIsNot(r1, r2)
        r1["nested"].append(123)
        self.assertEqual(s.scan("a", "b")[0]["value"], {"nested": [1, {"k": 2}]})

    def test_invalid_arguments_raise_valueerror(self):
        s = self.build()
        for bad in (1, 1.5, True, False, b"a", ["a"], {"a": 1}, object()):
            with self.assertRaises(ValueError, msg=bad):
                s.scan(bad)
            with self.assertRaises(ValueError, msg=bad):
                s.scan(None, bad)
            with self.assertRaises(ValueError, msg=bad):
                s.scan(bad, bad)
        # None is the legal "omitted" spelling for both bounds
        self.assertEqual([r["key"] for r in s.scan(None, None)],
                         [r["key"] for r in s.scan()])
        # inverted range
        with self.assertRaises(ValueError):
            s.scan("z", "a")
        with self.assertRaises(ValueError):
            s.scan("中", "B")
        # bad limits: bool explicitly rejected, no floats or negatives
        for bad in (True, False, -1, -10**9, 0.0, 1.0, 1.5, "1", b"1", [1], {"x": 1}):
            with self.assertRaises(ValueError, msg=bad):
                s.scan(limit=bad)

    def test_argument_errors_precede_log_validation_and_reads(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "x"}\n')  # corrupt terminated record
        with self.assertRaises(ValueError):
            s.scan(1)
        with self.assertRaises(ValueError):
            s.scan(None, 2)
        with self.assertRaises(ValueError):
            s.scan("z", "a")
        with self.assertRaises(ValueError):
            s.scan(limit=True)
        with self.assertRaises(ValueError):
            s.scan(limit=-1)
        # only with acceptable arguments does the corruption surface
        with self.assertRaises(WalCorruptionError):
            s.scan()
        with self.assertRaises(WalCorruptionError):
            s.scan("a", "z", 2)
        with self.assertRaises(WalCorruptionError):
            s.scan(limit=0)

    def test_corruption_raises_no_partial_result_and_keeps_memory(self):
        s = self.build()
        prefix = self.path.read_bytes()
        bad_tails = [
            b"   ",
            b"\n",
            b"  \n",
            b"\xff",
            b'{"op": "set", "key": "b", "value": NaN, "seq": 2}\n',
            b'{"op": "set", "op": "set", "key": "b", "value": 2, "seq": 2}\n',
            b'{"op": "set", "key": "b", "value": 2, "seq": 5}',
            b'{"op": "set", "key": "b", "seq": 2}',
            b'{"op": "bogus", "seq": 2}',
            b"not json\n",
            b"[1, 2]\n",
            b'{"op": "set", "key": "b", "value": 2, "seq": 2}extra',
        ]
        for tail in bad_tails:
            self.path.write_bytes(prefix)
            self.append_bytes(tail)
            with self.assertRaises(WalCorruptionError, msg=tail):
                s.scan()
            with self.assertRaises(WalCorruptionError, msg=tail):
                s.scan("a", "z", 1)
            # in-memory committed state and seq untouched (state is the
            # adopted view; a fresh query also refuses rather than serving
            # partial data -- asserted separately just above)
            self.assertEqual(s.commit_seq, 1, msg=tail)
            self.assertEqual(s.state["a"], 1, msg=tail)
            self.assertNotIn("b", s.state)

    def test_scan_is_strictly_read_only(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "frag"')  # interrupted tail
        before = self.path.read_bytes()
        size = len(before)
        for kwargs in (
            {},
            {"start_key": "a"},
            {"end_key": "z"},
            {"start_key": "a", "end_key": "z"},
            {"limit": 0},
            {"limit": 2},
            {"start_key": "B", "end_key": "中", "limit": 3},
        ):
            s.scan(**kwargs)
        self.assertEqual(self.path.read_bytes(), before)  # fragment bytes stay
        self.assertEqual(self.log_size(), size)
        self.assertEqual(s.state, {"B": 2, "a": 1, "aa": 3, "m": [True, None],
                                   "z": {"n": [1]}, "é": 5, "中": 4})
        self.assertEqual(s.commit_seq, 1)
        self.assertEqual(s.recover()["pending_count"], 2)

    def test_scan_then_append_still_drops_fragment(self):
        s = self.build()
        prefix_size = self.log_size()
        self.append_bytes('{"op": "set", "key": "frag"')
        s.scan()
        s.scan("a", "z", 1)
        s.set("n", 7)
        self.assertEqual(s.commit(), 2)
        self.assertEqual(s.commit_seq, 2)
        data = self.path.read_bytes()
        self.assertEqual(len(data), self.log_size())
        self.assertNotIn(b'"frag"', data)
        s2 = WalStore(self.path)
        # commit 2 adopts the pre-fragment pending batch too: "a" was
        # deleted and "tail" was set in that batch
        self.assertEqual(
            sorted(s2.state),
            sorted(["B", "aa", "m", "n", "tail", "z", "é", "中"]),
        )
        self.assertEqual(s2.commit_seq, 2)

    def test_deterministic_across_calls_and_reopens(self):
        s = self.build()
        cases = [
            {},
            {"start_key": "a"},
            {"end_key": "z"},
            {"start_key": "a", "end_key": "z"},
            {"limit": 3},
            {"start_key": "B", "end_key": "中", "limit": 2},
            {"limit": 0},
        ]
        for kwargs in cases:
            first = s.scan(**kwargs)
            for _ in range(3):
                self.assertEqual(s.scan(**kwargs), first)
                self.assertEqual(WalStore(self.path).scan(**kwargs), first)

    def test_closed_instance_raises_walclosederror(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.scan()
        with self.assertRaises(app.WalClosedError):
            s.scan("a", "z", 1)
        # close stays idempotent and the closed error beats bad arguments
        s.close()

    def test_readonly_instance_scans_without_lease(self):
        w = self.build()
        lock = Path(os.path.abspath(str(self.path)) + ".lock")
        r = WalStore(self.path, readonly=True)
        # a read-only scan agrees with a plain instance and with snapshot
        plain = WalStore(self.path)
        self.assertEqual(r.scan(), plain.scan())
        snap_state = plain.snapshot()["state"]
        self.assertEqual(
            r.scan(),
            [{"key": k, "value": snap_state[k]} for k in sorted(snap_state)],
        )
        self.assertEqual(
            [row["key"] for row in r.scan("a", "z", 2)],
            ["a", "aa"],
        )
        self.assertEqual(r.scan(limit=0), [])
        # scans while a live exclusive writer holds the lease are fine and
        # observe later commits
        self.assertEqual(w.commit(), 2)
        self.assertIn("tail", [row["key"] for row in r.scan()])
        self.assertNotIn("a", [row["key"] for row in r.scan()])
        self.assertFalse(lock.exists())  # read-only never creates the lease
        r.close()
        plain.close()
        w.close()

    def test_readonly_scan_creates_nothing_for_missing_log(self):
        r = WalStore(self.path, readonly=True)
        self.assertEqual(r.scan(), [])
        self.assertEqual(r.scan("a", "z", 5), [])
        r.close()
        self.assertFalse(self.path.exists())
        self.assertFalse(Path(os.path.abspath(str(self.path)) + ".lock").exists())

    def test_exclusive_scan_keeps_lease(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        self.assertEqual([r["key"] for r in s.scan()], ["a"])
        # the lease is still held: another exclusive opener is refused
        with self.assertRaises(app.WalBusyError):
            WalStore(self.path, exclusive=True)
        # and the holder can keep mutating afterwards
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        self.assertEqual([r["key"] for r in s.scan()], ["a", "b"])
        s.close()

    def test_scan_agrees_with_snapshot_filtering(self):
        s = self.build()
        state = s.snapshot()["state"]
        expected = [
            {"key": k, "value": state[k]}
            for k in sorted(state)
            if "a" <= k < "z"
        ][:2]
        self.assertEqual(s.scan("a", "z", 2), expected)
        # limit 0 short-circuits to [] regardless of a corrupt-free window
        self.assertEqual(s.scan("a", "z", 0), [])

    def test_legacy_log_scans_without_migration(self):
        self.write_lines(
            {"op": "set", "key": "legacy", "value": [1, 2], "seq": 1},
            {"op": "set", "key": "other", "value": {"x": 1}, "seq": 1},
            {"op": "commit", "seq": 1},
        )
        before = self.path.read_bytes()
        s = WalStore(self.path)
        self.assertEqual(
            s.scan(),
            [
                {"key": "legacy", "value": [1, 2]},
                {"key": "other", "value": {"x": 1}},
            ],
        )
        self.assertEqual(
            s.scan("l", None, 1),
            [{"key": "legacy", "value": [1, 2]}],
        )
        self.assertEqual(self.path.read_bytes(), before)

    def test_all_json_value_shapes_round_trip_through_scan(self):
        s = WalStore(self.path)
        values = {
            "i": 1, "f": 1.5, "s": "hi", "b": True, "n": None,
            "arr": [1, "two", False, None, {"x": []}],
            "obj": {"k": [1, 2]},
        }
        for k, v in values.items():
            s.set(k, v)
        s.commit()
        rows = {row["key"]: row["value"] for row in s.scan()}
        self.assertEqual(rows, values)
        self.assertIs(rows["b"], True)
        self.assertIsNone(rows["n"])

    def test_mutations_after_scan_follow_existing_rules(self):
        s = self.build()
        s.scan()
        s.scan("a", "z", 2)
        # pending tail from build() still rolls back by the old rules
        self.assertEqual(s.rollback(), 2)
        self.assertEqual(s.recover()["pending_count"], 0)
        # set/delete/commit/restore/apply_batch/repair all keep working
        self.assertEqual(s.commit(), 2)  # empty commit
        s.set("q", 8)
        self.assertEqual(s.commit(), 3)
        self.assertEqual(s.restore(1), 4)
        self.assertEqual(
            s.apply_batch(4, [{"op": "set", "key": "g", "value": 0}]),
            5,
        )
        self.assertIn("g", [r["key"] for r in s.scan()])
        self.append_bytes('{"op": "set", "key": "frag"')
        r = s.repair_tail()
        self.assertEqual(r.removed_bytes, len(b'{"op": "set", "key": "frag"'))
        reopened = WalStore(self.path)
        self.assertEqual(reopened.scan(), s.scan())


class PreAppendRevalidationTest(unittest.TestCase):
    """A write must re-validate the whole log it actually sees on disk.

    The store may stay open while another actor appends to or damages the
    log. Before any byte of a set/delete/commit is appended, the store must
    replay the log under the exact recover rules, re-determine the byte
    boundaries and last committed seq/state, and only then append -- never
    silently overwriting data the instance never knew about.
    """

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

    def base_store(self):
        # A settled store: {"a": 1} committed at seq 1, left open while the
        # test mutates the file out of band.
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        return s

    def record(self, row):
        return (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")

    # ----- category 1: external corruption blocks every write ----------------

    def test_external_corruption_blocks_set_delete_commit_without_change(self):
        corrupt_tails = [
            b'{"op": "set", "key": "x"}\n',        # parseable, missing value
            b"not-a-record\n",                     # terminated invalid JSON
            b"\n",                                 # blank record line
            b"   \n",                              # whitespace-only record
            b"\xff",                               # illegal UTF-8
            b'{"op": "set", "op": "set", "key": "q", "value": 1, "seq": 2}\n',
            b'{"op": "set", "key": "q", "seq": 2}\n',  # missing field
            b'{"op": "bogus", "seq": 2}\n',        # unknown op
            b'{"op": "set", "key": "q", "value": 1, "seq": 5}\n',  # seq jump
            b'{"op": "set", "key": 1, "value": 1, "seq": 2}\n',    # bad key
            b'{"op": "set", "key": "q", "value": NaN, "seq": 2}\n',  # NaN
            b'{"op": "set", "key": "q", "value": 1, "seq": 2}extra\n',
        ]
        s = self.base_store()
        good_prefix = self.path.read_bytes()
        for tail in corrupt_tails:
            self.path.write_bytes(good_prefix + tail)
            size = self.path.stat().st_size
            raw = self.path.read_bytes()
            for action in (
                lambda: s.set("b", 2),
                lambda: s.delete("a"),
                lambda: s.commit(),
            ):
                with self.assertRaises(WalCorruptionError, msg=tail):
                    action()
                # no append, no truncation, no adopted state or seq
                self.assertEqual(self.path.read_bytes(), raw, msg=tail)
                self.assertEqual(self.path.stat().st_size, size, msg=tail)
                self.assertEqual(s.state, {"a": 1}, msg=tail)
                self.assertEqual(s.commit_seq, 1, msg=tail)
            # rollback sees the same boundary: it validates before touching
            # bytes, so the corrupt file is left exactly in place
            with self.assertRaises(WalCorruptionError, msg=tail):
                s.rollback()
            self.assertEqual(self.path.read_bytes(), raw, msg=tail)
            # a reopen on the damaged log is rejected the same way
            with self.assertRaises(WalCorruptionError, msg=tail):
                WalStore(self.path)

    def test_corruption_does_not_latch_once_file_is_restored(self):
        s = self.base_store()
        good_prefix = self.path.read_bytes()
        self.append_bytes(b'{"op": "set", "key": "x"}\n')
        with self.assertRaises(WalCorruptionError):
            s.set("b", 2)
        # the external actor removes the bad bytes; the same open instance
        # revalidates and can write again with a continuous seq chain
        self.path.write_bytes(good_prefix)
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))

    # ----- category 2: legal external records keep existing semantics --------

    def test_external_pending_records_are_kept_and_join_the_new_batch(self):
        s = self.base_store()
        good_prefix = self.path.read_bytes()
        # a full legal, terminated but uncommitted batch written out of band
        ext_set = self.record(
            {"op": "set", "key": "ext", "value": 7, "seq": 2}
        )
        ext_del = self.record({"op": "delete", "key": "a", "seq": 2})
        self.append_bytes(ext_set + ext_del)

        # the new record continues from the file's last COMMITTED seq (1)+1;
        # it joins the external batch at seq 2 rather than clobbering it
        s.set("b", 2)
        new_row = self.record({"op": "set", "key": "b", "value": 2, "seq": 2})
        # external complete records are preserved byte for byte; the local
        # record is simply appended after them, nothing truncated
        self.assertEqual(self.path.read_bytes(),
                         good_prefix + ext_set + ext_del + new_row)
        self.assertEqual(
            s.pending_changes().changes,
            [
                {"op": "set", "key": "ext", "value": 7},
                {"op": "delete", "key": "a"},
                {"op": "set", "key": "b", "value": 2},
            ],
        )
        # committing seals the whole batch (external + local) at seq 2
        self.assertEqual(s.commit(), 2)
        self.assertEqual(s.state, {"b": 2, "ext": 7})
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"b": 2, "ext": 7}, 2))

    def test_external_committed_batch_is_observed_and_next_seq_follows(self):
        s = self.base_store()
        good_prefix = self.path.read_bytes()
        # an entire committed batch lands out of band while s is open
        self.append_bytes(
            self.record({"op": "set", "key": "ext", "value": 7, "seq": 2})
            + self.record({"op": "commit", "seq": 2})
        )
        # s still caches seq 1; the append replays and follows seq 3, never
        # reusing seq 2 and overwriting the unknown committed data
        s.set("n", 5)
        last = json.loads(self.path.read_bytes().splitlines()[-1])
        self.assertEqual(last, {"op": "set", "key": "n", "value": 5, "seq": 3})
        # the re-determined committed view and seq converge with the file
        self.assertEqual(s.state, {"a": 1, "ext": 7})
        self.assertEqual(s.commit_seq, 2)
        self.assertEqual(s.pending_changes().commit_seq, 2)
        self.assertEqual(s.commit(), 3)
        self.assertEqual(s.state, {"a": 1, "ext": 7, "n": 5})
        s2 = WalStore(self.path)
        self.assertEqual(
            (s2.state, s2.commit_seq),
            ({"a": 1, "ext": 7, "n": 5}, 3),
        )
        # original prefix bytes were never rewritten
        self.assertTrue(self.path.read_bytes().startswith(good_prefix))

    # ----- category 3: a recognisable tail fragment is the one truncation -----

    def test_external_tail_fragment_is_removed_then_append_continues(self):
        base = self.base_store()
        good_prefix = self.path.read_bytes()
        base.close()
        fragments = [
            '{"op": "set", "key": "x"',
            "{",
            '{"op": "commit", "seq": 2',
            '{"op": "set", "key": "hé'.encode("utf-8")[:-1],  # truncated UTF-8
            json.dumps({"op": "set", "key": "x", "value": 9, "seq": 2}),
        ]
        new_row = self.record({"op": "set", "key": "b", "value": 2, "seq": 2})
        for frag in fragments:
            raw = frag if isinstance(frag, bytes) else frag.encode("utf-8")
            self.path.write_bytes(good_prefix + raw)
            s = WalStore(self.path)  # open instance, then fragment appears
            s.set("b", 2)
            # only the fragment is gone; the new seq-2 record follows the
            # intact committed prefix byte for byte
            self.assertEqual(self.path.read_bytes(), good_prefix + new_row,
                             msg=frag)
            self.assertNotIn(b'"x"', self.path.read_bytes(), msg=frag)
            self.assertEqual(s.commit(), 2)
            s2 = WalStore(self.path)
            self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2),
                             msg=frag)

    def test_fragment_removal_keeps_complete_pending_records(self):
        s = self.base_store()
        good_prefix = self.path.read_bytes()
        pending = self.record({"op": "set", "key": "p", "value": 4, "seq": 2})
        self.append_bytes(pending + b'{"op": "set", "key": "x"')
        # delete goes through the same revalidation/truncation path
        s.delete("a")
        data = self.path.read_bytes()
        new_row = self.record({"op": "delete", "key": "a", "seq": 2})
        # the legal terminated pending record survives; only the fragment is
        # removed, and the local record is appended to the intact prefix
        self.assertEqual(data, good_prefix + pending + new_row)
        self.assertNotIn(b'"x"', data)
        self.assertEqual(s.pending_changes().pending_count, 2)
        self.assertEqual(s.commit(), 2)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"p": 4}, 2))

    # ----- per-op success: file bytes, seq, recovery --------------------------

    def test_set_success_appends_durable_seq_record(self):
        s = self.base_store()
        before = self.path.stat().st_size
        s.set("b", 2)
        row = self.record({"op": "set", "key": "b", "value": 2, "seq": 2})
        self.assertEqual(self.path.read_bytes()[before:], row)
        # state stays deferred until commit; seq of the open batch is 2
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        self.assertEqual(WalStore(self.path).pending_changes().changes,
                         [{"op": "set", "key": "b", "value": 2}])

    def test_delete_success_appends_durable_seq_record(self):
        s = self.base_store()
        before = self.path.stat().st_size
        s.delete("a")
        row = self.record({"op": "delete", "key": "a", "seq": 2})
        self.assertEqual(self.path.read_bytes()[before:], row)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        self.assertEqual(s.commit(), 2)
        self.assertEqual(WalStore(self.path).state, {})

    def test_commit_success_advances_seq_only_after_persist(self):
        s = self.base_store()
        before = self.path.stat().st_size
        self.assertEqual(s.commit(), 2)  # empty commit
        row = self.record({"op": "commit", "seq": 2})
        self.assertEqual(self.path.read_bytes()[before:], row)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 2))
        self.assertEqual(WalStore(self.path).commit_seq, 2)

    # ----- per-op sync failure: OSError, no adopted state/seq ----------------

    def test_set_sync_failure_propagates_and_adopts_nothing(self):
        s = self.base_store()
        size = self.path.stat().st_size
        with mock.patch("app.os.fsync", side_effect=OSError("disk on fire")):
            with self.assertRaises(OSError):
                s.set("b", 2)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        # the un-durable record was best-effort rolled back; reopen confirms
        self.assertEqual(self.path.stat().st_size, size)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        self.assertEqual(s2.recover()["pending_count"], 0)
        # no seq was consumed: retry seals exactly seq 2
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)

    def test_delete_sync_failure_propagates_and_adopts_nothing(self):
        s = self.base_store()
        size = self.path.stat().st_size
        with mock.patch("app.os.fsync", side_effect=OSError("disk on fire")):
            with self.assertRaises(OSError):
                s.delete("a")
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        self.assertEqual(self.path.stat().st_size, size)
        self.assertEqual(WalStore(self.path).recover()["pending_count"], 0)
        s.delete("a")
        self.assertEqual(s.commit(), 2)
        self.assertEqual(WalStore(self.path).state, {})

    def test_commit_sync_failure_propagates_and_keeps_old_commit(self):
        s = self.base_store()
        s.set("c", 3)  # durable pending record, seq 2
        size = self.path.stat().st_size
        with mock.patch("app.os.fsync", side_effect=OSError("disk on fire")):
            with self.assertRaises(OSError):
                s.commit()
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        # the failed commit boundary is gone; the pending set remains visible
        self.assertEqual(self.path.stat().st_size, size)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        self.assertEqual(s2.pending_changes().changes,
                         [{"op": "set", "key": "c", "value": 3}])
        # retry commits exactly seq 2
        self.assertEqual(s.commit(), 2)
        self.assertEqual(s.state, {"a": 1, "c": 3})

    def test_sync_failure_with_external_pending_loses_only_local_bytes(self):
        s = self.base_store()
        self.append_bytes(
            self.record({"op": "set", "key": "ext", "value": 7, "seq": 2})
        )
        ext_size = self.path.stat().st_size
        with mock.patch("app.os.fsync", side_effect=OSError("disk on fire")):
            with self.assertRaises(OSError):
                s.set("b", 2)
        # external pending bytes survive; only the local record is rolled back
        self.assertEqual(self.path.stat().st_size, ext_size)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        s2 = WalStore(self.path)
        self.assertEqual(s2.pending_changes().changes,
                         [{"op": "set", "key": "ext", "value": 7}])
        # retry joins the surviving external batch and seals at seq 2
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        self.assertEqual(s.state, {"a": 1, "b": 2, "ext": 7})

    # ----- lease / mode exception priority is unchanged ----------------------

    def test_readonly_and_closed_priority_beat_revalidation(self):
        s = self.base_store()
        ro = WalStore(self.path, readonly=True)
        s_close = self.base_store()
        self.append_bytes(b'{"op": "set", "key": "x"}\n')  # corrupt out of band
        # read-only refusal happens before the log is replayed
        with self.assertRaises(app.WalReadOnlyError):
            ro.set("b", 2)
        with self.assertRaises(app.WalReadOnlyError):
            ro.delete("a")
        with self.assertRaises(app.WalReadOnlyError):
            ro.commit()
        # closed refusal takes priority as well
        s_close.close()
        with self.assertRaises(app.WalClosedError):
            s_close.set("b", 2)
        with self.assertRaises(app.WalClosedError):
            s_close.commit()
        # the live writable instance still detects the corruption properly
        with self.assertRaises(WalCorruptionError):
            s.set("b", 2)

    def test_exclusive_holder_revalidates_external_changes(self):
        s = WalStore(self.path, exclusive=True)
        s.set("a", 1)
        s.commit()
        self.append_bytes(
            self.record({"op": "set", "key": "ext", "value": 7, "seq": 2})
            + b'{"op": "set", "key": "frag"'
        )
        # external pending kept, fragment dropped, write continues at seq 2
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq),
                         ({"a": 1, "b": 2, "ext": 7}, 2))
        s.close()


class IntegrityTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "store.wal"

    def tearDown(self):
        self.dir.cleanup()

    def read_rows(self):
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
        ]

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True))
                f.write("\n")

    def protected_store(self):
        """A committed protected log: {a: 1, b: 2} at seq 2."""
        s = WalStore(self.path, integrity=True)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)
        s.commit()
        return s

    # ----- parameter validation ---------------------------------------------

    def test_integrity_param_must_be_bool(self):
        for bad in (1, 0, "yes", None, 1.0):
            with self.assertRaises(ValueError):
                WalStore(self.path, integrity=bad)
        self.assertFalse(self.path.exists())

    # ----- chain start and format preservation -------------------------------

    def test_empty_log_starts_protected_chain(self):
        s = WalStore(self.path, integrity=True)
        s.set("a", 1)
        self.assertEqual(s.commit(), 1)
        rows = self.read_rows()
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIn("ic", row)
            self.assertIsInstance(row["ic"], str)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))

    def test_empty_existing_file_starts_protected_chain(self):
        self.path.write_text("")
        s = WalStore(self.path, integrity=True)
        s.set("a", 1)
        s.commit()
        self.assertIn("ic", self.read_rows()[0])

    def test_protected_log_stays_protected_in_default_mode(self):
        self.protected_store().close()
        s = WalStore(self.path)  # default mode
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        s.set("c", 3)
        self.assertEqual(s.commit(), 3)
        rows = self.read_rows()
        self.assertTrue(all("ic" in row for row in rows))
        s2 = WalStore(self.path, integrity=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2, "c": 3}, 3))

    def test_integrity_reopen_of_protected_log(self):
        self.protected_store().close()
        for _ in range(2):
            s = WalStore(self.path, integrity=True)
            self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
            r = s.recover()
            self.assertEqual(r["state"], {"a": 1, "b": 2})
            self.assertEqual(r["commit_seq"], 2)
            s.close()

    # ----- WalIntegrityError on fully unprotected legacy logs ----------------

    def test_integrity_rejects_fully_unprotected_log(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        before = self.path.read_bytes()
        with self.assertRaises(app.WalIntegrityError):
            WalStore(self.path, integrity=True)
        # nothing created, truncated, appended, or adopted
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises(app.WalIntegrityError):
            WalStore(self.path, integrity=True, readonly=True)
        self.assertEqual(self.path.read_bytes(), before)

    def test_integrity_error_is_unique_and_not_corruption(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        try:
            WalStore(self.path, integrity=True)
            self.fail("expected WalIntegrityError")
        except app.WalIntegrityError as exc:
            self.assertNotIsInstance(exc, WalCorruptionError)

    def test_default_mode_keeps_legacy_log_legacy(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)
        s.commit()
        rows = self.read_rows()
        self.assertTrue(all("ic" not in row for row in rows))
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))

    def test_failed_integrity_open_releases_lease(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        with self.assertRaises(app.WalIntegrityError):
            WalStore(self.path, exclusive=True, integrity=True)
        # the failed open must not leave the lease held
        s2 = WalStore(self.path, exclusive=True)
        s2.close()

    # ----- tamper detection ---------------------------------------------------

    def test_content_rewrite_detected(self):
        self.protected_store().close()
        rows = self.read_rows()
        rows[0]["value"] = 999  # rewrite a committed value in place
        self.write_rows(rows)
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path, integrity=True)

    def test_record_deletion_detected(self):
        self.protected_store().close()
        rows = self.read_rows()
        del rows[1]  # delete the commit record of seq 1
        self.write_rows(rows)
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    def test_record_insertion_detected(self):
        self.protected_store().close()
        rows = self.read_rows()
        rows.insert(1, dict(rows[1]))  # splice in a duplicate record
        self.write_rows(rows)
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    def test_record_reorder_detected(self):
        self.protected_store().close()
        rows = self.read_rows()
        rows[0], rows[1] = rows[1], rows[0]
        self.write_rows(rows)
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    def test_missing_integrity_metadata_detected(self):
        self.protected_store().close()
        rows = self.read_rows()
        del rows[1]["ic"]  # strip protection from one record
        self.write_rows(rows)
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    def test_mixed_protection_rejected_both_ways(self):
        # unprotected history followed by a protected record
        legacy = WalStore(self.path)
        legacy.set("a", 1)
        legacy.commit()
        legacy.close()
        other = Path(self.dir.name) / "other.wal"
        s = WalStore(other, integrity=True)
        s.set("b", 2)
        s.commit()
        s.close()
        protected_rows = [
            json.loads(line)
            for line in other.read_text(encoding="utf-8").splitlines()
        ]
        protected_line = other.read_text(encoding="utf-8").splitlines()[0]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(protected_line + "\n")
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path, integrity=True)
        # protected history followed by an unprotected record
        self.write_rows(protected_rows[:1])
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"op": "commit", "seq": 1}) + "\n")
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    # ----- every entry verifies the whole log --------------------------------

    def test_all_queries_verify_full_log(self):
        s = self.protected_store()
        rows = self.read_rows()
        rows[2]["value"] = 999  # corrupt a committed record out of band
        self.write_rows(rows)
        for call in (
            s.recover,
            s.audit,
            s.pending_changes,
            s.snapshot,
            s.history,
            lambda: s.diff(0, 1),
            s.scan,
            lambda: s.get("a"),
            lambda: s.contains("a"),
            lambda: s.set("c", 3),
            lambda: s.delete("a"),
            s.commit,
            s.rollback,
            s.repair_tail,
            lambda: s.restore(1),
            lambda: s.apply_batch(2, []),
        ):
            with self.assertRaises(WalCorruptionError):
                call()
        # no partial adoption: in-memory state and file are untouched
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(self.read_rows(), rows)

    def test_write_entries_verify_before_appending(self):
        s = self.protected_store()
        rows = self.read_rows()
        rows[0]["key"] = "tampered"
        self.write_rows(rows)
        size = self.path.stat().st_size
        with self.assertRaises(WalCorruptionError):
            s.set("c", 3)
        # nothing appended and no seq consumed
        self.assertEqual(self.path.stat().st_size, size)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))

    # ----- crash recovery, pending, rollback ----------------------------------

    def test_uncommitted_protected_records_pending_then_recovered(self):
        s = WalStore(self.path, integrity=True)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)  # process "dies" before commit
        s.close()
        s2 = WalStore(self.path, integrity=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        pc = s2.pending_changes()
        self.assertEqual(pc["commit_seq"], 1)
        self.assertEqual(pc["pending_count"], 1)
        # pending changes expose no integrity metadata
        self.assertEqual(pc["changes"], [{"op": "set", "key": "b", "value": 2}])
        # rollback clears them; retry continues the chain without gaps
        self.assertEqual(s2.rollback(), 1)
        s2.set("b", 2)
        self.assertEqual(s2.commit(), 2)
        s3 = WalStore(self.path, integrity=True)
        self.assertEqual((s3.state, s3.commit_seq), ({"a": 1, "b": 2}, 2))

    def test_repair_tail_clears_valid_protected_fragment(self):
        s = self.protected_store()
        rows = self.read_rows()
        chain = app._CHAIN_SEED
        for row in rows:
            core = json.dumps(
                {k: v for k, v in row.items() if k != "ic"}, sort_keys=True
            )
            chain = app._chain_digest(chain, core)
        nxt = {"op": "set", "key": "c", "value": 3, "seq": 3}
        nxt["ic"] = app._chain_digest(chain, json.dumps(nxt, sort_keys=True))
        # the record's terminator never became durable: interrupted write
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(nxt, sort_keys=True))
        res = s.repair_tail()
        self.assertEqual(res["removed_bytes"], len(json.dumps(nxt, sort_keys=True)))
        self.assertEqual(res["state"], {"a": 1, "b": 2})
        s2 = WalStore(self.path, integrity=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))

    def test_repair_tail_never_repairs_integrity_mismatch(self):
        s = self.protected_store()
        before = self.path.read_bytes()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"op": "set", "key": "c", "value": 3,
                                "seq": 3, "ic": "0" * 64}, sort_keys=True))
        with self.assertRaises(WalCorruptionError):
            s.repair_tail()
        # the file, state, and seq are untouched
        self.assertEqual(
            self.path.read_bytes(),
            before + json.dumps({"op": "set", "key": "c", "value": 3,
                                 "seq": 3, "ic": "0" * 64},
                                sort_keys=True).encode(),
        )
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))

    # ----- history/snapshot/diff consistency ----------------------------------

    def test_history_snapshot_diff_consistent_across_reopen(self):
        s = WalStore(self.path, integrity=True)
        s.set("a", {"n": [1, True]})
        s.commit()
        s.delete("a")
        s.set("b", 2)
        s.commit()
        h1, d1 = s.history(), s.diff(1, 2)
        snap1 = s.snapshot(1)
        s.close()
        s2 = WalStore(self.path, integrity=True)
        self.assertEqual(s2.history(), h1)
        self.assertEqual(s2.diff(1, 2), d1)
        self.assertEqual(s2.snapshot(1), snap1)
        self.assertEqual(s2.snapshot(1)["state"], {"a": {"n": [1, True]}})
        # mutating returned values never affects the store
        snap1["state"]["a"]["n"].append(9)
        h1[0]["changes"][0]["value"]["n"].append(9)
        self.assertEqual(s2.snapshot(1)["state"], {"a": {"n": [1, True]}})
        self.assertEqual(s2.history()[0]["changes"][0]["value"],
                         {"n": [1, True]})

    def test_restore_and_apply_batch_on_protected_log(self):
        s = self.protected_store()
        self.assertEqual(s.restore(1), 3)
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.apply_batch(3, [{"op": "set", "key": "z",
                                            "value": [1, {"x": True}]}]), 4)
        rows = self.read_rows()
        self.assertTrue(all("ic" in row for row in rows))
        s2 = WalStore(self.path, integrity=True)
        self.assertEqual(s2.state, {"a": 1, "z": [1, {"x": True}]})
        self.assertEqual(s2.commit_seq, 4)

    # ----- modes ---------------------------------------------------------------

    def test_readonly_integrity_verifies_and_queries(self):
        self.protected_store().close()
        ro = WalStore(self.path, readonly=True, integrity=True)
        self.assertEqual(ro.audit()["commit_seq"], 2)
        self.assertEqual(ro.snapshot()["state"], {"a": 1, "b": 2})
        self.assertEqual(len(ro.history()), 2)
        self.assertEqual(ro.scan(), [{"key": "a", "value": 1},
                                     {"key": "b", "value": 2}])
        for call in (lambda: ro.set("c", 3), ro.commit, ro.rollback,
                     ro.repair_tail, lambda: ro.restore(1),
                     lambda: ro.apply_batch(2, [])):
            with self.assertRaises(app.WalReadOnlyError):
                call()

    def test_exclusive_integrity_lease(self):
        s = WalStore(self.path, exclusive=True, integrity=True)
        s.set("a", 1)
        s.commit()
        with self.assertRaises(app.WalBusyError):
            WalStore(self.path, exclusive=True, integrity=True)
        with self.assertRaises(app.WalBusyError):
            WalStore(self.path, exclusive=True)
        s.close()
        s2 = WalStore(self.path, exclusive=True, integrity=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        s2.close()


class KeyVersionTest(unittest.TestCase):
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

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def build(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.set("b", 2)
        s.commit()  # 1: a, b touched at seq 1
        s.set("a", 10)
        s.commit()  # 2: a touched at seq 2
        s.delete("b")
        s.commit()  # 3: b deleted at seq 3
        s.set("c", 3)
        s.commit()  # 4: c touched at seq 4
        return s

    def test_missing_log_returns_zero_and_creates_nothing(self):
        s = WalStore(self.path)
        self.assertEqual(s.key_version("a"), 0)
        self.assertFalse(self.path.exists())

    def test_empty_log_returns_zero(self):
        self.path.touch()
        s = WalStore(self.path)
        self.assertEqual(s.key_version("a"), 0)

    def test_never_touched_key_returns_zero(self):
        s = self.build()
        self.assertEqual(s.key_version("never-seen"), 0)

    def test_versions_track_the_last_touching_commit(self):
        s = self.build()
        self.assertEqual(s.key_version("a"), 2)
        self.assertEqual(s.key_version("b"), 3)  # deleted key keeps delete seq
        self.assertEqual(s.key_version("c"), 4)

    def test_unrelated_commits_do_not_move_a_version(self):
        s = self.build()
        before = {k: s.key_version(k) for k in ("a", "b", "c", "z")}
        s.set("c", 33)
        s.commit()  # 5: only c touched
        self.assertEqual(s.key_version("c"), 5)
        for k in ("a", "b", "z"):
            self.assertEqual(s.key_version(k), before[k])

    def test_uncommitted_records_are_invisible(self):
        s = self.build()
        s.set("a", 999)  # pending, never committed
        s.delete("c")
        self.assertEqual(s.key_version("a"), 2)
        self.assertEqual(s.key_version("c"), 4)
        self.assertEqual(s.key_version("c"), 4)

    def test_tail_fragment_is_invisible(self):
        s = self.build()
        s.close()
        self.append_bytes('{"op": "set", "key": "a", "value": 5, "seq": 5')
        s2 = WalStore(self.path)
        self.assertEqual(s2.key_version("a"), 2)
        # a complete record whose terminator never became durable is a
        # discardable fragment too
        self.append_bytes(
            json.dumps({"op": "set", "key": "a", "value": 5, "seq": 5})
        )
        # the first fragment made the tail unrecognisable as one record;
        # rebuild a clean log with only the unterminated complete record
        s2.close()
        self.write_lines(
            {"op": "set", "key": "a", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1},
        )
        self.append_bytes('{"op": "set", "key": "a", "value": 5, "seq": 2}')
        s3 = WalStore(self.path)
        self.assertEqual(s3.key_version("a"), 1)

    def test_corruption_raises_walcorruptionerror(self):
        self.write_lines(
            {"op": "set", "key": "a", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1},
        )
        s = WalStore(self.path)
        self.append_bytes('{"op": "set", "key": "x"}\n')  # invalid record
        with self.assertRaises(WalCorruptionError):
            s.key_version("a")

    def test_closed_instance_raises_walclosederror_first(self):
        s = self.build()
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.key_version("a")
        with self.assertRaises(app.WalClosedError):
            s.key_version(123)  # closed beats the key validation

    def test_non_string_key_raises_valueerror_without_reading_log(self):
        s = self.build()
        before = self.path.read_bytes()
        for bad in (1, 1.5, None, b"a", ["a"], {"k": 1}, object()):
            with self.assertRaises(ValueError, msg=bad):
                s.key_version(bad)
        self.assertEqual(self.path.read_bytes(), before)

    def test_deterministic_across_calls_and_reopens(self):
        s = self.build()
        first = {k: s.key_version(k) for k in ("a", "b", "c", "z")}
        self.assertEqual(first, {k: s.key_version(k) for k in first})
        s2 = WalStore(self.path)
        self.assertEqual(first, {k: s2.key_version(k) for k in first})

    def test_readonly_instance_serves_versions(self):
        s = self.build()
        ro = WalStore(self.path, readonly=True)
        self.assertEqual(ro.key_version("a"), 2)
        self.assertEqual(ro.key_version("b"), 3)
        self.assertEqual(ro.key_version("z"), 0)

    def test_query_does_not_touch_log_or_state(self):
        s = self.build()
        before = self.path.read_bytes()
        audit_before = s.audit()
        s.key_version("a")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"a": 10, "c": 3}, 4))
        self.assertEqual(s.audit(), audit_before)


class ConditionalWriteTest(unittest.TestCase):
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

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def log_size(self):
        return self.path.stat().st_size if self.path.exists() else 0

    def build(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()  # 1: a at version 1
        s.set("b", 2)
        s.commit()  # 2: b at version 2
        return s

    def test_set_if_version_on_fresh_key_starts_chain(self):
        s = WalStore(self.path)
        self.assertFalse(self.path.exists())
        self.assertEqual(s.set_if_version("a", 0, {"n": [1]}), 1)
        self.assertEqual((s.state, s.commit_seq), ({"a": {"n": [1]}}, 1))
        self.assertEqual(s.key_version("a"), 1)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": {"n": [1]}}, 1))
        self.assertEqual(s2.key_version("a"), 1)

    def test_set_if_version_success_and_all_views_agree(self):
        s = self.build()
        new_seq = s.set_if_version("a", 1, {"n": [1, True]})
        self.assertEqual(new_seq, 3)
        expected_state = {"a": {"n": [1, True]}, "b": 2}
        self.assertEqual((s.state, s.commit_seq), (expected_state, 3))
        self.assertEqual(s.recover()["state"], expected_state)
        self.assertEqual(s.recover()["pending_count"], 0)
        self.assertEqual(s.snapshot()["state"], expected_state)
        self.assertEqual(s.snapshot(1)["state"], {"a": 1})
        self.assertEqual(
            s.history()[-1],
            {"commit_seq": 3,
             "changes": [{"op": "set", "key": "a",
                          "value": {"n": [1, True]}}]},
        )
        self.assertEqual(
            s.diff(2, 3)["changes"],
            [{"op": "set", "key": "a", "value": {"n": [1, True]}}],
        )
        self.assertEqual(
            s.scan(),
            [{"key": "a", "value": {"n": [1, True]}},
             {"key": "b", "value": 2}],
        )
        self.assertEqual(s.pending_changes()["changes"], [])
        self.assertEqual(s.key_version("a"), 3)
        self.assertEqual(s.key_version("b"), 2)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), (expected_state, 3))
        self.assertEqual(s2.key_version("a"), 3)

    def test_delete_if_version_success(self):
        s = self.build()
        self.assertEqual(s.delete_if_version("a", 1), 3)
        self.assertEqual((s.state, s.commit_seq), ({"b": 2}, 3))
        self.assertEqual(s.key_version("a"), 3)  # delete seq is kept
        self.assertFalse(s.contains("a"))
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"b": 2}, 3))
        self.assertEqual(s2.key_version("a"), 3)
        self.assertEqual(
            s2.history()[-1],
            {"commit_seq": 3, "changes": [{"op": "delete", "key": "a"}]},
        )

    def test_delete_if_version_on_never_seen_key(self):
        s = self.build()
        self.assertEqual(s.delete_if_version("ghost", 0), 3)
        self.assertEqual(s.key_version("ghost"), 3)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 3))

    def test_set_if_version_after_delete_uses_delete_seq(self):
        s = self.build()
        s.delete("a")
        s.commit()  # 3: a deleted at version 3
        self.assertEqual(s.key_version("a"), 3)
        self.assertEqual(s.set_if_version("a", 3, 99), 4)
        self.assertEqual(s.state["a"], 99)
        self.assertEqual(s.key_version("a"), 4)

    def test_log_bytes_match_manual_set_then_commit(self):
        s1 = WalStore(self.path)
        s1.set("seed", 0)
        s1.commit()
        self.assertEqual(s1.set_if_version("k", 0, {"n": [1, True]}), 2)

        other = Path(self.dir.name) / "manual.wal"
        s2 = WalStore(other)
        s2.set("seed", 0)
        s2.commit()
        s2.set("k", {"n": [1, True]})
        s2.commit()
        self.assertEqual(self.path.read_bytes(), other.read_bytes())

    def test_log_bytes_match_manual_delete_then_commit(self):
        s1 = WalStore(self.path)
        s1.set("k", 1)
        s1.commit()
        self.assertEqual(s1.delete_if_version("k", 1), 2)

        other = Path(self.dir.name) / "manual.wal"
        s2 = WalStore(other)
        s2.set("k", 1)
        s2.commit()
        s2.delete("k")
        s2.commit()
        self.assertEqual(self.path.read_bytes(), other.read_bytes())

    def test_conflict_raises_and_touches_nothing(self):
        s = self.build()
        before = self.path.read_bytes()
        # stale versions: a moved to 1, b to 2, "z" never existed
        for bad_call in (
            lambda: s.set_if_version("a", 0, 9),
            lambda: s.set_if_version("a", 2, 9),
            lambda: s.set_if_version("a", 99, 9),
            lambda: s.delete_if_version("a", 0),
            lambda: s.delete_if_version("b", 1),
            lambda: s.set_if_version("z", 1, 9),
            lambda: s.delete_if_version("z", 7),
        ):
            with self.assertRaises(app.WalConflictError):
                bad_call()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(s.recover()["pending_count"], 0)
        # the right versions still go through afterwards
        self.assertEqual(s.set_if_version("a", 1, 9), 3)
        self.assertEqual(s.delete_if_version("b", 2), 4)

    def test_conflict_is_judged_against_the_latest_commit(self):
        s1 = self.build()
        # a second writer commits out of band while s1 stays open
        s2 = WalStore(self.path)
        s2.set("a", 100)
        s2.commit()  # 3: a now at version 3
        with self.assertRaises(app.WalConflictError):
            s1.set_if_version("a", 1, 9)
        self.assertEqual(s1.set_if_version("a", 3, 9), 4)
        self.assertEqual(WalStore(self.path).state["a"], 9)

    def test_pending_records_raise_walpendingerror(self):
        s = self.build()
        s.set("c", 3)  # complete but uncommitted
        before = self.path.read_bytes()
        with self.assertRaises(WalPendingError):
            s.set_if_version("a", 1, 9)
        with self.assertRaises(WalPendingError):
            s.delete_if_version("a", 1)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        s.rollback()
        self.assertEqual(s.set_if_version("a", 1, 9), 3)

    def test_tail_fragment_raises_walpendingerror(self):
        s = self.build()
        s.close()
        self.append_bytes('{"op": "set", "key": "c", "value": 3, "seq": 3')
        s2 = WalStore(self.path)
        before = self.path.read_bytes()
        with self.assertRaises(WalPendingError):
            s2.set_if_version("a", 1, 9)
        with self.assertRaises(WalPendingError):
            s2.delete_if_version("a", 1)
        # the fragment is neither truncated nor appended over
        self.assertEqual(self.path.read_bytes(), before)
        s2.repair_tail()
        self.assertEqual(s2.set_if_version("a", 1, 9), 3)

    def test_corruption_raises_walcorruptionerror(self):
        self.write_lines(
            {"op": "set", "key": "a", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1},
        )
        s = WalStore(self.path)
        self.append_bytes('{"op": "set", "key": "x"}\n')  # invalid record
        with self.assertRaises(WalCorruptionError):
            s.set_if_version("a", 1, 2)
        with self.assertRaises(WalCorruptionError):
            s.delete_if_version("a", 1)

    def test_invalid_expected_seq_raises_valueerror_without_touching_anything(self):
        s = self.build()
        before = self.path.read_bytes()
        for bad in (True, False, -1, -10**9, 1.0, 1.5, "1", b"1", [1],
                    None, {"s": 1}, object()):
            with self.assertRaises(ValueError, msg=bad):
                s.set_if_version("a", bad, 9)
            with self.assertRaises(ValueError, msg=bad):
                s.delete_if_version("a", bad)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(s.recover()["pending_count"], 0)

    def test_invalid_key_or_value_raises_valueerror_without_touching_anything(self):
        s = self.build()
        before = self.path.read_bytes()
        for bad_key in (1, 1.5, None, b"a", ["a"], object()):
            with self.assertRaises(ValueError, msg=bad_key):
                s.set_if_version(bad_key, 1, 9)
            with self.assertRaises(ValueError, msg=bad_key):
                s.delete_if_version(bad_key, 1)
        cyclic = {}
        cyclic["self"] = cyclic
        for bad_value in (float("nan"), float("inf"), object(), cyclic,
                          {1: "non-string key"}):
            with self.assertRaises(ValueError, msg=bad_value):
                s.set_if_version("a", 1, bad_value)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))

    def test_closed_and_readonly_priorities(self):
        s = self.build()
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.set_if_version("a", 1, 9)
        with self.assertRaises(app.WalClosedError):
            s.delete_if_version("a", 1)
        ro = WalStore(self.path, readonly=True)
        with self.assertRaises(app.WalReadOnlyError):
            ro.set_if_version("a", 1, 9)
        with self.assertRaises(app.WalReadOnlyError):
            ro.delete_if_version("a", 1)
        # read-only refusal wins over argument validation
        with self.assertRaises(app.WalReadOnlyError):
            ro.set_if_version(1, "bad", object())
        ro.close()
        with self.assertRaises(app.WalClosedError):
            ro.set_if_version("a", 1, 9)

    def test_value_is_deep_copied_from_caller(self):
        s = WalStore(self.path)
        value = {"n": [1]}
        s.set_if_version("a", 0, value)
        value["n"].append(999)
        value["other"] = True
        self.assertEqual(s.state["a"], {"n": [1]})
        self.assertEqual(WalStore(self.path).state["a"], {"n": [1]})

    def test_oserror_on_commit_leaves_rollbackable_pending_tail(self):
        s = self.build()
        real_fsync = os.fsync
        calls = {"n": 0}

        def flaky(fd):
            calls["n"] += 1
            if calls["n"] == 2:  # the commit record's fsync
                raise OSError("disk on fire")
            return real_fsync(fd)

        with mock.patch("app.os.fsync", side_effect=flaky):
            with self.assertRaises(OSError):
                s.set_if_version("a", 1, 9)
        # the old committed view is still in place
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        # the durable but unsealed set record is observable and removable
        pending = s.pending_changes()
        self.assertEqual(pending["pending_count"], 1)
        self.assertEqual(
            pending["changes"], [{"op": "set", "key": "a", "value": 9}]
        )
        self.assertEqual(s.rollback(), 1)
        self.assertEqual(s.pending_changes()["pending_count"], 0)
        # a process terminated at that boundary recovers the prior commit
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))
        # the transient failure neither consumed a seq nor blocked a retry
        self.assertEqual(s2.set_if_version("a", 1, 9), 3)
        self.assertEqual(s2.state["a"], 9)

    def test_oserror_on_set_record_propagates_with_nothing_pending(self):
        s = self.build()
        before = self.path.read_bytes()
        with mock.patch("app.os.fsync",
                        side_effect=OSError("disk on fire")):
            with self.assertRaises(OSError):
                s.set_if_version("a", 1, 9)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(s.pending_changes()["pending_count"], 0)

    def test_seq_stays_monotonic_across_reopens(self):
        s = WalStore(self.path)
        self.assertEqual(s.set_if_version("a", 0, 1), 1)
        s2 = WalStore(self.path)
        self.assertEqual(s2.set_if_version("a", 1, 2), 2)
        s3 = WalStore(self.path)
        self.assertEqual(s3.delete_if_version("a", 2), 3)
        s4 = WalStore(self.path)
        self.assertEqual((s4.state, s4.commit_seq), ({}, 3))
        self.assertEqual(s4.key_version("a"), 3)
        self.assertEqual([e["commit_seq"] for e in s4.history()], [1, 2, 3])

    def test_exclusive_and_integrity_modes_keep_working(self):
        s = WalStore(self.path, exclusive=True, integrity=True)
        self.assertEqual(s.set_if_version("a", 0, 1), 1)
        self.assertEqual(s.key_version("a"), 1)
        with self.assertRaises(app.WalConflictError):
            s.delete_if_version("a", 0)
        self.assertEqual(s.delete_if_version("a", 1), 2)
        s.close()
        s2 = WalStore(self.path, integrity=True)
        self.assertEqual(s2.key_version("a"), 2)
        self.assertEqual((s2.state, s2.commit_seq), ({}, 2))
        rows = [
            json.loads(line)
            for line in self.path.read_text().splitlines()
        ]
        self.assertTrue(all("ic" in row for row in rows))


class ApplyIfVersionsTest(unittest.TestCase):
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
        s.commit()  # 1: a at version 1
        s.set("b", 2)
        s.commit()  # 2: b at version 2
        return s

    def test_success_and_all_views_agree(self):
        s = self.build()
        new_seq = s.apply_if_versions(
            {"a": 1, "b": 2},
            [
                {"op": "set", "key": "a", "value": {"n": [1, True]}},
                {"op": "delete", "key": "b"},
            ],
        )
        self.assertEqual(new_seq, 3)
        expected_state = {"a": {"n": [1, True]}}
        self.assertEqual((s.state, s.commit_seq), (expected_state, 3))
        self.assertEqual(s.recover()["state"], expected_state)
        self.assertEqual(s.snapshot()["state"], expected_state)
        self.assertEqual(s.snapshot(2)["state"], {"a": 1, "b": 2})
        self.assertEqual(
            s.history()[-1],
            {"commit_seq": 3,
             "changes": [
                 {"op": "set", "key": "a", "value": {"n": [1, True]}},
                 {"op": "delete", "key": "b"},
             ]},
        )
        self.assertEqual(
            s.diff(2, 3)["changes"],
            [{"op": "set", "key": "a", "value": {"n": [1, True]}},
             {"op": "delete", "key": "b"}],
        )
        self.assertEqual(
            s.scan(), [{"key": "a", "value": {"n": [1, True]}}]
        )
        self.assertEqual(s.pending_changes()["changes"], [])
        self.assertEqual(s.key_version("a"), 3)
        self.assertEqual(s.key_version("b"), 3)  # delete seq is kept
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), (expected_state, 3))
        self.assertEqual(s2.key_version("b"), 3)

    def test_same_key_may_appear_multiple_times_in_order(self):
        s = self.build()
        self.assertEqual(
            s.apply_if_versions(
                {"a": 1},
                [
                    {"op": "set", "key": "a", "value": 10},
                    {"op": "delete", "key": "a"},
                    {"op": "set", "key": "a", "value": 30},
                ],
            ),
            3,
        )
        self.assertEqual((s.state, s.commit_seq), ({"a": 30, "b": 2}, 3))
        self.assertEqual(s.key_version("a"), 3)
        self.assertEqual(
            s.history()[-1]["changes"],
            [{"op": "set", "key": "a", "value": 10},
             {"op": "delete", "key": "a"},
             {"op": "set", "key": "a", "value": 30}],
        )

    def test_empty_changes_still_commit_and_check_versions(self):
        s = self.build()
        self.assertEqual(s.apply_if_versions({"a": 1}, []), 3)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 3))
        self.assertEqual(s.history()[-1], {"commit_seq": 3, "changes": []})
        # a version mismatch is still a conflict even with no changes
        # (the empty commit did not touch "a", so its version stays 1)
        with self.assertRaises(app.WalConflictError):
            s.apply_if_versions({"a": 3}, [])
        self.assertEqual(s.commit_seq, 3)

    def test_mapping_may_hold_validation_only_keys(self):
        s = self.build()
        self.assertEqual(
            s.apply_if_versions(
                {"a": 1, "b": 2, "ghost": 0},
                [{"op": "set", "key": "a", "value": 9}],
            ),
            3,
        )
        self.assertEqual(s.state, {"a": 9, "b": 2})
        # a validation-only key that no longer matches blocks the batch
        with self.assertRaises(app.WalConflictError):
            s.apply_if_versions(
                {"a": 3, "ghost": 1},
                [{"op": "set", "key": "a", "value": 10}],
            )
        self.assertEqual((s.state, s.commit_seq), ({"a": 9, "b": 2}, 3))

    def test_deleted_key_version_is_the_delete_seq(self):
        s = self.build()
        s.delete("a")
        s.commit()  # 3: a deleted at version 3
        self.assertEqual(
            s.apply_if_versions(
                {"a": 3}, [{"op": "set", "key": "a", "value": 7}]
            ),
            4,
        )
        self.assertEqual(s.state["a"], 7)
        self.assertEqual(s.key_version("a"), 4)

    def test_conflict_raises_and_touches_nothing(self):
        s = self.build()
        before = self.path.read_bytes()
        with self.assertRaises(app.WalConflictError):
            s.apply_if_versions(
                {"a": 2, "b": 2},
                [{"op": "set", "key": "a", "value": 9}],
            )
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(s.pending_changes()["pending_count"], 0)

    def test_pending_records_raise_walpendingerror(self):
        s = self.build()
        s.set("c", 3)  # complete but uncommitted
        before = self.path.read_bytes()
        with self.assertRaises(WalPendingError):
            s.apply_if_versions(
                {"a": 1}, [{"op": "set", "key": "a", "value": 9}]
            )
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(s.rollback(), 1)

    def test_tail_fragment_raises_walpendingerror(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "c", "value":')
        before = self.path.read_bytes()
        with self.assertRaises(WalPendingError):
            s.apply_if_versions(
                {"a": 1}, [{"op": "set", "key": "a", "value": 9}]
            )
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corruption_raises_walcorruptionerror(self):
        s = self.build()
        self.append_bytes('{"op": "set", "key": "c", "value": 1, "seq": 3}\n'
                          '{"op": "commit", "seq": 4}\n')  # seq break
        with self.assertRaises(WalCorruptionError):
            s.apply_if_versions(
                {"a": 1}, [{"op": "set", "key": "a", "value": 9}]
            )
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))

    def test_invalid_expected_versions_raise_valueerror_first(self):
        s = self.build()
        bad = [
            None,
            [("a", 1)],
            {1: 1},
            {"a": True},
            {"a": -1},
            {"a": "1"},
            {"a": 1.0},
        ]
        for expected in bad:
            with self.assertRaises(ValueError):
                s.apply_if_versions(
                    expected, [{"op": "set", "key": "a", "value": 9}]
                )
        self.assertEqual(WalStore(self.path).commit_seq, 2)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))

    def test_invalid_changes_raise_valueerror_without_touching_log(self):
        bad_changes = [
            "not-a-list",
            [{"op": "set", "key": "a"}],
            [{"op": "set", "key": "a", "value": 1, "extra": 2}],
            [{"op": "delete", "key": "a", "value": 1}],
            [{"op": "commit"}],
            [{"op": "set", "key": 1, "value": 1}],
            [{"op": "set", "key": "a", "value": float("nan")}],
            [{"op": "set", "key": "a", "value": object()}],
            ["not-a-dict"],
        ]
        for changes in bad_changes:
            with self.assertRaises(ValueError):
                WalStore(self.path).apply_if_versions({"a": 0}, changes)
        # nothing was ever created
        self.assertFalse(self.path.exists())

    def test_change_key_missing_from_mapping_raises_valueerror(self):
        s = self.build()
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            s.apply_if_versions(
                {"a": 1},
                [{"op": "set", "key": "a", "value": 9},
                 {"op": "delete", "key": "b"}],
            )
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_closed_and_readonly_priorities(self):
        s = self.build()
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.apply_if_versions(None, None)
        ro = WalStore(self.path, readonly=True)
        with self.assertRaises(app.WalReadOnlyError):
            ro.apply_if_versions(None, None)
        ro.close()

    def test_value_is_deep_copied_from_caller(self):
        s = WalStore(self.path)
        value = {"n": [1]}
        s.apply_if_versions(
            {"a": 0}, [{"op": "set", "key": "a", "value": value}]
        )
        value["n"].append(2)
        self.assertEqual(s.state["a"], {"n": [1]})
        self.assertEqual(WalStore(self.path).state["a"], {"n": [1]})

    def test_log_bytes_match_manual_writes_then_commit(self):
        s1 = WalStore(self.path)
        s1.set("seed", 0)
        s1.commit()
        s1.apply_if_versions(
            {"seed": 1, "k": 0},
            [{"op": "set", "key": "k", "value": {"n": [1, True]}},
             {"op": "delete", "key": "seed"}],
        )

        other = Path(self.dir.name) / "manual.wal"
        s2 = WalStore(other)
        s2.set("seed", 0)
        s2.commit()
        s2.set("k", {"n": [1, True]})
        s2.delete("seed")
        s2.commit()
        self.assertEqual(self.path.read_bytes(), other.read_bytes())

    def test_oserror_on_commit_leaves_rollbackable_pending_tail(self):
        s = self.build()
        real_fsync = os.fsync
        calls = {"n": 0}

        def flaky(fd):
            calls["n"] += 1
            if calls["n"] == 2:  # the commit record's fsync
                raise OSError("disk on fire")
            return real_fsync(fd)

        with mock.patch("app.os.fsync", side_effect=flaky):
            with self.assertRaises(OSError):
                s.apply_if_versions(
                    {"a": 1}, [{"op": "set", "key": "a", "value": 9}]
                )
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        pending = s.pending_changes()
        self.assertEqual(pending["pending_count"], 1)
        self.assertEqual(
            pending["changes"], [{"op": "set", "key": "a", "value": 9}]
        )
        self.assertEqual(s.rollback(), 1)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))
        # the transient failure neither consumed a seq nor blocked a retry
        self.assertEqual(
            s2.apply_if_versions(
                {"a": 1}, [{"op": "set", "key": "a", "value": 9}]
            ),
            3,
        )

    def test_exclusive_and_integrity_modes_keep_working(self):
        s = WalStore(self.path, exclusive=True, integrity=True)
        self.assertEqual(
            s.apply_if_versions(
                {"a": 0}, [{"op": "set", "key": "a", "value": 1}]
            ),
            1,
        )
        with self.assertRaises(app.WalConflictError):
            s.apply_if_versions(
                {"a": 0}, [{"op": "delete", "key": "a"}]
            )
        self.assertEqual(
            s.apply_if_versions({"a": 1}, [{"op": "delete", "key": "a"}]), 2
        )
        s.close()
        s2 = WalStore(self.path, integrity=True)
        self.assertEqual((s2.state, s2.commit_seq), ({}, 2))
        self.assertEqual(s2.key_version("a"), 2)
        rows = [
            json.loads(line)
            for line in self.path.read_text().splitlines()
        ]
        self.assertTrue(all("ic" in row for row in rows))


class CommitIfSeqTest(unittest.TestCase):
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

    def append_bytes(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self.path.open("ab") as f:
            f.write(data)

    def build(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()  # 1
        s.set("b", 2)
        s.commit()  # 2
        return s

    def test_seals_pending_batch_and_all_views_agree(self):
        s = self.build()
        s.set("c", {"n": [1, True]})
        s.delete("a")
        self.assertEqual(s.recover()["pending_count"], 2)
        new_seq = s.commit_if_seq(2)
        self.assertEqual(new_seq, 3)
        expected_state = {"b": 2, "c": {"n": [1, True]}}
        self.assertEqual((s.state, s.commit_seq), (expected_state, 3))
        self.assertEqual(s.recover()["state"], expected_state)
        self.assertEqual(s.recover()["pending_count"], 0)
        self.assertEqual(s.snapshot()["state"], expected_state)
        self.assertEqual(s.snapshot(2)["state"], {"a": 1, "b": 2})
        self.assertEqual(
            s.history()[-1],
            {"commit_seq": 3,
             "changes": [{"op": "set", "key": "c", "value": {"n": [1, True]}},
                         {"op": "delete", "key": "a"}]},
        )
        self.assertEqual(
            s.diff(2, 3)["changes"],
            [{"op": "delete", "key": "a"},
             {"op": "set", "key": "c", "value": {"n": [1, True]}}],
        )
        self.assertEqual(
            s.scan(),
            [{"key": "b", "value": 2},
             {"key": "c", "value": {"n": [1, True]}}],
        )
        self.assertEqual(s.pending_changes()["changes"], [])
        self.assertEqual(s.key_version("a"), 3)
        self.assertEqual(s.key_version("c"), 3)
        self.assertEqual(s.key_version("b"), 2)
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), (expected_state, 3))
        self.assertEqual(s2.key_version("c"), 3)

    def test_empty_commit_when_nothing_pending(self):
        s = self.build()
        self.assertEqual(s.recover()["pending_count"], 0)
        self.assertEqual(s.commit_if_seq(2), 3)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 3))
        self.assertEqual(s.history()[-1], {"commit_seq": 3, "changes": []})
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 3))

    def test_empty_log_seals_from_seq_zero(self):
        s = WalStore(self.path)
        self.assertFalse(self.path.exists())
        s.set("a", 1)
        self.assertEqual(s.commit_if_seq(0), 1)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))

    def test_log_bytes_match_manual_commit(self):
        s1 = WalStore(self.path)
        s1.set("k", 1)
        s1.commit()
        s1.set("x", {"n": [1]})
        s1.delete("k")
        self.assertEqual(s1.commit_if_seq(1), 2)

        other = Path(self.dir.name) / "manual.wal"
        s2 = WalStore(other)
        s2.set("k", 1)
        s2.commit()
        s2.set("x", {"n": [1]})
        s2.delete("k")
        s2.commit()
        self.assertEqual(self.path.read_bytes(), other.read_bytes())

    def test_conflict_raises_and_touches_nothing(self):
        s = self.build()
        s.set("c", 3)  # complete but uncommitted
        before = self.path.read_bytes()
        for bad in (0, 1, 3, 99):
            with self.assertRaises(app.WalConflictError, msg=bad):
                s.commit_if_seq(bad)
        # nothing written, truncated, or consumed; the pending record stays
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        pending = s.pending_changes()
        self.assertEqual(pending["pending_count"], 1)
        self.assertEqual(
            pending["changes"], [{"op": "set", "key": "c", "value": 3}]
        )
        # the right seq still seals the same pending batch afterwards
        self.assertEqual(s.commit_if_seq(2), 3)
        self.assertEqual(s.state, {"a": 1, "b": 2, "c": 3})

    def test_conflict_is_judged_against_the_latest_commit(self):
        s1 = self.build()
        s1.set("c", 3)
        # a second writer commits out of band while s1 stays open
        s2 = WalStore(self.path)
        s2.set("d", 4)
        s2.commit()  # 3
        with self.assertRaises(app.WalConflictError):
            s1.commit_if_seq(2)
        # s1's pending record is still there, now sealed at the new base
        self.assertEqual(s1.commit_if_seq(3), 4)
        self.assertEqual(
            WalStore(self.path).state, {"a": 1, "b": 2, "c": 3, "d": 4}
        )

    def test_tail_fragment_raises_walpendingerror(self):
        s = self.build()
        s.set("c", 3)
        s.close()
        self.append_bytes('{"op": "set", "key": "d", "value": 4, "seq": 3')
        s2 = WalStore(self.path)
        before = self.path.read_bytes()
        with self.assertRaises(WalPendingError):
            s2.commit_if_seq(2)
        # the fragment is neither truncated nor appended over, and the
        # complete pending record survives untouched
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(s2.pending_changes()["pending_count"], 1)
        s2.repair_tail()
        self.assertEqual(s2.commit_if_seq(2), 3)
        self.assertEqual(s2.state, {"a": 1, "b": 2, "c": 3})

    def test_corruption_raises_walcorruptionerror(self):
        bad_tails = (
            b'\xff\xfe\n',                                  # illegal UTF-8
            '{"op": "set", "key": "x", "value": NaN, "seq": 3}\n',
            '{"op": "set", "key": "x", "key": "y", "value": 1, "seq": 3}\n',
            '{"op": "set", "key": "x", "seq": 3}\n',        # missing field
            '{"op": "set", "key": "x", "value": 1, "seq": 9}\n',  # seq break
        )
        for tail in bad_tails:
            with self.subTest(tail=tail):
                self.write_lines(
                    {"op": "set", "key": "a", "value": 1, "seq": 1},
                    {"op": "commit", "seq": 1},
                )
                s = WalStore(self.path)
                self.append_bytes(tail)
                before = self.path.read_bytes()
                with self.assertRaises(WalCorruptionError):
                    s.commit_if_seq(1)
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))

    def test_invalid_expected_seq_raises_valueerror_without_touching_anything(self):
        s = self.build()
        before = self.path.read_bytes()
        for bad in (True, False, -1, -10**9, 1.0, 1.5, "2", b"2", [2],
                    None, {"s": 2}, object()):
            with self.assertRaises(ValueError, msg=bad):
                s.commit_if_seq(bad)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(s.recover()["pending_count"], 0)

    def test_invalid_expected_seq_never_creates_the_log(self):
        s = WalStore(self.path)
        for bad in (True, -1, 0.0, "0"):
            with self.assertRaises(ValueError, msg=bad):
                s.commit_if_seq(bad)
        self.assertFalse(self.path.exists())

    def test_closed_and_readonly_priorities(self):
        s = self.build()
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.commit_if_seq(2)
        ro = WalStore(self.path, readonly=True)
        with self.assertRaises(app.WalReadOnlyError):
            ro.commit_if_seq(2)
        # read-only refusal wins over argument validation
        with self.assertRaises(app.WalReadOnlyError):
            ro.commit_if_seq("bad")
        ro.close()
        with self.assertRaises(app.WalClosedError):
            ro.commit_if_seq(2)

    def test_oserror_on_commit_leaves_rollbackable_pending_tail(self):
        s = self.build()
        s.set("c", 3)
        before_pending = s.pending_changes()["changes"]
        with mock.patch("app.os.fsync",
                        side_effect=OSError("disk on fire")):
            with self.assertRaises(OSError):
                s.commit_if_seq(2)
        # the old committed view is still in place
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        # the durable but unsealed set record is observable and removable
        pending = s.pending_changes()
        self.assertEqual(pending["pending_count"], 1)
        self.assertEqual(pending["changes"], before_pending)
        self.assertEqual(s.rollback(), 1)
        self.assertEqual(s.pending_changes()["pending_count"], 0)
        # a process terminated at that boundary recovers the prior commit
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))
        # the transient failure neither consumed a seq nor blocked a retry
        s2.set("c", 3)
        self.assertEqual(s2.commit_if_seq(2), 3)
        self.assertEqual(s2.state["c"], 3)

    def test_seq_stays_monotonic_across_reopens(self):
        s = WalStore(self.path)
        s.set("a", 1)
        self.assertEqual(s.commit_if_seq(0), 1)
        s2 = WalStore(self.path)
        s2.set("b", 2)
        self.assertEqual(s2.commit_if_seq(1), 2)
        s3 = WalStore(self.path)
        self.assertEqual(s3.commit_if_seq(2), 3)  # empty commit
        s4 = WalStore(self.path)
        self.assertEqual((s4.state, s4.commit_seq), ({"a": 1, "b": 2}, 3))
        self.assertEqual([e["commit_seq"] for e in s4.history()], [1, 2, 3])

    def test_exclusive_and_integrity_modes_keep_working(self):
        s = WalStore(self.path, exclusive=True, integrity=True)
        s.set("a", 1)
        self.assertEqual(s.commit_if_seq(0), 1)
        s.set("b", 2)
        with self.assertRaises(app.WalConflictError):
            s.commit_if_seq(0)
        self.assertEqual(s.commit_if_seq(1), 2)
        s.close()
        s2 = WalStore(self.path, integrity=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2}, 2))
        rows = [
            json.loads(line)
            for line in self.path.read_text().splitlines()
        ]
        self.assertTrue(all("ic" in row for row in rows))


class ApplyIdempotentTest(unittest.TestCase):
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

    def build(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()  # 1: {a:1}
        s.set("b", 2)
        s.commit()  # 2: {a:1, b:2}
        return s

    def test_first_use_commits_like_apply_batch(self):
        changes = [
            {"op": "set", "key": "c", "value": {"n": [1, True]}},
            {"op": "delete", "key": "a"},
        ]
        s = self.build()
        new_seq = s.apply_idempotent("req-1", 2, changes)
        self.assertEqual(new_seq, 3)
        self.assertEqual(s.commit_seq, 3)
        self.assertEqual(s.state, {"b": 2, "c": {"n": [1, True]}})
        self.assertEqual(s.recover()["pending_count"], 0)
        # the commit record carries the request identifier
        last_line = self.path.read_bytes().splitlines()[-1]
        self.assertEqual(
            json.loads(last_line),
            {"op": "commit", "request_id": "req-1", "seq": 3},
        )
        # every committed view matches the equivalent apply_batch run
        other = Path(self.dir.name) / "batch.wal"
        b = WalStore(other)
        b.set("a", 1)
        b.commit()
        b.set("b", 2)
        b.commit()
        self.assertEqual(b.apply_batch(2, changes), 3)
        self.assertEqual(s.state, b.state)
        self.assertEqual(s.history(), b.history())
        self.assertEqual(s.snapshot(), b.snapshot())
        # reopening recovers exactly the batch's committed view
        s2 = WalStore(self.path)
        self.assertEqual(
            (s2.state, s2.commit_seq), ({"b": 2, "c": {"n": [1, True]}}, 3)
        )
        self.assertEqual(s2.history(), b.history())

    def test_retry_returns_original_seq_without_writing(self):
        changes = [{"op": "set", "key": "c", "value": 3}]
        s = self.build()
        self.assertEqual(s.apply_idempotent("req-1", 2, changes), 3)
        bytes_after = self.path.read_bytes()
        # an exact retry appends nothing, changes nothing, consumes no seq
        self.assertEqual(s.apply_idempotent("req-1", 2, changes), 3)
        self.assertEqual(self.path.read_bytes(), bytes_after)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2, "c": 3}, 3))
        # later commits land; the retry still returns the original seq
        self.assertEqual(s.apply_batch(3, [{"op": "set", "key": "d", "value": 4}]), 4)
        self.assertEqual(s.apply_idempotent("req-1", 2, changes), 3)
        self.assertEqual(s.commit_seq, 4)
        self.assertEqual(s.state, {"a": 1, "b": 2, "c": 3, "d": 4})
        # a different identifier is a fresh request, not a retry
        self.assertEqual(s.apply_idempotent("req-2", 4, changes), 5)
        self.assertEqual(s.commit_seq, 5)

    def test_retry_survives_reopen(self):
        changes = [
            {"op": "set", "key": "c", "value": [1, {"x": None}]},
            {"op": "delete", "key": "a"},
        ]
        s = self.build()
        self.assertEqual(s.apply_idempotent("req-1", 2, changes), 3)
        s.close()
        bytes_after = self.path.read_bytes()
        s2 = WalStore(self.path)
        self.assertEqual(s2.apply_idempotent("req-1", 2, changes), 3)
        self.assertEqual(self.path.read_bytes(), bytes_after)
        self.assertEqual(s2.commit_seq, 3)
        s2.close()

    def test_empty_changes_commit_is_idempotent(self):
        s = self.build()
        self.assertEqual(s.apply_idempotent("req-empty", 2, []), 3)
        bytes_after = self.path.read_bytes()
        self.assertEqual(s.apply_idempotent("req-empty", 2, []), 3)
        self.assertEqual(self.path.read_bytes(), bytes_after)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 3))
        self.assertEqual(s.history()[-1], {"commit_seq": 3, "changes": []})

    def test_same_id_different_parameters_conflicts(self):
        s = self.build()
        changes = [
            {"op": "set", "key": "c", "value": 3},
            {"op": "delete", "key": "a"},
        ]
        self.assertEqual(s.apply_idempotent("req-1", 2, changes), 3)
        bytes_after = self.path.read_bytes()
        conflicting = [
            # a different base_seq
            ("req-1", 1, changes),
            ("req-1", 3, changes),
            # a changed value
            ("req-1", 2, [{"op": "set", "key": "c", "value": 4},
                          {"op": "delete", "key": "a"}]),
            # a changed key
            ("req-1", 2, [{"op": "set", "key": "z", "value": 3},
                          {"op": "delete", "key": "a"}]),
            # reordered changes
            ("req-1", 2, [{"op": "delete", "key": "a"},
                          {"op": "set", "key": "c", "value": 3}]),
            # a delete where the set was
            ("req-1", 2, [{"op": "delete", "key": "c"},
                          {"op": "delete", "key": "a"}]),
            # fewer / more changes
            ("req-1", 2, [{"op": "set", "key": "c", "value": 3}]),
            ("req-1", 2, changes + [{"op": "set", "key": "e", "value": 5}]),
        ]
        for request_id, base_seq, retry_changes in conflicting:
            with self.subTest(base_seq=base_seq, changes=retry_changes):
                with self.assertRaises(app.WalConflictError):
                    s.apply_idempotent(request_id, base_seq, retry_changes)
        # no rejection wrote, truncated, consumed a seq, or moved state
        self.assertEqual(self.path.read_bytes(), bytes_after)
        self.assertEqual(
            (s.state, s.commit_seq), ({"b": 2, "c": 3}, 3)
        )

    def test_comparison_distinguishes_json_numbers_and_booleans(self):
        s = WalStore(self.path)
        changes = [{"op": "set", "key": "k", "value": {"n": [1, True]}}]
        self.assertEqual(s.apply_idempotent("req-1", 0, changes), 1)
        # JSON numbers compare numerically across int/float
        self.assertEqual(
            s.apply_idempotent(
                "req-1", 0, [{"op": "set", "key": "k", "value": {"n": [1.0, True]}}]
            ),
            1,
        )
        # ...but a boolean is never a number, nested or not
        with self.assertRaises(app.WalConflictError):
            s.apply_idempotent(
                "req-1", 0, [{"op": "set", "key": "k", "value": {"n": [1, 1]}}]
            )
        with self.assertRaises(app.WalConflictError):
            s.apply_idempotent(
                "req-1", 0, [{"op": "set", "key": "k", "value": {"n": [True, True]}}]
            )
        self.assertEqual(s.commit_seq, 1)

    def test_fresh_id_with_stale_base_conflicts(self):
        s = self.build()
        before = self.path.read_bytes()
        with self.assertRaises(app.WalConflictError):
            s.apply_idempotent("req-1", 1, [])
        with self.assertRaises(app.WalConflictError):
            s.apply_idempotent("req-1", 99, [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        # the identifier stayed unused: the up-to-date base succeeds
        self.assertEqual(s.apply_idempotent("req-1", 2, []), 3)

    def test_argument_validation_precedes_log_access(self):
        missing = Path(self.dir.name) / "missing.wal"
        s = WalStore(missing)
        s.close()  # closed before any argument check
        with self.assertRaises(app.WalClosedError):
            s.apply_idempotent("", "not-an-int", "not-a-list")
        self.assertFalse(missing.exists())
        s = WalStore(missing, readonly=True)
        with self.assertRaises(app.WalReadOnlyError):
            s.apply_idempotent("", "not-an-int", "not-a-list")
        s.close()
        self.assertFalse(missing.exists())
        s = WalStore(missing)
        bad_request_ids = ["", None, 0, 1, b"x", ["x"], {"x": 1}, True]
        for request_id in bad_request_ids:
            with self.subTest(request_id=request_id):
                with self.assertRaises(ValueError):
                    s.apply_idempotent(request_id, 0, [])
        bad_bases = [True, False, -1, "0", 0.5, None]
        for base in bad_bases:
            with self.subTest(base=base):
                with self.assertRaises(ValueError):
                    s.apply_idempotent("req-1", base, [])
        bad_changes = [
            "not-a-list",
            42,
            [{"op": "set", "key": "k"}],
            [{"op": "set", "key": "k", "value": 1, "extra": 2}],
            [{"op": "delete", "key": "k", "value": 1}],
            [{"op": "bogus", "key": "k"}],
            [{"op": "set", "key": 1, "value": 1}],
            [{"op": "set", "key": "k", "value": float("nan")}],
            ["not-a-dict"],
        ]
        for changes in bad_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    s.apply_idempotent("req-1", 0, changes)
        # every rejection happened before the log was created
        self.assertFalse(missing.exists())
        s.close()

    def test_pending_records_block_then_free_the_identifier(self):
        s = self.build()
        s.set("c", 3)  # complete but uncommitted record
        with self.assertRaises(WalPendingError):
            s.apply_idempotent("req-1", 2, [{"op": "set", "key": "d", "value": 4}])
        # even an already-used identifier reports the unsettled log first
        s.rollback()
        self.assertEqual(s.apply_idempotent("req-1", 2, []), 3)
        s.set("c", 3)
        with self.assertRaises(WalPendingError):
            s.apply_idempotent("req-1", 2, [])
        # rolling the pending records back settles the log; the retry then
        # returns the original seq
        self.assertEqual(s.rollback(), 1)
        self.assertEqual(s.apply_idempotent("req-1", 2, []), 3)
        self.assertEqual(s.commit_seq, 3)

    def test_unfinished_tail_blocks_until_repaired(self):
        s = self.build()
        s.close()
        with self.path.open("ab") as f:
            f.write(b'{"op": "set", "key": "c"')  # interrupted write
        s = WalStore(self.path)
        with self.assertRaises(WalPendingError):
            s.apply_idempotent("req-1", 2, [])
        r = s.repair_tail()
        self.assertGreater(r["removed_bytes"], 0)
        self.assertEqual(s.apply_idempotent("req-1", 2, []), 3)
        s.close()

    def test_commit_failure_leaves_identifier_unused(self):
        changes = [{"op": "set", "key": "c", "value": 3}]
        s = WalStore(self.path)
        calls = {"n": 0}
        real_fsync = os.fsync

        def flaky(fd):
            calls["n"] += 1
            if calls["n"] == 3:
                # the change record and the created file's directory entry
                # are durable; the commit record's sync fails
                raise OSError("simulated fsync failure")
            return real_fsync(fd)

        with mock.patch("app.os.fsync", side_effect=flaky):
            with self.assertRaises(OSError):
                s.apply_idempotent("req-1", 0, changes)
        # the commit never became durable: state and seq did not move, the
        # change record is pending, and the identifier is still free
        self.assertEqual((s.state, s.commit_seq), ({}, 0))
        p = s.pending_changes()
        self.assertEqual(p["changes"], changes)
        with self.assertRaises(WalPendingError):
            s.apply_idempotent("req-1", 0, changes)
        self.assertEqual(s.rollback(), 1)
        self.assertEqual(s.apply_idempotent("req-1", 0, changes), 1)
        self.assertEqual(s.state, {"c": 3})
        # the now-durable identifier answers retries with the original seq
        self.assertEqual(s.apply_idempotent("req-1", 0, changes), 1)
        s.close()

    def test_corruption_beats_everything(self):
        s = self.build()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "bogus", "seq": 3}\n')
        before = self.path.read_bytes()
        with self.assertRaises(WalCorruptionError):
            s.apply_idempotent("req-1", 2, [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        s.close()

    def test_request_id_stays_out_of_read_views(self):
        s = WalStore(self.path)
        self.assertEqual(
            s.apply_idempotent("req-1", 0, [{"op": "set", "key": "k", "value": 1}]),
            1,
        )
        self.assertEqual(s.history(), [
            {"commit_seq": 1, "changes": [{"op": "set", "key": "k", "value": 1}]}
        ])
        self.assertEqual(s.snapshot(), {"state": {"k": 1}, "commit_seq": 1})
        self.assertEqual(s.pending_changes()["changes"], [])
        a = s.audit()
        self.assertEqual((a["commit_seq"], a["pending_count"]), (1, 0))
        self.assertEqual(s.scan(), [{"key": "k", "value": 1}])
        s.close()

    def test_caller_changes_are_deep_copied(self):
        s = WalStore(self.path)
        changes = [{"op": "set", "key": "k", "value": {"n": [1]}}]
        self.assertEqual(s.apply_idempotent("req-1", 0, changes), 1)
        changes[0]["value"]["n"].append(2)
        # the stored request still compares against the original values
        with self.assertRaises(app.WalConflictError):
            s.apply_idempotent("req-1", 0, changes)
        self.assertEqual(
            s.apply_idempotent(
                "req-1", 0, [{"op": "set", "key": "k", "value": {"n": [1]}}]
            ),
            1,
        )
        self.assertEqual(s.state, {"k": {"n": [1]}})
        s.close()

    def test_integrity_protected_log(self):
        s = WalStore(self.path, integrity=True)
        changes = [{"op": "set", "key": "k", "value": 1}]
        self.assertEqual(s.apply_idempotent("req-1", 0, changes), 1)
        bytes_after = self.path.read_bytes()
        self.assertEqual(s.apply_idempotent("req-1", 0, changes), 1)
        self.assertEqual(self.path.read_bytes(), bytes_after)
        s.close()
        # the protected chain still verifies on reopen, marker included
        s2 = WalStore(self.path, integrity=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"k": 1}, 1))
        self.assertEqual(s2.apply_idempotent("req-1", 0, changes), 1)
        self.assertEqual(s2.apply_idempotent("req-2", 1, []), 2)
        s2.close()
        for line in self.path.read_bytes().splitlines():
            self.assertIn(b'"ic"', line)

    def test_legacy_and_mixed_logs_keep_working(self):
        # a legacy log (no request_id markers) accepts idempotent commits
        s = self.build()
        self.assertEqual(s.apply_idempotent("req-1", 2, []), 3)
        # plain commits may follow idempotent ones in the same log
        self.assertEqual(s.commit(), 4)
        self.assertEqual(s.apply_idempotent("req-1", 2, []), 3)
        self.assertEqual(s.commit_seq, 4)
        s.close()
        s2 = WalStore(self.path)
        self.assertEqual(s2.commit_seq, 4)
        self.assertEqual(s2.apply_idempotent("req-1", 2, []), 3)
        s2.close()

    def test_malformed_request_id_records_are_corruption(self):
        # a non-string request_id on a commit record
        self.write_lines(
            {"op": "set", "key": "a", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1, "request_id": 5},
        )
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)
        # an empty request_id
        self.write_lines(
            {"op": "set", "key": "a", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1, "request_id": ""},
        )
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)
        # a request_id on a non-commit record breaks the field set
        self.write_lines(
            {"op": "set", "key": "a", "value": 1, "seq": 1, "request_id": "r"},
            {"op": "commit", "seq": 1},
        )
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)
        # the same request_id committed twice can only be tampering
        self.write_lines(
            {"op": "set", "key": "a", "value": 1, "seq": 1},
            {"op": "commit", "seq": 1, "request_id": "r"},
            {"op": "set", "key": "b", "value": 2, "seq": 2},
            {"op": "commit", "seq": 2, "request_id": "r"},
        )
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    def test_strict_tail_store(self):
        s = self.build()
        s.close()
        s = WalStore(self.path, strict_tail=True)
        with self.path.open("ab") as f:
            f.write(b'{"op": "set", "key": "c"')  # out-of-band tail fragment
        with self.assertRaises(app.WalTailError):
            s.apply_idempotent("req-1", 2, [])
        s.repair_tail()
        self.assertEqual(s.apply_idempotent("req-1", 2, []), 3)
        s.close()


# ---------------------------------------------------------------------------
# Crash-consistency acceptance suite.
#
# A child process is genuinely terminated (os._exit, no cleanup) at every
# record/commit boundary of every write entry point, on three log inputs
# (empty file, legacy-format log, integrity-protected log). The parent then
# reopens the log and judges the outcome purely through public return
# values, public exceptions, and the log's byte boundaries (compared against
# a crash-free reference run of the same public API calls).
# ---------------------------------------------------------------------------

# Child-side harness. Parameters arrive as argv: repo dir, log path, and a
# JSON spec {"op": ..., ...op args..., "crash": {...}}. The crash hook
# terminates the process with exit code 3 exactly at the requested boundary:
#   {"kind": "after_record", "n": N}  -- die right after the Nth record's
#     fsync returned (the record is durable, no in-memory adoption yet);
#   {"kind": "mid_record", "record": M, "cut": C} -- die while writing the
#     Mth record: only a prefix of its bytes (C bytes, or all-but-|C| when
#     C < 0) is flushed and fsynced, never the record terminator.
_CRASH_CHILD_SCRIPT = r"""
import json
import os
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
import app

log_path = sys.argv[2]
spec = json.loads(sys.argv[3])
crash = spec.get("crash")

real_fsync = os.fsync
counter = {"n": 0}

if crash and crash["kind"] == "after_record":
    target = crash["n"]

    def fsync_hook(fd):
        real_fsync(fd)
        counter["n"] += 1
        if counter["n"] == target:
            # The record is durable; die before any in-memory adoption,
            # exactly like a kill at this write boundary.
            os._exit(3)

    os.fsync = fsync_hook
elif crash and crash["kind"] == "mid_record":
    target = crash["record"]
    cut = crash["cut"]
    real_open = pathlib.Path.open

    class CrashingWriter:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._wrapped.close()
            return False

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def write(self, data):
            counter["n"] += 1
            if counter["n"] == target:
                # Die mid-record: only a prefix of the record's bytes ever
                # reaches the disk, without the record terminator.
                piece = data[:cut] if cut >= 0 else data[: len(data) + cut]
                self._wrapped.write(piece)
                self._wrapped.flush()
                real_fsync(self._wrapped.fileno())
                os._exit(3)
            return self._wrapped.write(data)

    def open_hook(self, *args, **kwargs):
        f = real_open(self, *args, **kwargs)
        mode = kwargs.get("mode") or (args[0] if args else "r")
        if "a" in mode and str(self) == log_path:
            return CrashingWriter(f)
        return f

    pathlib.Path.open = open_hook

store = app.WalStore(log_path)
op = spec["op"]
if op == "set":
    store.set("k", {"v": [1, 2]})
elif op == "delete":
    store.delete("a")
elif op == "commit":
    store.set("p", 9)
    store.commit()
elif op == "commit_if_seq":
    store.set("p", 9)
    store.commit_if_seq(spec["expected"])
elif op == "restore":
    store.restore(spec["target"])
elif op == "apply_batch":
    store.apply_batch(spec["base"], spec["changes"])
elif op == "apply_idempotent":
    store.apply_idempotent(spec["request_id"], spec["base"], spec["changes"])
elif op == "set_if_version":
    store.set_if_version("k", spec["expected"], 5)
elif op == "delete_if_version":
    store.delete_if_version("a", spec["expected"])
elif op == "apply_if_versions":
    store.apply_if_versions(spec["versions"], spec["changes"])
else:
    raise SystemExit("unknown op: %r" % (op,))
"""


class _CrashTestBase(unittest.TestCase):
    """Shared fixtures: log inputs, operation table, reference byte layout."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.path = self.root / "store.wal"

    def tearDown(self):
        self.dir.cleanup()

    # ----- log inputs --------------------------------------------------------

    def _baseline(self, setup):
        """(commit_seq, state, history) of the log before the operation."""
        if setup == "empty":
            return 0, {}, []
        return 2, {"a": 1, "b": 2}, [
            (1, [{"op": "set", "key": "a", "value": 1}]),
            (2, [{"op": "set", "key": "b", "value": 2}]),
        ]

    def _build_setup(self, path, setup):
        if setup == "empty":
            # An existing but empty log file.
            Path(path).write_bytes(b"")
        elif setup == "legacy":
            s = WalStore(path)
            s.set("a", 1)
            s.commit()
            s.set("b", 2)
            s.commit()
            s.close()
        elif setup == "integrity":
            s = WalStore(path, integrity=True)
            s.set("a", 1)
            s.commit()
            s.set("b", 2)
            s.commit()
            s.close()
        else:
            raise AssertionError(setup)

    # ----- operation table ---------------------------------------------------

    def _op_def(self, setup, op):
        """(child spec, record list, final committed state) for one operation.

        The record list has one ("change", change_dict) entry per set/delete
        record and one ("commit", None) entry for the sealing commit record,
        in log write order.
        """
        seq0, state0, _history0 = self._baseline(setup)
        without_a = {k: v for k, v in state0.items() if k != "a"}
        if op == "set":
            return (
                {"op": "set"},
                [("change", {"op": "set", "key": "k", "value": {"v": [1, 2]}})],
                {**state0, "k": {"v": [1, 2]}},
            )
        if op == "delete":
            return (
                {"op": "delete"},
                [("change", {"op": "delete", "key": "a"})],
                dict(without_a),
            )
        if op == "commit":
            return (
                {"op": "commit"},
                [
                    ("change", {"op": "set", "key": "p", "value": 9}),
                    ("commit", None),
                ],
                {**state0, "p": 9},
            )
        if op == "commit_if_seq":
            return (
                {"op": "commit_if_seq", "expected": seq0},
                [
                    ("change", {"op": "set", "key": "p", "value": 9}),
                    ("commit", None),
                ],
                {**state0, "p": 9},
            )
        if op == "restore":
            if setup == "empty":
                return ({"op": "restore", "target": 0}, [("commit", None)], {})
            return (
                {"op": "restore", "target": 1},
                [("change", {"op": "delete", "key": "b"}), ("commit", None)],
                {"a": 1},
            )
        if op == "apply_batch":
            changes = [
                {"op": "set", "key": "x", "value": 1},
                {"op": "delete", "key": "a"},
            ]
            return (
                {"op": "apply_batch", "base": seq0, "changes": changes},
                [
                    ("change", changes[0]),
                    ("change", changes[1]),
                    ("commit", None),
                ],
                {**without_a, "x": 1},
            )
        if op == "apply_idempotent":
            changes = [
                {"op": "set", "key": "x", "value": 1},
                {"op": "delete", "key": "a"},
            ]
            return (
                {
                    "op": "apply_idempotent",
                    "request_id": "req-crash",
                    "base": seq0,
                    "changes": changes,
                },
                [
                    ("change", changes[0]),
                    ("change", changes[1]),
                    ("commit", None),
                ],
                {**without_a, "x": 1},
            )
        if op == "set_if_version":
            return (
                {"op": "set_if_version", "expected": 0},
                [("change", {"op": "set", "key": "k", "value": 5}), ("commit", None)],
                {**state0, "k": 5},
            )
        if op == "delete_if_version":
            return (
                {
                    "op": "delete_if_version",
                    "expected": 0 if setup == "empty" else 1,
                },
                [("change", {"op": "delete", "key": "a"}), ("commit", None)],
                dict(without_a),
            )
        if op == "apply_if_versions":
            changes = [
                {"op": "set", "key": "x", "value": [3]},
                {"op": "delete", "key": "a"},
            ]
            versions = {"x": 0, "a": 0 if setup == "empty" else 1}
            return (
                {
                    "op": "apply_if_versions",
                    "versions": versions,
                    "changes": changes,
                },
                [
                    ("change", changes[0]),
                    ("change", changes[1]),
                    ("commit", None),
                ],
                {**without_a, "x": [3]},
            )
        raise AssertionError(op)

    def _apply_spec(self, store, spec):
        """In-process twin of the child script's operation dispatch."""
        op = spec["op"]
        if op == "set":
            store.set("k", {"v": [1, 2]})
        elif op == "delete":
            store.delete("a")
        elif op == "commit":
            store.set("p", 9)
            store.commit()
        elif op == "commit_if_seq":
            store.set("p", 9)
            store.commit_if_seq(spec["expected"])
        elif op == "restore":
            store.restore(spec["target"])
        elif op == "apply_batch":
            store.apply_batch(spec["base"], spec["changes"])
        elif op == "apply_idempotent":
            store.apply_idempotent(
                spec["request_id"], spec["base"], spec["changes"]
            )
        elif op == "set_if_version":
            store.set_if_version("k", spec["expected"], 5)
        elif op == "delete_if_version":
            store.delete_if_version("a", spec["expected"])
        elif op == "apply_if_versions":
            store.apply_if_versions(spec["versions"], spec["changes"])
        else:
            raise AssertionError(op)

    # ----- crash points and reference layout ---------------------------------

    def _crash_points(self, records):
        """Every termination boundary of one operation's record sequence."""
        points = [
            ("mid", 1, 5),  # partial JSON prefix of the first record
            ("mid", 1, -1),  # complete first record, terminator never durable
        ]
        points += [("after", n) for n in range(1, len(records) + 1)]
        if len(records) > 1 and records[-1][0] == "commit":
            # The commit record itself interrupted: partial JSON, and a
            # complete commit record whose terminator never became durable.
            points += [("mid", len(records), 5), ("mid", len(records), -1)]
        return points

    def _reference_layout(self, setup, spec, nrecords):
        """Byte layout of a crash-free run of the same public API calls.

        Returns (ref_bytes, setup_bytes, offsets) where offsets[i] is the
        byte offset just past record i's terminator (offsets[0] is the end
        of the pre-operation setup prefix).
        """
        ref = self.root / "reference.wal"
        if ref.exists():
            ref.unlink()
        self._build_setup(ref, setup)
        setup_bytes = ref.read_bytes()
        s = WalStore(ref)
        self._apply_spec(s, spec)
        s.close()
        ref_bytes = ref.read_bytes()
        offsets = [len(setup_bytes)]
        pos = len(setup_bytes)
        for _ in range(nrecords):
            pos = ref_bytes.index(b"\n", pos) + 1
            offsets.append(pos)
        self.assertEqual(pos, len(ref_bytes))
        ref.unlink()
        return ref_bytes, setup_bytes, offsets

    def _run_crash_child(self, path, spec):
        repo = os.path.dirname(os.path.abspath(app.__file__))
        proc = subprocess.run(
            [sys.executable, "-c", _CRASH_CHILD_SCRIPT, repo, str(path), json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            proc.returncode,
            3,
            "child did not terminate at the requested boundary: "
            "rc=%r stdout=%r stderr=%r" % (proc.returncode, proc.stdout, proc.stderr),
        )


class CrashBoundaryTest(_CrashTestBase):
    """Terminate the process at every record/commit boundary of every op."""

    ALL_OPS = (
        "set",
        "delete",
        "commit",
        "commit_if_seq",
        "restore",
        "apply_batch",
        "apply_idempotent",
        "set_if_version",
        "delete_if_version",
        "apply_if_versions",
    )

    def _key_version(self, history, key):
        version = 0
        for seq, changes in history:
            for change in changes:
                if change["key"] == key:
                    version = seq
        return version

    def _verify_views(
        self,
        path,
        setup,
        exp_state,
        exp_seq,
        exp_pending,
        exp_history,
        valid_bytes,
        committed_bytes,
        tail_bytes,
    ):
        """Every read-only entry agrees on the replayed committed view."""
        open_modes = [dict(), dict(readonly=True)]
        if setup == "integrity":
            open_modes.append(dict(integrity=True))
            open_modes.append(dict(integrity=True, readonly=True))
        bytes_before = path.read_bytes()
        self.assertEqual(len(bytes_before), valid_bytes + tail_bytes)
        for mode in open_modes:
            s = WalStore(path, **mode)
            try:
                r = s.recover()
                self.assertEqual(r["state"], exp_state)
                self.assertEqual(r["commit_seq"], exp_seq)
                self.assertEqual(r["pending_count"], len(exp_pending))
                self.assertEqual((s.state, s.commit_seq), (exp_state, exp_seq))
                # the result is an independent deep copy
                for key in r["state"]:
                    r["state"][key] = "mutated"
                self.assertEqual(s.recover()["state"], exp_state)

                snap = s.snapshot()
                self.assertEqual(snap["state"], exp_state)
                self.assertEqual(snap["commit_seq"], exp_seq)
                self.assertEqual(s.snapshot(0)["state"], {})
                self.assertEqual(s.snapshot(exp_seq)["state"], exp_state)

                self.assertEqual(
                    s.history(),
                    [
                        {"commit_seq": seq, "changes": changes}
                        for seq, changes in exp_history
                    ],
                )

                d = s.diff(0, exp_seq)
                self.assertEqual(d["from_seq"], 0)
                self.assertEqual(d["to_seq"], exp_seq)
                self.assertEqual(
                    d["changes"],
                    [
                        {"op": "set", "key": k, "value": v}
                        for k, v in sorted(exp_state.items())
                    ],
                )
                self.assertEqual(s.diff(exp_seq, exp_seq)["changes"], [])

                self.assertEqual(
                    s.scan(),
                    [
                        {"key": k, "value": v}
                        for k, v in sorted(exp_state.items())
                    ],
                )

                p = s.pending_changes()
                self.assertEqual(p["commit_seq"], exp_seq)
                self.assertEqual(p["pending_count"], len(exp_pending))
                self.assertEqual(p["changes"], exp_pending)

                a = s.audit()
                self.assertEqual(a["state"], exp_state)
                self.assertEqual(a["commit_seq"], exp_seq)
                self.assertEqual(a["pending_count"], len(exp_pending))
                self.assertEqual(a["valid_bytes"], valid_bytes)
                self.assertEqual(a["committed_bytes"], committed_bytes)
                self.assertEqual(a["tail_bytes"], tail_bytes)

                for key, value in exp_state.items():
                    self.assertEqual(s.get(key), value)
                    self.assertTrue(s.contains(key))
                self.assertFalse(s.contains("\x00-absent"))
                # only committed batches move a key's version: pending
                # records and a discarded tail fragment are invisible
                for change in exp_pending:
                    self.assertEqual(
                        s.key_version(change["key"]),
                        self._key_version(exp_history, change["key"]),
                    )
                if exp_seq and not exp_pending:
                    for _seq, changes in exp_history[-1:]:
                        for change in changes:
                            self.assertEqual(
                                s.key_version(change["key"]), exp_seq
                            )
            finally:
                s.close()
        # none of the read-only entries created, truncated, or rewrote bytes
        self.assertEqual(path.read_bytes(), bytes_before)

    def _run_crash_scenario(
        self, setup, spec, records, final, ref_bytes, setup_bytes, offsets, point
    ):
        seq0, state0, history0 = self._baseline(setup)
        path = self.root / "crash.wal"
        if path.exists():
            path.unlink()
        self._build_setup(path, setup)
        self.assertEqual(path.read_bytes(), setup_bytes)

        kind = point[0]
        crash_spec = dict(spec)
        if kind == "after":
            crash_spec["crash"] = {"kind": "after_record", "n": point[1]}
            durable = records[: point[1]]
            frag = b""
        else:
            _mid, m, cut = point
            crash_spec["crash"] = {"kind": "mid_record", "record": m, "cut": cut}
            durable = records[: m - 1]
            line = ref_bytes[offsets[m - 1] : offsets[m]]
            frag = line[:cut] if cut >= 0 else line[: len(line) + cut]
        self._run_crash_child(path, crash_spec)

        k = len(durable)
        committed = bool(durable) and durable[-1][0] == "commit"
        # The crashed process left exactly the durable prefix of the
        # reference run's records (plus any unfinished fragment) on disk.
        self.assertEqual(path.read_bytes(), ref_bytes[: offsets[k]] + frag)

        if committed:
            exp_state, exp_seq = final, seq0 + 1
            exp_pending = []
            exp_history = history0 + [
                (seq0 + 1, [c for t, c in records if t == "change"])
            ]
        else:
            # state is exactly the last complete commit; the unfinished
            # commit never advanced the seq
            exp_state, exp_seq = state0, seq0
            exp_pending = [c for t, c in durable if t == "change"]
            exp_history = history0
        valid_bytes = offsets[k]
        if committed:
            committed_bytes = offsets[k]
        else:
            committed_bytes = len(setup_bytes) if setup != "empty" else 0
        self._verify_views(
            path,
            setup,
            exp_state,
            exp_seq,
            exp_pending,
            exp_history,
            valid_bytes,
            committed_bytes,
            len(frag),
        )
        self._verify_follow_up(
            path,
            setup,
            spec,
            records,
            final,
            ref_bytes,
            setup_bytes,
            offsets,
            k,
            committed,
            exp_pending,
        )

    def _verify_follow_up(
        self,
        path,
        setup,
        spec,
        records,
        final,
        ref_bytes,
        setup_bytes,
        offsets,
        k,
        committed,
        exp_pending,
    ):
        """Byte boundaries, pending records, and seqs after the reopen."""
        seq0, state0, _history0 = self._baseline(setup)
        if committed:
            # The commit was durable: the log equals the crash-free run and
            # the next commit keeps climbing without reusing a seq.
            self.assertEqual(path.read_bytes(), ref_bytes)
            s = WalStore(path)
            self.assertEqual(s.commit(), seq0 + 2)
            self.assertEqual(s.state, final)
            s.close()
            s2 = WalStore(path)
            self.assertEqual((s2.state, s2.commit_seq), (final, seq0 + 2))
            s2.close()
            return
        if spec["op"] in ("commit", "commit_if_seq") and k == len(records) - 1:
            # Only the sealing commit was lost: committing after the reopen
            # seals the same batch with the same seq, producing exactly the
            # crash-free run's bytes.
            s = WalStore(path)
            self.assertEqual(s.commit(), seq0 + 1)
            self.assertEqual(s.state, final)
            s.close()
            self.assertEqual(path.read_bytes(), ref_bytes)
            return
        # Rollback removes the pending records (and any tail fragment)
        # in place; the committed prefix is never rewritten.
        s = WalStore(path)
        self.assertEqual(s.rollback(), len(exp_pending))
        self.assertEqual(path.read_bytes(), setup_bytes)
        self.assertEqual((s.state, s.commit_seq), (state0, seq0))
        # The next append continues the seq chain without a skip or reuse.
        s.set("z", 1)
        self.assertEqual(s.commit(), seq0 + 1)
        self.assertEqual(s.state, {**state0, "z": 1})
        s.close()
        # Byte-identical to a clean follow-up run of the same public calls.
        follow = self.root / "follow.wal"
        if follow.exists():
            follow.unlink()
        self._build_setup(follow, setup)
        sf = WalStore(follow)
        sf.set("z", 1)
        sf.commit()
        sf.close()
        self.assertEqual(path.read_bytes(), follow.read_bytes())
        follow.unlink()
        s2 = WalStore(path)
        self.assertEqual((s2.state, s2.commit_seq), ({**state0, "z": 1}, seq0 + 1))
        s2.close()
        # The log format is preserved across the crash and the follow-up:
        # legacy logs never gain integrity fields, protected logs keep them.
        data = path.read_bytes()
        if setup == "legacy":
            self.assertNotIn(b'"ic"', data)
        elif setup == "integrity":
            for line in data.splitlines():
                self.assertIn(b'"ic"', line)

    def _check_op_crash_boundaries(self, op):
        for setup in ("empty", "legacy", "integrity"):
            spec, records, final = self._op_def(setup, op)
            ref_bytes, setup_bytes, offsets = self._reference_layout(
                setup, spec, len(records)
            )
            for point in self._crash_points(records):
                with self.subTest(setup=setup, op=op, point=point):
                    self._run_crash_scenario(
                        setup, spec, records, final, ref_bytes, setup_bytes,
                        offsets, point,
                    )

    def test_set_crash_boundaries(self):
        self._check_op_crash_boundaries("set")

    def test_delete_crash_boundaries(self):
        self._check_op_crash_boundaries("delete")

    def test_commit_crash_boundaries(self):
        self._check_op_crash_boundaries("commit")

    def test_commit_if_seq_crash_boundaries(self):
        self._check_op_crash_boundaries("commit_if_seq")

    def test_restore_crash_boundaries(self):
        self._check_op_crash_boundaries("restore")

    def test_apply_batch_crash_boundaries(self):
        self._check_op_crash_boundaries("apply_batch")

    def test_apply_idempotent_crash_boundaries(self):
        self._check_op_crash_boundaries("apply_idempotent")

    def test_set_if_version_crash_boundaries(self):
        self._check_op_crash_boundaries("set_if_version")

    def test_delete_if_version_crash_boundaries(self):
        self._check_op_crash_boundaries("delete_if_version")

    def test_apply_if_versions_crash_boundaries(self):
        self._check_op_crash_boundaries("apply_if_versions")

    def test_commit_seq_monotonic_across_repeated_crashes(self):
        for setup in ("empty", "legacy", "integrity"):
            with self.subTest(setup=setup):
                path = self.root / "monotonic.wal"
                if path.exists():
                    path.unlink()
                self._build_setup(path, setup)
                seq0, _state0, _h = self._baseline(setup)
                expected = seq0
                # Three rounds of "commit record durable, then die": each
                # reopen recovers exactly one more committed seq.
                for round_no in range(3):
                    spec = {
                        "op": "apply_batch",
                        "base": expected,
                        "changes": [
                            {"op": "set", "key": "r%d" % round_no, "value": round_no}
                        ],
                        "crash": {"kind": "after_record", "n": 2},
                    }
                    self._run_crash_child(path, spec)
                    expected += 1
                    s = WalStore(path)
                    self.assertEqual(s.commit_seq, expected)
                    self.assertEqual(s.recover()["pending_count"], 0)
                    s.close()
                # Die with the commit record complete but unterminated:
                # the seq must not advance and the next commit reuses no
                # committed seq -- it takes exactly expected + 1.
                spec = {
                    "op": "apply_batch",
                    "base": expected,
                    "changes": [{"op": "set", "key": "lost", "value": 1}],
                    "crash": {"kind": "mid_record", "record": 2, "cut": -1},
                }
                self._run_crash_child(path, spec)
                s = WalStore(path)
                self.assertEqual(s.commit_seq, expected)
                p = s.pending_changes()
                self.assertEqual(p["commit_seq"], expected)
                self.assertEqual(
                    p["changes"], [{"op": "set", "key": "lost", "value": 1}]
                )
                self.assertEqual(s.rollback(), 1)
                self.assertEqual(s.commit(), expected + 1)
                seqs = [entry["commit_seq"] for entry in s.history()]
                self.assertEqual(seqs, list(range(1, expected + 2)))
                s.close()


class CrashFsyncFailureTest(_CrashTestBase):
    """An OSError from fsync at any record boundary keeps the old commit."""

    def _run_fsync_failure(self, setup, spec, records, final, ref_bytes,
                           setup_bytes, offsets, fail_at):
        seq0, state0, _history0 = self._baseline(setup)
        path = self.root / "fsyncfail.wal"
        if path.exists():
            path.unlink()
        self._build_setup(path, setup)
        s = WalStore(path)
        calls = {"n": 0}
        real_fsync = os.fsync

        def flaky(fd):
            calls["n"] += 1
            if calls["n"] == fail_at:
                raise OSError("simulated fsync failure")
            return real_fsync(fd)

        with mock.patch("app.os.fsync", side_effect=flaky):
            with self.assertRaises(OSError):
                self._apply_spec(s, spec)
        # The old committed state and seq did not move early.
        self.assertEqual((s.state, s.commit_seq), (state0, seq0))
        # The failed record was truncated back; the durable prefix stays.
        durable = records[: fail_at - 1]
        self.assertEqual(path.read_bytes(), ref_bytes[: offsets[fail_at - 1]])
        p = s.pending_changes()
        self.assertEqual(p["commit_seq"], seq0)
        self.assertEqual(p["changes"], [c for t, c in durable if t == "change"])
        # The durable-but-unsealed prefix follows the existing rules:
        # observable via pending_changes, removable via rollback.
        self.assertEqual(s.rollback(), len(p["changes"]))
        self.assertEqual(path.read_bytes(), setup_bytes)
        # The transient failure neither consumed a seq nor blocked a retry.
        self._apply_spec(s, spec)
        if records[-1][0] != "commit":
            # set/delete write no commit record of their own; seal the
            # retried record so the committed view is observable.
            s.commit()
        self.assertEqual((s.state, s.commit_seq), (final, seq0 + 1))
        s.close()
        mode = {"integrity": True} if setup == "integrity" else {}
        s2 = WalStore(path, **mode)
        self.assertEqual((s2.state, s2.commit_seq), (final, seq0 + 1))
        self.assertEqual(s2.recover()["pending_count"], 0)
        s2.close()

    def test_fsync_failure_at_every_record_boundary(self):
        for setup in ("legacy", "integrity"):
            for op in CrashBoundaryTest.ALL_OPS:
                spec, records, final = self._op_def(setup, op)
                ref_bytes, setup_bytes, offsets = self._reference_layout(
                    setup, spec, len(records)
                )
                for fail_at in range(1, len(records) + 1):
                    with self.subTest(setup=setup, op=op, fail_at=fail_at):
                        self._run_fsync_failure(
                            setup, spec, records, final, ref_bytes,
                            setup_bytes, offsets, fail_at,
                        )


class CrashCorruptionStabilityTest(_CrashTestBase):
    """Corruption after termination: every read-only entry raises, cleanly."""

    def _corruptions(self, setup):
        """(name, corrupt(path)) pairs; seq 3 is the next seq after setup."""
        cases = [
            ("invalid_utf8", lambda p: self._append_bytes(p, b"\xff\xfe\n")),
            (
                "nonstandard_json_constant",
                lambda p: self._append_bytes(
                    p, b'{"op": "set", "key": "x", "value": NaN, "seq": 3}\n'
                ),
            ),
            (
                "duplicate_fields",
                lambda p: self._append_bytes(
                    p,
                    b'{"op": "set", "key": "x", "key": "y", "value": 1, "seq": 3}\n',
                ),
            ),
            (
                "wrong_field_set",
                lambda p: self._append_bytes(
                    p,
                    b'{"op": "set", "key": "x", "value": 1, "seq": 3, "extra": 1}\n',
                ),
            ),
            (
                "missing_field",
                lambda p: self._append_bytes(
                    p, b'{"op": "set", "key": "x", "seq": 3}\n'
                ),
            ),
            (
                "unknown_op",
                lambda p: self._append_bytes(p, b'{"op": "bogus", "seq": 3}\n'),
            ),
            (
                "seq_jump",
                lambda p: self._append_bytes(
                    p, b'{"op": "set", "key": "x", "value": 1, "seq": 5}\n'
                ),
            ),
            ("blank_record", lambda p: self._append_bytes(p, b"\n")),
            ("non_object", lambda p: self._append_bytes(p, b"[1, 2]\n")),
            (
                "seq_tampered_in_place",
                lambda p: self._replace_bytes(
                    p, b'"seq": 1', b'"seq": 2'
                ),
            ),
        ]
        if setup == "integrity":
            cases += [
                (
                    "bad_digest_record",
                    lambda p: self._append_bytes(
                        p,
                        b'{"op": "set", "key": "x", "value": 1, "seq": 3, '
                        b'"ic": "' + b"0" * 64 + b'"}\n',
                    ),
                ),
                (
                    "value_tampered_in_place",
                    lambda p: self._replace_bytes(
                        p, b'"value": 1', b'"value": 2'
                    ),
                ),
                (
                    "digest_tampered_in_place",
                    lambda p: self._tamper_first_digest(p),
                ),
            ]
        return cases

    @staticmethod
    def _tamper_first_digest(path):
        path = Path(path)
        data = path.read_bytes()
        marker = b'"ic": "'
        i = data.index(marker) + len(marker)
        replacement = b"0" if data[i : i + 1] != b"0" else b"1"
        path.write_bytes(data[:i] + replacement + data[i + 1 :])

    @staticmethod
    def _append_bytes(path, data):
        with Path(path).open("ab") as f:
            f.write(data)

    @staticmethod
    def _replace_bytes(path, old, new):
        path = Path(path)
        data = path.read_bytes()
        assert old in data, (old, data)
        path.write_bytes(data.replace(old, new, 1))

    def test_corruption_raises_everywhere_without_side_effects(self):
        for setup in ("legacy", "integrity"):
            seq0, state0, _history0 = self._baseline(setup)
            for name, corrupt in self._corruptions(setup):
                with self.subTest(setup=setup, corruption=name):
                    path = self.root / "corrupt.wal"
                    if path.exists():
                        path.unlink()
                    self._build_setup(path, setup)
                    s = WalStore(path)  # live instance on the good log
                    corrupt(path)
                    corrupted_bytes = path.read_bytes()
                    read_only_calls = [
                        lambda: s.recover(),
                        lambda: s.snapshot(),
                        lambda: s.snapshot(1),
                        lambda: s.history(),
                        lambda: s.diff(0, 1),
                        lambda: s.scan(),
                        lambda: s.pending_changes(),
                        lambda: s.audit(),
                        lambda: s.get("a"),
                        lambda: s.contains("a"),
                        lambda: s.key_version("a"),
                    ]
                    for call in read_only_calls:
                        with self.assertRaises(WalCorruptionError):
                            call()
                    # no partial result was adopted into memory
                    self.assertEqual((s.state, s.commit_seq), (state0, seq0))
                    # the queries never truncated or rewrote the log
                    self.assertEqual(path.read_bytes(), corrupted_bytes)
                    mutating_calls = [
                        lambda: s.set("q", 1),
                        lambda: s.delete("a"),
                        lambda: s.commit(),
                        lambda: s.commit_if_seq(seq0),
                        lambda: s.rollback(),
                        lambda: s.restore(1),
                        lambda: s.apply_batch(seq0, []),
                        lambda: s.apply_idempotent("rid", seq0, []),
                        lambda: s.set_if_version("a", 1, 2),
                        lambda: s.delete_if_version("a", 1),
                        lambda: s.apply_if_versions({}, []),
                        lambda: s.repair_tail(),
                    ]
                    for call in mutating_calls:
                        with self.assertRaises(WalCorruptionError):
                            call()
                    self.assertEqual((s.state, s.commit_seq), (state0, seq0))
                    self.assertEqual(path.read_bytes(), corrupted_bytes)
                    s.close()
                    with self.assertRaises(WalCorruptionError):
                        WalStore(path)
                    self.assertEqual(path.read_bytes(), corrupted_bytes)


class CrashExceptionPriorityTest(_CrashTestBase):
    """Error kinds and their priority on logs a crashed process left behind."""

    def test_pending_batch_blocks_settled_only_entries(self):
        # A complete, terminated but uncommitted batch -- exactly what a
        # crash after the change records but before the commit leaves.
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)
        s.delete("a")
        settled_only = [
            lambda: s.restore(1),
            lambda: s.apply_batch(1, []),
            lambda: s.apply_idempotent("rid", 1, []),
            lambda: s.apply_if_versions({}, []),
            lambda: s.set_if_version("a", 1, 9),
            lambda: s.delete_if_version("a", 1),
        ]
        for call in settled_only:
            with self.assertRaises(WalPendingError):
                call()
        # WalPendingError keeps priority over a stale version/base.
        stale = [
            lambda: s.apply_batch(99, []),
            lambda: s.apply_idempotent("rid", 99, []),
            lambda: s.set_if_version("a", 7, 9),
            lambda: s.delete_if_version("a", 7),
            lambda: s.apply_if_versions({"a": 7}, []),
        ]
        for call in stale:
            with self.assertRaises(WalPendingError):
                call()
        # Nothing was written, truncated, or adopted by the rejections.
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        self.assertEqual(s.recover()["pending_count"], 2)
        # commit and rollback remain available to settle the log.
        self.assertEqual(s.rollback(), 2)
        self.assertEqual(s.set_if_version("a", 1, 9), 2)
        self.assertEqual((s.state, s.commit_seq), ({"a": 9}, 2))
        s.close()

    def test_conflict_on_settled_log(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        before = self.path.read_bytes()
        conflicting = [
            lambda: s.apply_batch(0, [{"op": "set", "key": "b", "value": 2}]),
            lambda: s.apply_idempotent("rid", 0, []),
            lambda: s.set_if_version("a", 7, 9),
            lambda: s.delete_if_version("a", 0),
            lambda: s.apply_if_versions({"a": 0}, [{"op": "delete", "key": "a"}]),
            lambda: s.commit_if_seq(0),
            lambda: s.commit_if_seq(2),
        ]
        for call in conflicting:
            with self.assertRaises(app.WalConflictError):
                call()
        # No rejection wrote, truncated, consumed a seq, or moved state.
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        # The up-to-date preconditions still succeed with the very next seq.
        self.assertEqual(s.delete_if_version("a", 1), 2)
        self.assertEqual((s.state, s.commit_seq), ({}, 2))
        s.close()

    def test_readonly_and_closed_priorities(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)  # pending record stays in the log
        ro = WalStore(self.path, readonly=True)
        mutating = [
            lambda: ro.set("q", 1),
            lambda: ro.delete("a"),
            lambda: ro.commit(),
            lambda: ro.commit_if_seq(1),
            lambda: ro.rollback(),
            lambda: ro.restore(1),
            lambda: ro.apply_batch(1, []),
            lambda: ro.apply_idempotent("rid", 1, []),
            lambda: ro.set_if_version("a", 1, 9),
            lambda: ro.delete_if_version("a", 1),
            lambda: ro.apply_if_versions({}, []),
            lambda: ro.repair_tail(),
        ]
        for call in mutating:
            with self.assertRaises(app.WalReadOnlyError):
                call()
        # The read-only instance still observes the committed view and the
        # pending count, and never took the write lease.
        self.assertEqual((ro.state, ro.commit_seq), ({"a": 1}, 1))
        self.assertEqual(ro.recover()["pending_count"], 1)
        # WalClosedError beats WalReadOnlyError once closed.
        ro.close()
        for call in [lambda: ro.recover(), lambda: ro.set("q", 1)]:
            with self.assertRaises(app.WalClosedError):
                call()
        ro.close()  # close itself stays idempotent
        # The writer is unaffected and settles the log itself.
        self.assertEqual(s.rollback(), 1)
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.recover()
        with self.assertRaises(app.WalClosedError):
            s.commit()
        s.close()

    def test_integrity_open_of_legacy_log_still_rejected(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        before = self.path.read_bytes()
        for mode in (dict(integrity=True), dict(integrity=True, readonly=True)):
            with self.assertRaises(app.WalIntegrityError):
                WalStore(self.path, **mode)
            # nothing created, truncated, appended, or adopted
            self.assertEqual(self.path.read_bytes(), before)
        # The default mode keeps reading and extending the legacy format.
        s = WalStore(self.path)
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
        s.close()
        self.assertNotIn(b'"ic"', self.path.read_bytes())
        # An empty file may still start a protected chain.
        empty = self.root / "empty.wal"
        empty.write_bytes(b"")
        s2 = WalStore(empty, integrity=True)
        s2.set("x", 1)
        self.assertEqual(s2.commit(), 1)
        s2.close()
        self.assertIn(b'"ic"', empty.read_bytes())

    def test_corruption_beats_pending(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)  # complete pending record
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "bogus", "seq": 2}\n')  # terminated corruption
        with self.assertRaises(WalCorruptionError):
            s.apply_batch(1, [])
        with self.assertRaises(WalCorruptionError):
            s.restore(1)
        with self.assertRaises(WalCorruptionError):
            s.recover()
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        s.close()

    def test_exclusive_lease_after_crash(self):
        # A crash left a pending batch behind; an exclusive opener recovers
        # it, holds the lease against a second exclusive opener, and can
        # settle the log.
        spec, records, final = self._op_def("legacy", "commit")
        self._build_setup(self.path, "legacy")
        crash_spec = dict(spec)
        crash_spec["crash"] = {"kind": "after_record", "n": 1}
        self._run_crash_child(self.path, crash_spec)
        s = WalStore(self.path, exclusive=True)
        self.assertEqual((s.state, s.commit_seq), ({"a": 1, "b": 2}, 2))
        self.assertEqual(s.recover()["pending_count"], 1)
        with self.assertRaises(app.WalBusyError):
            WalStore(self.path, exclusive=True)
        # A read-only instance may open alongside the lease holder.
        ro = WalStore(self.path, readonly=True)
        self.assertEqual((ro.state, ro.commit_seq), ({"a": 1, "b": 2}, 2))
        ro.close()
        self.assertEqual(s.commit(), 3)
        self.assertEqual(s.state, {"a": 1, "b": 2, "p": 9})
        s.close()
        s2 = WalStore(self.path, exclusive=True)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1, "b": 2, "p": 9}, 3))
        s2.close()


class CapacityLimitTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "store.wal"

    def tearDown(self):
        self.dir.cleanup()

    def _size(self):
        return self.path.stat().st_size if self.path.exists() else 0

    def _required_bytes(self, integrity, fn):
        """Capture the capacity error's required_bytes for a planned call."""
        s = WalStore(self.path, integrity=integrity, max_bytes=0)
        try:
            with self.assertRaises(WalCapacityError) as ctx:
                fn(s)
            return ctx.exception.required_bytes
        finally:
            s.close()

    def test_max_bytes_validation(self):
        for bad in (True, False, -1, -100, 1.5, "10", object()):
            with self.assertRaises(ValueError):
                WalStore(self.path, max_bytes=bad)
        # A rejected call never creates or touches the log.
        self.assertFalse(self.path.exists())
        # Omitted and None both mean unlimited.
        for kw in ({}, {"max_bytes": None}):
            s = WalStore(self.path, **kw)
            for i in range(5):
                s.set("k%d" % i, "x" * 100)
            s.commit()
            s.close()
            self.path.unlink()

    def test_max_bytes_zero_allows_open_and_queries(self):
        s = WalStore(self.path, max_bytes=0)
        self.assertEqual((s.state, s.commit_seq), ({}, 0))
        self.assertEqual(s.recover()["commit_seq"], 0)
        self.assertFalse(self.path.exists())
        s.close()

    def test_set_rejected_atomically(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        size = self._size()
        s = WalStore(self.path, max_bytes=size)
        before = (dict(s.state), s.commit_seq)
        with self.assertRaises(WalCapacityError) as ctx:
            s.set("b", 2)
        err = ctx.exception
        self.assertEqual(err.limit, size)
        self.assertIsInstance(err.required_bytes, int)
        self.assertGreater(err.required_bytes, size)
        # Nothing changed: file bytes, seq, state, pending count.
        self.assertEqual(self._size(), size)
        self.assertEqual((dict(s.state), s.commit_seq), before)
        self.assertEqual(s.pending_changes()["pending_count"], 0)
        s.close()
        # required_bytes is the exact file length had the write succeeded.
        s = WalStore(self.path)
        s.set("b", 2)
        s.close()
        self.assertEqual(self._size(), err.required_bytes)

    def test_delete_and_commit_rejected(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        size = self._size()
        s = WalStore(self.path, max_bytes=size)
        with self.assertRaises(WalCapacityError):
            s.delete("a")
        with self.assertRaises(WalCapacityError):
            s.commit()
        self.assertEqual(self._size(), size)
        self.assertEqual((dict(s.state), s.commit_seq), ({"a": 1}, 1))
        s.close()

    def test_exact_fit_allowed(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        size = self._size()
        need = self._required_bytes(False, lambda st: st.set("b", 2))
        s = WalStore(self.path, max_bytes=need)
        s.set("b", 2)  # exactly at the limit: allowed
        self.assertEqual(self._size(), need)
        # The commit record needs more room and is refused; the pending
        # record stays and can still be sealed once the limit allows.
        with self.assertRaises(WalCapacityError) as ctx:
            s.commit()
        self.assertEqual(s.pending_changes()["pending_count"], 1)
        s.close()
        s = WalStore(self.path, max_bytes=ctx.exception.required_bytes)
        self.assertEqual(s.commit(), 2)
        self.assertEqual(dict(s.state), {"a": 1, "b": 2})
        self.assertEqual(self._size(), ctx.exception.required_bytes)
        s.close()

    def test_apply_batch_prechecked_as_a_whole(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        size = self._size()
        changes = [
            {"op": "set", "key": "x", "value": "héllo"},
            {"op": "delete", "key": "a"},
            {"op": "set", "key": "y", "value": [1, 2.5, True, None]},
        ]
        need = self._required_bytes(False, lambda st: st.apply_batch(1, changes))
        # Room for a strict prefix of the batch is not enough: nothing is
        # written unless the whole batch plus its commit fits.
        s = WalStore(self.path, max_bytes=need - 1)
        with self.assertRaises(WalCapacityError) as ctx:
            s.apply_batch(1, changes)
        self.assertEqual(ctx.exception.required_bytes, need)
        self.assertEqual(self._size(), size)
        self.assertEqual(s.commit_seq, 1)
        self.assertEqual(s.pending_changes()["pending_count"], 0)
        s.close()
        s = WalStore(self.path, max_bytes=need)
        self.assertEqual(s.apply_batch(1, changes), 2)
        self.assertEqual(self._size(), need)
        self.assertEqual(s.commit_seq, 2)
        s.close()

    def test_empty_batch_commit_still_checked(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        size = self._size()
        s = WalStore(self.path, max_bytes=size)
        with self.assertRaises(WalCapacityError):
            s.apply_batch(1, [])
        self.assertEqual(self._size(), size)
        self.assertEqual(s.commit_seq, 1)
        s.close()

    def test_conditional_entries_rejected(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        size = self._size()
        s = WalStore(self.path, max_bytes=size)
        with self.assertRaises(WalCapacityError):
            s.set_if_version("b", 0, 2)
        with self.assertRaises(WalCapacityError):
            s.delete_if_version("a", 1)
        with self.assertRaises(WalCapacityError):
            s.apply_if_versions({"a": 1}, [{"op": "set", "key": "a", "value": 9}])
        with self.assertRaises(WalCapacityError):
            s.apply_idempotent("req-1", 1, [{"op": "set", "key": "b", "value": 2}])
        with self.assertRaises(WalCapacityError):
            s.restore(0)
        self.assertEqual(self._size(), size)
        self.assertEqual((dict(s.state), s.commit_seq), ({"a": 1}, 1))
        self.assertEqual(s.pending_changes()["pending_count"], 0)
        s.close()

    def test_commit_if_seq_rejected(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.set("b", 2)  # pending record to seal
        s.close()
        size = self._size()
        s = WalStore(self.path, max_bytes=size)
        with self.assertRaises(WalCapacityError) as ctx:
            s.commit_if_seq(1)
        self.assertEqual(ctx.exception.limit, size)
        # The pending record is untouched and can still be sealed later.
        self.assertEqual(s.pending_changes()["pending_count"], 1)
        self.assertEqual(self._size(), size)
        s.close()
        s = WalStore(self.path, max_bytes=ctx.exception.required_bytes)
        self.assertEqual(s.commit_if_seq(1), 2)
        self.assertEqual(dict(s.state), {"a": 1, "b": 2})
        s.close()

    def test_idempotent_retry_not_limited(self):
        s = WalStore(self.path)
        s.apply_idempotent("req-1", 0, [{"op": "set", "key": "k", "value": "v"}])
        s.close()
        # A retry of an already-durable request appends nothing, so the
        # capacity limit never applies to it.
        s = WalStore(self.path, max_bytes=0)
        self.assertEqual(
            s.apply_idempotent("req-1", 0, [{"op": "set", "key": "k", "value": "v"}]),
            1,
        )
        s.close()

    def test_over_limit_log_opens_and_reads(self):
        s = WalStore(self.path)
        s.set("a", "x" * 100)
        s.commit()
        s.close()
        size = self._size()
        for kw in ({"max_bytes": 1}, {"max_bytes": size - 1}):
            s = WalStore(self.path, **kw)
            self.assertEqual(dict(s.state), {"a": "x" * 100})
            self.assertEqual(s.commit_seq, 1)
            self.assertEqual(s.get("a"), "x" * 100)
            self.assertTrue(s.contains("a"))
            self.assertEqual(len(s.scan()), 1)
            self.assertEqual(s.snapshot()["commit_seq"], 1)
            self.assertEqual(len(s.history()), 1)
            self.assertEqual(s.key_version("a"), 1)
            self.assertEqual(s.audit()["valid_bytes"], size)
            v = s.view()
            self.assertEqual(v.get("a"), "x" * 100)
            s.close()
        ro = WalStore(self.path, readonly=True, max_bytes=1)
        self.assertEqual(ro.get("a"), "x" * 100)
        ro.close()

    def test_rollback_and_repair_tail_unrestricted(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        committed_size = self._size()
        # Add a complete pending record and an unfinished tail fragment
        # out of band.
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "b", "value": 2, "seq": 2}\n')
        s = WalStore(self.path, max_bytes=0)
        self.assertEqual(s.rollback(), 1)
        self.assertEqual(self._size(), committed_size)
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "b"')
        frag_size = self._size()
        r = s.repair_tail()
        self.assertEqual(r["removed_bytes"], frag_size - committed_size)
        self.assertEqual(self._size(), committed_size)
        s.close()

    def test_tail_fragment_not_truncated_on_rejection(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        size = self._size()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "b"')  # incomplete JSON prefix
        frag_size = self._size()
        s = WalStore(self.path, max_bytes=size)
        with self.assertRaises(WalCapacityError):
            s.set("c", 3)
        # The recognisable tail fragment survives the capacity refusal.
        self.assertEqual(self._size(), frag_size)
        s.close()
        # A fitting write still discards the fragment exactly as before.
        need = self._required_bytes(False, lambda st: st.set("c", 3))
        s = WalStore(self.path, max_bytes=need)
        s.set("c", 3)
        self.assertEqual(self._size(), need)
        s.close()

    def test_exception_priority_preserved(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        s.close()
        size = self._size()
        # Read-only and closed instances keep their priorities.
        ro = WalStore(self.path, readonly=True, max_bytes=0)
        with self.assertRaises(app.WalReadOnlyError):
            ro.set("b", 2)
        ro.close()
        s = WalStore(self.path, max_bytes=0)
        s.close()
        with self.assertRaises(app.WalClosedError):
            s.set("b", 2)
        # Pending records and conflicts keep priority over capacity.
        s = WalStore(self.path, max_bytes=size)
        s2 = WalStore(self.path)
        s2.set("b", 2)  # complete pending record via a second handle
        s2.close()
        with self.assertRaises(WalPendingError):
            s.apply_batch(1, [{"op": "set", "key": "c", "value": 3}])
        s.rollback()
        with self.assertRaises(app.WalConflictError):
            s.apply_batch(7, [{"op": "set", "key": "c", "value": 3}])
        # Corruption keeps priority over capacity.
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "bogus", "seq": 2}\n')
        with self.assertRaises(WalCorruptionError):
            s.set("c", 3)
        s.close()

    def test_protected_log_required_bytes(self):
        s = WalStore(self.path, integrity=True)
        s.set("a", 1)
        s.commit()
        s.close()
        size = self._size()
        need = self._required_bytes(
            True, lambda st: st.apply_batch(1, [{"op": "set", "key": "b", "value": 2}])
        )
        self.assertGreater(need, size)
        s = WalStore(self.path, max_bytes=need - 1)
        with self.assertRaises(WalCapacityError) as ctx:
            st_changes = [{"op": "set", "key": "b", "value": 2}]
            s.apply_batch(1, st_changes)
        self.assertEqual(ctx.exception.required_bytes, need)
        self.assertEqual(self._size(), size)
        s.close()
        # The prediction is exact even with integrity metadata chained.
        s = WalStore(self.path, integrity=True, max_bytes=need)
        self.assertEqual(s.apply_batch(1, [{"op": "set", "key": "b", "value": 2}]), 2)
        self.assertEqual(self._size(), need)
        s.close()
        # The protected log still verifies cleanly on reopen.
        s = WalStore(self.path, integrity=True)
        self.assertEqual(dict(s.state), {"a": 1, "b": 2})
        s.close()

    def test_legacy_log_compatible(self):
        # A log written by an older version (no max_bytes notion) opens and
        # keeps working; the limit only constrains new appends.
        with self.path.open("w", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "a", "value": 1, "seq": 1}\n')
            f.write('{"op": "commit", "seq": 1}\n')
        size = self._size()
        s = WalStore(self.path, max_bytes=size)
        self.assertEqual(dict(s.state), {"a": 1})
        with self.assertRaises(WalCapacityError):
            s.set("b", 2)
        s.close()
        s = WalStore(self.path)
        s.set("b", 2)
        s.commit()
        self.assertEqual(dict(s.state), {"a": 1, "b": 2})
        s.close()

    def test_exclusive_and_strict_tail_combine(self):
        s = WalStore(self.path, exclusive=True, strict_tail=True)
        s.set("a", 1)
        s.commit()
        s.close()
        size = self._size()
        s = WalStore(self.path, exclusive=True, strict_tail=True, max_bytes=size)
        with self.assertRaises(WalCapacityError):
            s.set("b", 2)
        self.assertEqual(self._size(), size)
        s.close()
        # A recognised tail fragment keeps raising WalTailError (strict
        # policy priority) rather than WalCapacityError.
        s = WalStore(self.path, strict_tail=True, max_bytes=0)
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "b"')
        with self.assertRaises(app.WalTailError):
            s.set("c", 3)
        s.close()


if __name__ == "__main__":
    unittest.main()
