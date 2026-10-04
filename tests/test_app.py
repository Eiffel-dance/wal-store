import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app
from app import WalCorruptionError, WalPendingError, WalStore


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
        # holder keeps its own lease, state, seq, and file bytes
        self.assertEqual(s.state, {"a": 1})
        self.assertEqual(s.commit_seq, 1)
        s.set("b", 2)
        self.assertEqual(s.commit(), 2)
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


if __name__ == "__main__":
    unittest.main()
