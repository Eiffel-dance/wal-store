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


class ValidationTest(unittest.TestCase):
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

    def test_empty_string_key_is_allowed(self):
        s = WalStore(self.path)
        s.set("", 1)
        s.delete("")
        s.set("", {"": [""]})
        self.assertEqual(s.commit(), 1)
        self.assertEqual(s.state, {"": {"": [""]}})

    def test_non_string_keys_rejected(self):
        s = WalStore(self.path)
        for bad_key in (1, 1.5, None, True, b"a", ("a",), ["a"], {"a": 1}):
            with self.assertRaises(ValueError, msg=bad_key):
                s.set(bad_key, 1)
            with self.assertRaises(ValueError, msg=bad_key):
                s.delete(bad_key)
        self.assertFalse(self.path.exists())  # log never created
        self.assertEqual((s.state, s.commit_seq), ({}, 0))

    def test_non_json_values_rejected(self):
        s = WalStore(self.path)

        class Custom:
            pass

        bad_values = [
            float("nan"),
            float("inf"),
            float("-inf"),
            {"nested": float("nan")},
            [1, float("inf")],
            (1, 2),
            {"t": (1,)},
            {1, 2},
            frozenset([1]),
            b"bytes",
            Custom(),
            {"obj": Custom()},
            {1: "non-string key"},
            {"ok": {2: "nested non-string key"}},
        ]
        for bad in bad_values:
            with self.assertRaises(ValueError, msg=bad):
                s.set("k", bad)
        self.assertFalse(self.path.exists())
        self.assertEqual((s.state, s.commit_seq), ({}, 0))

    def test_circular_reference_rejected(self):
        s = WalStore(self.path)
        cyc_list = [1]
        cyc_list.append(cyc_list)
        cyc_dict = {}
        cyc_dict["self"] = cyc_dict
        for bad in (cyc_list, cyc_dict, {"wrap": cyc_list}):
            with self.assertRaises(ValueError):
                s.set("k", bad)
        self.assertFalse(self.path.exists())

    def test_valid_nested_values_accepted(self):
        s = WalStore(self.path)
        value = {"a": [1, 2.5, True, False, None, "x"], "b": {"c": []}}
        s.set("k", value)
        s.commit()
        self.assertEqual(s.state, {"k": value})
        self.assertEqual(WalStore(self.path).state, {"k": value})

    def test_failed_validation_preserves_prior_state(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.assertRaises(ValueError):
            s.set("b", float("nan"))
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))
        # log content unchanged: no partial record appended
        s2 = WalStore(self.path)
        self.assertEqual((s2.state, s2.commit_seq), ({"a": 1}, 1))
        self.assertEqual(s2.recover()["pending_count"], 0)

    def test_recovery_rejects_non_string_key(self):
        self.write_lines(
            '{"op": "set", "key": 5, "value": 1, "seq": 1}',
            '{"op": "commit", "seq": 1}',
        )
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)
        self.write_lines(
            '{"op": "delete", "key": null, "seq": 1}',
            '{"op": "commit", "seq": 1}',
        )
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

    def test_recovery_rejects_non_finite_and_bad_constants(self):
        for line in (
            '{"op": "set", "key": "a", "value": NaN, "seq": 1}',
            '{"op": "set", "key": "a", "value": Infinity, "seq": 1}',
            '{"op": "set", "key": "a", "value": -Infinity, "seq": 1}',
            '{"op": "set", "key": "a", "value": [1, NaN], "seq": 1}',
        ):
            self.write_lines(line, '{"op": "commit", "seq": 1}')
            with self.assertRaises(WalCorruptionError, msg=line):
                WalStore(self.path)

    def test_recovery_corruption_does_not_touch_live_state(self):
        s = WalStore(self.path)
        s.set("a", 1)
        s.commit()
        with self.path.open("a", encoding="utf-8") as f:
            f.write('{"op": "set", "key": "b", "value": NaN, "seq": 2}\n')
        with self.assertRaises(WalCorruptionError):
            s.recover()
        self.assertEqual((s.state, s.commit_seq), ({"a": 1}, 1))


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
            f.write('{"op": "set", "key": "x"')  # truncated pending record
        with self.assertRaises(WalCorruptionError):
            WalStore(self.path)

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


if __name__ == "__main__":
    unittest.main()
