import json
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
