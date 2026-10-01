import copy
import json
from pathlib import Path


class WalCorruptionError(ValueError):
    """Raised when the write-ahead log is truncated or contains an invalid record."""


class RecoverResult(dict):
    """Plain recovery result: supports both result.state and result["state"] access."""
    __getattr__ = dict.__getitem__


_REQUIRED_FIELDS = {
    "set": {"op", "key", "value", "seq"},
    "delete": {"op", "key", "seq"},
    "commit": {"op", "seq"},
}


class WalStore:
    def __init__(self, path):
        self.path = Path(path)
        self.state = {}
        self.commit_seq = 0
        self.recover()

    def _append(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def set(self, key, value):
        self._append({"op": "set", "key": key, "value": value, "seq": self.commit_seq + 1})

    def delete(self, key):
        self._append({"op": "delete", "key": key, "seq": self.commit_seq + 1})

    def commit(self):
        self.commit_seq += 1
        self._append({"op": "commit", "seq": self.commit_seq})
        self.recover()
        return self.commit_seq

    @staticmethod
    def _parse_record(line):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise WalCorruptionError("malformed JSON record") from exc
        if not isinstance(row, dict):
            raise WalCorruptionError("record is not a JSON object")
        op = row.get("op")
        required = _REQUIRED_FIELDS.get(op)
        if required is None or set(row) != required:
            raise WalCorruptionError("unknown op or invalid record fields")
        seq = row["seq"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
            raise WalCorruptionError("seq must be a positive integer")
        if op != "commit":
            try:
                hash(row["key"])
            except TypeError as exc:
                raise WalCorruptionError("record key is not hashable") from exc
        return row

    def recover(self):
        # Everything is built in locals first; self.state/self.commit_seq are
        # only replaced once the whole log has been validated, so corruption
        # never exposes partial results or clobbers the in-memory snapshot.
        state = {}
        commit_seq = 0
        pending = {}
        pending_count = 0
        if self.path.exists():
            try:
                text = self.path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise WalCorruptionError("log is not valid UTF-8") from exc
            for line in text.splitlines():
                row = self._parse_record(line)
                op = row["op"]
                seq = row["seq"]
                if op == "commit":
                    if seq != commit_seq + 1:
                        raise WalCorruptionError("commit seq jumped or repeated")
                    # Applying in pending insertion order is equivalent to record
                    # order when a key is modified several times in one commit.
                    for key, (kind, value) in pending.items():
                        if kind == "delete":
                            state.pop(key, None)
                        else:
                            state[key] = value
                    pending = {}
                    pending_count = 0
                    commit_seq = seq
                else:
                    if seq != commit_seq + 1:
                        raise WalCorruptionError("modification seq does not match commit")
                    if op == "set":
                        pending[row["key"]] = ("set", row["value"])
                    else:
                        pending[row["key"]] = ("delete", None)
                    pending_count += 1
        self.state = state
        self.commit_seq = commit_seq
        return RecoverResult(
            state=copy.deepcopy(state),
            commit_seq=commit_seq,
            pending_count=pending_count,
        )
