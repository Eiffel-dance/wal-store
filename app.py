import copy
import json
import os
from pathlib import Path


class WalCorruptionError(ValueError):
    """Raised when the write-ahead log fails validation."""


class RecoveryResult(dict):
    """Plain result mapping; keys are also readable as attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


_SCHEMAS = {
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
        """Durably append one log record.

        Raises OSError if the record cannot be written or synced; the
        un-durable tail is best-effort truncated back so a failed write can
        never masquerade as a committed record on reopen.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, sort_keys=True) + "\n"
        created = not self.path.exists()
        with self.path.open("a", encoding="utf-8") as f:
            saved_size = f.tell()
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                try:
                    f.truncate(saved_size)
                    f.flush()
                    os.fsync(f.fileno())
                except OSError:
                    pass
                raise
        if created:
            self._fsync_parent_dir()

    def _fsync_parent_dir(self):
        """Best-effort persistence of a freshly created log file's directory entry."""
        try:
            fd = os.open(str(self.path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            # Some filesystems do not support fsync on directories; the log
            # contents themselves have already been synced.
            pass
        finally:
            os.close(fd)

    def set(self, key, value):
        self._append(
            {"op": "set", "key": key, "value": value, "seq": self.commit_seq + 1}
        )

    def delete(self, key):
        self._append({"op": "delete", "key": key, "seq": self.commit_seq + 1})

    def commit(self):
        # Persist the commit boundary first and only adopt the new seq/state
        # once it is durable: a failed write must neither consume the seq nor
        # present a committed state.
        seq = self.commit_seq + 1
        self._append({"op": "commit", "seq": seq})
        self.recover()
        return self.commit_seq

    def _replay(self):
        candidate = {}
        pending = []
        committed = 0
        if self.path.exists():
            try:
                text = self.path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise WalCorruptionError("log is not valid UTF-8") from exc
            for line in text.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise WalCorruptionError("invalid JSON record: %r" % line) from exc
                if not isinstance(row, dict):
                    raise WalCorruptionError("record is not an object: %r" % (row,))
                op = row.get("op")
                if not isinstance(op, str):
                    raise WalCorruptionError("op is not a string: %r" % (op,))
                schema = _SCHEMAS.get(op)
                if schema is None:
                    raise WalCorruptionError("unknown op: %r" % (op,))
                if set(row) != schema:
                    raise WalCorruptionError(
                        "bad fields for op %r: %r" % (op, sorted(row))
                    )
                seq = row["seq"]
                if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
                    raise WalCorruptionError("seq is not a positive integer: %r" % (seq,))
                if seq != committed + 1:
                    raise WalCorruptionError(
                        "seq %r does not follow committed seq %r" % (seq, committed)
                    )
                if op == "commit":
                    # Apply this batch in original set/delete order.
                    for p in pending:
                        if p["op"] == "delete":
                            candidate.pop(p["key"], None)
                        else:
                            candidate[p["key"]] = p["value"]
                    pending = []
                    committed = seq
                else:
                    pending.append(row)
        return candidate, committed, len(pending)

    def recover(self):
        # Replay purely into local objects first; raising WalCorruptionError
        # must never partially replace the current in-memory state.
        candidate, committed, pending_count = self._replay()
        self.state = candidate
        self.commit_seq = committed
        return RecoveryResult(
            state=copy.deepcopy(candidate),
            commit_seq=committed,
            pending_count=pending_count,
        )
