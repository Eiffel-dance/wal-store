import copy
import json
import math
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

# bool is an int subclass but remains acceptable (it round-trips as JSON
# true/false).


# Sentinel distinguishing "no default given to get()" from a caller passing
# default=None, which is an ordinary (and storable) value.
_UNSET = object()


def _validate_key(key):
    if not isinstance(key, str):
        raise ValueError("key must be a string, got %r" % (type(key).__name__,))


def _validate_value(root):
    # Iterative walk: deep nesting must raise ValueError rather than crashing
    # with RecursionError, and only true back-references (cycles) are rejected
    # -- containers that merely share a non-cyclic sub-object are allowed.
    stack = [(root, False)]
    on_path = set()
    while stack:
        value, exiting = stack.pop()
        if exiting:
            on_path.discard(id(value))
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("float values must be finite, got %r" % (value,))
            continue
        if isinstance(value, (str, int)) or value is None:
            continue
        if isinstance(value, (dict, list)):
            obj_id = id(value)
            if obj_id in on_path:
                raise ValueError("value contains a circular reference")
            on_path.add(obj_id)
            stack.append((value, True))
            if isinstance(value, dict):
                for k, v in value.items():
                    if not isinstance(k, str):
                        raise ValueError(
                            "object keys must be strings, got %r"
                            % (type(k).__name__,)
                        )
                    stack.append((v, False))
            else:
                for item in value:
                    stack.append((item, False))
            continue
        raise ValueError(
            "value must be composed of JSON-compatible types, got %r"
            % (type(value).__name__,)
        )


def _reject_constant(constant):
    # NaN / Infinity / -Infinity are non-standard JSON constants; a record
    # containing them is corruption, never silently parsed as a float.
    raise WalCorruptionError("non-standard JSON constant %r" % (constant,))


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
        # Validate fully before touching the log: a rejected call must never
        # create, truncate, or append to the file or alter in-memory state.
        _validate_key(key)
        _validate_value(value)
        self._append(
            {"op": "set", "key": key, "value": value, "seq": self.commit_seq + 1}
        )

    def delete(self, key):
        _validate_key(key)
        self._append({"op": "delete", "key": key, "seq": self.commit_seq + 1})

    def commit(self):
        # Persist the commit boundary first and only adopt the new seq/state
        # once it is durable: a failed write must neither consume the seq nor
        # present a committed state.
        seq = self.commit_seq + 1
        self._append({"op": "commit", "seq": seq})
        self.recover()
        return self.commit_seq

    def _committed_view(self):
        # Replay the log purely into local objects, exactly like recovery, but
        # never adopt the result: a query must observe the last durable commit
        # even when newer uncommitted records sit in the tail, and a corrupt
        # record (in the tail or the committed region) raises WalCorruptionError
        # without partially replacing the current in-memory state.
        candidate, committed, _pending = self._replay()
        return candidate, committed

    def get(self, key, default=_UNSET):
        _validate_key(key)
        if default is not _UNSET:
            _validate_value(default)
        state, _committed = self._committed_view()
        if key in state:
            # An independent deep copy: mutating a returned nested dict or list
            # must not affect state, later queries, or a reopened store.
            return copy.deepcopy(state[key])
        if default is not _UNSET:
            # A stored None (key in state) and a missing key never collapse:
            # presence above returns the stored value, even when it is None.
            return copy.deepcopy(default)
        raise KeyError(key)

    def contains(self, key):
        _validate_key(key)
        state, _committed = self._committed_view()
        return key in state

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
                    row = json.loads(line, parse_constant=_reject_constant)
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
                if op != "commit":
                    # Same strict rules as the write entry point: string keys
                    # and strict-JSON values only; non-conforming records in
                    # either the pending tail or the committed region abort the
                    # whole recovery before the candidate state is adopted.
                    try:
                        _validate_key(row["key"])
                        if op == "set":
                            _validate_value(row["value"])
                    except WalCorruptionError:
                        raise
                    except ValueError as exc:
                        raise WalCorruptionError(str(exc)) from exc
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
