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

# Line boundaries recognised by str.splitlines(); a record write always ends
# with exactly one of them (json.dumps emits the record, then "\n").
_LINE_BOUNDARIES = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")

# Proper prefixes of the JSON literals a record value can contain.
_LITERAL_PREFIXES = ("true", "false", "null")

# Remainders after an already-parsed integer part that can still grow into a
# valid JSON number when more bytes arrive (e.g. "1." -> "1.5", "1e" -> "1e5").
_NUMBER_TAILS = frozenset({".", "e", "E", "e+", "e-", "E+", "E-"})


def _is_incomplete_record(exc, line):
    """Whether a JSON parse failure is explained solely by end of input.

    A record whose write was interrupted is a strict prefix of a valid JSON
    text: the decoder only fails because the bytes stop. Genuine corruption
    (unrecognised tokens, misplaced characters, trailing data, bad escapes)
    never qualifies, so it cannot be mistaken for a discardable fragment.
    """
    rest = line[exc.pos:]
    msg = exc.msg
    if msg == "Unterminated string starting at":
        # The string scan ran into the end of the input.
        return True
    if msg == "Invalid \\uXXXX escape":
        # A \uXXXX escape cut short by end of input; one containing a non-hex
        # digit could never come out of a valid write and stays corruption.
        digits = rest[2:]
        return len(digits) < 4 and all(c in _HEX_DIGITS for c in digits)
    if msg == "Expecting value":
        if not rest:
            return True
        if any(lit.startswith(rest) for lit in _LITERAL_PREFIXES):
            return True
        # A lone minus sign is the start of a number.
        return rest == "-"
    if msg in (
        "Expecting ',' delimiter",
        "Expecting ':' delimiter",
        "Expecting property name enclosed in double quotes",
    ):
        if not rest:
            return True
        # "1." / "1e" / "1e+" etc.: the decoder stopped inside a number.
        return msg == "Expecting ',' delimiter" and rest in _NUMBER_TAILS
    return False


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


def _reject_duplicate_keys(pairs):
    # A JSON object with repeated member names is corruption even when the
    # repeated values are identical; the default last-wins behaviour would
    # silently mutate replay semantics. The hook runs for every object,
    # including nested ones, while the surrounding record is being parsed,
    # i.e. before any candidate state can be adopted.
    seen = set()
    for key, _value in pairs:
        if key in seen:
            raise WalCorruptionError("duplicate field name in record: %r" % (key,))
        seen.add(key)
    return dict(pairs)


class WalStore:
    def __init__(self, path):
        self.path = Path(path)
        self.state = {}
        self.commit_seq = 0
        # Byte length of the durable log prefix as judged by the latest
        # replay; anything beyond it is a discarded tail fragment.
        self._valid_size = None
        self.recover()

    def _drop_tail_fragment(self):
        """Remove a discarded tail fragment left by an interrupted write.

        Only bytes beyond the last complete record (as judged by the latest
        replay) are removed, so a later append can never re-consume the
        fragment as part of a new record; the durable prefix is never
        rewritten.
        """
        if self._valid_size is None or not self.path.exists():
            return
        if self.path.stat().st_size <= self._valid_size:
            return
        with self.path.open("r+b") as f:
            f.truncate(self._valid_size)
            f.flush()
            os.fsync(f.fileno())

    def _append(self, row):
        """Durably append one log record.

        Raises OSError if the record cannot be written or synced; the
        un-durable tail is best-effort truncated back so a failed write can
        never masquerade as a committed record on reopen.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, sort_keys=True) + "\n"
        self._drop_tail_fragment()
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
            # The appended record is complete and durable, so the durable
            # prefix now extends to the new end of the file.
            self._valid_size = f.tell()
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

    def rollback(self):
        # Validate the log by the exact recovery rules before touching
        # anything. The replay builds only local objects, so corruption raises
        # WalCorruptionError without partially replacing state or commit_seq.
        # committed_size is the byte offset just past the last commit record;
        # every complete set/delete beyond it is the uncommitted batch, and any
        # bytes beyond valid_size are the discardable interrupted-write tail.
        (
            _candidate,
            _committed,
            pending_count,
            _valid_size,
            committed_size,
        ) = self._replay()
        if not self.path.exists() or self.path.stat().st_size <= committed_size:
            # Nothing sits beyond the committed prefix (pending_count is then
            # necessarily 0); no truncation is needed.
            self._valid_size = committed_size
            return pending_count
        # Shrink the log in place: bytes of the committed prefix are never
        # rewritten or reordered, only the tail beyond the last commit is
        # removed. Persist before returning, and let an OSError from open,
        # truncate, flush, or fsync propagate -- a failed truncation must
        # never masquerade as a successful rollback. state and commit_seq are
        # not touched: they already equal the last committed view.
        with self.path.open("r+b") as f:
            f.truncate(committed_size)
            f.flush()
            os.fsync(f.fileno())
        self._valid_size = committed_size
        return pending_count

    def _committed_view(self):
        # Replay the log purely into local objects, exactly like recovery, but
        # never adopt the result: a query must observe the last durable commit
        # even when newer uncommitted records sit in the tail, and a corrupt
        # record (in the tail or the committed region) raises WalCorruptionError
        # without partially replacing the current in-memory state.
        candidate, committed, _pending, _valid_size, _committed_size = self._replay()
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

    def snapshot(self, target_seq=None):
        """Read-only view of the committed state at a past commit boundary.

        The whole log is parsed under the exact recovery rules first -- any
        corruption anywhere in it raises WalCorruptionError, never a partial
        snapshot -- and only batches up to the target commit are replayed
        into the result. Records committed after the target and the
        uncommitted tail never appear in it. target_seq defaults to the
        latest committed seq; 0 yields the empty state. The log, the public
        state, and commit_seq are never touched.
        """
        history = []
        _candidate, committed, _pending, _valid_size, _committed_size = self._replay(
            snapshots=history
        )
        if target_seq is None:
            target_seq = committed
        elif (
            isinstance(target_seq, bool)
            or not isinstance(target_seq, int)
            or target_seq < 0
        ):
            raise ValueError(
                "target_seq must be a non-negative integer, got %r" % (target_seq,)
            )
        if target_seq > committed:
            raise ValueError(
                "target_seq %r exceeds latest committed seq %r"
                % (target_seq, committed)
            )
        if target_seq == 0:
            state = {}
        else:
            # Commit seqs are contiguous from 1, so the target's view is
            # exactly the entry recorded at that commit boundary. It is
            # already a deep copy private to this call, so the caller may
            # mutate the result freely without affecting the store.
            state = dict(history)[target_seq]
        return RecoveryResult(state=state, commit_seq=target_seq)

    def _replay(self, snapshots=None):
        # snapshots: optional caller-provided list; when given, one
        # (seq, deep-copy-of-state) entry per durable commit record is
        # appended, in commit order, so historical committed views can be
        # served without re-parsing the log. Purely observational: the
        # replay itself, its return value, and the log are unaffected.
        candidate = {}
        pending = []
        committed = 0
        valid_size = 0
        # Byte offset just past the last durable commit record (0 when there
        # is no committed prefix); rollback truncates at exactly this point.
        committed_size = 0
        if self.path.exists():
            data = self.path.read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                # A write interrupted mid-character leaves a truncated UTF-8
                # sequence at the very end of the file; only that tail is
                # discarded. Invalid bytes anywhere else are corruption.
                if exc.reason == "unexpected end of data" and exc.end == len(data):
                    text = data[: exc.start].decode("utf-8")
                else:
                    raise WalCorruptionError("log is not valid UTF-8") from exc
            # splitlines() recognises every Unicode line boundary, while a
            # literal U+2028/U+2029 inside a JSON string is emitted escaped by
            # json.dumps, so a legal record can never be fragmented. An empty
            # file yields no segments at all (empty store). Only the final
            # segment can lack a terminator, and the terminator is the
            # boundary that makes a record adoptable: a complete record write
            # always ends with its newline, so an unterminated final segment
            # is the unfinished tail of an interrupted write -- even when its
            # bytes happen to parse as a complete, fully valid record.
            segments = text.splitlines(keepends=True)
            last = len(segments) - 1
            for i, seg in enumerate(segments):
                if seg.endswith("\r\n"):
                    line, terminated = seg[:-2], True
                elif seg[-1:] in _LINE_BOUNDARIES:
                    line, terminated = seg[:-1], True
                else:
                    line, terminated = seg, False
                # Any fragment that remains empty or whitespace-only -- a
                # blank line between records, a bare empty record, trailing
                # whitespace, or a second terminator at end of file -- is
                # corruption and is rejected up front, before any candidate
                # state can be adopted.
                if not line.strip():
                    raise WalCorruptionError("empty record in log")
                try:
                    row = json.loads(
                        line,
                        parse_constant=_reject_constant,
                        object_pairs_hook=_reject_duplicate_keys,
                    )
                except WalCorruptionError:
                    raise
                except (json.JSONDecodeError, TypeError) as exc:
                    if (
                        i == last
                        and not terminated
                        and _is_incomplete_record(exc, line)
                    ):
                        # Interrupted write: the partial trailing record is
                        # discarded -- it is not replayed, not counted as
                        # pending, and its bytes stay behind valid_size.
                        break
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
                if not terminated:
                    # The record's terminator never became durable, so the
                    # write is unfinished: the fragment is discarded whole --
                    # not applied, not counted as pending, and its bytes stay
                    # beyond valid_size for a later append or rollback to
                    # remove. Validation above has already run, so only a
                    # fragment that is recognisably one complete record
                    # reaches this point; anything else raised already.
                    break
                if op == "commit":
                    # Apply this batch in original set/delete order.
                    for p in pending:
                        if p["op"] == "delete":
                            candidate.pop(p["key"], None)
                        else:
                            candidate[p["key"]] = p["value"]
                    pending = []
                    committed = seq
                    if snapshots is not None:
                        # An independent deep copy taken at the commit
                        # boundary; later batches must never mutate it.
                        snapshots.append((seq, copy.deepcopy(candidate)))
                else:
                    pending.append(row)
                # The segment round-trips to its original bytes, so this is
                # the exact byte offset just past the record's terminator.
                valid_size += len(seg.encode("utf-8"))
                if op == "commit":
                    committed_size = valid_size
        self._valid_size = valid_size
        return candidate, committed, len(pending), valid_size, committed_size

    def recover(self):
        # Replay purely into local objects first; raising WalCorruptionError
        # must never partially replace the current in-memory state.
        candidate, committed, pending_count, _valid_size, _committed_size = self._replay()
        self.state = candidate
        self.commit_seq = committed
        return RecoveryResult(
            state=copy.deepcopy(candidate),
            commit_seq=committed,
            pending_count=pending_count,
        )
