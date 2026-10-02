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


# Characters str.splitlines() treats as line boundaries. json.dumps escapes
# every control character inside strings, so a legal record never contains
# any of these raw; the final record is terminated exactly when the log text
# ends with one of them.
_LINE_BOUNDARY_CHARS = frozenset("\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")

_LITERALS = {"t": "true", "f": "false", "n": "null"}


def _scan_json_string(text, i):
    """Scan a JSON string token starting at text[i] == '"'.

    Returns (status, next_index): "complete" when the closing quote was
    found, "prefix" when the input ended inside the token, and "invalid"
    when the token violates the JSON grammar.
    """
    n = len(text)
    j = i + 1
    while True:
        if j >= n:
            return "prefix", j
        c = text[j]
        if c == '"':
            return "complete", j + 1
        if c == "\\":
            j += 1
            if j >= n:
                return "prefix", j
            esc = text[j]
            if esc in '"\\/bfnrt':
                j += 1
            elif esc == "u":
                j += 1
                for _ in range(4):
                    if j >= n:
                        return "prefix", j
                    if text[j] not in _HEX_DIGITS:
                        return "invalid", i
                    j += 1
            else:
                return "invalid", i
        elif ord(c) < 0x20:
            # Control characters must be escaped; json.loads agrees.
            return "invalid", i
        else:
            j += 1


def _scan_json_number(text, i):
    """Scan a JSON number token starting at text[i]; same protocol as
    _scan_json_string."""
    n = len(text)
    j = i
    if j < n and text[j] == "-":
        j += 1
    if j >= n:
        return "prefix", j
    if text[j] == "0":
        j += 1
    elif "1" <= text[j] <= "9":
        while j < n and "0" <= text[j] <= "9":
            j += 1
    else:
        return "invalid", i
    if j < n and text[j] == ".":
        j += 1
        if j >= n:
            return "prefix", j
        if not "0" <= text[j] <= "9":
            return "invalid", i
        while j < n and "0" <= text[j] <= "9":
            j += 1
    if j < n and text[j] in "eE":
        j += 1
        if j < n and text[j] in "+-":
            j += 1
        if j >= n:
            return "prefix", j
        if not "0" <= text[j] <= "9":
            return "invalid", i
        while j < n and "0" <= text[j] <= "9":
            j += 1
    return "complete", j


def _json_prefix_status(text):
    """Classify text against the JSON grammar without building a value.

    Returns "complete" when text is exactly one valid JSON document
    (surrounding whitespace allowed), "prefix" when it is a strict prefix
    of some valid JSON document -- more bytes could still complete it --
    and "invalid" otherwise. Only "prefix" fragments can be the torn tail
    of an interrupted write. Iterative, so deeply nested fragments cannot
    hit the recursion limit.
    """
    n = len(text)
    i = 0
    stack = []  # state to enter once the nested value being parsed completes
    state = "value"
    while True:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            # Ran out of input: a finished top-level value means the text
            # was complete; anything else is a truncated prefix.
            return "complete" if state == "end" else "prefix"
        c = text[i]
        if state == "value":
            if c == "{":
                stack.append("obj_after_value")
                state = "obj_key_or_end"
                i += 1
            elif c == "[":
                stack.append("arr_after_value")
                state = "arr_value_or_end"
                i += 1
            elif c == '"':
                status, i = _scan_json_string(text, i)
                if status != "complete":
                    return status
                state = stack[-1] if stack else "end"
            elif c in _LITERALS:
                word = _LITERALS[c]
                rest = text[i:i + len(word)]
                if rest == word:
                    i += len(word)
                    state = stack[-1] if stack else "end"
                elif word.startswith(rest):
                    return "prefix"
                else:
                    return "invalid"
            elif c == "-" or "0" <= c <= "9":
                status, i = _scan_json_number(text, i)
                if status != "complete":
                    return status
                state = stack[-1] if stack else "end"
            else:
                return "invalid"
        elif state == "arr_value_or_end":
            if c == "]":
                stack.pop()
                i += 1
                state = stack[-1] if stack else "end"
            else:
                state = "value"
        elif state == "obj_key_or_end" or state == "obj_key":
            if state == "obj_key_or_end" and c == "}":
                stack.pop()
                i += 1
                state = stack[-1] if stack else "end"
            elif c == '"':
                status, i = _scan_json_string(text, i)
                if status != "complete":
                    return status
                state = "obj_colon"
            else:
                return "invalid"
        elif state == "obj_colon":
            if c != ":":
                return "invalid"
            i += 1
            state = "value"
        elif state == "obj_after_value":
            if c == "}":
                stack.pop()
                i += 1
                state = stack[-1] if stack else "end"
            elif c == ",":
                i += 1
                state = "obj_key"
            else:
                return "invalid"
        elif state == "arr_after_value":
            if c == "]":
                stack.pop()
                i += 1
                state = stack[-1] if stack else "end"
            elif c == ",":
                i += 1
                state = "value"
            else:
                return "invalid"
        else:  # "end": a complete document followed by more content
            return "invalid"


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
        candidate, committed, _pending, _torn = self._replay()
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
        torn_bytes = 0
        if self.path.exists():
            data = self.path.read_bytes()
            tail = b""
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                # A write interrupted mid-character leaves a truncated
                # multi-byte sequence at the very end of the file; those
                # bytes belong to the torn fragment. Invalid UTF-8 anywhere
                # else -- or any trailing byte sequence that is not a
                # truncation of a valid character -- is corruption.
                if exc.end == len(data) and exc.reason == "unexpected end of data":
                    tail = data[exc.start:]
                    text = data[:exc.start].decode("utf-8")
                else:
                    raise WalCorruptionError("log is not valid UTF-8") from exc
            # splitlines() recognises every Unicode line boundary, while a
            # literal U+2028/U+2029 inside a JSON string is emitted escaped by
            # json.dumps, so a legal record can never be fragmented. It absorbs
            # the single terminator that ends the final record, and an empty
            # file yields no lines at all (empty store); any fragment that
            # remains empty or whitespace-only -- a blank line between records,
            # a bare empty record, or a second terminator at end of file -- is
            # therefore corruption and is rejected up front, before any
            # candidate state can be adopted.
            lines = text.splitlines()
            torn_bytes = len(tail)
            if lines and text[-1] not in _LINE_BOUNDARY_CHARS:
                # The final record is unterminated. A complete record stays
                # readable without its newline; a fragment left by an
                # interrupted write -- valid JSON so far, missing the rest
                # only because the input ended -- is discarded: it is not
                # replayed, not counted as pending, and does not advance the
                # seq. Anything else (a blank fragment, broken JSON, or a
                # parseable record with illegal fields or seq) remains
                # corruption and is rejected by the loop below.
                last = lines[-1]
                if last.strip() and _json_prefix_status(last) == "prefix":
                    torn_bytes += len(last.encode("utf-8"))
                    lines = lines[:-1]
            for line in lines:
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
        return candidate, committed, len(pending), torn_bytes

    def _truncate_tail(self, nbytes):
        """Remove a torn fragment left by an interrupted write.

        Only ever shortens the log: every byte of the legal prefix is left
        untouched, so a later append extends the last complete record
        instead of merging with the fragment and re-consuming it.
        """
        with self.path.open("r+b") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.truncate(max(size - nbytes, 0))
            f.flush()
            os.fsync(f.fileno())

    def recover(self):
        # Replay purely into local objects first; raising WalCorruptionError
        # must never partially replace the current in-memory state.
        candidate, committed, pending_count, torn_bytes = self._replay()
        if torn_bytes:
            # The discarded fragment is dropped physically as well: once
            # replay has classified it, removing exactly those bytes keeps a
            # later append from extending the fragment into a corrupt record.
            self._truncate_tail(torn_bytes)
        self.state = candidate
        self.commit_seq = committed
        return RecoveryResult(
            state=copy.deepcopy(candidate),
            commit_seq=committed,
            pending_count=pending_count,
        )
