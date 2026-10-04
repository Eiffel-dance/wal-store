import copy
import errno
import fcntl
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


class WalCorruptionError(ValueError):
    """Raised when the write-ahead log fails validation."""


class WalBusyError(Exception):
    """Raised when an exclusive write lease is already held for the log path."""


class WalClosedError(Exception):
    """Raised when any public method is called on a closed WalStore."""


class WalReadOnlyError(Exception):
    """Raised when a mutating method is called on a read-only WalStore."""


class WalPendingError(Exception):
    """Raised by restore()/apply_batch() when the log is not settled.

    Either complete, terminated set/delete records wait in an uncommitted
    batch, or an interrupted write leaves an unfinished tail fragment.
    Neither entry splices a new batch into either state; the caller must
    commit or roll the pending records back (or repair the fragment) first.
    """


class WalConflictError(Exception):
    """Raised by apply_batch() when the declared base seq is stale.

    The caller-declared base commit seq does not match the log's latest
    committed seq, so the batch's compare-and-swap precondition fails.
    Nothing is written, truncated, or adopted; the caller is expected to
    re-read the current state and retry with the up-to-date base.
    """


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


def _json_equal(a, b):
    """Deep equality that keeps JSON's number/boolean distinction.

    Python counts True as 1, so a plain == would judge the JSON values
    1 and true (and 1.0 and true) the same; restore must treat them as
    different values. The rule is applied recursively so a boolean
    nested in an object or array still forces a set record. Every other
    pair follows ordinary equality, under which the JSON number family
    (int and float) compares numerically.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_json_equal(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _json_equal(x, y) for x, y in zip(a, b)
        )
    return a == b


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


# Side-channel directory for the identity-based lease locks. It never holds
# WAL bytes: per-real-object locks are named by device and inode, and the
# concurrent-first-open interlock locks by the canonical target path. It is
# shared by every spelling (symlink, hard link, or normalized alias) of a log.
_LOCK_DIR = os.path.join(tempfile.gettempdir(), "walstore_locks")


class WalStore:
    def __init__(self, path, exclusive=False, readonly=False):
        # Validate every parameter before touching the path in any way: a
        # rejected call must never read, create, truncate, or append to the
        # log or its lock file.
        if not isinstance(exclusive, bool):
            raise ValueError(
                "exclusive must be a bool, got %r" % (type(exclusive).__name__,)
            )
        if not isinstance(readonly, bool):
            raise ValueError(
                "readonly must be a bool, got %r" % (type(readonly).__name__,)
            )
        if exclusive and readonly:
            raise ValueError(
                "exclusive and readonly cannot both be True: a read-only "
                "store never takes the write lease"
            )
        self.path = Path(path)
        self.state = {}
        self.commit_seq = 0
        # Byte length of the durable log prefix as judged by the latest
        # replay; anything beyond it is a discarded tail fragment.
        self._valid_size = None
        self._closed = False
        self._readonly = readonly
        # Legacy per-normalized-path lease descriptor (kept for the exact
        # historical sibling-lock behaviour) plus the real-object identity
        # lease descriptors:
        #   _identity_fd  flock on a file named by the log's (dev, ino)
        #   _gate_fd      serializes concurrent first creation of a missing log
        #   _pin_fd       open descriptor on the leased object, anchoring its
        #                 identity for the lifetime of the lease
        # _leased_dev/_leased_ino record the one real object under lease.
        self._lock_fd = None
        self._identity_fd = None
        self._gate_fd = None
        self._pin_fd = None
        self._leased_dev = None
        self._leased_ino = None
        if exclusive:
            # The lease is taken before the log is ever read: a conflicting
            # open raises WalBusyError without creating, truncating, or
            # appending to the log. A read-only instance never takes it and
            # may open a path leased by a live exclusive writer.
            self._acquire_lease()
        self.recover()

    @staticmethod
    def _busy_error(path):
        return WalBusyError(
            "log path is exclusively leased: %r" % (str(path),)
        )

    def _take_flock(self, lock_path):
        """Open ``lock_path`` and take a non-blocking exclusive flock.

        Returns the held descriptor. A lock already held by another open
        file description maps to WalBusyError; every other filesystem
        failure (including an unresolvable lock location) stays an OSError.
        """
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise self._busy_error(self.path) from None
            raise
        return fd

    def _identity_stat(self):
        """Identity of the log's real object, following every symlink.

        Returns (st_dev, st_ino), or None when no object (not even a
        dangling symlink) sits at the path. Hard links to one inode resolve
        identically; distinct symlink chains reaching one file resolve
        identically. Only metadata is read, never WAL content. Any other
        stat failure propagates as OSError.
        """
        try:
            st = os.stat(os.fspath(self.path))
        except FileNotFoundError:
            return None
        return (st.st_dev, st.st_ino)

    @staticmethod
    def _identity_lock_path(dev, ino):
        digest = hashlib.sha256(
            ("dev=%d:ino=%d" % (dev, ino)).encode("ascii")
        ).hexdigest()
        return os.path.join(_LOCK_DIR, "ino-" + digest + ".lock")

    def _canonical_missing_target(self):
        """Deterministic path naming a log that does not exist yet.

        The longest existing ancestor is resolved through symlinks (so
        aliases converging on one missing target agree), and the remaining
        non-existent components -- including the target of a dangling
        symlink -- are appended verbatim. The result only names a lock; it
        never creates the log.
        """
        cur = os.path.abspath(os.fspath(self.path))
        missing_parts = []
        while not os.path.lexists(cur):
            missing_parts.append(os.path.basename(cur))
            parent = os.path.dirname(cur)
            if parent == cur:  # filesystem root
                break
            cur = parent
        # realpath follows symlinks in the existing prefix (and a dangling
        # terminal link) without requiring the final target to exist.
        base = os.path.realpath(cur)
        if missing_parts:
            return os.path.join(base, *reversed(missing_parts))
        return base

    def _create_gate_lock_path(self):
        target = self._canonical_missing_target()
        digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
        return os.path.join(_LOCK_DIR, "new-" + digest + ".lock")

    def _lock_identity(self, dev, ino):
        """Take the per-real-object flock for (dev, ino)."""
        fd = self._take_flock(self._identity_lock_path(dev, ino))
        self._identity_fd = fd
        self._leased_dev = dev
        self._leased_ino = ino

    def _pin_object(self):
        """Open and pin the leased object, asserting its identity matches."""
        fd = os.open(os.fspath(self.path), os.O_RDONLY)
        try:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != (self._leased_dev, self._leased_ino):
                raise OSError(
                    errno.ESTALE,
                    "wal object identity changed under the exclusive lease: %r"
                    % (str(self.path),),
                )
        except OSError:
            os.close(fd)
            raise
        self._pin_fd = fd

    def _acquire_lease(self):
        """Take the exclusive write lease for this store's real log object.

        The lease has three cooperating flocks, all held per open file
        description so two exclusive instances conflict even within one
        process, and all released by the kernel when the holder dies:

        * a legacy sibling lock derived from the normalized absolute path,
          preserving the original same-spelling mutual exclusion;
        * an identity lock named by the real object's (st_dev, st_ino), so
          symlink and hard-link aliases of one existing log compete for the
          very same lease instead of each getting its own path lock;
        * for a log that does not exist yet, a creation gate named by the
          canonical target path that serializes concurrent first opens
          across aliases. After acquiring it the path is re-resolved: the
          first opener creates the log and adopts its inode lock, while any
          alias opener that finds the object already there -- or created by
          the first opener -- takes (or loses on) the inode lock.

        Nothing here reads, creates, truncates, or appends WAL bytes; only
        metadata and side-channel lock files in a shared lock directory are
        touched. A conflict raises WalBusyError; an unresolvable identity or
        failing file operation propagates the underlying OSError.
        """
        # 1. Legacy per-path sibling lock: unchanged path, message, and
        # mutual exclusion for identical normalized spellings.
        self._lock_fd = self._take_flock(
            os.path.abspath(os.fspath(self.path)) + ".lock"
        )
        try:
            identity = self._identity_stat()
            if identity is not None:
                # 2a. Existing real object: one lease per (device, inode).
                self._lock_identity(*identity)
                self._pin_object()
                return
            # 2b. The log does not exist yet: serialize concurrent first
            # creation across every alias of the same target, then resolve
            # the real object the winner is about to bring into existence.
            self._gate_fd = self._take_flock(self._create_gate_lock_path())
            identity = self._identity_stat()
            if identity is not None:
                # A racing alias opener (same gate) created it first; compete
                # for the identity lock like any existing-object opener.
                self._lock_identity(*identity)
                self._pin_object()
            # else: this opener won the gate and owns the first create. The
            # (dev, ino) identity lock is adopted atomically in _append once
            # the log file is actually created; the gate is held until then.
        except BaseException:
            self._release_lease_fds()
            raise

    def _release_lease_fds(self):
        """Release every lease-side descriptor; tolerant of partial opens."""
        fd = self._lock_fd
        self._lock_fd = None
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
        fd = self._identity_fd
        self._identity_fd = None
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
        fd = self._gate_fd
        self._gate_fd = None
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
        fd = self._pin_fd
        self._pin_fd = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        self._leased_dev = None
        self._leased_ino = None

    def _adopt_created_identity(self, write_fd):
        """Adopt the inode lock for a log the gate winner just created.

        Runs while the creation gate is still held: no alias opener can have
        slipped past its own gate, so the fresh inode cannot already be
        leased. The object is pinned before the gate is dropped.
        """
        st = os.fstat(write_fd.fileno())
        dev, ino = st.st_dev, st.st_ino
        fd = self._take_flock(self._identity_lock_path(dev, ino))
        self._identity_fd = fd
        self._leased_dev = dev
        self._leased_ino = ino
        # Pin via an independent descriptor (write_fd closes at _append's
        # end) before releasing the creation gate.
        self._pin_object()
        gate = self._gate_fd
        self._gate_fd = None
        if gate is not None:
            try:
                fcntl.flock(gate, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(gate)

    def _verify_leased_object(self, f):
        """Guard every write-mode WAL open against an identity swap.

        An exclusive store only ever mutates the exact (dev, ino) it leased;
        if the path was replaced (unlinked and re-created as another object
        through an alias) the write aborts with OSError rather than touching
        a log this instance never leased. Non-exclusive stores are unaffected.
        """
        if self._leased_dev is None:
            return
        st = os.fstat(f.fileno())
        if (st.st_dev, st.st_ino) != (self._leased_dev, self._leased_ino):
            raise OSError(
                errno.ESTALE,
                "wal object identity changed under the exclusive lease: %r"
                % (str(self.path),),
            )

    def close(self):
        """Release the write lease (if any) and close the store.

        Idempotent: closing an already-closed store is a no-op, and close
        itself never raises WalClosedError. After close, every other public
        method raises WalClosedError.
        """
        if self._closed:
            return
        self._closed = True
        self._release_lease_fds()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        # Best-effort lease release at garbage collection; the kernel
        # releases the flock on process death in any case.
        try:
            self.close()
        except Exception:
            pass

    def _check_open(self):
        if self._closed:
            raise WalClosedError("WalStore is closed")

    def _check_writable(self):
        # Closed takes priority: a closed read-only (or writable) instance
        # reports WalClosedError from every public method but close(). Only an
        # open instance is allowed to refuse with WalReadOnlyError.
        self._check_open()
        if self._readonly:
            raise WalReadOnlyError("WalStore was opened read-only")

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
            self._authorize_write_handle(f)
            f.truncate(self._valid_size)
            f.flush()
            os.fsync(f.fileno())

    def _authorize_write_handle(self, f):
        """Pin every write-mode WAL handle to the leased real object.

        If the creation gate is still held (the log did not exist when this
        exclusive store opened), opening for write is the first-create
        moment: adopt that object's inode lease then. Otherwise the handle
        must target the exact (dev, ino) the store leased; an identity swap
        aborts with OSError. Non-exclusive stores hold neither and are
        untouched.
        """
        if self._gate_fd is not None and self._identity_fd is None:
            self._adopt_created_identity(f)
        else:
            self._verify_leased_object(f)

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
            self._authorize_write_handle(f)
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
        self._check_writable()
        _validate_key(key)
        _validate_value(value)
        self._append(
            {"op": "set", "key": key, "value": value, "seq": self.commit_seq + 1}
        )

    def delete(self, key):
        self._check_writable()
        _validate_key(key)
        self._append({"op": "delete", "key": key, "seq": self.commit_seq + 1})

    def commit(self):
        # Persist the commit boundary first and only adopt the new seq/state
        # once it is durable: a failed write must neither consume the seq nor
        # present a committed state.
        self._check_writable()
        seq = self.commit_seq + 1
        self._append({"op": "commit", "seq": seq})
        self.recover()
        return self.commit_seq

    def rollback(self):
        # Validate the log by the exact recovery rules before touching
        self._check_writable()
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
            self._authorize_write_handle(f)
            f.truncate(committed_size)
            f.flush()
            os.fsync(f.fileno())
        self._valid_size = committed_size
        return pending_count

    def restore(self, target_seq):
        """Commit the state of a past commit boundary as a fresh new state.

        Read-only with respect to history: no earlier record is rewritten
        or reordered, and snapshot/history still serve every old commit.
        The target state is reached by one new set/delete batch sealed by
        one new commit, whose seq is latest commit_seq + 1 even when the
        target equals the current state (an empty batch then) or is 0
        (an empty state). Returns the new commit_seq.

        target_seq must be a non-boolean non-negative integer no greater
        than the latest committed seq; type errors, negatives, and
        out-of-range values raise ValueError before the log is touched.
        The log is then validated under the exact recover/snapshot rules
        (corruption anywhere raises WalCorruptionError), and must be
        settled at the last commit: complete uncommitted set/delete
        records or a trailing unfinished fragment make it raise
        WalPendingError with the file, state, and commit_seq untouched.
        Every change record and the commit are appended and fsynced one
        at a time as in set/commit; an OSError propagates with the old
        committed state and seq in place, already-written records simply
        forming the batch's pending tail (rollback clears them).
        """
        self._check_writable()
        if (
            isinstance(target_seq, bool)
            or not isinstance(target_seq, int)
            or target_seq < 0
        ):
            raise ValueError(
                "target_seq must be a non-negative integer, got %r"
                % (target_seq,)
            )
        # Full recover/snapshot validation first, purely into local
        # objects: corruption raises before any truncation or append.
        history_views = []
        candidate, committed, pending_count, _valid_size, _committed_size = (
            self._replay(snapshots=history_views)
        )
        if target_seq > committed:
            raise ValueError(
                "target_seq %r exceeds latest committed seq %r"
                % (target_seq, committed)
            )
        # The log must be settled: neither a complete, terminated
        # uncommitted batch nor an unfinished tail fragment may be spliced
        # into. A tail fragment is recover/audit-recognised garbage the
        # next append would truncate away; refusing here keeps restore
        # from appending (or truncating) anything before validation has
        # fully passed, and forces the caller to commit, rollback, or
        # repair explicitly.
        file_size = self.path.stat().st_size if self.path.exists() else 0
        if pending_count > 0 or file_size > self._valid_size:
            raise WalPendingError(
                "log is not settled at commit %r: %r pending record(s), "
                "an unfinished tail fragment is present"
                % (committed, pending_count)
            )
        current = candidate
        target = {} if target_seq == 0 else dict(history_views)[target_seq]
        # Merge the two key sets and write deterministically in Unicode
        # (code point) key order. A key only in the target, or present
        # with a different value under JSON-type-aware comparison, is a
        # set; a key present now but absent in the target is a delete.
        new_seq = committed + 1
        for key in sorted(set(current) | set(target)):
            if key not in target:
                self._append({"op": "delete", "key": key, "seq": new_seq})
            elif key not in current or not _json_equal(current[key], target[key]):
                # Deep copy: the adopted value must not share objects with
                # the replay's snapshot, which the caller could mutate.
                self._append(
                    {
                        "op": "set",
                        "key": key,
                        "value": copy.deepcopy(target[key]),
                        "seq": new_seq,
                    }
                )
        self._append({"op": "commit", "seq": new_seq})
        # Adopt the new committed view only once its commit boundary is
        # durable, exactly as commit() does.
        self.recover()
        return self.commit_seq

    def apply_batch(self, base_seq, changes):
        """Atomically commit an ordered batch of set/delete changes.

        Every change is appended to the log first, in the caller's order,
        and the batch is sealed by one new commit record whose seq is the
        latest committed seq + 1 -- exactly as if the caller had issued the
        same set/delete records one at a time and then committed. Returns
        the new commit_seq. An empty changes collection is an empty commit:
        no change records, one commit record, seq advancing by one.

        base_seq is the caller-declared compare-and-swap precondition: it
        must be a non-boolean non-negative integer and must equal the log's
        latest committed seq, otherwise the call raises ValueError or the
        unique WalConflictError respectively. Each change must be a dict
        carrying exactly the existing record semantics -- {"op": "set",
        "key", "value"} or {"op": "delete", "key"} -- with a string key and
        a value under the existing JSON-compatibility rules; any other
        shape raises ValueError. All argument validation completes before
        the log is touched.

        The log is then validated under the exact recover rules (any
        corruption raises WalCorruptionError) and must be settled at the
        last commit: complete uncommitted set/delete records or a
        recover/audit-recognisable unfinished tail fragment raise
        WalPendingError. None of these rejections writes, truncates, or
        alters in-memory state. A read-only instance raises
        WalReadOnlyError and a closed one WalClosedError before any of the
        above, exactly as the other mutating entries do.

        The write phase follows the same single-writer lease, per-record
        flush+fsync, and seq monotonicity rules as set/delete/commit: an
        OSError from any change record or the final commit propagates
        unchanged, the already-durable prefix stays observable through
        pending_changes and removable through rollback, and neither the
        in-memory state nor commit_seq advances. A process terminated at
        any write boundary recovers to the last complete commit on reopen;
        the unsealed part of the batch is never applied. Values are
        deep-copied into the written records, so neither the caller's
        changes collection nor the returned seq shares mutable nested
        objects with the store.
        """
        self._check_writable()
        if (
            isinstance(base_seq, bool)
            or not isinstance(base_seq, int)
            or base_seq < 0
        ):
            raise ValueError(
                "base_seq must be a non-negative integer, got %r" % (base_seq,)
            )
        if not isinstance(changes, (list, tuple)):
            raise ValueError(
                "changes must be a list of change records, got %r"
                % (type(changes).__name__,)
            )
        # Validate and normalize every change up front: a rejected call
        # must never create, truncate, or append to the log or alter
        # in-memory state. Values are deep-copied so the written records
        # can never share mutable nested objects with the caller's
        # collection.
        normalized = []
        for index, change in enumerate(changes):
            if not isinstance(change, dict):
                raise ValueError(
                    "change %r must be a dict, got %r"
                    % (index, type(change).__name__)
                )
            op = change.get("op")
            if op == "set":
                if set(change) != {"op", "key", "value"}:
                    raise ValueError(
                        "set change %r must carry exactly op, key and value, "
                        "got %r" % (index, sorted(change))
                    )
                _validate_key(change["key"])
                _validate_value(change["value"])
                normalized.append(
                    {
                        "op": "set",
                        "key": change["key"],
                        "value": copy.deepcopy(change["value"]),
                    }
                )
            elif op == "delete":
                if set(change) != {"op", "key"}:
                    raise ValueError(
                        "delete change %r must carry exactly op and key, "
                        "got %r" % (index, sorted(change))
                    )
                _validate_key(change["key"])
                normalized.append({"op": "delete", "key": change["key"]})
            else:
                raise ValueError(
                    "change %r has unknown op %r: only 'set' and 'delete' "
                    "records may be batched" % (index, op)
                )
        # Full recover-rule validation of the log, purely into local
        # objects: corruption raises WalCorruptionError before anything is
        # written, truncated, or adopted.
        (
            _candidate,
            committed,
            pending_count,
            _valid_size,
            _committed_size,
        ) = self._replay()
        # The log must be settled at the last commit before a new batch is
        # spliced on: neither complete uncommitted records nor a
        # recognisable unfinished tail fragment may be present (same rule
        # as restore).
        file_size = self.path.stat().st_size if self.path.exists() else 0
        if pending_count > 0 or file_size > self._valid_size:
            raise WalPendingError(
                "log is not settled at commit %r: %r pending record(s), "
                "an unfinished tail fragment is present"
                % (committed, pending_count)
            )
        if base_seq != committed:
            raise WalConflictError(
                "base_seq %r does not match latest committed seq %r"
                % (base_seq, committed)
            )
        # Append each change record and then the sealing commit, one
        # durable record at a time, exactly as restore does. An OSError
        # propagates with the old committed state and seq in place; the
        # already-written records form the batch's pending tail.
        new_seq = committed + 1
        for change in normalized:
            record = dict(change)
            record["seq"] = new_seq
            self._append(record)
        self._append({"op": "commit", "seq": new_seq})
        # Adopt the new committed view only once its commit boundary is
        # durable, exactly as commit() does.
        self.recover()
        return self.commit_seq

    def _committed_view(self):
        # Replay the log purely into local objects, exactly like recovery, but
        # never adopt the result: a query must observe the last durable commit
        # even when newer uncommitted records sit in the tail, and a corrupt
        # record (in the tail or the committed region) raises WalCorruptionError
        # without partially replacing the current in-memory state.
        candidate, committed, _pending, _valid_size, _committed_size = self._replay()
        return candidate, committed

    def get(self, key, default=_UNSET):
        self._check_open()
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
        self._check_open()
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
        self._check_open()
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

    def history(self, since_seq=0, until_seq=None):
        """Read-only export of committed change batches, in commit order.

        Argument form is validated first: since_seq must be a non-boolean
        non-negative integer, and until_seq -- when given (None means the
        latest committed seq) -- likewise, with since_seq <= until_seq;
        any type error, negative value, or inverted range raises
        ValueError. The whole log is then parsed under the exact recovery
        rules -- UTF-8, JSON, field sets, duplicate fields, seq
        continuity, the terminator adoption boundary, and JSON values --
        with a trailing interrupted-write fragment ignored by the recover
        rules and uncommitted set/delete records left in the log only;
        any other corruption raises WalCorruptionError with no partial
        result. Once the latest committed seq is known, a since_seq above
        it or an until_seq beyond it also raises ValueError.

        Returns a list, in commit order, of one entry per commit with
        since_seq < commit_seq <= until_seq: {"commit_seq": seq,
        "changes": [...]}, where changes preserves the batch's log write
        order -- set items carry op/key/value, delete items op/key -- and
        an empty commit yields an empty changes list. Every returned
        object and nested value is an independent deep copy; the log, the
        public state, commit_seq, and pending_count are never touched, and
        repeated calls on the same log prefix, including after a reopen,
        return identical results.
        """
        self._check_open()
        if (
            isinstance(since_seq, bool)
            or not isinstance(since_seq, int)
            or since_seq < 0
        ):
            raise ValueError(
                "since_seq must be a non-negative integer, got %r" % (since_seq,)
            )
        if until_seq is not None and (
            isinstance(until_seq, bool)
            or not isinstance(until_seq, int)
            or until_seq < 0
        ):
            raise ValueError(
                "until_seq must be a non-negative integer, got %r" % (until_seq,)
            )
        if until_seq is not None and since_seq > until_seq:
            raise ValueError(
                "since_seq %r exceeds until_seq %r" % (since_seq, until_seq)
            )
        batches = []
        _candidate, committed, _pending, _valid_size, _committed_size = self._replay(
            batches=batches
        )
        if until_seq is None:
            until_seq = committed
        if since_seq > committed:
            raise ValueError(
                "since_seq %r exceeds latest committed seq %r"
                % (since_seq, committed)
            )
        if until_seq > committed:
            raise ValueError(
                "until_seq %r exceeds latest committed seq %r"
                % (until_seq, committed)
            )
        # Commit seqs are contiguous from 1, so the recorded batches are
        # already in commit order; the changes were deep-copied when the
        # batch was recorded, so the caller may mutate the result freely.
        return [
            {"commit_seq": seq, "changes": changes}
            for seq, changes in batches
            if since_seq < seq <= until_seq
        ]

    def pending_changes(self):
        """Read-only view of the complete uncommitted records after the last commit.

        The whole log is parsed under the exact recovery rules -- UTF-8,
        JSON objects, duplicate fields, the per-op field sets, positive
        integer seq continuity, key/value types, and the terminator
        adoption boundary -- with a trailing interrupted-write fragment
        (an unfinished JSON prefix, a complete record whose terminator
        never became durable, or truncated UTF-8 bytes) discarded exactly
        as recover does: it is neither returned nor counted. Every fully
        written (terminated) set/delete record beyond the last commit is
        returned in log order even though it is not committed; any other
        corruption anywhere in the log raises WalCorruptionError with no
        partial result and no change to the public state, commit_seq, or
        the file.

        Returns RecoveryResult with commit_seq (the seq of the last
        complete commit record, 0 when there is none), pending_count
        (always equal to len(changes)), and changes -- set items carry
        op/key/value, delete items op/key; every nested value is an
        independent deep copy, so mutating the result never affects the
        store, a later commit/rollback, a reopen, or a repeated call.
        With no pending records the fields are the current seq, zero, and
        an empty list. The log is never truncated, rewritten, or appended
        to; no seq is consumed and the public state stays at the last
        committed view.
        """
        self._check_open()
        changes = []
        # As in audit, the replay is purely observational: restore the
        # cached accepted-prefix boundary so this query can never
        # influence a later append's decision to drop a tail fragment.
        # The replay itself builds only local objects and never touches
        # self.state, so WalCorruptionError propagates with nothing
        # adopted.
        saved_valid_size = self._valid_size
        try:
            (
                _candidate,
                committed,
                pending_count,
                _valid_size,
                _committed_size,
            ) = self._replay(pending_out=changes)
        finally:
            self._valid_size = saved_valid_size
        return RecoveryResult(
            commit_seq=committed,
            pending_count=pending_count,
            changes=changes,
        )

    def audit(self):
        """Read-only audit of the accepted prefix and any trailing fragment.

        Parses the log under exactly the same rules as recover -- UTF-8,
        JSON, field sets, seq continuity, and the terminator adoption
        boundary -- but never adopts the result: state and commit_seq are
        left untouched and the log is never truncated, rewritten, or
        appended to. Returns the recover triple plus three byte counters:
          valid_bytes     bytes from the start through the end of the last
                         terminated, accepted record (the committed prefix
                         plus any uncommitted set/delete records);
          committed_bytes byte offset just past the last commit record
                         (0 when there is no commit record);
          tail_bytes     file size minus valid_bytes: the unfinished
                         trailing write fragment only.
        A final segment without its terminator counts solely in tail_bytes
        when recover would recognise it either as an unfinished JSON prefix
        or as one complete record passing every structural, field, and seq
        check; it is never applied, counted as pending, or allowed to
        advance commit_seq, and truncated UTF-8 bytes are likewise counted
        only as fragment bytes. Anything else after the accepted prefix
        (blank lines, trailing whitespace, illegal UTF-8, non-standard JSON
        constants, duplicate fields, a wrong field set, a seq break, or any
        other unrecognisable, non-unfinished content) raises
        WalCorruptionError with no partial result and no change to the
        in-memory state. Repeated calls, including on a reopened instance,
        return identical fields and values.
        """
        self._check_open()
        # _replay caches the accepted-prefix length on the instance for a
        # later append; audit is purely observational and must not influence
        # that decision, so restore whatever was cached before (the replay
        # itself builds only local objects and never touches self.state).
        saved_valid_size = self._valid_size
        try:
            (
                candidate,
                committed,
                pending_count,
                valid_size,
                committed_size,
            ) = self._replay()
        finally:
            self._valid_size = saved_valid_size
        file_size = self.path.stat().st_size if self.path.exists() else 0
        return RecoveryResult(
            state=copy.deepcopy(candidate),
            commit_seq=committed,
            pending_count=pending_count,
            valid_bytes=valid_size,
            committed_bytes=committed_size,
            tail_bytes=file_size - valid_size,
        )

    def repair_tail(self):
        """Remove an unfinished write fragment left at the log's end.

        Validates the whole log under exactly the same rules as recover and
        audit -- UTF-8, JSON, record fields, seq continuity, the terminator
        adoption boundary, and duplicate fields -- building only local
        objects first. Anything after the accepted prefix that is not
        provably the single trailing fragment of an interrupted write (blank
        lines, trailing whitespace, illegal UTF-8, non-standard JSON
        constants, duplicate fields, a wrong field set, a seq break, an
        unknown op, or any other unrecognisable content) raises
        WalCorruptionError with the file, state, and commit_seq untouched.

        When a fragment exists it is removed in place: the log is truncated
        strictly to audit's valid_bytes, so no earlier byte is ever
        rewritten and no complete (terminated) set/delete record --
        uncommitted ones included -- is removed; they stay pending and keep
        following rollback's semantics. An OSError from opening,
        truncating, flushing, or syncing propagates with the in-memory
        commit state untouched. A missing log, an empty log, or a log with
        no trailing fragment reports removed_bytes 0 and never creates a
        file. Returns the recover triple -- state being an independent deep
        copy of the post-repair replayable view -- plus removed_bytes, the
        raw number of tail bytes removed; on return the repaired log is
        durable and a reopen recovers exactly the accepted prefix.
        """
        self._check_writable()
        # The replay raises WalCorruptionError before the file or the
        # adopted state can be touched, and its accepted-prefix boundary is
        # the same valid_bytes audit reports.
        (
            candidate,
            committed,
            pending_count,
            valid_size,
            _committed_size,
        ) = self._replay()
        file_size = self.path.stat().st_size if self.path.exists() else 0
        removed_bytes = file_size - valid_size
        if removed_bytes <= 0:
            # No unfinished fragment (a missing or empty log included):
            # nothing is created, opened, truncated, or synced.
            self.state = candidate
            self.commit_seq = committed
            self._valid_size = valid_size
            return RecoveryResult(
                state=copy.deepcopy(candidate),
                commit_seq=committed,
                pending_count=pending_count,
                removed_bytes=0,
            )
        # Shrink the log in place strictly to the accepted prefix. Only the
        # tail fragment is removed; prefix bytes are never rewritten or
        # reordered. Let any OSError from open/truncate/flush/fsync
        # propagate -- a failed repair must never masquerade as success --
        # and adopt the replayed view only once the truncation is durable.
        with self.path.open("r+b") as f:
            self._authorize_write_handle(f)
            f.truncate(valid_size)
            f.flush()
            os.fsync(f.fileno())
        self.state = candidate
        self.commit_seq = committed
        self._valid_size = valid_size
        return RecoveryResult(
            state=copy.deepcopy(candidate),
            commit_seq=committed,
            pending_count=pending_count,
            removed_bytes=removed_bytes,
        )

    def _replay(self, snapshots=None, batches=None, pending_out=None):
        # snapshots: optional caller-provided list; when given, one
        # (seq, deep-copy-of-state) entry per durable commit record is
        # appended, in commit order, so historical committed views can be
        # served without re-parsing the log. batches: likewise, one
        # (seq, changes) entry per commit record, changes holding the
        # batch's set/delete records in their original log order (set
        # items carry op/key/value, delete items op/key; values are deep
        # copies private to this replay). pending_out: likewise, filled
        # once at the end with the records of the single trailing
        # uncommitted batch in log order (set items carry op/key/value,
        # delete items op/key; values are deep copies private to this
        # replay). All three are purely observational: the replay itself,
        # its return value, and the log are unaffected.
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
                    if batches is not None:
                        # The batch's changes in original log order, with
                        # values deep-copied so the recorded entry can
                        # never be mutated through the replay's state.
                        changes = []
                        for p in pending:
                            if p["op"] == "delete":
                                changes.append({"op": "delete", "key": p["key"]})
                            else:
                                changes.append(
                                    {
                                        "op": "set",
                                        "key": p["key"],
                                        "value": copy.deepcopy(p["value"]),
                                    }
                                )
                        batches.append((seq, changes))
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
        if pending_out is not None:
            # The trailing uncommitted batch in original log order,
            # stripped to op/key(/value) and with values deep-copied so
            # the caller can never mutate the replay's records or the
            # parsed log objects.
            for p in pending:
                if p["op"] == "delete":
                    pending_out.append({"op": "delete", "key": p["key"]})
                else:
                    pending_out.append(
                        {
                            "op": "set",
                            "key": p["key"],
                            "value": copy.deepcopy(p["value"]),
                        }
                    )
        self._valid_size = valid_size
        return candidate, committed, len(pending), valid_size, committed_size

    def recover(self):
        # Replay purely into local objects first; raising WalCorruptionError
        # must never partially replace the current in-memory state.
        self._check_open()
        candidate, committed, pending_count, _valid_size, _committed_size = self._replay()
        self.state = candidate
        self.commit_seq = committed
        return RecoveryResult(
            state=copy.deepcopy(candidate),
            commit_seq=committed,
            pending_count=pending_count,
        )
