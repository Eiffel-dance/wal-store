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


class WalTailError(Exception):
    """Raised by a strict-tail store when a recover-recognised tail exists.

    When a WalStore is opened with ``strict_tail=True`` the recovery rules
    still run in full first, but the trailing fragment the default mode
    silently discards is instead a hard error: construction (and every
    later entry that replays the log) refuses rather than accepting a log
    whose last write was interrupted. The fragment is neither truncated,
    appended over, nor adopted as state or commit_seq, and an exclusive
    lease is never left behind.

    Attributes:
        valid_bytes: accepted-prefix length, identical to audit()'s
            ``valid_bytes``.
        tail_bytes: number of raw trailing fragment bytes, identical to
            audit()'s ``tail_bytes``.
        tail_kind: one of ``"truncated_utf8"`` (the fragment ends in a cut
            multi-byte UTF-8 sequence), ``"incomplete_json"`` (an
            unfinished JSON prefix), or ``"missing_terminator"`` (a
            complete, fully valid record whose terminator never became
            durable). Truncated UTF-8 takes precedence.
    """

    def __init__(self, valid_bytes, tail_bytes, tail_kind, path=None):
        self.valid_bytes = valid_bytes
        self.tail_bytes = tail_bytes
        self.tail_kind = tail_kind
        location = "" if path is None else " at %r" % (str(path),)
        super().__init__(
            "log%s has an unfinished tail fragment: kind=%r, "
            "valid_bytes=%r, tail_bytes=%r"
            % (location, tail_kind, valid_bytes, tail_bytes)
        )


class WalIntegrityError(Exception):
    """Raised when integrity mode is requested for an unprotected log.

    integrity=True refuses to adopt a non-empty log whose records carry no
    integrity metadata at all: silently starting a protected chain after a
    legacy history would make that history indistinguishable from a
    truncated protected one. The log is neither created, truncated, nor
    appended to, and no in-memory state changes; the caller must either
    open the log in the default mode (which keeps the legacy format) or
    start a fresh protected log at another path.
    """


class WalBusyError(Exception):
    """Raised when an exclusive write lease is already held for the log path."""


class WalClosedError(Exception):
    """Raised when any public method is called on a closed WalStore."""


class WalReadOnlyError(Exception):
    """Raised when a mutating method is called on a read-only WalStore."""


class WalPendingError(Exception):
    """Raised by restore()/apply_batch()/apply_if_versions()/commit_if_seq()/apply_idempotent() when the log is not settled.

    Either complete, terminated set/delete records wait in an uncommitted
    batch, or an interrupted write leaves an unfinished tail fragment.
    Neither entry splices a new batch into either state; the caller must
    commit or roll the pending records back (or repair the fragment) first.
    commit_if_seq() is the exception that seals complete pending records
    instead of refusing them; it raises only for the unfinished fragment.
    """


class WalConflictError(Exception):
    """Raised by apply_batch()/apply_if_versions()/commit_if_seq()/apply_idempotent() when a declared precondition is stale.

    The caller-declared base commit seq or per-key expected version does
    not match the log's latest committed state, so the batch's
    compare-and-swap precondition fails. Nothing is written, truncated, or
    adopted; the caller is expected to re-read the current state and retry
    with the up-to-date base.
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

# Field name carrying a protected record's integrity metadata. Legacy
# records never contain it; a protected log carries it on every record.
_INTEGRITY_FIELD = "ic"

# Domain-separated seed of the integrity hash chain: the prev-chain value
# assumed for the first record of a protected log.
_CHAIN_SEED = hashlib.sha256(b"walstore-integrity-chain-v1").hexdigest()


def _chain_digest(prev_chain, core_line):
    """Integrity metadata of one protected record.

    Chains the record's canonical core line (every field except the
    integrity metadata itself, JSON-serialised with sorted keys) onto the
    previous record's digest, so the digest covers the record's content,
    its commit seq, and its position in the operation order: rewriting
    content, deleting or inserting a record, or reordering records breaks
    the chain at the first touched boundary and is detected when the log
    is verified.
    """
    return hashlib.sha256(
        (prev_chain + "\n" + core_line).encode("utf-8")
    ).hexdigest()


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


def _version_of_key(batches, key):
    """Commit seq of the last committed batch that set or deleted ``key``.

    ``batches`` is the (seq, changes) list recorded by a replay; only
    committed batches appear in it, so pending records and discarded tail
    fragments can never move a key's version. A key no committed batch
    ever touched has version 0, and a deleted key keeps the seq of the
    commit that deleted it.
    """
    version = 0
    for seq, changes in batches:
        for change in changes:
            if change["key"] == key:
                version = seq
    return version


def _validate_expected_seq(expected_seq):
    if (
        isinstance(expected_seq, bool)
        or not isinstance(expected_seq, int)
        or expected_seq < 0
    ):
        raise ValueError(
            "expected_seq must be a non-negative integer, got %r"
            % (expected_seq,)
        )


def _validate_request_id(request_id):
    # The persistent request identifier of apply_idempotent: a non-empty
    # string, validated before the log is ever read or created.
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(
            "request_id must be a non-empty string, got %r" % (request_id,)
        )


def _changes_equal(a, b):
    """Order-sensitive equality of two normalized change sequences.

    Change order, keys, and delete entries must match exactly; set values
    are compared under the JSON-type-aware rules of _json_equal, so JSON
    numbers and booleans -- nested or not -- are never conflated, while
    the JSON number family (int and float) still compares numerically.
    """
    if len(a) != len(b):
        return False
    for change_a, change_b in zip(a, b):
        if change_a["op"] != change_b["op"] or change_a["key"] != change_b["key"]:
            return False
        if change_a["op"] == "set" and not _json_equal(
            change_a["value"], change_b["value"]
        ):
            return False
    return True


def _validate_expected_versions(expected_versions):
    # The version precondition mapping of apply_if_versions: string keys,
    # each mapped to a non-boolean non-negative integer under the
    # key_version rules (0 for a key no committed batch ever touched, the
    # deleting commit's seq for a deleted key).
    if not isinstance(expected_versions, dict):
        raise ValueError(
            "expected_versions must be a dict mapping keys to versions, "
            "got %r" % (type(expected_versions).__name__,)
        )
    for key, version in expected_versions.items():
        if not isinstance(key, str):
            raise ValueError(
                "expected_versions keys must be strings, got %r"
                % (type(key).__name__,)
            )
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version < 0
        ):
            raise ValueError(
                "version for key %r must be a non-negative integer, got %r"
                % (key, version)
            )


def _normalize_changes(changes):
    # Validate and normalize an ordered set/delete change collection up
    # front: a rejected call must never create, truncate, or append to the
    # log or alter in-memory state. Values are deep-copied so the written
    # records can never share mutable nested objects with the caller's
    # collection.
    if not isinstance(changes, (list, tuple)):
        raise ValueError(
            "changes must be a list of change records, got %r"
            % (type(changes).__name__,)
        )
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
    return normalized


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


class WalView:
    """Fixed committed-read view of one WAL commit boundary.

    Created only through ``WalStore.view``; a view holds its own
    deep-copied key/value state at exactly one commit seq and never reads
    the log again, so later commits on the store, closing (or dropping)
    the originating instance, and even out-of-band modification or
    corruption of the log file cannot invalidate it or change a value it
    serves. It offers no write entry point: it produces no WAL bytes, no
    pending records, and no new seq, takes no lease, and needs no close.

    The view serves ``commit_seq`` (the boundary it was built at, 0 for
    the empty state), ``state`` (the complete committed key/value state
    at that boundary), and the same get/contains/scan read semantics the
    store provides over its latest committed view. Every value handed to
    a caller is an independent deep copy, so mutating a returned object
    never affects the view or a later read; repeated reads at the same
    boundary return equal values for the lifetime of the view.
    """

    __slots__ = ("commit_seq", "_state")

    def __init__(self, commit_seq, state):
        # Built only from a replay's private boundary snapshot; copy once
        # more defensively so the view can never share mutable nested
        # objects with the replay or the constructing call.
        self.commit_seq = commit_seq
        self._state = copy.deepcopy(state)

    @property
    def state(self):
        """The boundary's complete key/value state, an independent copy.

        A fresh deep copy on every access: mutating the returned mapping
        or a nested value never affects the view, and a later access
        returns the unchanged boundary state.
        """
        return copy.deepcopy(self._state)

    def get(self, key, default=_UNSET):
        # The same key validation, default-value validation, stored-None
        # handling, and KeyError semantics as WalStore.get, served from
        # the frozen boundary rather than a fresh replay.
        _validate_key(key)
        if default is not _UNSET:
            _validate_value(default)
        if key in self._state:
            return copy.deepcopy(self._state[key])
        if default is not _UNSET:
            return copy.deepcopy(default)
        raise KeyError(key)

    def contains(self, key):
        # The same string-key validation as WalStore.contains; presence
        # is decided solely at the fixed boundary.
        _validate_key(key)
        return key in self._state

    def scan(self, start_key=None, end_key=None, limit=None):
        """Deterministic half-open range scan of the fixed boundary.

        Identical argument rules, Unicode code-point ordering, half-open
        ``[start_key, end_key)`` window, limit cap, item shape, and
        independent deep-copy guarantees as ``WalStore.scan``, frozen at
        the view's commit boundary: every call on the same view returns
        the same sequence.
        """
        if start_key is not None and not isinstance(start_key, str):
            raise ValueError(
                "start_key must be a string or omitted, got %r"
                % (type(start_key).__name__,)
            )
        if end_key is not None and not isinstance(end_key, str):
            raise ValueError(
                "end_key must be a string or omitted, got %r"
                % (type(end_key).__name__,)
            )
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError(
                "limit must be a non-negative integer or omitted, got %r"
                % (limit,)
            )
        if (
            start_key is not None
            and end_key is not None
            and start_key > end_key
        ):
            raise ValueError(
                "start_key %r exceeds end_key %r" % (start_key, end_key)
            )
        keys = sorted(self._state)
        if start_key is not None:
            keys = [key for key in keys if key >= start_key]
        if end_key is not None:
            keys = [key for key in keys if key < end_key]
        if limit is not None:
            keys = keys[:limit]
        return [
            {"key": key, "value": copy.deepcopy(self._state[key])}
            for key in keys
        ]


class WalStore:
    def __init__(
        self, path, exclusive=False, readonly=False, integrity=False,
        strict_tail=False,
    ):
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
        if not isinstance(integrity, bool):
            raise ValueError(
                "integrity must be a bool, got %r" % (type(integrity).__name__,)
            )
        if not isinstance(strict_tail, bool):
            raise ValueError(
                "strict_tail must be a bool, got %r"
                % (type(strict_tail).__name__,)
            )
        self.path = Path(path)
        self.state = {}
        self.commit_seq = 0
        # Byte length of the durable log prefix as judged by the latest
        # replay; anything beyond it is a discarded tail fragment.
        self._valid_size = None
        self._closed = False
        self._readonly = readonly
        # Integrity protection mode. _integrity_requested records the
        # constructor's integrity flag; _protected is the effective mode --
        # True when integrity was requested or the log was recognised as
        # protected on replay, so a protected log opened in the default
        # mode keeps its protected format. _chain_head is the integrity
        # chain digest just past the accepted prefix (the seed for an
        # empty or missing log); the next protected record chains onto it.
        self._integrity_requested = integrity
        self._protected = integrity
        self._chain_head = _CHAIN_SEED
        # Strict tail policy: when True every replay that finds a
        # recover-recognisable unfinished tail fragment raises WalTailError
        # instead of discarding the fragment. The flag never changes what
        # counts as a valid record, as corruption, or as a fragment -- only
        # whether a recognised fragment is accepted.
        self._strict_tail = strict_tail
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
        try:
            self.recover()
        except BaseException:
            # A failed open (corruption, or WalIntegrityError when
            # integrity=True meets a fully unprotected legacy log) must not
            # leak the lease: the instance is never handed to the caller.
            self._release_lease_fds()
            raise

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

    def _revalidate_for_append(self):
        """Validate the whole log by the exact recover rules before appending.

        A store can stay open while its log is appended to or damaged out of
        band, so the boundaries cached when the store opened are never trusted
        for a write: the log is replayed from the head exactly as recover()
        does, re-determining valid_bytes, committed_bytes, and the last
        committed state. The replay builds only local objects, so a terminated
        invalid record, a blank/whitespace record, illegal UTF-8, duplicate or
        missing fields, an unknown op, a bad key/value, or a seq break raises
        WalCorruptionError without appending, truncating, or adopting anything
        -- state, commit_seq, and the byte boundary a later rollback observes
        are all left untouched.

        The only bytes allowed beyond the replayed valid_bytes are the single
        trailing interrupted-write fragment the recovery rules recognise; that
        fragment is removed in place exactly as before, while every complete
        record before it -- legal uncommitted set/delete records included -- is
        preserved byte for byte. Returns the replayed committed view and the
        last committed seq in the file; the next record must use
        ``committed + 1`` rather than any seq cached at open time.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        saved_boundary = self._valid_size
        saved_chain_head = self._chain_head
        try:
            (
                candidate,
                committed,
                _pending_count,
                valid_size,
                _committed_size,
            ) = self._replay()
            file_size = self.path.stat().st_size if self.path.exists() else 0
            if file_size > valid_size:
                # The replay has already proved the bytes beyond valid_size
                # are exactly one unfinished tail fragment -- anything else
                # raised WalCorruptionError above -- so removing them can
                # never discard a complete, terminated record.
                with self.path.open("r+b") as f:
                    self._authorize_write_handle(f)
                    f.truncate(valid_size)
                    f.flush()
                    os.fsync(f.fileno())
        except BaseException:
            # Neither a failed validation nor a failed fragment removal
            # adopts a boundary: state, commit_seq, the cached
            # accepted-prefix edge, and the cached integrity chain head
            # stay exactly as on entry. Every later write revalidates from
            # the head in any case.
            self._valid_size = saved_boundary
            self._chain_head = saved_chain_head
            raise
        return candidate, committed

    def _append(self, row):
        """Durably append one log record after head-to-tail revalidation.

        The whole log is validated under the exact recover rules first (see
        _revalidate_for_append); WalCorruptionError propagates with the file
        and in-memory state untouched. The record's seq is stamped from the
        seq replayed out of the file -- the file's last committed seq plus
        one -- never from a value cached when the store opened, so a batch
        committed out of band while the store was open extends the chain
        instead of being overwritten.

        Raises OSError if the record cannot be written or synced; the
        un-durable tail is best-effort truncated back so a failed write can
        never masquerade as a committed record on reopen, and no new state is
        adopted before the record is durable.
        """
        candidate, committed = self._revalidate_for_append()
        record = dict(row)
        record["seq"] = committed + 1
        if self._protected:
            # Chain the record onto the integrity digest replayed out of
            # the accepted prefix (the seed for an empty log), exactly as
            # the replay recomputes it: the canonical core line is the
            # record without its integrity metadata, JSON-serialised with
            # sorted keys.
            core_line = json.dumps(record, sort_keys=True)
            record[_INTEGRITY_FIELD] = _chain_digest(self._chain_head, core_line)
        line = json.dumps(record, sort_keys=True) + "\n"
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
            new_size = f.tell()
        # Adopt only once the appended record is complete and durable.
        # candidate is the last *committed* view -- pending records never
        # join it -- so a set/delete append keeps the state deferred until
        # commit exactly as before, while a batch committed out of band
        # converges memory with what a reopen would recover.
        self._valid_size = new_size
        self.state = candidate
        self.commit_seq = committed
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
        # The seq is stamped inside _append from a fresh head-to-tail replay,
        # not from the open-time cached commit_seq.
        self._append({"op": "set", "key": key, "value": value})

    def delete(self, key):
        self._check_writable()
        _validate_key(key)
        self._append({"op": "delete", "key": key})

    def commit(self):
        # Re-validate the whole log and persist the commit boundary first; the
        # seq is stamped from the replayed last commit, and the new seq/state
        # are adopted only once the record is durable: a failed validation or
        # write must neither consume a seq nor present a committed state.
        self._check_writable()
        self._append({"op": "commit"})
        self.recover()
        return self.commit_seq

    def commit_if_seq(self, expected_seq):
        """Atomically seal the pending batch, guarded by the latest commit seq.

        The conditional counterpart of commit(): the caller declares with
        ``expected_seq`` the latest committed seq it observed before
        deciding to seal, and the pending batch is committed only when the
        log's latest committed seq still equals it. On success one commit
        record -- seq = latest committed seq + 1 -- is appended, sealing
        every complete, terminated set/delete record written after the
        last commit in its original log order, exactly as if commit() had
        been called; with no pending records the call is an empty commit
        (one commit record, seq advancing by one). Returns the new
        commit_seq; state, recover, snapshot, history, diff, scan,
        pending_changes, key_version, and a reopen then all reflect the
        same committed result.

        ``expected_seq`` must be a non-boolean non-negative integer; every
        argument error raises ValueError before the log is read or
        created. A closed instance raises WalClosedError and a read-only
        one WalReadOnlyError before any argument is checked, exactly as
        the other mutating entries do.

        The log is then validated under the exact recover rules (any
        corruption -- illegal UTF-8, non-standard JSON, duplicate fields,
        bad field sets, a seq break -- raises WalCorruptionError). A
        recover/audit-recognisable unfinished tail fragment raises
        WalPendingError and must be cleared with repair_tail first; the
        file, the in-memory state, every seq, and the complete pending
        records all stay untouched. Complete pending records themselves
        are exactly what this entry seals, so they never raise. A latest
        committed seq that no longer equals ``expected_seq`` raises
        WalConflictError, the comparison always made against the latest
        complete commit. None of these rejections writes, truncates, or
        consumes a seq.

        The write phase follows the same single-writer lease, per-record
        flush+fsync, and seq monotonicity rules as commit(): an OSError
        from the commit record's write or sync propagates unchanged, the
        already-durable set/delete records stay observable through
        pending_changes and removable through rollback, neither the
        in-memory state nor commit_seq advances, and a process terminated
        at any write boundary recovers to the last complete commit on
        reopen with no seq skipped. A successful call never rewrites old
        log bytes.
        """
        self._check_writable()
        _validate_expected_seq(expected_seq)
        # Full recover-rule validation of the log, purely into local
        # objects: corruption raises WalCorruptionError before anything is
        # written, truncated, or adopted.
        (
            _candidate,
            committed,
            _pending_count,
            _valid_size,
            _committed_size,
        ) = self._replay()
        # An unfinished tail fragment must be repaired first: appending
        # over it would silently discard bytes the caller may still owe a
        # decision on. Complete pending records are not an error here --
        # sealing them is the whole point of this entry.
        file_size = self.path.stat().st_size if self.path.exists() else 0
        if file_size > self._valid_size:
            raise WalPendingError(
                "log has an unfinished tail fragment at commit %r: "
                "repair_tail() must clear it before the batch can be sealed"
                % (committed,)
            )
        if committed != expected_seq:
            raise WalConflictError(
                "expected_seq %r does not match latest committed seq %r"
                % (expected_seq, committed)
            )
        # Append the sealing commit, stamped from a fresh head-to-tail
        # replay inside _append; the pending records already carry the new
        # seq, so the commit seals them in order. Adopt the new committed
        # view only once the boundary is durable, exactly as commit()
        # does.
        self._append({"op": "commit"})
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
        normalized = _normalize_changes(changes)
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

    def apply_idempotent(self, request_id, base_seq, changes):
        """Atomically commit a batch under a persistent request identifier.

        The retry-safe counterpart of apply_batch: ``request_id`` is a
        caller-chosen identifier persisted on the batch's commit record,
        so a caller whose response was lost after the commit became
        durable can repeat the call without advancing the seq twice. On
        first use of an identifier the call behaves exactly like
        apply_batch(base_seq, changes): the log must be settled at the
        last commit, ``base_seq`` must equal the latest committed seq,
        the changes are appended in the caller's order and sealed by one
        new commit record -- carrying ``request_id`` -- whose seq is the
        latest committed seq + 1. Returns the new commit_seq; state,
        recover, snapshot, history, diff, scan, pending_changes,
        key_version, and a reopen then all reflect the same committed
        result as the equivalent apply_batch call.

        When ``request_id`` already names a complete durable commit, the
        call is a retry: if ``base_seq`` and the change sequence are
        exactly those of the recorded request -- compared in order, by
        key, by delete entry, and by nested value under the
        JSON-type-aware rules (JSON numbers and booleans are never
        conflated) -- the original commit seq is returned even when later
        commits have since landed; no byte is appended, no state changes,
        and no seq is consumed. The same identifier with a different
        base_seq or a different change sequence raises WalConflictError,
        as does a fresh identifier whose base_seq no longer equals the
        latest committed seq. An identifier counts as used only once its
        commit record is durable: a batch whose change records were
        written but never sealed leaves the identifier free, and after
        the pending records are rolled back the same identifier may be
        used again.

        ``request_id`` must be a non-empty string, ``base_seq`` a
        non-boolean non-negative integer, and ``changes`` follows the
        apply_batch rules (an ordered list or tuple of {"op": "set",
        "key", "value"} / {"op": "delete", "key"} records with string
        keys and JSON-compatible values, deep-copied into the written
        records). Every argument error raises ValueError before the log
        is read or created; a closed instance raises WalClosedError and
        a read-only one WalReadOnlyError before any argument is checked,
        exactly as the other mutating entries do.

        The log is then validated under the exact recover rules (any
        corruption raises WalCorruptionError) and must be settled at the
        last commit: complete uncommitted set/delete records or a
        recover/audit-recognisable unfinished tail fragment raise
        WalPendingError. None of these rejections writes, truncates, or
        alters in-memory state, and none consumes a seq.

        The write phase follows the same single-writer lease, per-record
        flush+fsync, and seq monotonicity rules as apply_batch: an
        OSError from any change record or the final commit propagates
        unchanged, the already-durable prefix stays observable through
        pending_changes and removable through rollback, neither the
        in-memory state nor commit_seq advances, and a process terminated
        at any write boundary recovers to the last complete commit on
        reopen -- with the request identifier free again when its commit
        never became durable, and answered with the original seq when it
        did.
        """
        self._check_writable()
        _validate_request_id(request_id)
        if (
            isinstance(base_seq, bool)
            or not isinstance(base_seq, int)
            or base_seq < 0
        ):
            raise ValueError(
                "base_seq must be a non-negative integer, got %r" % (base_seq,)
            )
        # Validate and normalize every change up front, exactly as
        # apply_batch does: a rejected call must never create, truncate,
        # or append to the log or alter in-memory state.
        normalized = _normalize_changes(changes)
        # Full recover-rule validation of the log, purely into local
        # objects: corruption raises WalCorruptionError before anything is
        # written, truncated, or adopted. The replay also collects the
        # durable request-id occupancy records.
        idempotent = []
        (
            _candidate,
            committed,
            pending_count,
            _valid_size,
            _committed_size,
        ) = self._replay(idempotent_out=idempotent)
        # The log must be settled at the last commit before a new batch is
        # spliced on: neither complete uncommitted records nor a
        # recognisable unfinished tail fragment may be present (same rule
        # as restore and apply_batch).
        file_size = self.path.stat().st_size if self.path.exists() else 0
        if pending_count > 0 or file_size > self._valid_size:
            raise WalPendingError(
                "log is not settled at commit %r: %r pending record(s), "
                "an unfinished tail fragment is present"
                % (committed, pending_count)
            )
        for stored_id, stored_seq, stored_changes in idempotent:
            if stored_id != request_id:
                continue
            # The identifier already names a complete durable commit. An
            # exact retry -- the base_seq the commit was built on (its seq
            # minus one) and the identical change sequence -- is answered
            # with the original seq without appending a byte, changing
            # state, or consuming a seq, even when later commits have
            # landed since. Anything else under the same identifier is a
            # conflict.
            if stored_seq - 1 == base_seq and _changes_equal(
                stored_changes, normalized
            ):
                return stored_seq
            raise WalConflictError(
                "request_id %r is already used by commit %r with different "
                "parameters" % (request_id, stored_seq)
            )
        if base_seq != committed:
            raise WalConflictError(
                "base_seq %r does not match latest committed seq %r"
                % (base_seq, committed)
            )
        # Append each change record and then the sealing commit -- which
        # carries the request identifier, so the occupancy becomes durable
        # exactly with the commit boundary -- one durable record at a
        # time, exactly as apply_batch does. An OSError propagates with
        # the old committed state and seq in place; the already-written
        # records form the batch's pending tail and the identifier stays
        # unused until the commit record is durable.
        new_seq = committed + 1
        for change in normalized:
            record = dict(change)
            record["seq"] = new_seq
            self._append(record)
        self._append({"op": "commit", "seq": new_seq, "request_id": request_id})
        # Adopt the new committed view only once its commit boundary is
        # durable, exactly as commit() does.
        self.recover()
        return self.commit_seq

    def set_if_version(self, key, expected_seq, value):
        """Optimistically write one set record, guarded by the key's version.

        The single-writer compare-and-swap entry for one key: the caller
        declares with ``expected_seq`` the key_version it observed before
        deciding to write, and the write goes through only when the key's
        version in the latest complete commit still equals it. On success
        one set record and one commit record are appended -- exactly as if
        set(key, value) were followed by commit() -- and the new commit
        seq is returned; state, recover, snapshot, history, diff, scan,
        pending_changes, and key_version then all reflect the same
        committed result.

        ``key`` must be a string and ``value`` follows the existing
        JSON-compatibility rules (it is deep-copied into the written
        record, so the store never shares mutable nested objects with the
        caller). ``expected_seq`` must be a non-boolean non-negative
        integer: 0 requires the key to be absent from every committed
        batch, and any larger value requires the key's last committed
        touch -- set or delete -- to carry exactly that commit seq. Every
        argument error raises ValueError before the log is touched; a
        closed instance raises WalClosedError and a read-only one
        WalReadOnlyError before any argument is checked, exactly as the
        other mutating entries do.

        The log is then validated under the exact recover rules (any
        corruption raises WalCorruptionError) and must be settled at the
        last commit: complete uncommitted set/delete records or a
        recover/audit-recognisable unfinished tail fragment raise
        WalPendingError, and a key version that no longer matches raises
        WalConflictError, the comparison always made against the latest
        complete commit. None of these rejections appends, truncates, or
        alters in-memory state.

        The write phase follows the same single-writer lease, per-record
        flush+fsync, and seq monotonicity rules as set/commit: an OSError
        from either record propagates unchanged, a set record already
        durable but not yet sealed stays observable through
        pending_changes and removable through rollback, neither state nor
        commit_seq advances, and a process terminated at any write
        boundary recovers to the previous commit on reopen.
        """
        self._check_writable()
        _validate_key(key)
        _validate_expected_seq(expected_seq)
        _validate_value(value)
        self._commit_if_version(
            key,
            expected_seq,
            {"op": "set", "key": key, "value": copy.deepcopy(value)},
        )
        return self.commit_seq

    def delete_if_version(self, key, expected_seq):
        """Optimistically write one delete record, guarded by the key's version.

        The delete counterpart of set_if_version: the write goes through
        only when the key's version in the latest complete commit equals
        ``expected_seq`` -- for a live key the seq of the commit that last
        set it, for an already-deleted key the seq of the commit that
        deleted it, and 0 for a key no committed batch ever touched. On
        success one delete record and one commit record are appended --
        exactly as if delete(key) were followed by commit() -- and the new
        commit seq is returned; state, recover, snapshot, history, diff,
        scan, pending_changes, and key_version then all reflect the same
        committed result.

        ``key`` must be a string and ``expected_seq`` a non-boolean
        non-negative integer; every argument error raises ValueError
        before the log is touched, a closed instance raises
        WalClosedError, and a read-only one WalReadOnlyError, with the
        same priority as the other mutating entries. The log is validated
        under the exact recover rules (corruption raises
        WalCorruptionError) and must be settled at the last commit:
        complete uncommitted records or a recognisable unfinished tail
        fragment raise WalPendingError, and a stale expected_seq raises
        WalConflictError against the latest complete commit. None of
        these rejections appends, truncates, or alters in-memory state.

        Durability follows the set/commit rules: each record is flushed
        and fsynced as it is written, an OSError propagates with the old
        committed state and seq in place, a delete record already durable
        but not yet sealed stays observable through pending_changes and
        removable through rollback, and a process terminated at any write
        boundary recovers to the previous commit on reopen.
        """
        self._check_writable()
        _validate_key(key)
        _validate_expected_seq(expected_seq)
        self._commit_if_version(
            key, expected_seq, {"op": "delete", "key": key}
        )
        return self.commit_seq

    def _commit_if_version(self, key, expected_seq, record):
        """Validate, check the version precondition, and commit one record.

        Shared write path of set_if_version/delete_if_version; every
        argument has already been validated and every value deep-copied by
        the caller. The log is replayed under the exact recover rules
        purely into local objects, so corruption raises WalCorruptionError
        before anything is written, truncated, or adopted. The log must
        be settled at the last commit (same rule as restore and
        apply_batch): neither complete uncommitted records nor a
        recognisable unfinished tail fragment may be present. The
        precondition compares expected_seq against the key's version in
        the latest complete commit only -- pending records and a
        discardable tail fragment never move it.
        """
        batches = []
        (
            _candidate,
            committed,
            pending_count,
            _valid_size,
            _committed_size,
        ) = self._replay(batches=batches)
        file_size = self.path.stat().st_size if self.path.exists() else 0
        if pending_count > 0 or file_size > self._valid_size:
            raise WalPendingError(
                "log is not settled at commit %r: %r pending record(s), "
                "an unfinished tail fragment is present"
                % (committed, pending_count)
            )
        version = _version_of_key(batches, key)
        if version != expected_seq:
            raise WalConflictError(
                "key %r is at version %r, not expected version %r"
                % (key, version, expected_seq)
            )
        # Append the change record and then the sealing commit, one
        # durable record at a time, exactly as apply_batch does. An
        # OSError propagates with the old committed state and seq in
        # place; an already-written change record forms the pending tail.
        new_seq = committed + 1
        record = dict(record)
        record["seq"] = new_seq
        self._append(record)
        self._append({"op": "commit", "seq": new_seq})
        # Adopt the new committed view only once its commit boundary is
        # durable, exactly as commit() does.
        self.recover()

    def apply_if_versions(self, expected_versions, changes):
        """Atomically commit a batch guarded by many keys' versions.

        The multi-key compare-and-swap entry: ``expected_versions`` maps
        each guarded key to the key_version the caller observed before
        deciding to write, and the whole batch goes through as one new
        commit only when every mapped key's version in the latest complete
        commit still equals its declared value. On success the changes are
        appended in the caller's order -- the same key may appear several
        times, each record acting on the batch's running state -- and the
        batch is sealed by one new commit record whose seq is the latest
        committed seq + 1, exactly as apply_batch does. Returns the new
        commit_seq; state, recover, snapshot, history, diff, scan,
        key_version, and a reopen all reflect the complete batch. An empty
        changes collection is an empty commit: no change records, one
        commit record, seq advancing by one.

        ``expected_versions`` must be a dict with string keys and
        non-boolean non-negative integer versions under the key_version
        rules: 0 requires the key to be absent from every committed batch,
        and a deleted key keeps the seq of the commit that deleted it.
        ``changes`` must be a list or tuple whose items carry exactly the
        existing record semantics -- {"op": "set", "key", "value"} or
        {"op": "delete", "key"} -- with string keys and values under the
        existing JSON-compatibility rules (deep-copied into the written
        records). Every key touched by changes must appear in
        expected_versions; the mapping may additionally name keys used
        only for validation. Every argument error raises ValueError before
        the log is read or created; a closed instance raises
        WalClosedError and a read-only one WalReadOnlyError before any
        argument is checked, exactly as the other mutating entries do.

        The log is then validated under the exact recover rules (any
        corruption raises WalCorruptionError) and must be settled at the
        last commit: complete uncommitted set/delete records or a
        recover/audit-recognisable unfinished tail fragment raise
        WalPendingError. Every version comparison is made against the
        latest complete commit at validation time -- pending records and a
        discardable tail fragment never move a key's version -- and any
        mismatch raises WalConflictError. None of these rejections
        appends, truncates, or alters in-memory state, and no seq is
        consumed.

        The write phase follows the same single-writer lease, per-record
        flush+fsync, and seq monotonicity rules as apply_batch: an OSError
        from any change record or the final commit propagates unchanged,
        the already-durable prefix stays observable through
        pending_changes and removable through rollback, neither the
        in-memory state nor commit_seq advances, and a process terminated
        at any write boundary recovers to the last complete commit on
        reopen with the unsealed part of the batch never applied.
        """
        self._check_writable()
        _validate_expected_versions(expected_versions)
        normalized = _normalize_changes(changes)
        # Every key the batch touches must be guarded by the mapping; the
        # mapping itself may name additional validation-only keys.
        for change in normalized:
            if change["key"] not in expected_versions:
                raise ValueError(
                    "change key %r is missing from expected_versions"
                    % (change["key"],)
                )
        # Full recover-rule validation of the log, purely into local
        # objects: corruption raises WalCorruptionError before anything is
        # written, truncated, or adopted.
        batches = []
        (
            _candidate,
            committed,
            pending_count,
            _valid_size,
            _committed_size,
        ) = self._replay(batches=batches)
        # The log must be settled at the last commit before a new batch is
        # spliced on: neither complete uncommitted records nor a
        # recognisable unfinished tail fragment may be present (same rule
        # as restore, apply_batch, and the single-key conditional writes).
        file_size = self.path.stat().st_size if self.path.exists() else 0
        if pending_count > 0 or file_size > self._valid_size:
            raise WalPendingError(
                "log is not settled at commit %r: %r pending record(s), "
                "an unfinished tail fragment is present"
                % (committed, pending_count)
            )
        # Compare every guarded key against its version in the latest
        # complete commit only.
        for key, expected in expected_versions.items():
            version = _version_of_key(batches, key)
            if version != expected:
                raise WalConflictError(
                    "key %r is at version %r, not expected version %r"
                    % (key, version, expected)
                )
        # Append each change record and then the sealing commit, one
        # durable record at a time, exactly as apply_batch does. An
        # OSError propagates with the old committed state and seq in
        # place; the already-written records form the batch's pending
        # tail.
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

    def key_version(self, key):
        """Read-only per-key version of the last committed write.

        Returns the commit seq of the most recent complete commit whose
        batch set or deleted ``key``: a key no committed batch ever
        touched yields 0, a deleted key keeps the seq of the commit that
        deleted it, and commits that never touched the key leave its
        version unchanged. Only the last complete commit boundary is
        observed -- complete but uncommitted set/delete records and a
        discardable trailing write fragment are invisible, so the version
        always names a committed state the caller could have read.

        The whole log is validated under the exact recover rules first:
        a missing or empty log yields 0 (and is never created), any
        corruption raises WalCorruptionError with no partial result, a
        closed instance raises WalClosedError before the key is even
        validated, and a non-string key raises ValueError before the log
        is read. The query never appends, truncates, or rewrites the log
        and never changes state, commit_seq, pending_count, or the cached
        accepted-prefix boundary a later append relies on; repeated calls
        on the same acceptable log prefix, including after a reopen,
        return the same version. It is the read half of
        set_if_version/delete_if_version: the seq it returns is exactly
        the expected_seq those entries compare against.
        """
        self._check_open()
        _validate_key(key)
        batches = []
        # Purely observational replay, exactly as in scan, diff, and
        # pending_changes: restore the cached accepted-prefix boundary so
        # the query can never influence a later append's decision to drop
        # a tail fragment.
        saved_valid_size = self._valid_size
        try:
            self._replay(batches=batches)
        finally:
            self._valid_size = saved_valid_size
        return _version_of_key(batches, key)

    def scan(self, start_key=None, end_key=None, limit=None):
        """Read-only deterministic range scan of the last committed view.

        Returns the committed items whose keys lie in the half-open range
        [start_key, end_key), sorted by Unicode code point and capped at
        limit items; each item is exactly {"key": key, "value": value} with
        an independent deep copy of the value. An omitted bound means
        unbounded on that side, an omitted limit means no cap, and a zero
        limit returns an empty list. With no matching keys -- including a
        missing or empty log -- the result is an empty list and no file is
        created.

        start_key and end_key must each be a string or omitted, start_key
        must not exceed end_key, and limit must be a non-boolean
        non-negative integer (or omitted); every such rejection is a
        ValueError raised before the log is read. The whole log is then
        validated under the exact recover rules: corruption anywhere --
        the committed region, terminated uncommitted records, or an
        unrecognisable tail -- raises WalCorruptionError with no partial
        result, while a single trailing interrupted-write fragment is
        ignored by the recover rules. Uncommitted set/delete records are
        never visible: only the state of the last complete commit is
        served.

        The scan never appends, truncates, or rewrites the log and never
        changes state, commit_seq, pending_count, or the cached
        accepted-prefix boundary; a read-only instance scans without
        taking the write lease and an exclusive instance keeps its lease
        untouched. Repeated scans of the same acceptable log prefix,
        including after a reopen, return exactly the same sequence.
        """
        # Closed takes priority, exactly as every other public query does.
        self._check_open()
        # Reject every argument before the log is ever read: a rejected
        # call must never read, create, truncate, or append to the file.
        if start_key is not None and not isinstance(start_key, str):
            raise ValueError(
                "start_key must be a string or omitted, got %r"
                % (type(start_key).__name__,)
            )
        if end_key is not None and not isinstance(end_key, str):
            raise ValueError(
                "end_key must be a string or omitted, got %r"
                % (type(end_key).__name__,)
            )
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError(
                "limit must be a non-negative integer or omitted, got %r"
                % (limit,)
            )
        if (
            start_key is not None
            and end_key is not None
            and start_key > end_key
        ):
            raise ValueError(
                "start_key %r exceeds end_key %r" % (start_key, end_key)
            )
        # Validate the whole log under the exact recover rules, purely into
        # local objects: corruption raises WalCorruptionError with no
        # partial result and no change to state or commit_seq. As in audit
        # and pending_changes, restore the cached accepted-prefix boundary
        # so an observational scan can never influence a later append's
        # decision to drop a tail fragment. The committed view contains
        # only the last complete commit: pending records and a discarded
        # tail fragment are invisible by construction.
        saved_valid_size = self._valid_size
        try:
            state, _committed = self._committed_view()
        finally:
            self._valid_size = saved_valid_size
        # Sort by Unicode code point first, then apply the half-open
        # [start_key, end_key) window and the limit cap in that order, so
        # limit always means the first limit ordered matches.
        keys = sorted(state)
        if start_key is not None:
            keys = [key for key in keys if key >= start_key]
        if end_key is not None:
            keys = [key for key in keys if key < end_key]
        if limit is not None:
            keys = keys[:limit]
        return [
            {"key": key, "value": copy.deepcopy(state[key])} for key in keys
        ]

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

    def view(self, target_seq=None):
        """Open a fixed committed-read view at a past commit boundary.

        Unlike snapshot() -- which returns one deep-copied mapping and
        re-validates the log on every later query -- view() replays the
        log once under the exact recover rules and captures the target
        boundary's complete state inside the returned WalView, which
        never reads the log again. Subsequent commits on this store,
        closing this store, and out-of-band modification or corruption of
        the log file therefore cannot invalidate the view or change a
        value it serves. The view offers no write entry point: the call
        appends no WAL bytes, creates no pending records, consumes no
        seq, and takes no write lease.

        target_seq omitted or None selects the newest complete commit
        visible to this replay; 0 selects the empty state. Any other
        value must be a non-boolean non-negative integer no greater than
        the latest committed seq, otherwise ValueError is raised. A
        missing or empty log (and a log containing only an uncommitted
        batch or discarded tail fragment) yields the seq-0 empty view.

        The whole log is validated before the view is built, exactly as
        recover/snapshot require: corruption anywhere raises
        WalCorruptionError with no partial view, a single trailing
        interrupted-write fragment is ignored under the recover rules,
        complete but uncommitted set/delete records are invisible, and on
        a strict_tail instance a recognised tail fragment raises
        WalTailError instead. integrity=True on a non-empty unprotected
        legacy log still raises WalIntegrityError with the same
        precedence. A closed instance raises WalClosedError before any
        argument is checked. None of these rejections writes, truncates,
        appends, or adopts anything, and the call never changes the
        results of audit, recover, snapshot, history, diff, scan,
        pending_changes, or key_version (the observational replay
        restores the cached accepted-prefix boundary exactly as those
        entries do). Repeated calls with the same target_seq on the same
        log boundary return equal views, and mutating an object one view
        returns never affects a later read or another view.
        """
        self._check_open()
        history_views = []
        # Purely observational replay, exactly as in scan, diff,
        # pending_changes, audit, and key_version: restore the cached
        # accepted-prefix boundary so building a view can never influence
        # a later append's decision to drop a tail fragment. The replay
        # builds only local objects and never touches self.state, so
        # WalCorruptionError/WalTailError/WalIntegrityError propagate with
        # nothing adopted.
        saved_valid_size = self._valid_size
        try:
            (
                _candidate,
                committed,
                _pending_count,
                _valid_size,
                _committed_size,
            ) = self._replay(snapshots=history_views)
        finally:
            self._valid_size = saved_valid_size
        if target_seq is None:
            target_seq = committed
        elif (
            isinstance(target_seq, bool)
            or not isinstance(target_seq, int)
            or target_seq < 0
        ):
            raise ValueError(
                "target_seq must be a non-negative integer, got %r"
                % (target_seq,)
            )
        if target_seq > committed:
            raise ValueError(
                "target_seq %r exceeds latest committed seq %r"
                % (target_seq, committed)
            )
        if target_seq == 0:
            boundary_state = {}
        else:
            # Commit seqs are contiguous from 1; the recorded snapshot is
            # already a deep copy private to this replay. WalView copies
            # once more in its constructor.
            boundary_state = dict(history_views)[target_seq]
        return WalView(target_seq, boundary_state)

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

    def diff(self, from_seq, to_seq):
        """Read-only key-level diff between two committed states.

        from_seq names the baseline committed state and to_seq the target
        state; seq 0 is the empty object on either side. Both must be
        non-boolean non-negative integers with from_seq <= to_seq -- every
        such form error (booleans, non-integers, negatives, an inverted
        range) raises ValueError before the log is ever read, and a closed
        instance raises WalClosedError before any of these checks. The
        whole log is then validated under the exact recover rules: any
        corruption anywhere raises WalCorruptionError with no partial
        result, a trailing interrupted-write fragment is ignored by the
        recover rules, and complete but uncommitted set/delete records
        never enter the diff. Once the latest committed seq is known, a
        from_seq or to_seq beyond it also raises ValueError; an empty log
        (latest seq 0) therefore only accepts diff(0, 0).

        Returns a RecoveryResult with from_seq, to_seq, and changes: the
        minimal key-level modifications turning the baseline into the
        target, at most one entry per key, sorted by Unicode code point.
        A key added or holding a different value (compared with the
        JSON-type-aware rules, so 1 and true -- nested or not -- differ)
        yields {"op": "set", "key", "value"}; a key missing from the
        target yields {"op": "delete", "key"}. Identical states yield an
        empty changes list. The result and every nested value are
        independent deep copies: mutating them never affects the store, a
        later commit, or a repeated call.

        The diff is purely observational: it never appends, truncates,
        reorders, or rewrites the log, never creates a missing log file,
        and never changes state, commit_seq, pending_count, or the cached
        accepted-prefix boundary a later append relies on. It runs on
        read-only instances (without taking the write lease) and on
        exclusive instances (leaving the lease untouched), and repeated
        calls on the same acceptable log prefix, including after a
        reopen, return identical results.
        """
        # Closed takes priority, exactly as every other public query does.
        self._check_open()
        # Reject every argument form error before the log is ever read: a
        # rejected call must never read, create, truncate, or append to
        # the file.
        if (
            isinstance(from_seq, bool)
            or not isinstance(from_seq, int)
            or from_seq < 0
        ):
            raise ValueError(
                "from_seq must be a non-negative integer, got %r" % (from_seq,)
            )
        if (
            isinstance(to_seq, bool)
            or not isinstance(to_seq, int)
            or to_seq < 0
        ):
            raise ValueError(
                "to_seq must be a non-negative integer, got %r" % (to_seq,)
            )
        if from_seq > to_seq:
            raise ValueError(
                "from_seq %r exceeds to_seq %r" % (from_seq, to_seq)
            )
        # Validate the whole log under the exact recover rules, purely
        # into local objects: corruption raises WalCorruptionError with no
        # partial result and no change to state or commit_seq. As in
        # audit, scan, and pending_changes, restore the cached
        # accepted-prefix boundary so an observational diff can never
        # influence a later append's decision to drop a tail fragment.
        history_views = []
        saved_valid_size = self._valid_size
        try:
            _candidate, committed, _pending, _valid_size, _committed_size = (
                self._replay(snapshots=history_views)
            )
        finally:
            self._valid_size = saved_valid_size
        if from_seq > committed:
            raise ValueError(
                "from_seq %r exceeds latest committed seq %r"
                % (from_seq, committed)
            )
        if to_seq > committed:
            raise ValueError(
                "to_seq %r exceeds latest committed seq %r"
                % (to_seq, committed)
            )
        # Commit seqs are contiguous from 1, so each boundary's view is
        # exactly the snapshot recorded at that commit; 0 is the empty
        # object. The snapshots are already deep copies private to this
        # call.
        views = dict(history_views)
        base = {} if from_seq == 0 else views[from_seq]
        target = {} if to_seq == 0 else views[to_seq]
        # Merge the two key sets and emit deterministically in Unicode
        # (code point) key order, at most one entry per key: a key only
        # in the target, or present with a different value under
        # JSON-type-aware comparison, is a set; a key present in the
        # baseline but absent from the target is a delete.
        changes = []
        for key in sorted(set(base) | set(target)):
            if key not in target:
                changes.append({"op": "delete", "key": key})
            elif key not in base or not _json_equal(base[key], target[key]):
                # Deep copy: the returned value must not share objects
                # with anything the store or a later call could mutate.
                changes.append(
                    {"op": "set", "key": key, "value": copy.deepcopy(target[key])}
                )
        return RecoveryResult(from_seq=from_seq, to_seq=to_seq, changes=changes)

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
            ) = self._replay(strict=False)
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
        # the same valid_bytes audit reports. repair_tail always uses the
        # default discard policy: on a strict-tail store it is the explicit
        # escape hatch that clears a fragment every other entry refuses.
        (
            candidate,
            committed,
            pending_count,
            valid_size,
            _committed_size,
        ) = self._replay(strict=False)
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

    def _replay(
        self,
        snapshots=None,
        batches=None,
        pending_out=None,
        strict=None,
        idempotent_out=None,
    ):
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
        # replay). idempotent_out: likewise, one (request_id, seq,
        # changes) entry per durable commit record carrying a request_id
        # marker, in commit order, so apply_idempotent can recognise a
        # retry of an already-committed request. All four are purely
        # observational: the replay itself, its return value, and the log
        # are unaffected.
        #
        # strict selects the tail policy: None means the store's own
        # strict_tail setting (every replaying public entry), False forces
        # the default discard behaviour (audit and repair_tail always keep
        # reporting/clearing the fragment), and True forces refusal. When
        # the effective policy is strict, a recognised trailing fragment
        # raises WalTailError here -- after every other validation has
        # passed, so corruption and WalIntegrityError keep priority --
        # before any candidate state, boundary, or chain head is adopted.
        candidate = {}
        pending = []
        committed = 0
        valid_size = 0
        # Byte offset just past the last durable commit record (0 when there
        # is no committed prefix); rollback truncates at exactly this point.
        committed_size = 0
        # Integrity chain state for protected logs: log_protected is decided
        # by the first parsed record (None until then) and must agree with
        # every later record -- protected and unprotected records may never
        # mix. chain is the running digest the next record must chain onto.
        log_protected = None
        chain = _CHAIN_SEED
        # Request identifiers of durable idempotent commits already seen in
        # this replay: the API persists each identifier at most once, so a
        # repetition can only come from out-of-band tampering.
        seen_request_ids = set()
        if strict is None:
            strict = self._strict_tail
        # Kind of the single recover-recognised trailing fragment, if one
        # is found: "truncated_utf8", "incomplete_json", or
        # "missing_terminator". Truncated UTF-8 is recorded the moment the
        # cut trailing bytes are stripped, so it always takes precedence
        # over the JSON-level classification of the bytes before them.
        tail_kind = None
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
                    tail_kind = "truncated_utf8"
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
                        # Truncated UTF-8 earlier in the same fragment keeps
                        # precedence over the JSON-level classification.
                        if tail_kind is None:
                            tail_kind = "incomplete_json"
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
                # The first parsed record decides the log's format; every
                # later record must carry the same protection. Mixing
                # protected and unprotected records is corruption, never a
                # format upgrade.
                has_integrity = _INTEGRITY_FIELD in row
                if log_protected is None:
                    log_protected = has_integrity
                elif has_integrity != log_protected:
                    raise WalCorruptionError(
                        "protected and unprotected records are mixed in the log"
                    )
                expected_fields = (
                    schema | {_INTEGRITY_FIELD} if log_protected else schema
                )
                fields = set(row)
                if op == "commit" and "request_id" in fields:
                    # Optional idempotent-commit marker: a non-empty string,
                    # unique across the whole log. Only commit records may
                    # carry it; a malformed or repeated identifier is
                    # corruption, never a discardable fragment.
                    request_id = row["request_id"]
                    if not isinstance(request_id, str) or not request_id:
                        raise WalCorruptionError(
                            "request_id is not a non-empty string: %r"
                            % (request_id,)
                        )
                    if request_id in seen_request_ids:
                        raise WalCorruptionError(
                            "duplicate request_id in log: %r" % (request_id,)
                        )
                    seen_request_ids.add(request_id)
                    fields.discard("request_id")
                if fields != expected_fields:
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
                if log_protected:
                    # Verify the record's integrity metadata before it can be
                    # accepted -- or even recognised as a discardable
                    # unfinished fragment: an integrity mismatch is always
                    # corruption, never a repairable tail. The digest chains
                    # the canonical core line (the record without its
                    # metadata) onto the previous record's digest, covering
                    # content, commit seq, and operation order.
                    core_line = json.dumps(
                        {k: v for k, v in row.items() if k != _INTEGRITY_FIELD},
                        sort_keys=True,
                    )
                    expected_ic = _chain_digest(chain, core_line)
                    if row[_INTEGRITY_FIELD] != expected_ic:
                        raise WalCorruptionError(
                            "integrity metadata mismatch for seq %r" % (seq,)
                        )
                if not terminated:
                    # The record's terminator never became durable, so the
                    # write is unfinished: the fragment is discarded whole --
                    # not applied, not counted as pending, and its bytes stay
                    # beyond valid_size for a later append or rollback to
                    # remove. Validation above has already run, so only a
                    # fragment that is recognisably one complete record
                    # reaches this point; anything else raised already. The
                    # integrity chain head is NOT advanced past it either:
                    # the fragment's bytes are removed by the next append or
                    # rollback, so the next record must chain onto the last
                    # adopted (terminated) record, exactly as valid_size
                    # only covers the accepted prefix.
                    if tail_kind is None:
                        tail_kind = "missing_terminator"
                    break
                if log_protected:
                    chain = expected_ic
                if op == "commit":
                    if batches is not None or idempotent_out is not None:
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
                        if batches is not None:
                            batches.append((seq, changes))
                        if idempotent_out is not None and "request_id" in row:
                            # The durable occupancy record of an idempotent
                            # commit: the request identifier, the commit seq
                            # that sealed its batch, and the batch itself as
                            # an independent deep copy.
                            idempotent_out.append(
                                (row["request_id"], seq, copy.deepcopy(changes))
                            )
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
        # Adopt the format the log itself proved to have. A protected log
        # stays protected even when opened in the default mode; a log with
        # no parsed records at all (missing, empty, or only an
        # unrecognisable interrupted-write prefix) keeps the mode the store
        # already had, so an integrity=True store starts a protected chain
        # on an empty log. A non-empty log that is fully unprotected is a
        # legacy log: the default mode keeps reading and extending it in
        # the legacy format, but integrity=True refuses it with the unique
        # WalIntegrityError -- before any state, boundary, or chain head is
        # adopted.
        if log_protected is True:
            self._protected = True
        elif log_protected is False:
            if self._integrity_requested:
                raise WalIntegrityError(
                    "log is not empty and carries no integrity protection: %r"
                    % (str(self.path),)
                )
            self._protected = False
        if strict and tail_kind is not None:
            # Strict tail policy: full validation above has already passed
            # (corruption -- including a protected-log integrity mismatch --
            # raised WalCorruptionError, and a legacy log under
            # integrity=True raised WalIntegrityError, both taking
            # precedence), so the bytes beyond valid_size are exactly the
            # one recover-recognised trailing fragment. Refuse before any
            # candidate state, boundary, or chain head is adopted: nothing
            # is truncated, appended, or created, and a constructor-side
            # refusal drops the exclusive lease in __init__. tail_kind was
            # set only while parsing bytes of an existing file, so data is
            # bound here.
            raise WalTailError(
                valid_size, len(data) - valid_size, tail_kind, self.path
            )
        self._chain_head = chain
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
