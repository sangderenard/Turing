"""Structured SSA value ids: high bits carry the group and its flags.

Every id in this compiler used to be a bare integer whose *meaning* lived
somewhere else -- an accounting dict, a side table, or nothing at all but a
convention.  Three independent numbering sources exist (``id(node)`` memory
addresses from ProcessGraph ingestion, ``tensor_identity()``'s counter for
AbstractTensor tracing, and ``GLOBAL_MONOTONIC_IDS`` for compiler-invented
values), and passes told them apart by guessing at magnitude: "at or above a
billion means the compiler made it."  That convention was never enforced and
has already been violated -- ``ssa_llvm_backend`` places synthetic history
buffers at exactly ``1_000_000_000``, the same point ``GLOBAL_MONOTONIC_IDS``
starts issuing from, "deliberately far above any real SSA numbering."  Two
ranges, one number, no way to tell which one owns it.

An id here is instead self-describing: the high bits state which group
minted it and which properties hold of it, the low bits are its serial
within that space.  Nothing has to look a value up to know what kind of
thing it is, the ranges cannot silently overlap because each group owns its
own bit, and the concordance can group and label rows straight from the
prefix.

Flags are independent bits, not an enumeration, because the properties are
genuinely independent: a compiler-minted value can also be shared across
call frames, and a tensor-identity token can be either.  An enumeration
would force those combinations into separate, unrelated group numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bit 63 is the sign bit and stays clear so an id is a positive integer in
# every backend that carries it as a signed 64-bit word.  Bit 62 is left
# free as headroom for a flag added later without moving the serial field.
MINTED = 1 << 61
"""The compiler invented this value; it names nothing the source declared."""

SHARED = 1 << 60
"""More than one call frame genuinely holds this id (a reference count over
the whole module said so, not an accounting label a pass chose to write).
Enrichment of a value carrying this bit must keep every frame's occurrence
synchronized; without it, one frame's copy can go stale and later revert a
correct fact -- the ``restore`` round-trip of 2026-09-17/18."""

TENSOR = 1 << 59
"""Originated as a ``tensor_identity()`` token on the AbstractTensor tracing
path, rather than from ProcessGraph structural ingestion."""

HISTORY = 1 << 58
"""A synthetic history-watch buffer, never a value the program owns."""

HISTORY_COUNT = 1 << 57
"""The sample-count companion of a ``HISTORY`` ring, so the two synthetic
buffers a watched value needs are told apart by a bit rather than by a
second numeric base that has to be kept clear of the first."""

FLAG_BITS = MINTED | SHARED | TENSOR | HISTORY | HISTORY_COUNT
SERIAL_BITS = (1 << 57) - 1

_FLAG_NAMES: tuple[tuple[int, str], ...] = (
    (MINTED, "minted"),
    (SHARED, "shared"),
    (TENSOR, "tensor"),
    (HISTORY, "history"),
    (HISTORY_COUNT, "history-count"),
)


def compose(serial: int, flags: int = 0) -> int:
    """One id from a serial and its flags.

    The serial must fit the serial field; a caller that overflows it is
    minting more values than this space was sized for and must be told,
    not silently wrapped into another group's bits.
    """
    serial = int(serial)
    if serial < 0 or serial > SERIAL_BITS:
        raise ValueError(
            f"serial {serial} does not fit the id serial field "
            f"(0..{SERIAL_BITS})"
        )
    if int(flags) & ~FLAG_BITS:
        raise ValueError(f"unknown id flag bits set: {int(flags):#x}")
    return serial | int(flags)


def serial_of(value_id: int) -> int:
    """The serial this id carries, with its group/flag bits removed."""
    return int(value_id) & SERIAL_BITS


def flags_of(value_id: int) -> int:
    """Only the flag bits of this id."""
    return int(value_id) & FLAG_BITS


def has_flag(value_id: int, flag: int) -> bool:
    return bool(int(value_id) & int(flag))


def with_flag(value_id: int, flag: int) -> int:
    """This id with `flag` also set -- its serial and other flags intact."""
    if int(flag) & ~FLAG_BITS:
        raise ValueError(f"unknown id flag bits set: {int(flag):#x}")
    return int(value_id) | int(flag)


def flag_names(value_id: int) -> tuple[str, ...]:
    """Every flag this id carries, in a stable order."""
    return tuple(
        name for bit, name in _FLAG_NAMES if int(value_id) & bit
    )


def is_legacy(value_id: int) -> bool:
    """Whether this id predates the structured space.

    An id with no flag bits at all is one of the unstructured numbers the
    older sources still hand out (a ProcessGraph ``id()`` address, a raw
    tensor-identity token, a pre-migration monotonic value).  It is not an
    error -- the migration is staged -- but it is a value whose group
    cannot be read off the number, and the concordance says so plainly
    rather than claiming a group it does not know.
    """
    return flags_of(value_id) == 0


def describe(value_id: int) -> str:
    """A short, glanceable label: the group/flags, then the serial.

    ``minted|shared#47`` reads at once; ``4899916394579099695`` does not.
    This is what the concordance prints, so a reader never has to decode a
    prefix by hand or consult another table to learn what a row is.
    """
    names = flag_names(value_id)
    if not names:
        return f"legacy#{int(value_id)}"
    return f"{'|'.join(names)}#{serial_of(value_id)}"


def explain(value_id: int) -> str:
    """One id expanded into its sections, for reading by eye.

    ``describe`` is the one-line form that belongs inside a message; this is
    the full decode for when an id turns up somewhere raw -- a traceback, a
    debugger, someone else's log -- and the question is simply "what IS
    this".  Every flag is listed whether or not it is set, so a reader sees
    the whole vocabulary rather than inferring it from the ones that
    happen to be present.
    """
    value_id = int(value_id)
    rows = [f"{value_id}  ->  {describe(value_id)}"]
    for bit, name in _FLAG_NAMES:
        shift = bit.bit_length() - 1
        mark = "set" if value_id & bit else "."
        rows.append(f"  bit {shift:<2} {name:<14} {mark}")
    serial = serial_of(value_id)
    rows.append(f"  serial     {serial} (0x{serial:X})")
    unknown = value_id & ~(FLAG_BITS | SERIAL_BITS)
    if unknown:
        # Bits outside both fields mean this number was never composed by
        # this module -- say so rather than quietly presenting a serial
        # that dropped part of the value on the floor.
        rows.append(f"  UNCLAIMED BITS 0x{unknown:X} -- not a composed id")
    return "\n".join(rows)


def label(value_id: int) -> str:
    """How the concordance prints one id inline.

    A structured id shows its group and flags, because that is the part a
    reader needs and the digits are not.  An id that has no flags yet shows
    its bare number: during the staged migration most ids are still
    unstructured, and prefixing every one of them with ``legacy#`` would
    add noise to every line while saying nothing the reader can act on.
    Use :func:`describe` where the distinction itself matters.
    """
    return str(int(value_id)) if is_legacy(value_id) else describe(value_id)


@dataclass(frozen=True)
class IdGroup:
    """One group's rows, as the concordance presents them together."""

    flags: int
    label: str
    value_ids: tuple[int, ...]


def group_by_prefix(value_ids) -> tuple[IdGroup, ...]:
    """Every id gathered under its own group/flag prefix.

    Presentation, not analysis: rows that share a prefix belong to the same
    space and are worth reading together, and a group that should be empty
    but is not (``history`` ids appearing among a function's own formals,
    say) is visible immediately rather than buried in serial order.
    """
    grouped: dict[int, list[int]] = {}
    for value_id in value_ids:
        grouped.setdefault(flags_of(value_id), []).append(int(value_id))
    ordered = []
    for flags in sorted(grouped):
        names = flag_names(flags)
        ordered.append(IdGroup(
            flags=flags,
            label="|".join(names) if names else "legacy",
            value_ids=tuple(sorted(grouped[flags])),
        ))
    return tuple(ordered)


if __name__ == "__main__":  # pragma: no cover - operator convenience
    # ``python -m src.compiler.id_space 2305843010213707500`` -- paste an id
    # straight from a traceback or a log and read what it is, without
    # writing a scratch script to import this module first.
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.compiler.id_space <id> [<id> ...]")
        raise SystemExit(2)
    for argument in sys.argv[1:]:
        try:
            print(explain(int(argument, 0)))
        except ValueError:
            print(f"{argument}: not an integer")
        print()
