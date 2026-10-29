"""LZ77 match coding of an octet string, encoded and decoded in parallel.

Packet layout::

    output length | group count | command octets   (varints) | commands | literals

A *group* is ``(literal count, match length, match distance)``: that many
literals copied from the literal stream, then ``match length`` octets copied
from ``distance`` back in the output.  The last group carries the trailing
literals and a zero-length match.  Commands are the groups' three fields as
little-endian base-128 varints.

Encoding is tensor composition:

1. every position's 4-gram, as one exact integer;
2. a stable sort of (gram, position) gives every position the most recent
   earlier position with the same gram -- its match candidate at distance
   ``d``;
3. a run of consecutive positions sharing one candidate distance is one
   match: each 4-gram in the run is equal, so every octet through the run's
   last gram is.  Run ends are found by pointer doubling;
4. the greedy parse -- a match where one starts, else a literal -- is the
   orbit of position 0 under ``i -> i + token length``, read by pointer
   doubling as in ``octet_huffman``;
5. groups, literals and varints are compacted with prefix sums and scatter.

Decoding parses every varint at once (an octet below 128 ends one), places
every output position in its group with one prefix sum, and resolves each
copied octet to the literal it ultimately copies by pointer doubling over
``o -> o - distance``.  A malformed packet refuses with ``ValueError``.
"""

from __future__ import annotations

from ..abstraction import AbstractTensor
from .bitstream import tensor_octets_to_bytes

#: Shortest match worth a group.
MIN_MATCH = 4
_VARINT_LIMIT = 8   # octets: values stay below 2**56


def _tensor(values) -> AbstractTensor:
    return AbstractTensor.tensor(list(values)).to_dtype("int64")


def _int(value: AbstractTensor) -> int:
    return int(value.item())


def _zeros(count: int, like: AbstractTensor) -> AbstractTensor:
    return AbstractTensor.zeros((count,), cls=type(like)).to_dtype("int64")


def _arange(count: int, like: AbstractTensor) -> AbstractTensor:
    return AbstractTensor.arange(count, cls=type(like)).to_dtype("int64")


def _place(count: int, index: AbstractTensor, values: AbstractTensor) -> AbstractTensor:
    """``count`` zeros with ``values`` summed in at ``index``."""

    if index.shape[0] == 0:
        return _zeros(count, values if values.shape[0] else _tensor([0]))
    return AbstractTensor.scatter(
        _zeros(count, values), index.to_dtype("int64"), values.to_dtype("int64"),
        dim=0,
    ).to_dtype("int64")


def _orbit(
    jump: AbstractTensor, steps: int, start: int = 0,
) -> tuple[AbstractTensor, int]:
    """``jump**t(start)`` for every ``t < steps``, and ``jump**steps(start)``."""

    ordinals = _arange(steps, jump)
    starts = ordinals * 0 + start
    end = _tensor([start])
    power = 1
    while power <= steps:
        taken = (ordinals // power) % 2
        starts = starts + taken * (jump[starts] - starts)
        if steps & power:
            end = jump[end]
        jump = jump[jump]
        power *= 2
    return starts, _int(end[0])


def _hops_to_end(jump: AbstractTensor, end: int) -> AbstractTensor:
    """Hops from each position to the absorbing ``end`` (Wyllie ranking)."""

    hops = (_arange(jump.shape[0], jump) - end != 0).to_dtype("int64")
    for _ in range(max(1, (jump.shape[0]).bit_length())):
        hops = hops + hops[jump]
        jump = jump[jump]
    return hops


def _varints(values: AbstractTensor) -> AbstractTensor:
    widths = values * 0 + 1
    for octet in range(1, _VARINT_LIMIT):
        widths = widths + (values >= 2 ** (7 * octet)).to_dtype("int64")
    starts = widths.cumsum(dim=0) - widths
    total = _int(widths.sum())
    index, contribution = [], []
    for octet in range(_VARINT_LIMIT):
        live = (widths > octet).to_dtype("int64")
        more = (widths > octet + 1).to_dtype("int64")
        digit = (values // (2 ** (7 * octet))) % 128 + 128 * more
        index.append((starts + octet) * live + total * (1 - live))
        contribution.append(digit * live)
    return _place(
        total + 1,
        AbstractTensor.cat(index, dim=0),
        AbstractTensor.cat(contribution, dim=0),
    )[:total]


def _compact(mask: AbstractTensor, values: AbstractTensor) -> AbstractTensor:
    """``values`` where ``mask``, in order."""

    count = _int(mask.sum())
    slots = mask.cumsum(dim=0) - mask
    return _place(
        count + 1, slots * mask + count * (1 - mask), values * mask,
    )[:count]


#: Earlier occurrences of a position's 4-gram tried as its match.
CANDIDATES = 2
#: Longest match: matches are measured by binary search up to this.
MAX_MATCH = 1 << 10
_MODULI = (2147483647, 2147483629)
_BASE = 257


def _powers(base: int, modulus: int, count: int, like) -> AbstractTensor:
    """``base**k % modulus`` for ``k < count``, by binary exponent."""

    exponents = _arange(count, like)
    value = exponents * 0 + 1
    square = base % modulus
    bit = 1
    while bit < max(count, 2):
        taken = (exponents // bit) % 2
        value = (value * (1 + taken * (square - 1))) % modulus
        square = square * square % modulus
        bit *= 2
    return value


def _prefix_hashes(octets: AbstractTensor, modulus: int):
    """``H[k]``, the hash of ``octets[:k]``, and the base powers."""

    count = octets.shape[0]
    powers = _powers(_BASE, modulus, count + 1, octets)
    inverse = _powers(pow(_BASE, modulus - 2, modulus), modulus, count, octets)
    weighted = (octets * inverse) % modulus
    sums = AbstractTensor.cat(
        (_tensor([0]), weighted.cumsum(dim=0) % modulus), dim=0,
    )
    shift = AbstractTensor.cat((_tensor([0]), powers[:count]), dim=0)
    # H[k] = sum b[j] B^(k-1-j) = B^(k-1) * sum b[j] B^-j.
    return (sums * shift) % modulus, powers


def _longest_matches(octets, order, sorted_gram, positions, total):
    """Each position's longest match among its ``CANDIDATES`` most recent
    earlier occurrences of its 4-gram: ``(length, distance)``, 0 for none.

    Lengths are found by binary search on two polynomial prefix hashes and
    then checked octet by octet; a position whose check fails falls back to
    the exact shared-distance run below it.
    """

    grams = positions.shape[0]
    tables = [_prefix_hashes(octets, modulus) for modulus in _MODULI]
    best_length = positions * 0
    best_distance = positions * 0
    for back in range(1, min(CANDIDATES, grams - 1) + 1):
        same = ((sorted_gram[back:] - sorted_gram[:-back]) == 0).to_dtype("int64")
        candidate = _place(grams, order[back:], (order[:-back] + 1) * same) - 1
        has = (candidate >= 0).to_dtype("int64")
        source = candidate * has
        limit = (total - positions).minimum(MAX_MATCH) * has
        length = positions * 0
        step = MAX_MATCH
        while step:
            trial = length + step
            fits = (trial <= limit).to_dtype("int64")
            safe = trial * fits
            equal = fits
            for (hashes, powers), modulus in zip(tables, _MODULI):
                here = (hashes[positions + safe] - hashes[positions] * powers[safe]) % modulus
                there = (hashes[source + safe] - hashes[source] * powers[safe]) % modulus
                equal = equal * ((here - there) == 0).to_dtype("int64")
            length = length + step * equal
            step //= 2
        better = (length > best_length).to_dtype("int64") * (length >= MIN_MATCH).to_dtype("int64")
        best_length = best_length + better * (length - best_length)
        best_distance = best_distance + better * ((positions - candidate) - best_distance)
    return best_length, best_distance


def _verified(octets, match_at, lengths, distances, total):
    """Whether every matched octet equals the octet it copies."""

    if match_at.shape[0] == 0:
        return True
    marks = _place(
        total + 1,
        AbstractTensor.cat((match_at, match_at + lengths), dim=0),
        AbstractTensor.cat((distances, distances * -1), dim=0),
    ).cumsum(dim=0)[:total]
    covered = (marks > 0).to_dtype("int64")
    places = _arange(total, octets)
    source = places - marks * covered
    return _int(((octets - octets[source]) * covered).abs().sum()) == 0


def _header(*values: int) -> bytes:
    out = bytearray()
    for value in values:
        while True:
            low, value = value & 0x7F, value >> 7
            out.append(low | (0x80 if value else 0))
            if not value:
                break
    return bytes(out)


def _read_header(packet: bytes, count: int) -> tuple[list[int], int]:
    values, at = [], 0
    for _ in range(count):
        value = shift = 0
        while True:
            if at >= len(packet) or shift > 7 * _VARINT_LIMIT:
                raise ValueError("LZ packet header is truncated")
            octet = packet[at]
            at += 1
            value |= (octet & 0x7F) << shift
            shift += 7
            if not octet & 0x80:
                break
        values.append(value)
    return values, at


def lz_encode(data: bytes, dictionary: bytes = b"") -> bytes:
    """The match-coded packet of ``data``.

    ``dictionary`` is text both sides already hold: matches may copy from
    it as though it preceded ``data``, and none of it is written.
    """

    data, dictionary = bytes(data), bytes(dictionary)
    n, known = len(data), len(dictionary)
    if n == 0:
        return _header(0, 0, 0)
    total = known + n
    octets = _tensor(dictionary + data)
    grams = total - MIN_MATCH + 1
    token = octets * 0 + 1
    match_length = octets * 0
    match_distance = octets * 0
    if grams > 1:
        gram = octets[:grams] * 0
        for offset in range(MIN_MATCH):
            gram = gram * 256 + octets[offset:offset + grams]
        positions = _arange(grams, octets)
        order = octets.ensure_tensor(AbstractTensor.topk(
            -(gram * grams + positions), k=grams, dim=0,
        ).indices).to_dtype("int64")
        sorted_gram = gram[order]
        length, distance = _longest_matches(
            octets, order, sorted_gram, positions, total,
        )
        has = (length > 0).to_dtype("int64")
        pad = _zeros(total - grams, octets)
        token = AbstractTensor.cat((length * has + (1 - has), pad + 1), dim=0)
        match_length = AbstractTensor.cat((length, pad), dim=0)
        match_distance = AbstractTensor.cat((distance, pad), dim=0)

    for _attempt in range(2):
        # The greedy parse: the orbit of ``known`` (the first octet of
        # ``data``) under i -> i + token length.
        jump = AbstractTensor.cat(
            (_arange(total, octets) + token, _tensor([total])), dim=0,
        )
        steps = _int(_hops_to_end(jump, total)[known])
        starts, end = _orbit(jump, steps, known)
        if end != total:
            raise AssertionError("LZ parse did not end on the input length")
        is_match = (match_length[starts] > 0).to_dtype("int64")
        match_at = _compact(is_match, starts)
        lengths = match_length[match_at]
        distances = match_distance[match_at]
        if _verified(octets, match_at, lengths, distances, total):
            break
        # A hash collision measured a match too long: code literally.
        token = octets * 0 + 1
        match_length = octets * 0
        match_distance = octets * 0
    ends = match_at + lengths
    previous_end = AbstractTensor.cat((_tensor([known]), ends[:-1]), dim=0) if (
        match_at.shape[0]
    ) else _tensor([0])[:0]
    trailing = total - (_int(ends[-1]) if match_at.shape[0] else known)
    literal_counts = AbstractTensor.cat(
        (match_at - previous_end, _tensor([trailing])), dim=0,
    )
    groups = literal_counts.shape[0]
    fields = AbstractTensor.stack((
        literal_counts,
        AbstractTensor.cat((lengths, _tensor([0])), dim=0),
        AbstractTensor.cat((distances, _tensor([0])), dim=0),
    ), dim=1).reshape(-1)
    commands = tensor_octets_to_bytes(_varints(fields))

    covered = _place(
        total + 1,
        AbstractTensor.cat((match_at, ends), dim=0),
        AbstractTensor.cat((lengths * 0 + 1, lengths * 0 - 1), dim=0),
    ).cumsum(dim=0)[known:total]
    literal = (covered == 0).to_dtype("int64")
    literals = tensor_octets_to_bytes(_compact(literal, octets[known:]))
    return _header(n, groups, len(commands)) + commands + literals


def lz_decode(packet: bytes, dictionary: bytes = b"") -> bytes:
    """Inverse of :func:`lz_encode` with the same ``dictionary``; refuses a
    malformed packet."""

    packet, dictionary = bytes(packet), bytes(dictionary)
    (n, groups, command_bytes), at = _read_header(packet, 3)
    if n == 0:
        if groups or command_bytes or len(packet) != at:
            raise ValueError("empty LZ packet carries data")
        return b""
    if at + command_bytes > len(packet):
        raise ValueError("LZ packet commands overrun the packet")
    known = len(dictionary)
    total = known + n
    commands = _tensor(packet[at:at + command_bytes])
    literals = _tensor(dictionary + packet[at + command_bytes:])

    final = (commands < 128).to_dtype("int64")
    if command_bytes == 0 or _int(final[-1]) != 1 or _int(final.sum()) != 3 * groups:
        raise ValueError("LZ packet commands are not whole varints")
    which = final.cumsum(dim=0) - final
    places = _arange(command_bytes, commands)
    stops = _compact(final, places)
    first = AbstractTensor.cat((_tensor([0]), stops[:-1] + 1), dim=0)
    within = places - first[which]
    if _int(within.max()) >= _VARINT_LIMIT:
        raise ValueError("LZ packet varint is too long")
    fields = _place(
        3 * groups, which, (commands % 128) * (2 ** (7 * within)),
    ).reshape(groups, 3)
    literal_counts, lengths, distances = fields[:, 0], fields[:, 1], fields[:, 2]
    sizes = literal_counts + lengths
    if (
        _int(sizes.sum()) != n
        or _int(literal_counts.sum()) != len(packet) - at - command_bytes
        or _int(((lengths > 0).to_dtype("int64") * (distances < 1).to_dtype("int64")).sum())
    ):
        raise ValueError("LZ packet groups disagree with its lengths")

    # Output positions 0..known-1 are the dictionary: literals that are
    # their own source.  Literal octets follow it in ``literals``.
    group_start = sizes.cumsum(dim=0) - sizes + known
    literal_start = literal_counts.cumsum(dim=0) - literal_counts + known
    output = _arange(total, commands)
    group = (
        _place(total + 1, group_start, sizes * 0 + 1).cumsum(dim=0)[:total] - 1
    ).maximum(0)
    in_data = (output >= known).to_dtype("int64")
    offset = (output - group_start[group]) * in_data
    is_literal = 1 - in_data * (offset >= literal_counts[group]).to_dtype("int64")
    source = output - distances[group] * (1 - is_literal)
    if _int(source.min()) < 0:
        raise ValueError("LZ packet copies from before its output")
    literal_index = (
        (literal_start[group] + offset) * in_data + output * (1 - in_data)
    ) * is_literal
    value = literals[literal_index] * is_literal
    for _ in range(max(1, total.bit_length())):
        source = source[source]
    if _int(is_literal[source].min()) != 1:
        raise ValueError("LZ packet copy does not reach a literal")
    return tensor_octets_to_bytes(value[source][known:])


__all__ = ["MIN_MATCH", "lz_decode", "lz_encode"]
