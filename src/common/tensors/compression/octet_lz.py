"""LZ77 match coding of an octet string, encoded and decoded in parallel.

Packet layout::

    output length (8) | group count (8) | command octets (8) | commands | literals

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


def _orbit(jump: AbstractTensor, steps: int) -> tuple[AbstractTensor, int]:
    """``jump**t(0)`` for every ``t < steps``, and ``jump**steps(0)``."""

    ordinals = _arange(steps, jump)
    starts = ordinals * 0
    end = _tensor([0])
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


def lz_encode(data: bytes) -> bytes:
    """The match-coded packet of ``data``."""

    data = bytes(data)
    n = len(data)
    header = n.to_bytes(8, "little")
    if n == 0:
        return header + bytes(16)
    octets = _tensor(data)
    grams = n - MIN_MATCH + 1
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
        repeated = ((sorted_gram[1:] - sorted_gram[:-1]) == 0).to_dtype("int64")
        candidate = _place(
            grams, order[1:], (order[:-1] + 1) * repeated,
        ) - 1
        has = (candidate >= 0).to_dtype("int64")
        distance = (positions - candidate) * has
        # A run continues while the next position matches at the same
        # distance; its last position is the first that does not continue.
        continues = AbstractTensor.cat((
            has[:-1] * has[1:] * ((distance[1:] - distance[:-1]) == 0).to_dtype("int64"),
            has[-1:] * 0,
        ), dim=0)
        run_end = positions + continues
        for _ in range(max(1, grams.bit_length())):
            run_end = run_end[run_end]
        length = (run_end - positions + MIN_MATCH) * has
        pad = _zeros(n - grams, octets)
        token = AbstractTensor.cat((length * has + (1 - has), pad + 1), dim=0)
        match_length = AbstractTensor.cat((length, pad), dim=0)
        match_distance = AbstractTensor.cat((distance, pad), dim=0)

    # The greedy parse: the orbit of 0 under i -> i + token length.
    jump = AbstractTensor.cat((_arange(n, octets) + token, _tensor([n])), dim=0)
    steps = _int(_hops_to_end(jump, n)[0])
    starts, end = _orbit(jump, steps)
    if end != n:
        raise AssertionError("LZ parse did not end on the input length")
    is_match = (match_length[starts] > 0).to_dtype("int64")
    match_at = _compact(is_match, starts)
    lengths = match_length[match_at]
    distances = match_distance[match_at]
    ends = match_at + lengths
    previous_end = AbstractTensor.cat((_tensor([0]), ends[:-1]), dim=0) if (
        match_at.shape[0]
    ) else _tensor([0])[:0]
    trailing = n - (_int(ends[-1]) if match_at.shape[0] else 0)
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
        n + 1,
        AbstractTensor.cat((match_at, ends), dim=0),
        AbstractTensor.cat((lengths * 0 + 1, lengths * 0 - 1), dim=0),
    ).cumsum(dim=0)[:n]
    literal = (covered == 0).to_dtype("int64")
    literals = tensor_octets_to_bytes(_compact(literal, octets))
    return (
        header + groups.to_bytes(8, "little")
        + len(commands).to_bytes(8, "little") + commands + literals
    )


def lz_decode(packet: bytes) -> bytes:
    """Inverse of :func:`lz_encode`; refuses a malformed packet."""

    packet = bytes(packet)
    if len(packet) < 24:
        raise ValueError("LZ packet has no header")
    n = int.from_bytes(packet[:8], "little")
    groups = int.from_bytes(packet[8:16], "little")
    command_bytes = int.from_bytes(packet[16:24], "little")
    if n == 0:
        if groups or command_bytes or len(packet) != 24:
            raise ValueError("empty LZ packet carries data")
        return b""
    if 24 + command_bytes > len(packet):
        raise ValueError("LZ packet commands overrun the packet")
    commands = _tensor(packet[24:24 + command_bytes])
    literals = _tensor(packet[24 + command_bytes:]) if (
        len(packet) > 24 + command_bytes
    ) else commands[:0]

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
        or _int(literal_counts.sum()) != len(packet) - 24 - command_bytes
        or _int(((lengths > 0).to_dtype("int64") * (distances < 1).to_dtype("int64")).sum())
    ):
        raise ValueError("LZ packet groups disagree with its lengths")

    group_start = sizes.cumsum(dim=0) - sizes
    literal_start = literal_counts.cumsum(dim=0) - literal_counts
    group = _place(n + 1, group_start, sizes * 0 + 1).cumsum(dim=0)[:n] - 1
    output = _arange(n, commands)
    offset = output - group_start[group]
    is_literal = (offset < literal_counts[group]).to_dtype("int64")
    source = output - distances[group] * (1 - is_literal)
    if _int(source.min()) < 0:
        raise ValueError("LZ packet copies from before its output")
    literal_index = (literal_start[group] + offset) * is_literal
    value = literals[literal_index] * is_literal if literals.shape[0] else output * 0
    for _ in range(max(1, n.bit_length())):
        source = source[source]
    if _int(is_literal[source].min()) != 1:
        raise ValueError("LZ packet copy does not reach a literal")
    return tensor_octets_to_bytes(value[source])


__all__ = ["MIN_MATCH", "lz_decode", "lz_encode"]
