"""Canonical Huffman codebooks expressed with AbstractTensor operations.

There is deliberately no NumPy/SciPy conversion path in this module. Tables,
symbol lookup, ranks, code construction, and codeword expansion remain on the
caller's AbstractTensor backend. Python integers describe static format limits
and tensor shapes only.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..abstraction import AbstractTensor


def _require_tensor(value, name: str) -> AbstractTensor:
    if not isinstance(value, AbstractTensor):
        raise TypeError(f"{name} must be an AbstractTensor")
    return value


def _truth(value: AbstractTensor) -> bool:
    """Cross the tensor/host boundary only for structural validation."""
    return bool(value.item())


def symbol_frequencies(
    symbols: AbstractTensor,
    alphabet: AbstractTensor,
    *,
    valid: AbstractTensor | None = None,
) -> AbstractTensor:
    """Count an explicit integer alphabet through tensor comparison/reduction."""
    values = _require_tensor(symbols, "symbols")
    vocabulary = _require_tensor(alphabet, "alphabet")
    if vocabulary.ndims() != 1 or vocabulary.shape[0] < 1:
        raise ValueError("alphabet must be a nonempty one-dimensional tensor")
    if not _truth(((vocabulary % 1) == 0).all()):
        raise ValueError("alphabet must contain integers")
    same_alphabet_symbol = (
        vocabulary.unsqueeze(1) - vocabulary.unsqueeze(0)
    ) == 0
    if not _truth(
        (
            same_alphabet_symbol.to_dtype("int64").sum(dim=1)
            == 1
        ).all()
    ):
        raise ValueError("alphabet symbols must be unique")

    flat = values.flatten()
    if valid is None:
        validity = (flat * 0 + 1).to_dtype("int64")
    else:
        mask = _require_tensor(valid, "valid")
        if mask.shape != values.shape:
            raise ValueError("valid mask must match the symbol tensor")
        validity = mask.flatten().to_dtype("int64")
    matches = (
        flat.unsqueeze(1) - vocabulary.unsqueeze(0)
    ) == 0
    match_counts = (
        matches.to_dtype("int64") * validity.unsqueeze(1)
    ).sum(dim=1)
    if not _truth((match_counts == validity).all()):
        raise ValueError("a valid symbol lies outside the alphabet")
    return (
        matches.to_dtype("int64") * validity.unsqueeze(1)
    ).sum(dim=0)


def canonical_codes_from_lengths(
    code_lengths: AbstractTensor,
    *,
    max_bits: int,
) -> AbstractTensor:
    """Return the canonical integer code assigned to every dense symbol.

    A zero length means that the symbol is absent. Codes of equal length retain
    symbol order. All numerical work is tensor composition: equality masks,
    reductions, broadcasting, indexing, and the canonical-code recurrence.
    """
    lengths = _require_tensor(code_lengths, "code_lengths")
    if lengths.ndims() != 1:
        raise ValueError("code_lengths must be one-dimensional")
    if max_bits < 1:
        raise ValueError("max_bits must be positive")

    valid_range = (lengths >= 0) & (lengths <= max_bits)
    if not _truth(valid_range.all()):
        raise ValueError(f"code lengths must be between zero and {max_bits}")

    counts = AbstractTensor.stack(
        [(lengths == width).sum() for width in range(1, max_bits + 1)],
        dim=0,
    ).reshape(-1)

    next_codes = []
    code = counts[0] * 0
    for width in range(1, max_bits + 1):
        if width > 1:
            code = (code + counts[width - 2]) * 2
        next_codes.append(code)
    first_code = AbstractTensor.stack(next_codes, dim=0).reshape(-1)

    symbol_index = AbstractTensor.arange(
        lengths.shape[0], cls=type(lengths)
    )
    # Valuewise predicates intentionally do not perform implicit broadcasting.
    # Form the broadcasted matrices arithmetically, then compare to scalars.
    earlier_symbol = (
        symbol_index.unsqueeze(1) - symbol_index.unsqueeze(0)
    ) > 0
    same_length = (
        lengths.unsqueeze(1) - lengths.unsqueeze(0)
    ) == 0
    rank_within_length = (
        earlier_symbol & same_length
    ).to_dtype("int64").sum(dim=1)

    present = (lengths > 0).to_dtype("int64")
    safe_length_index = (
        lengths.maximum(1) - 1
    ).to_dtype("int64")
    return (
        first_code[safe_length_index] + rank_within_length
    ) * present


def _validated_frequency_state(
    frequencies: AbstractTensor,
) -> tuple[AbstractTensor, AbstractTensor, int]:
    """Validate frequencies and return weights, active mask, and count."""
    weights = _require_tensor(frequencies, "frequencies")
    if weights.ndims() != 1 or weights.shape[0] < 1:
        raise ValueError("frequencies must contain at least one symbol")
    if not _truth((weights >= 0).all()):
        raise ValueError("frequencies cannot be negative")
    if not _truth(((weights % 1) == 0).all()):
        raise ValueError("frequencies must be integer counts")
    active = (weights > 0).to_dtype("int64")
    active_count = int(active.sum().item())
    if active_count < 1:
        raise ValueError("at least one symbol needs positive frequency")
    return weights, active, active_count


def _ascending_order(
    weights: AbstractTensor,
) -> AbstractTensor:
    """Return stable ascending indices using the shared tensor top-k op."""
    count = weights.shape[0]
    index = AbstractTensor.arange(count, cls=type(weights))
    # Validated integer weights permit an exact lexicographic scalar key.
    priority = weights * (count + 1) + index
    raw = AbstractTensor.topk(-priority, k=count, dim=0).indices
    return weights.ensure_tensor(raw).to_dtype("int64")


def length_limited_huffman_code_lengths(
    frequencies: AbstractTensor,
    *,
    max_bits: int,
) -> AbstractTensor:
    """Build optimal Huffman lengths no greater than ``max_bits``.

    This is the Package-Merge construction of Larmore and Hirschberg. Each
    package carries a tensor membership row over source symbols. Repeated
    package, concatenate, stable top-k ordering, and a final membership
    reduction produce all leaf depths without a host-side tree.
    """
    weights, active, active_count = _validated_frequency_state(frequencies)
    if max_bits < 1:
        raise ValueError("max_bits must be positive")
    if active_count > 2 ** max_bits:
        raise ValueError(
            f"{active_count} active symbols cannot fit in {max_bits} bits"
        )
    if active_count == 1:
        return active

    symbol_count = weights.shape[0]
    source_index = AbstractTensor.arange(symbol_count, cls=type(weights))
    inactive_priority = (
        (weights.sum() + 1) * (symbol_count + 1) + symbol_count
    )
    source_priority = (
        weights * (symbol_count + 1)
        + source_index
        + (1 - active) * inactive_priority
    )
    selected_raw = AbstractTensor.topk(
        -source_priority, k=active_count, dim=0
    ).indices
    selected = weights.ensure_tensor(selected_raw).to_dtype("int64")
    leaves = weights[selected]
    identity = (
        source_index.unsqueeze(1) - source_index.unsqueeze(0)
    ) == 0
    leaf_membership = identity.to_dtype("int64")[selected]

    current_weights = leaves
    current_membership = leaf_membership
    for _ in range(1, max_bits):
        pair_count = current_weights.shape[0] // 2
        paired = AbstractTensor.arange(
            0, pair_count * 2, 2, cls=type(current_weights)
        ).to_dtype("int64")
        package_weights = (
            current_weights[paired] + current_weights[paired + 1]
        )
        package_membership = (
            current_membership[paired]
            + current_membership[paired + 1]
        )
        combined_weights = AbstractTensor.cat(
            (leaves, package_weights), dim=0
        )
        combined_membership = AbstractTensor.cat(
            (leaf_membership, package_membership), dim=0
        )
        order = _ascending_order(combined_weights)
        current_weights = combined_weights[order]
        current_membership = combined_membership[order]

    selected_count = 2 * active_count - 2
    if current_weights.shape[0] < selected_count:
        raise ValueError("length-limited package set is infeasible")
    lengths = current_membership[:selected_count].sum(dim=0)

    # These checks protect the canonical constructor from a malformed package
    # implementation while remaining backend-independent.
    if not _truth((lengths <= max_bits).all()):
        raise RuntimeError("Package-Merge produced an overlong code")
    kraft_units = (
        2 ** (max_bits - lengths.maximum(1))
    ) * (lengths > 0).to_dtype("int64")
    if not _truth(kraft_units.sum() == 2 ** max_bits):
        raise RuntimeError("Package-Merge produced an incomplete prefix tree")
    return lengths


def huffman_code_lengths(
    frequencies: AbstractTensor,
    *,
    max_bits: int | None = None,
) -> AbstractTensor:
    """Build unconstrained optimal Huffman lengths from integer frequencies.

    Active tree nodes occupy fixed tensor slots. Each slot carries a membership
    row identifying the source symbols beneath that node. Every merge selects
    two minimum weighted slots, increments the depths of their combined
    membership, and writes the merged node back into the first slot. No Python
    priority queue or numerical array participates.
    """
    if max_bits is not None:
        return length_limited_huffman_code_lengths(
            frequencies, max_bits=max_bits
        )
    weights, active, active_count = _validated_frequency_state(frequencies)
    if active_count == 1:
        # A one-symbol alphabet still needs one transmitted bit per symbol;
        # an empty codeword would make symbol count unknowable from the wire.
        return active

    symbol_count = weights.shape[0]
    index = AbstractTensor.arange(symbol_count, cls=type(weights))
    identity = (
        index.unsqueeze(1) - index.unsqueeze(0)
    ) == 0
    membership = (
        identity.to_dtype("int64") * active.unsqueeze(1)
    )
    lengths = AbstractTensor.zeros((symbol_count,), cls=type(weights))

    # Integer counts permit a lexicographic scalar priority: weight first,
    # stable slot identity second. This makes ties deterministic on all backends.
    priority_scale = symbol_count + 1
    inactive_priority = (
        (weights.sum() + 1) * priority_scale + symbol_count
    )
    for _ in range(active_count - 1):
        priority = (
            weights * priority_scale + index
            + (1 - active) * inactive_priority
        )
        selected_raw = AbstractTensor.topk(
            -priority, k=2, dim=0
        ).indices
        selected = weights.ensure_tensor(selected_raw).to_dtype("int64")
        merged_membership = membership[selected].sum(dim=0)
        merged_weight = weights[selected].sum()
        lengths = lengths + merged_membership

        first = ((index - selected[0]) == 0).to_dtype("int64")
        second = ((index - selected[1]) == 0).to_dtype("int64")
        active = active * (1 - second)
        weights = weights * (1 - first) + merged_weight * first
        membership = (
            membership * (1 - first).unsqueeze(1)
            + merged_membership.unsqueeze(0) * first.unsqueeze(1)
        )
    return lengths


@dataclass(frozen=True)
class HuffmanCodewords:
    """Tensor representation of variable-length codewords.

    ``bits`` and ``valid`` have shape ``symbols.shape + (max_bits,)``. Bits are
    MSB-first and right-padding positions are zero. Keeping the validity mask
    explicit preserves codeword boundaries until the bitstream compaction
    stage, where BitBitBuffer provenance can be attached without rediscovering
    symbol ownership.
    """

    codes: AbstractTensor
    lengths: AbstractTensor
    bits: AbstractTensor
    valid: AbstractTensor
    max_bits: int

    @property
    def bit_count(self) -> AbstractTensor:
        return self.lengths.sum()


@dataclass(frozen=True)
class CanonicalHuffmanTable:
    """Canonical code and length tensors for dense or explicit alphabets."""

    codes: AbstractTensor
    lengths: AbstractTensor
    max_bits: int
    symbols: AbstractTensor | None = None

    @classmethod
    def from_code_lengths(
        cls,
        code_lengths: AbstractTensor,
        *,
        max_bits: int = 16,
        symbols: AbstractTensor | None = None,
        validate: bool = True,
    ) -> "CanonicalHuffmanTable":
        lengths = _require_tensor(code_lengths, "code_lengths")
        codes = canonical_codes_from_lengths(lengths, max_bits=max_bits)
        table = cls(
            codes=codes,
            lengths=lengths,
            max_bits=max_bits,
            symbols=symbols,
        )
        if validate:
            table.validate()
        return table

    @classmethod
    def from_frequencies(
        cls,
        frequencies: AbstractTensor,
        *,
        max_bits: int | None = None,
        symbols: AbstractTensor | None = None,
    ) -> "CanonicalHuffmanTable":
        """Build an optimal table, optionally rejecting overlong codes."""
        lengths = huffman_code_lengths(
            frequencies, max_bits=max_bits
        )
        observed_max = int(lengths.max().item())
        return cls.from_code_lengths(
            lengths,
            max_bits=max_bits or observed_max,
            symbols=symbols,
        )

    @classmethod
    def from_samples(
        cls,
        samples: AbstractTensor,
        alphabet: AbstractTensor,
        *,
        valid: AbstractTensor | None = None,
        max_bits: int | None = None,
    ) -> "CanonicalHuffmanTable":
        """Build a table from masked symbol samples and an explicit alphabet."""
        frequencies = symbol_frequencies(
            samples, alphabet, valid=valid
        )
        return cls.from_frequencies(
            frequencies,
            max_bits=max_bits,
            symbols=alphabet,
        )

    @property
    def alphabet(self) -> AbstractTensor:
        """Return explicit symbols or the implicit dense integer alphabet."""
        if self.symbols is not None:
            return self.symbols
        return AbstractTensor.arange(
            self.lengths.shape[0], cls=type(self.lengths)
        )

    def validate(self) -> None:
        """Validate shape, canonical range, and the Kraft prefix bound."""
        _require_tensor(self.codes, "codes")
        _require_tensor(self.lengths, "lengths")
        if self.codes.shape != self.lengths.shape or self.lengths.ndims() != 1:
            raise ValueError("codes and lengths must be equal one-dimensional tensors")
        if self.max_bits < 1:
            raise ValueError("max_bits must be positive")

        valid_range = (
            (self.lengths >= 0) & (self.lengths <= self.max_bits)
        )
        if not _truth(valid_range.all()):
            raise ValueError("table contains an invalid code length")

        present = (self.lengths > 0).to_dtype("int64")
        units = (
            2 ** (self.max_bits - self.lengths.maximum(1))
        ) * present
        if not _truth(units.sum() <= 2 ** self.max_bits):
            raise ValueError("code lengths violate the Kraft prefix bound")

        fits_length = (
            self.codes
            < (2 ** self.lengths.maximum(1))
        ) | (self.lengths == 0)
        if not _truth(fits_length.all()):
            raise ValueError("canonical code does not fit its declared length")

        if self.symbols is not None:
            symbols = _require_tensor(self.symbols, "symbols")
            if symbols.shape != self.lengths.shape:
                raise ValueError("explicit symbols must align with code lengths")
            if not _truth(((symbols % 1) == 0).all()):
                raise ValueError("Huffman symbols must be integers")
            same_symbol = (
                symbols.unsqueeze(1) - symbols.unsqueeze(0)
            ) == 0
            if not _truth(
                (same_symbol.to_dtype("int64").sum(dim=1) == 1).all()
            ):
                raise ValueError("explicit Huffman symbols must be unique")

    def lookup(
        self,
        symbols: AbstractTensor,
        *,
        validate: bool = True,
    ) -> tuple[AbstractTensor, AbstractTensor]:
        """Gather code integers and lengths for a dense symbol tensor."""
        values = _require_tensor(symbols, "symbols")
        if self.symbols is None:
            indices = values.to_dtype("int64")
            in_range = (values >= 0) & (values < self.lengths.shape[0])
            if validate and not _truth(in_range.all()):
                raise ValueError("symbol lies outside the Huffman alphabet")
            codes = self.codes[indices]
            lengths = self.lengths[indices]
        else:
            flat = values.flatten()
            matches = (
                flat.unsqueeze(1) - self.symbols.unsqueeze(0)
            ) == 0
            if validate and not _truth(
                (matches.to_dtype("int64").sum(dim=1) == 1).all()
            ):
                raise ValueError("symbol lies outside the Huffman alphabet")
            numeric_matches = matches.to_dtype("int64")
            codes = (
                numeric_matches * self.codes.unsqueeze(0)
            ).sum(dim=1).reshape(values.shape)
            lengths = (
                numeric_matches * self.lengths.unsqueeze(0)
            ).sum(dim=1).reshape(values.shape)
        if validate and not _truth((lengths > 0).all()):
            raise ValueError("symbol has no code in this Huffman table")
        return codes, lengths

    def encode_codewords(
        self,
        symbols: AbstractTensor,
        *,
        validate: bool = True,
    ) -> HuffmanCodewords:
        """Expand symbols into parallel, MSB-first padded codeword tensors."""
        codes, lengths = self.lookup(symbols, validate=validate)
        positions = AbstractTensor.arange(
            self.max_bits, cls=type(lengths)
        )
        exponent = (
            lengths.unsqueeze(-1) - 1 - positions
        ).maximum(0)
        divisor = 2 ** exponent
        quotient = codes.unsqueeze(-1) // divisor
        bits = quotient % 2
        valid = (lengths.unsqueeze(-1) - positions) > 0
        bits = bits * valid.to_dtype("int64")
        return HuffmanCodewords(
            codes=codes,
            lengths=lengths,
            bits=bits,
            valid=valid,
            max_bits=self.max_bits,
        )


__all__ = [
    "CanonicalHuffmanTable",
    "HuffmanCodewords",
    "canonical_codes_from_lengths",
    "huffman_code_lengths",
    "length_limited_huffman_code_lengths",
    "symbol_frequencies",
]
