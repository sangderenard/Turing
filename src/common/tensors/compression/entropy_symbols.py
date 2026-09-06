"""Format-neutral entropy symbols derived from coefficient events."""

from __future__ import annotations

from dataclasses import dataclass

from ..abstraction import AbstractTensor
from .coefficient_events import (
    BlockCoefficientEvents,
    SignedMagnitudeFields,
)


def _truth(value: AbstractTensor) -> bool:
    return bool(value.item())


@dataclass(frozen=True)
class EntropySymbolSequence:
    """Symbols and their raw payload fields with fixed-capacity validity."""

    symbols: AbstractTensor
    payloads: AbstractTensor
    payload_lengths: AbstractTensor
    valid: AbstractTensor

    @property
    def count(self) -> AbstractTensor:
        return self.valid.to_dtype("int64").sum()

    def compact(self) -> "EntropySymbolSequence":
        """Compact valid entries through prefix sums and one tensor scatter."""
        symbols = self.symbols.flatten()
        payloads = self.payloads.flatten()
        lengths = self.payload_lengths.flatten()
        valid = self.valid.flatten().to_dtype("int64")
        if not (
            symbols.shape == payloads.shape
            == lengths.shape == valid.shape
        ):
            raise ValueError("entropy symbol fields must have equal shapes")

        capacity = symbols.shape[0]
        count = valid.sum()
        ranks = valid.cumsum(dim=0) - 1
        scratch = capacity
        destinations = (
            ranks * valid + scratch * (1 - valid)
        ).to_dtype("int64")

        def compact_field(field: AbstractTensor) -> AbstractTensor:
            target = AbstractTensor.zeros(
                (capacity + 1,), cls=type(field)
            )
            target = AbstractTensor.scatter(
                target,
                destinations,
                field * valid,
                dim=0,
            )
            return target[:capacity]

        slots = AbstractTensor.arange(capacity, cls=type(symbols))
        compact_valid = (count - slots) > 0
        return EntropySymbolSequence(
            symbols=compact_field(symbols),
            payloads=compact_field(payloads),
            payload_lengths=compact_field(lengths),
            valid=compact_valid,
        )

    def to_symbol_list(self) -> list[int]:
        """Materialize the valid symbol prefix at a host boundary."""
        values = self.symbols.flatten().tolist()
        validity = self.valid.flatten().tolist()
        return [
            int(value)
            for value, present in zip(values, validity)
            if bool(present)
        ]


@dataclass(frozen=True)
class BlockEntropySymbols:
    """Separate DC and AC entropy streams for reversible coefficient blocks."""

    dc: EntropySymbolSequence
    ac: EntropySymbolSequence
    original_shape: tuple[int, ...]
    coefficient_count: int
    max_magnitude_bits: int
    ac_category_radix: int

    @property
    def ac_alphabet_size(self) -> int:
        """Dense alphabet size including token zero for end-of-block."""
        return (
            (self.coefficient_count - 1) * self.ac_category_radix + 1
        )

    def ac_payload_lengths(
        self, alphabet: AbstractTensor
    ) -> AbstractTensor:
        """Return the magnitude-bit width encoded by every AC token."""
        present = (alphabet > 0).to_dtype("int64")
        return (
            (alphabet - 1) % self.ac_category_radix
        ) * present


def ac_entropy_tokens(
    zero_runs: AbstractTensor,
    categories: AbstractTensor,
    valid: AbstractTensor,
    *,
    category_radix: int,
) -> AbstractTensor:
    """Encode branchless AC run/category tokens as ordinary tensor dataflow."""
    if zero_runs.shape != categories.shape or zero_runs.shape != valid.shape:
        raise ValueError("AC token inputs must have equal shapes")
    if category_radix < 2:
        raise ValueError("category_radix must leave room for nonzero categories")
    return (
        1 + zero_runs * category_radix + categories
    ) * valid.to_dtype("int64")


def coefficient_events_to_entropy_symbols(
    events: BlockCoefficientEvents,
) -> BlockEntropySymbols:
    """Map coefficient events to reversible, format-neutral entropy tokens.

    DC symbols are magnitude categories. AC token zero means end-of-block.
    Every nonzero AC token is ``1 + zero_run * radix + category``. The radix
    is one greater than the maximum category, making the mapping collision-free
    while leaving a format adapter free to choose a different wire syntax.
    """
    if events.dc.max_bits != events.ac.max_bits:
        raise ValueError("DC and AC magnitude limits must agree")
    max_bits = events.ac.max_bits
    radix = max_bits + 1
    ac_count = events.coefficient_count - 1
    if events.ac_valid.shape[1] != ac_count:
        raise ValueError("AC event capacity does not match coefficient count")

    dc_valid = (events.dc.categories * 0 + 1) > 0
    dc = EntropySymbolSequence(
        symbols=events.dc.categories,
        payloads=events.dc.payloads,
        payload_lengths=events.dc.categories,
        valid=dc_valid,
    )

    slots = AbstractTensor.arange(ac_count + 1, cls=type(events.ac_valid))
    event_count = events.event_counts
    ac_event_valid = events.ac_valid.to_dtype("int64")
    event_tokens = ac_entropy_tokens(
        events.ac_zero_runs,
        events.ac.categories,
        events.ac_valid,
        category_radix=radix,
    )
    event_payloads = events.ac.payloads * ac_event_valid
    event_lengths = events.ac.categories * ac_event_valid

    # Append one spare slot, then place end-of-block token zero immediately
    # after the final event. Validity includes events plus that terminator.
    zero_column = AbstractTensor.zeros(
        (event_tokens.shape[0], 1), cls=type(event_tokens)
    )
    padded_tokens = AbstractTensor.cat(
        (event_tokens, zero_column), dim=1
    )
    padded_payloads = AbstractTensor.cat(
        (event_payloads, zero_column), dim=1
    )
    padded_lengths = AbstractTensor.cat(
        (event_lengths, zero_column), dim=1
    )
    valid = (event_count.unsqueeze(1) + 1 - slots.unsqueeze(0)) > 0
    eob = (
        slots.unsqueeze(0) - event_count.unsqueeze(1)
    ) == 0
    # Token zero is already present in the padded/unused region. Multiplication
    # makes the intended event/EOB ownership explicit for future translators.
    symbols = padded_tokens * eob.logical_not().to_dtype("int64")
    ac = EntropySymbolSequence(
        symbols=symbols,
        payloads=padded_payloads,
        payload_lengths=padded_lengths,
        valid=valid,
    )
    return BlockEntropySymbols(
        dc=dc,
        ac=ac,
        original_shape=events.original_shape,
        coefficient_count=events.coefficient_count,
        max_magnitude_bits=max_bits,
        ac_category_radix=radix,
    )


def entropy_symbols_to_coefficient_events(
    streams: BlockEntropySymbols,
) -> BlockCoefficientEvents:
    """Recover coefficient events exactly from the neutral entropy streams."""
    dc = streams.dc
    ac = streams.ac
    if not (
        dc.symbols.shape == dc.payloads.shape
        == dc.payload_lengths.shape == dc.valid.shape
    ):
        raise ValueError("DC entropy fields must have equal shapes")
    if not (
        ac.symbols.shape == ac.payloads.shape
        == ac.payload_lengths.shape == ac.valid.shape
    ):
        raise ValueError("AC entropy fields must have equal shapes")
    if ac.symbols.ndims() != 2:
        raise ValueError("AC entropy symbols must be block-by-slot")
    if ac.symbols.shape[1] != streams.coefficient_count:
        raise ValueError("AC stream capacity does not match coefficient count")

    if not _truth(dc.valid.all()):
        raise ValueError("every block requires one DC symbol")
    if not _truth((dc.symbols == dc.payload_lengths).all()):
        raise ValueError("DC symbol and payload category disagree")
    dc_present = dc.symbols > 0
    dc_fields = SignedMagnitudeFields(
        categories=dc.symbols,
        payloads=dc.payloads,
        valid=dc_present,
        max_bits=streams.max_magnitude_bits,
    )

    valid = ac.valid.to_dtype("int64")
    eob = ((ac.symbols == 0).to_dtype("int64") * valid)
    if not _truth((eob.sum(dim=1) == 1).all()):
        raise ValueError("each block requires exactly one end-of-block token")
    event_valid = (
        (ac.symbols > 0).to_dtype("int64") * valid
    )
    event_count = event_valid.sum(dim=1)
    slots = AbstractTensor.arange(
        ac.symbols.shape[1], cls=type(ac.symbols)
    )
    expected_valid = (
        event_count.unsqueeze(1) + 1 - slots.unsqueeze(0)
    ) > 0
    expected_eob = (
        slots.unsqueeze(0) - event_count.unsqueeze(1)
    ) == 0
    if not _truth((expected_valid == ac.valid).all()):
        raise ValueError("AC validity must be one contiguous symbol prefix")
    if not _truth(
        (expected_eob.to_dtype("int64") == eob).all()
    ):
        raise ValueError("end-of-block must follow the final AC event")

    ac_count = streams.coefficient_count - 1
    event_tokens = ac.symbols[:, :ac_count]
    compact_valid = event_valid[:, :ac_count]
    raw = (event_tokens - 1) * compact_valid
    runs = (raw // streams.ac_category_radix) * compact_valid
    categories = (raw % streams.ac_category_radix) * compact_valid
    if not _truth(
        (
            (categories <= streams.max_magnitude_bits)
            & ((categories > 0) | (compact_valid == 0))
        ).all()
    ):
        raise ValueError("AC token contains an invalid magnitude category")
    if not _truth((ac.payload_lengths[:, :ac_count] == categories).all()):
        raise ValueError("AC token and payload category disagree")

    ac_fields = SignedMagnitudeFields(
        categories=categories,
        payloads=ac.payloads[:, :ac_count] * compact_valid,
        valid=compact_valid > 0,
        max_bits=streams.max_magnitude_bits,
    )
    event_positions = (
        (runs + 1) * compact_valid
    ).cumsum(dim=1) - 1
    if not _truth(
        (
            (event_positions < ac_count)
            | (compact_valid == 0)
        ).all()
    ):
        raise ValueError("AC zero runs extend beyond the coefficient block")
    last_plus_one = (
        (event_positions + 1) * compact_valid
    ).max(dim=1)
    trailing_zeros = ac_count - last_plus_one
    return BlockCoefficientEvents(
        dc=dc_fields,
        ac=ac_fields,
        ac_zero_runs=runs,
        ac_valid=compact_valid > 0,
        trailing_zeros=trailing_zeros,
        original_shape=streams.original_shape,
        coefficient_count=streams.coefficient_count,
    )


__all__ = [
    "BlockEntropySymbols",
    "EntropySymbolSequence",
    "ac_entropy_tokens",
    "coefficient_events_to_entropy_symbols",
    "entropy_symbols_to_coefficient_events",
]
