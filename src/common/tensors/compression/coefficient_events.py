"""Format-neutral coefficient event extraction in AbstractTensor.

Block transforms commonly produce one DC coefficient followed by an ordered
sequence of AC coefficients. This module records the information needed by
run-length and entropy coders without assigning that information to JPEG, a
container, or any particular wire format.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..abstraction import AbstractTensor


@dataclass(frozen=True)
class SignedMagnitudeFields:
    """Reversible category and payload representation of signed integers."""

    categories: AbstractTensor
    payloads: AbstractTensor
    valid: AbstractTensor
    max_bits: int


def encode_signed_magnitudes(
    values: AbstractTensor,
    *,
    max_bits: int = 31,
    validate: bool = True,
) -> SignedMagnitudeFields:
    """Split signed integers into magnitude categories and unsigned payloads.

    Category zero represents zero. For each nonzero value, ``category`` is the
    number of magnitude bits. Positive payloads are ordinary binary magnitude;
    negative payloads use the complemented lower half of that category. This
    is a useful entropy-coding primitive, not a format-specific wire encoding.
    """
    if not isinstance(values, AbstractTensor):
        raise TypeError("values must be an AbstractTensor")
    if max_bits < 1:
        raise ValueError("max_bits must be positive")
    magnitudes = values.abs()
    if validate and not bool((magnitudes < 2 ** max_bits).all().item()):
        raise ValueError(
            f"signed magnitude exceeds the {max_bits}-bit limit"
        )
    powers = 2 ** AbstractTensor.arange(max_bits, cls=type(values))
    categories = (
        (magnitudes.unsqueeze(-1) - powers) >= 0
    ).to_dtype("int64").sum(dim=-1)
    valid = magnitudes > 0
    numeric_valid = valid.to_dtype("int64")
    positive = (values > 0).to_dtype("int64")
    negative_payload = (2 ** categories) - 1 + values
    payloads = (
        positive * magnitudes + (1 - positive) * negative_payload
    ) * numeric_valid
    return SignedMagnitudeFields(
        categories=categories,
        payloads=payloads,
        valid=valid,
        max_bits=max_bits,
    )


def decode_signed_magnitudes(
    fields: SignedMagnitudeFields,
) -> AbstractTensor:
    """Reconstruct signed integers exactly from category/payload fields."""
    categories = fields.categories
    payloads = fields.payloads
    if categories.shape != payloads.shape or categories.shape != fields.valid.shape:
        raise ValueError("signed magnitude fields must have equal shapes")
    # Multiplying by validity maps category zero to exponent zero without a
    # backend-dependent scalar clamp.
    threshold = 2 ** (
        (categories - 1) * fields.valid.to_dtype("int64")
    )
    positive = (payloads - threshold >= 0).to_dtype("int64")
    negative = payloads - ((2 ** categories) - 1)
    decoded = positive * payloads + (1 - positive) * negative
    return decoded * fields.valid.to_dtype("int64")


@dataclass(frozen=True)
class BlockCoefficientEvents:
    """Reversible event fields collected from scanned coefficient blocks.

    AC arrays use fixed-capacity event slots. ``ac_valid`` identifies occupied
    slots, so backends can process every block in parallel without losing the
    variable event count required by a later serializer.
    """

    dc: SignedMagnitudeFields
    ac: SignedMagnitudeFields
    ac_zero_runs: AbstractTensor
    ac_valid: AbstractTensor
    trailing_zeros: AbstractTensor
    original_shape: tuple[int, ...]
    coefficient_count: int

    @property
    def event_counts(self) -> AbstractTensor:
        return self.ac_valid.to_dtype("int64").sum(dim=1)


def concatenate_block_coefficient_events(
    events,
) -> BlockCoefficientEvents:
    """Concatenate compatible block-event batches without decoding them."""

    events = tuple(events)
    if not events:
        raise ValueError("at least one block-event batch is required")
    first = events[0]
    if any(event.coefficient_count != first.coefficient_count for event in events):
        raise ValueError("block-event coefficient counts must match")
    if any(event.dc.max_bits != first.dc.max_bits for event in events):
        raise ValueError("DC magnitude widths must match")
    if any(event.ac.max_bits != first.ac.max_bits for event in events):
        raise ValueError("AC magnitude widths must match")

    def join_signed(name: str) -> SignedMagnitudeFields:
        fields = tuple(getattr(event, name) for event in events)
        return SignedMagnitudeFields(
            categories=AbstractTensor.cat(
                tuple(field.categories for field in fields), dim=0
            ),
            payloads=AbstractTensor.cat(
                tuple(field.payloads for field in fields), dim=0
            ),
            valid=AbstractTensor.cat(
                tuple(field.valid for field in fields), dim=0
            ),
            max_bits=fields[0].max_bits,
        )

    block_count = sum(event.dc.categories.shape[0] for event in events)
    return BlockCoefficientEvents(
        dc=join_signed("dc"),
        ac=join_signed("ac"),
        ac_zero_runs=AbstractTensor.cat(
            tuple(event.ac_zero_runs for event in events), dim=0
        ),
        ac_valid=AbstractTensor.cat(
            tuple(event.ac_valid for event in events), dim=0
        ),
        trailing_zeros=AbstractTensor.cat(
            tuple(event.trailing_zeros for event in events), dim=0
        ),
        original_shape=(block_count, first.coefficient_count),
        coefficient_count=first.coefficient_count,
    )


def slice_block_coefficient_events(
    events: BlockCoefficientEvents,
    start: int,
    stop: int,
) -> BlockCoefficientEvents:
    """Take a contiguous block range while retaining event field alignment."""

    start, stop = int(start), int(stop)
    block_count = events.dc.categories.shape[0]
    if start < 0 or stop < start or stop > block_count:
        raise ValueError("block-event slice is outside the available range")

    def slice_signed(fields: SignedMagnitudeFields) -> SignedMagnitudeFields:
        return SignedMagnitudeFields(
            categories=fields.categories[start:stop],
            payloads=fields.payloads[start:stop],
            valid=fields.valid[start:stop],
            max_bits=fields.max_bits,
        )

    return BlockCoefficientEvents(
        dc=slice_signed(events.dc),
        ac=slice_signed(events.ac),
        ac_zero_runs=events.ac_zero_runs[start:stop],
        ac_valid=events.ac_valid[start:stop],
        trailing_zeros=events.trailing_zeros[start:stop],
        original_shape=(stop - start, events.coefficient_count),
        coefficient_count=events.coefficient_count,
    )


def _collect_flat_block_coefficient_events(
    flat: AbstractTensor,
    dc_differences: AbstractTensor,
    *,
    original_shape: tuple[int, ...],
    max_magnitude_bits: int,
) -> BlockCoefficientEvents:
    """Shared parallel event extraction once DC boundaries are established."""

    coefficient_count = original_shape[-1]
    dc_fields = encode_signed_magnitudes(
        dc_differences,
        max_bits=max_magnitude_bits,
        validate=False,
    )

    ac_values = flat[:, 1:]
    ac_count = coefficient_count - 1
    positions = AbstractTensor.arange(ac_count, cls=type(flat))
    nonzero = ac_values != 0
    numeric_nonzero = nonzero.to_dtype("int64")

    # Rank each nonzero and scatter it directly into the compact block row.
    # Positions are compacted alongside values. Differencing adjacent compact
    # positions yields zero runs without Python-carried scan state.
    ranks = numeric_nonzero.cumsum(dim=1) - 1
    slots = positions
    block_count = flat.shape[0]
    scratch = block_count * ac_count
    block_offsets = (
        AbstractTensor.arange(block_count, cls=type(flat)) * ac_count
    ).unsqueeze(1)
    destinations = (
        (block_offsets + ranks) * numeric_nonzero
        + scratch * (1 - numeric_nonzero)
    ).to_dtype("int64")

    def compact(field: AbstractTensor) -> AbstractTensor:
        target = AbstractTensor.zeros(
            (scratch + 1,), cls=type(field)
        )
        target = AbstractTensor.scatter(
            target,
            destinations.flatten(),
            (field * numeric_nonzero).flatten(),
            dim=0,
        )
        return target[:scratch].reshape(block_count, ac_count)

    compact_values = compact(ac_values)
    event_counts = numeric_nonzero.sum(dim=1)
    ac_valid = (event_counts.unsqueeze(1) - slots.unsqueeze(0)) > 0
    compact_positions_plus_one = compact(
        (positions.unsqueeze(0) + 1) * numeric_nonzero
    )
    prior_positions_plus_one = AbstractTensor.cat(
        (
            AbstractTensor.zeros(
                (block_count, 1), cls=type(compact_positions_plus_one)
            ),
            compact_positions_plus_one[:, :-1],
        ),
        dim=1,
    )
    compact_runs = (
        compact_positions_plus_one - prior_positions_plus_one - 1
    ) * ac_valid.to_dtype("int64")
    last_position_plus_one = (
        compact_positions_plus_one * ac_valid.to_dtype("int64")
    ).max(dim=1)
    trailing_zeros = ac_count - last_position_plus_one
    ac_fields = encode_signed_magnitudes(
        compact_values,
        max_bits=max_magnitude_bits,
        validate=False,
    )
    return BlockCoefficientEvents(
        dc=dc_fields,
        ac=ac_fields,
        ac_zero_runs=compact_runs,
        ac_valid=ac_valid,
        trailing_zeros=trailing_zeros,
        original_shape=original_shape,
        coefficient_count=coefficient_count,
    )


def collect_block_coefficient_events(
    coefficients: AbstractTensor,
    *,
    max_magnitude_bits: int = 31,
    previous_dc=0,
) -> BlockCoefficientEvents:
    """Collect DC differences and compact nonzero AC events in parallel."""
    if not isinstance(coefficients, AbstractTensor):
        raise TypeError("coefficients must be an AbstractTensor")
    if coefficients.ndims() < 2 or coefficients.shape[-1] < 2:
        raise ValueError(
            "coefficients must contain blocks with DC and at least one AC value"
        )
    original_shape = tuple(coefficients.shape)
    coefficient_count = original_shape[-1]
    flat = coefficients.reshape(-1, coefficient_count)
    dc_values = flat[:, 0]
    prior_dc = flat.ensure_tensor(previous_dc).reshape(-1)[:1]
    dc_differences = AbstractTensor.cat(
        (
            dc_values[:1] - prior_dc,
            dc_values[1:] - dc_values[:-1],
        ),
        dim=0,
    )
    return _collect_flat_block_coefficient_events(
        flat,
        dc_differences,
        original_shape=original_shape,
        max_magnitude_bits=max_magnitude_bits,
    )


def collect_component_block_coefficient_events(
    coefficients: AbstractTensor,
    *,
    max_magnitude_bits: int = 31,
    previous_dc=(0,),
) -> BlockCoefficientEvents:
    """Collect several independent component streams in one tensor program.

    ``coefficients`` has shape ``(components, ..., coefficients_per_block)``.
    DC prediction resets for each leading component, while all remaining
    event extraction is one wider parallel operation.
    """

    if not isinstance(coefficients, AbstractTensor):
        raise TypeError("coefficients must be an AbstractTensor")
    if coefficients.ndims() < 3 or coefficients.shape[-1] < 2:
        raise ValueError(
            "component coefficients need a component axis and block axis"
        )
    component_count = coefficients.shape[0]
    previous_dc = tuple(previous_dc)
    if len(previous_dc) != component_count:
        raise ValueError("previous_dc must contain one value per component")

    coefficient_count = coefficients.shape[-1]
    component_flat = coefficients.reshape(
        component_count, -1, coefficient_count
    )
    dc_values = component_flat[:, :, 0]
    prior_dc = AbstractTensor.stack(
        tuple(
            component_flat.ensure_tensor(value).reshape(1)
            for value in previous_dc
        ),
        dim=0,
    )
    dc_differences = AbstractTensor.cat(
        (
            dc_values[:, :1] - prior_dc,
            dc_values[:, 1:] - dc_values[:, :-1],
        ),
        dim=1,
    ).reshape(-1)
    flat = component_flat.reshape(-1, coefficient_count)
    return _collect_flat_block_coefficient_events(
        flat,
        dc_differences,
        original_shape=tuple(coefficients.shape),
        max_magnitude_bits=max_magnitude_bits,
    )


def reconstruct_block_coefficients(
    events: BlockCoefficientEvents,
) -> AbstractTensor:
    """Reconstruct the scanned coefficient tensor exactly from event fields."""
    dc_differences = decode_signed_magnitudes(events.dc).flatten()
    dc_values = dc_differences.cumsum(dim=0)

    ac_count = events.coefficient_count - 1
    if events.ac_zero_runs.shape != events.ac_valid.shape:
        raise ValueError("AC runs and validity mask must have equal shapes")
    if events.ac.categories.shape != events.ac_valid.shape:
        raise ValueError("AC magnitude fields and validity mask must align")
    if events.ac_valid.shape[1] != ac_count:
        raise ValueError("AC event capacity does not match coefficient count")

    valid = events.ac_valid.to_dtype("int64")
    event_values = decode_signed_magnitudes(events.ac) * valid
    event_positions = (
        (events.ac_zero_runs + 1) * valid
    ).cumsum(dim=1) - 1
    destinations = AbstractTensor.arange(
        ac_count, cls=type(event_positions)
    )
    placement = (
        (
            event_positions.unsqueeze(2)
            - destinations.reshape(1, 1, ac_count)
        ) == 0
    ).to_dtype("int64") * valid.unsqueeze(2)
    ac_values = (
        placement * event_values.unsqueeze(2)
    ).sum(dim=1)
    flat = AbstractTensor.cat((dc_values.unsqueeze(1), ac_values), dim=1)
    return flat.reshape(events.original_shape)


__all__ = [
    "BlockCoefficientEvents",
    "SignedMagnitudeFields",
    "collect_component_block_coefficient_events",
    "concatenate_block_coefficient_events",
    "collect_block_coefficient_events",
    "decode_signed_magnitudes",
    "encode_signed_magnitudes",
    "reconstruct_block_coefficients",
    "slice_block_coefficient_events",
]
