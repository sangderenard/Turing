import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.compression.coefficient_events import (
    collect_block_coefficient_events,
    decode_signed_magnitudes,
    encode_signed_magnitudes,
    reconstruct_block_coefficients,
)


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_signed_magnitude_fields_are_exact(backend):
    expected = [-255, -5, -1, 0, 1, 5, 255]
    with AT.use_backend(backend):
        fields = encode_signed_magnitudes(AT.tensor(expected), max_bits=8)
        decoded = decode_signed_magnitudes(fields)

    assert fields.categories.tolist() == [8, 3, 1, 0, 1, 3, 8]
    assert fields.payloads.tolist() == [0, 2, 0, 0, 1, 5, 255]
    assert decoded.tolist() == expected


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_block_coefficient_events_collect_all_entropy_fields(backend):
    coefficients = [
        [10, 0, 0, -3, 0, 2, 0, 0],
        [13, 0, 0, 0, 0, 0, 0, 0],
        [8, 4, 0, 0, 0, -1, 0, 6],
    ]
    with AT.use_backend(backend):
        events = collect_block_coefficient_events(
            AT.tensor(coefficients), max_magnitude_bits=8
        )

    assert decode_signed_magnitudes(events.dc).tolist() == [10, 3, -5]
    assert events.event_counts.tolist() == [2, 0, 3]
    assert events.ac_zero_runs.tolist() == [
        [2, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 3, 1, 0, 0, 0, 0],
    ]
    assert events.trailing_zeros.tolist() == [2, 7, 0]
    assert events.ac.categories.tolist() == [
        [2, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [3, 1, 3, 0, 0, 0, 0],
    ]
    assert events.ac.payloads.tolist() == [
        [0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [4, 0, 6, 0, 0, 0, 0],
    ]


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_block_coefficient_event_round_trip_is_exact(backend):
    coefficients = [
        [[10, 0, 0, -3, 0, 2, 0, 0]],
        [[13, 0, 0, 0, 0, 0, 0, 0]],
        [[8, 4, 0, 0, 0, -1, 0, 6]],
    ]
    with AT.use_backend(backend):
        original = AT.tensor(coefficients)
        events = collect_block_coefficient_events(
            original, max_magnitude_bits=8
        )
        reconstructed = reconstruct_block_coefficients(events)

    assert reconstructed.shape == (3, 1, 8)
    assert reconstructed.tolist() == coefficients


def test_signed_magnitude_limit_is_explicit():
    with AT.use_backend("numpy"):
        with pytest.raises(ValueError, match="8-bit limit"):
            encode_signed_magnitudes(AT.tensor([256]), max_bits=8)


def test_block_dc_predictor_can_continue_across_streaming_batches():
    with AT.use_backend("numpy"):
        first = collect_block_coefficient_events(
            AT.tensor([[3, 0, 0], [6, 0, 0]]),
            max_magnitude_bits=8,
        )
        second = collect_block_coefficient_events(
            AT.tensor([[8, 0, 0]]),
            max_magnitude_bits=8,
            previous_dc=6,
        )

    assert decode_signed_magnitudes(first.dc).tolist() == [3, 3]
    assert decode_signed_magnitudes(second.dc).tolist() == [2]
