"""Falsifiable predictions for the singularity-stencil experiment.

Every test states what topology predicts *before* measuring it, using the
reference field ``F_a(z) = z**2 - a`` with zeros at ``+-sqrt(a)``.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from src.common.singularity_stencil import (  # noqa: E402
    add_fields,
    box_stencil,
    circle_stencil,
    decompose,
    detection_rate,
    naive_grid_zeros,
    phase_diagram,
    predicted_winding,
    smooth_noise,
    with_hole,
    z2_minus_a,
)


def test_degenerate_zero_has_charge_two_with_centre_removed():
    # a = 0: F = z**2, a double zero at the origin.  Remove the centre entirely.
    f = with_hole(z2_minus_a(0.0), (0.0, 0.0), radius=0.25)
    assert torch.isnan(f(torch.tensor([0.0]), torch.tensor([0.0]))[0]).all()
    for r in (0.3, 1.0, 10.0):
        rep = circle_stencil(f, (0.0, 0.0), r)
        assert rep.reliable and rep.winding == 2


def test_orientation_and_simple_zero():
    rep = circle_stencil(lambda x, y: (x, y), (0.0, 0.0), 1.0)
    assert rep.winding == 1
    # conj(z)**2 winds the other way.
    rep = circle_stencil(lambda x, y: (x * x - y * y, -2.0 * x * y), (0.0, 0.0), 1.0)
    assert rep.winding == -2


def test_phase_diagram_matches_prediction_r_equals_sqrt_a():
    a = torch.linspace(0.0, 1.0, 81, dtype=torch.float64)
    r = torch.linspace(0.02, 1.2, 119, dtype=torch.float64)
    w, mag, step = phase_diagram(a, r, n=512)
    pred = predicted_winding(a, r)

    # Exclude only the thin band where the contour passes through a zero.
    gap = (r.view(-1, 1) - torch.sqrt(a).view(1, -1)).abs()
    away = gap > 0.02
    assert away.float().mean() > 0.95
    assert torch.equal(torch.round(w)[away], pred[away])
    # Off the band the measurement is an exact integer, not merely close.
    assert (w[away] - torch.round(w[away])).abs().max() < 1e-9
    # And it is only 0 or 2 -- never an intermediate value -- everywhere it is reliable.
    ok = step < math.pi / 2
    assert set(torch.round(w[ok]).unique().tolist()) <= {0.0, 2.0}


def test_magnitude_is_continuous_while_winding_is_discrete():
    # Along a sweep of r at fixed a, min|F| on the ring varies smoothly and
    # dips to 0 at r = sqrt(a); the winding is piecewise constant and jumps.
    a = torch.tensor([0.25], dtype=torch.float64)
    r = torch.linspace(0.05, 1.0, 200, dtype=torch.float64)
    w, mag, _ = phase_diagram(a, r, n=512)
    w, mag = w[:, 0], mag[:, 0]
    jumps = (torch.round(w[1:]) != torch.round(w[:-1])).nonzero().flatten()
    assert len(jumps) == 1
    r_jump = float(r[jumps[0] + 1])
    assert abs(r_jump - 0.5) < 0.01
    assert float(mag.min()) < 0.01 and abs(float(r[mag.argmin()]) - 0.5) < 0.01


def test_stencil_finds_zeros_a_coarse_pointwise_detector_misses():
    a = 0.3  # zeros at +-0.5477, between the grid points of a 0.5-spaced grid
    f = z2_minus_a(a)
    assert naive_grid_zeros(f, extent=2.0, spacing=0.5, eps=0.02).shape[0] == 0
    rep = circle_stencil(f, (0.0, 0.0), 1.0, n=16)  # a very coarse stencil
    assert rep.reliable and rep.winding == 2


def test_winding_survives_smooth_noise_below_ring_margin():
    # On r = 1 with a = 0.3, |F| >= r**2 - a = 0.7.  Noise well below that
    # margin cannot change the degree (Rouche); far above it, it can.
    rates = detection_rate(0.3, 1.0, sigmas=[0.05, 0.1, 3.0], trials=48, seed=1)
    assert rates[0] == 1.0 and rates[1] == 1.0
    assert rates[2] < 0.9


def test_decomposition_splits_two_into_one_plus_one():
    a = 0.3
    f = z2_minus_a(a)
    dec = decompose(f, (-2.0, -2.0, 2.0, 2.0), min_size=1e-3)
    assert dec.total == 2
    assert not dec.unresolved
    assert sorted(c.winding for c in dec.charges) == [1, 1]
    xs = sorted(c.center[0] for c in dec.charges)
    s = math.sqrt(a)
    assert abs(xs[0] + s) < 2e-3 and abs(xs[1] - s) < 2e-3
    for c in dec.charges:
        assert abs(c.center[1]) < 2e-3


def test_decomposition_with_noise_and_missing_centre_conserves_charge():
    gen = torch.Generator().manual_seed(7)
    f = add_fields(z2_minus_a(0.3), smooth_noise(0.05, generator=gen))
    f = with_hole(f, (0.0, 0.0), 0.1)  # the centre is NaN; zeros are elsewhere
    dec = decompose(f, (-2.0, -2.0, 2.0, 2.0), min_size=1e-3)
    assert dec.total == 2
    assert sum(c.winding for c in dec.charges) == 2
    assert all(abs(c.winding) == 1 for c in dec.charges)


def test_coalescing_zeros_read_as_one_charge_until_resolved():
    # A pair of zeros at +-0.01: every stencil wider than the pair reports a
    # single charge-2 object; only stencils finer than the separation
    # deconstruct it into 1 + 1.
    f = z2_minus_a(1e-4)
    for r in (0.02, 0.1, 1.0):
        assert circle_stencil(f, (0.0, 0.0), r).winding == 2
    assert circle_stencil(f, (0.0, 0.0), 0.005).winding == 0
    assert circle_stencil(f, (0.01, 0.0), 0.005).winding == 1
    fine = decompose(f, (-1.0, -1.0, 1.0, 1.0), min_size=1e-3)
    assert sorted(c.winding for c in fine.charges) == [1, 1]


def test_box_contours_are_additive():
    f = z2_minus_a(0.3)
    parent = box_stencil(f, (-1.0, -1.0, 1.0, 1.0))
    left = box_stencil(f, (-1.0, -1.0, 0.01, 1.0))
    right = box_stencil(f, (0.01, -1.0, 1.0, 1.0))
    assert parent.winding == left.winding + right.winding == 2
    assert left.winding == right.winding == 1
