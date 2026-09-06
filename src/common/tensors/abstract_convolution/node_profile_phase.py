"""Node-local phase evolution driven by REAL measured operation time.

``spectral_propagator.propagate`` moves influence THROUGH a graph on one
shared clock ``t``. This module gives each NODE its own clock instead,
advanced only by how long that node's own real work actually took -- not a
synthetic per-step increment, not a global tick shared by everything.

The measurement substrate is ``src/compiler/shell_telemetry.py``'s
``TelemetryChannel`` -- the same channel the compiler's own trace and
profile records already flow through (its own docstring: "a single record
stream with a kind, not four streams"). ``PROFILE`` is one of its kinds
precisely for this: "how long something took after the fact." Nothing new
is invented here; a node's clock is one ``TelemetryChannel`` consumer among
however many others are already watching the same channel.

Every call to :meth:`NodePhaseClock.tick` runs one real operation, measures
it with ``time.perf_counter_ns()``, and appends exactly one sample -- phase
and intensity both derive from that one real number, and nothing is
aggregated, averaged, resampled, or dropped. That is what "without losing
any detail" means concretely: the recorded trajectory has exactly as many
samples as operations actually ran, in the order they actually finished.

Turning multiple nodes' trajectories into "3D spectral data" is a single
BATCHED ``fft(axis=-1)`` over a stacked ``(node, sample)`` tensor -- not a
Python loop over nodes calling ``.fft()`` one at a time. A per-node loop
would serialize what is naturally one vectorized call and, worse, invites
collapsing each row's phase into a scalar along the way; stacking first and
transforming once keeps every node's phase trajectory fully independent
through the entire computation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

from ..abstraction import AbstractTensor
from ....compiler.shell_telemetry import TelemetryChannel


@dataclass
class NodePhaseClock:
    """One node's local oscillator, advanced only by real profiled time.

    ``omega`` is this node's own angular frequency: how much phase it
    accumulates per SECOND of measured execution time. This is deliberately
    independent of any graph-travel frequency in ``spectral_propagator`` --
    a node's intrinsic rate of internal change need not match how fast
    influence travels between nodes.

    A tick's measured duration is used twice, because it is the one real
    quantity that tick produced: it is both the elapsed time the phase
    clock advances by, and that sample's intensity (a slower operation is
    recorded as a stronger source at this node, not a separate synthetic
    weight).
    """

    node: int
    omega: float
    channel: TelemetryChannel
    _elapsed_s: float = field(default=0.0, init=False, repr=False)
    _phase: list = field(default_factory=list, init=False, repr=False)
    _intensity: list = field(default_factory=list, init=False, repr=False)

    def tick(self, operation: Callable[[], None]) -> int:
        """Run ``operation`` once, profile it for real, advance phase.

        Returns the measured duration in nanoseconds. ``operation`` is
        called with no arguments and its return value is discarded --
        this clock only cares how long it took, not what it produced.
        """

        started = time.perf_counter_ns()
        operation()
        nanoseconds = time.perf_counter_ns() - started
        self.channel.profile(
            f"node-{self.node}", nanoseconds=nanoseconds, node=self.node,
        )
        seconds = nanoseconds / 1_000_000_000.0
        self._elapsed_s += seconds
        self._phase.append(self.omega * self._elapsed_s)
        self._intensity.append(seconds)
        return nanoseconds

    @property
    def sample_count(self) -> int:
        return len(self._phase)

    @property
    def elapsed_seconds(self) -> float:
        return self._elapsed_s

    def trajectory(self) -> AbstractTensor:
        """This node's complex trajectory, one sample per real tick, in
        recorded order -- every sample kept, none aggregated or dropped."""

        if not self._phase:
            raise ValueError(
                f"node {self.node}: no ticks recorded yet; trajectory is "
                "undefined for zero real samples"
            )
        phase = AbstractTensor.get_tensor(self._phase)
        intensity = AbstractTensor.get_tensor(self._intensity)
        return AbstractTensor.complex(
            intensity * phase.cos(), intensity * phase.sin(),
        )


def local_spectrum(trajectory: AbstractTensor) -> Tuple[AbstractTensor, AbstractTensor]:
    """One (batch of) trajectory's own frequency decomposition.

    ``trajectory`` may be a single node's ``(sample,)`` complex trajectory
    or a stacked ``(node, sample)`` one -- ``fft``'s default ``axis=-1``
    transforms the last axis and leaves any leading node axis alone, so a
    stacked call is a single batched transform, not node-count Python
    iterations.

    Returns ``(phase, intensity)`` over frequency bins, using ``atan2`` for
    phase -- the primitive the dtype/spectral manifesto's phase 1 added
    specifically so a complex tensor's own angle is recoverable without a
    native ``angle()`` (manifesto section 1.2: "conj, angle -- do not exist
    anywhere"; ``atan2`` does, since ``aff20f1``).
    """

    spectrum = trajectory.fft()
    real, imag = AbstractTensor.real(spectrum), AbstractTensor.imag(spectrum)
    intensity = (real * real + imag * imag).sqrt()
    phase = imag.atan2(real)
    return phase, intensity


def spectral_cube(
    clocks: Sequence[NodePhaseClock],
) -> Tuple[AbstractTensor, AbstractTensor]:
    """Stack synchronized node clocks into ``(node, frequency)`` phase and
    intensity cubes -- the "3D spectral data" this module produces.

    Every clock must have recorded the SAME number of ticks (a synchronized
    event loop: one profiled operation per node per round). An unequal
    count is refused rather than padded or truncated -- either would
    fabricate or discard real measured detail, exactly what "without losing
    any detail" rules out. Call sites with genuinely asynchronous nodes
    should read ``NodePhaseClock.trajectory()``/``local_spectrum`` per node
    individually instead of forcing them into one cube.

    The stack-then-transform order matters: every trajectory is assembled
    first, THEN one batched ``fft(axis=-1)`` runs over the whole
    ``(node, sample)`` tensor inside :func:`local_spectrum` -- not a Python
    loop calling ``.fft()`` once per node, which would serialize a
    naturally vectorized operation and risk collapsing each row down before
    the transform ever saw the others.
    """

    if not clocks:
        raise ValueError("spectral_cube requires at least one clock")
    counts = {clock.sample_count for clock in clocks}
    if len(counts) > 1:
        raise ValueError(
            f"clocks recorded different tick counts {sorted(counts)}; "
            "padding or truncating would fabricate or discard real "
            "measured data. Stack only synchronized clocks."
        )
    trajectories = AbstractTensor.stack([clock.trajectory() for clock in clocks])
    return local_spectrum(trajectories)


__all__ = ["NodePhaseClock", "local_spectrum", "spectral_cube"]
