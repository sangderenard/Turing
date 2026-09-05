"""Time-stepped dye transport whose fixed point is the influence field.

``InfluenceField.table()`` is the converged answer and ``trace()`` is the order
the transports happened in. Neither is a *state*: nothing in them says where
the dye is right now. This module is the third view of the same computation --
a running solver with dye actually in the pipes.

It is a solver rather than an effect, and that distinction is the whole point.
A scrolling gradient painted along an edge shows motion without transporting
anything: the colour at the far end is known before the animation starts, the
concentration never varies along the pipe, and mixing never happens because
there is nothing to mix. What a viewer would be reading is a decoration whose
parameters happen to have been sampled from data.

Here the dye is state. Each pipe holds cells of power sums, advected by upwind
transport; a node sums what arrives and pushes it onward. Mixing is addition of
power sums -- the same merge operator ``Moments.__add__`` provides -- so a
junction mixes correctly by construction rather than by a blend chosen to look
plausible. Hue shifts along a pipe because the mean of the local distribution
genuinely moves as new dye joins it, not because a hue was ramped.

Emission is periodic and phase-staggered per source. A single impulse would
drain and leave a dead network; emitting everything in lockstep would produce a
global pulse the program does not have. Staggering is also what makes the
picture legible: packets from different origins arrive at a junction at
different moments, so a viewer sees which source is contributing when.

The correctness property is that the two views agree. Integrating what passes
through each node over one emission period reproduces ``table()``, because the
per-traversal attenuation and back-edge decay here are the same constants the
field transports with. So the animation is the computation, and
``integration_error`` measures that claim rather than asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..compiler.influence_field import (
    BACK_EDGE_ROLES,
    DYNAMIC,
    SPECTRUM_END,
    CATEGORIES,
    FORK_ROLES,
    MAX_DISPERSION,
    RECURRENT,
    InfluenceField,
    Moments,
)

# Power-sum slots carried by every cell.
S0, S1, S2 = 0, 1, 2
MOMENTS = 3


@dataclass(frozen=True, slots=True)
class FlowSettings:
    """Transport rates. Physical constants come from the field's contract."""

    # Cells per unit of layout distance; more cells means finer packets and a
    # sharper front, at linear cost.
    cell_density: float = 0.08
    min_cells: int = 4
    max_cells: int = 96
    # Cells advanced per second. Upwind advection carries a shift of
    # ``flow_speed * dt`` cells per step, and that shift decides how sharp a
    # packet stays: at exactly 1.0 the scheme degenerates to ``new[x] =
    # old[x-1]``, a pure translation with no numerical diffusion at all, so a
    # released drop of dye keeps its edges however far it travels. Below 1.0
    # the first-order scheme smears, and over a long pipe a drop spreads into
    # a uniform tint. Pairing this with the driver's dt so the product lands on
    # 1.0 is what makes discrete dye packets visible rather than a wash.
    flow_speed: float = 120.0
    # Emission period in seconds, and how much of it a source is actually open.
    emission_period: float = 2.0
    emission_duty: float = 0.18


class InfluenceFlow:
    """Dye in pipes: emission, advection, junction mixing, and decay.

    The network is taken from an ``InfluenceField`` -- its edges, roles, and
    allotted source hues -- so the solver and the analysis cannot disagree
    about topology or about which origin owns which colour.
    """

    def __init__(
        self,
        field: InfluenceField,
        *,
        lengths: Mapping[tuple[Any, Any], float] | None = None,
        settings: FlowSettings | None = None,
    ) -> None:
        self.field = field
        self.settings = settings or FlowSettings()
        contract = field.contract
        self.categories = tuple(contract.categories)
        self._category_index = {
            name: index for index, name in enumerate(self.categories)
        }

        self.edges: list[tuple[Any, Any, str]] = [
            (source, target, role)
            for source, outgoing in field._outgoing.items()
            for target, role in outgoing
        ]
        self.nodes: list[Any] = sorted(
            {key for edge in self.edges for key in edge[:2]}
            | {source.key for source in field.sources},
            key=str,
        )
        self._node_index = {key: index for index, key in enumerate(self.nodes)}

        counts = []
        for source, target, _ in self.edges:
            span = 1.0 if lengths is None else float(
                lengths.get((source, target), 1.0)
            )
            counts.append(int(max(
                self.settings.min_cells,
                min(self.settings.max_cells, span * self.settings.cell_density),
            )))
        self.cell_count = max(counts) if counts else self.settings.min_cells

        shape = (len(self.edges), self.cell_count, len(self.categories), MOMENTS)
        self.pipes = np.zeros(shape, dtype=np.float64)
        self.arrivals = np.zeros(
            (len(self.nodes), len(self.categories), MOMENTS), dtype=np.float64
        )
        self.integrated = np.zeros_like(self.arrivals)
        self.time = 0.0

        # Per-edge transport factors, taken from the contract so the solver
        # decays exactly as the field's transport does.
        forks: dict[Any, int] = {}
        for source, _, role in self.edges:
            if role in FORK_ROLES:
                forks[source] = forks.get(source, 0) + 1
        outgoing: dict[Any, int] = {}
        for source, _, _ in self.edges:
            outgoing[source] = outgoing.get(source, 0) + 1
        dividing = contract.fan_out == "divide"

        self.edge_factor = np.ones(len(self.edges), dtype=np.float64)
        self.edge_is_back = np.zeros(len(self.edges), dtype=bool)
        # Share of a junction's outflow that enters this pipe. Under ``divide``
        # a tee splits its dye rather than cloning it, which is what the
        # rendered picture has always depicted; under ``copy`` every pipe
        # receives the junction's whole quantity.
        self.edge_split = np.ones(len(self.edges), dtype=np.float64)
        for index, (source, _, role) in enumerate(self.edges):
            factor = contract.attenuation
            if role in BACK_EDGE_ROLES:
                factor *= contract.decay
                self.edge_is_back[index] = True
            if role in FORK_ROLES and forks.get(source) and not dividing:
                factor /= forks[source]
            self.edge_factor[index] = factor
            if dividing:
                self.edge_split[index] = 1.0 / max(1, outgoing.get(source, 1))

        self._edge_source = np.asarray(
            [self._node_index[edge[0]] for edge in self.edges], dtype=np.int64
        )
        self._edge_target = np.asarray(
            [self._node_index[edge[1]] for edge in self.edges], dtype=np.int64
        )

        # Emitters. When the producing IR knows the order its values actually
        # come into existence, ink is released in that order: node k opens at
        # phase k/N of the period, so the release sweeps the program exactly as
        # the program runs. Nothing has to be reconstructed and no loop needs
        # special handling -- an unrolled body simply has more nodes, and each
        # of them activates when it activates.
        self.emitters: list[tuple[int, int, float, float]] = []
        order = [
            key for key in getattr(field, "activation_order", ())
            if key in self._node_index
        ]
        if order:
            self.activation_length = len(order)
            category = self._category_index.get(
                DYNAMIC, next(iter(self._category_index.values()))
            )
            for rank, key in enumerate(order):
                self.emitters.append((
                    self._node_index[key],
                    category,
                    # Hue is the node's place in the activation order, so the
                    # colour of a drop says when in the run it was released.
                    SPECTRUM_END * rank / max(1, len(order) - 1),
                    rank / len(order),
                ))
        else:
            # No authored order: fall back to the origins, staggered against
            # one another so they never pulse in lockstep.
            self.activation_length = 0
            for source in field.sources:
                if source.category not in self._category_index:
                    continue
                self.emitters.append((
                    self._node_index[source.key],
                    self._category_index[source.category],
                    source.hue,
                    (source.ordinal * 0.6180339887) % 1.0,
                ))

        self._recurrent_index = self._category_index.get(RECURRENT)

        # Dye injected from outside, waiting for the next step to carry it in.
        # When something is actually observing the program, this is where its
        # events land, and the periodic emitters are silenced: a source that
        # opens on a clock and a source that opens because a region really ran
        # must not both be running, or the picture mixes an observation with a
        # rehearsal of one and nothing distinguishes them.
        self._injected: dict[tuple[int, int], Moments] = {}
        self.observed = False

    def observe(self) -> None:
        """Stop emitting on a clock; carry only what is injected."""

        self.observed = True
        self.emitters = []

    def inject(self, key: Any, hue: float, weight: float,
               category: str = DYNAMIC) -> bool:
        """Release dye at one node because something happened there.

        Returns whether the node is part of this network -- an event naming
        something the field does not contain is dropped and reported, rather
        than silently colouring nothing.
        """

        node = self._node_index.get(key)
        if node is None:
            return False
        slot = self._category_index.get(category)
        if slot is None:
            return False
        held = self._injected.get((node, slot))
        deposit = (held or Moments()).deposited(float(hue), float(weight))
        self._injected[(node, slot)] = deposit
        return True

    def step(self, dt: float) -> None:
        """Advance emission, advection, and junction mixing by ``dt``."""

        settings = self.settings
        self.arrivals.fill(0.0)

        # Emission. A source is open for part of its period; the phase offset
        # is what staggers origins against one another.
        period = max(1e-6, settings.emission_period)
        duty = max(1e-6, settings.emission_duty)
        for node_index, category, hue, phase in self.emitters:
            position = ((self.time / period) + phase) % 1.0
            if position >= duty:
                continue
            # Raised-cosine valve rather than a hard gate. A gate is a
            # discontinuity, so the least difference in how two implementations
            # carry time -- float64 here, a float32 uniform on the GPU -- lands
            # on opposite sides of the boundary and differs by a whole emission
            # quantum. The window integrates to exactly one unit per period, so
            # the convergence property is unchanged.
            envelope = (1.0 - math.cos(2.0 * math.pi * position / duty)) / duty
            amount = envelope * dt / period
            slot = self.arrivals[node_index, category]
            slot[S0] += amount
            slot[S1] += amount * hue
            slot[S2] += amount * hue * hue

        # Injected dye enters exactly like emitted dye: as arrivals, before
        # the junctions push. Nothing downstream can tell the difference,
        # which is the point -- transport does not care why a drop exists.
        if self._injected:
            for (node, slot), moments in self._injected.items():
                self.arrivals[node, slot, S0] += moments.s0
                self.arrivals[node, slot, S1] += moments.s1
                self.arrivals[node, slot, S2] += moments.s2
            self._injected.clear()

        # Advection. Upwind transport at a uniform rate: the tail cell leaves
        # the pipe, everything else shifts toward it. First order, so the front
        # spreads a little as it travels -- which is what dispersion in a real
        # pipe does anyway, and it keeps the scheme unconditionally stable.
        shift = min(1.0, settings.flow_speed * dt)
        outflow = self.pipes[:, -1] * shift
        self.pipes[:, 1:] = (
            self.pipes[:, 1:] * (1.0 - shift) + self.pipes[:, :-1] * shift
        )
        self.pipes[:, 0] *= 1.0 - shift

        # Pipes discharge into their target nodes, decayed by the same factor
        # the field transports with.
        decayed = outflow * self.edge_factor[:, None, None]
        np.add.at(self.arrivals, self._edge_target, decayed)

        # Influence that crossed a back edge is loop-carried from here on, so
        # it is reclassified exactly as the field's transport reclassifies it.
        if self._recurrent_index is not None:
            back = np.flatnonzero(self.edge_is_back)
            for index in back:
                target = self._edge_target[index]
                moved = decayed[index].copy()
                carried = moved.sum(axis=0)
                self.arrivals[target] -= moved
                self.arrivals[target, self._recurrent_index] += carried

        self.integrated += self.arrivals

        # Junctions push what arrived into every outgoing pipe's head cell.
        # Power sums add, so a junction mixes by the field's merge operator.
        #
        # Not scaled by ``shift``. Arrivals are a quantity, not a density: the
        # advection step already removed exactly ``outflow`` from each pipe, so
        # adding less than the full arrival here would leak mass every hop and
        # the network would starve however long it ran.
        #
        # ``edge_split`` is a different thing entirely -- how a junction
        # apportions its outflow between the pipes leaving it. Handing every
        # outgoing pipe the whole arrival manufactures dye at each tee, which
        # is invisible on an acyclic graph but compounds without bound once the
        # network has a cycle.
        self.pipes[:, 0] += (
            self.arrivals[self._edge_source] * self.edge_split[:, None, None]
        )
        self.time += dt

    def cell_readings(self, edge_index: int) -> np.ndarray:
        """Collapse one pipe's cells to ``(cells, 4)``: hue, dispersion, weight, staging.

        This is the same collapse the palette was baked from, applied to live
        state instead of converged totals -- so concentration genuinely varies
        along the pipe and the optics have something real to modulate.
        """

        cells = self.pipes[edge_index]
        totals = cells[..., S0]
        weight = totals.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(totals > 0, cells[..., S1] / np.maximum(totals, 1e-12), 0.0)
            variance = np.maximum(
                0.0,
                np.where(
                    totals > 0, cells[..., S2] / np.maximum(totals, 1e-12), 0.0
                ) - mean * mean,
            )
        blended = np.where(
            weight[:, None] > 0, totals / np.maximum(weight[:, None], 1e-12), 0.0
        )
        hue = (mean * blended).sum(axis=1)
        dispersion = np.minimum(
            1.0, np.sqrt((variance * blended).sum(axis=1)) / MAX_DISPERSION
        )
        baked = self._category_index.get("baked")
        staging = (
            np.zeros_like(weight) if baked is None
            else np.where(weight > 0, totals[:, baked] / np.maximum(weight, 1e-12), 0.0)
        )
        return np.stack([hue, dispersion, weight, staging], axis=1)

    def integration_error(self) -> float:
        """Relative disagreement between integrated flow and ``table()``.

        The field deposits one unit per source in total, while the solver emits
        one unit per source *per period*, so the integral is normalised by the
        periods elapsed before the two are comparable. Without that they differ
        by a factor of the run length, which looks like disagreement and is
        only a difference of units.

        The field also retires cursors below ``epsilon`` and the solver does
        not, so exact equality is not expected; this reports the gap.
        """

        periods = self.time / max(1e-9, self.settings.emission_period)
        if periods <= 0.0:
            return 0.0
        readings = {reading.key: reading for reading in self.field.table()}
        reference = 0.0
        difference = 0.0
        for key, node_index in self._node_index.items():
            reading = readings.get(key)
            for name, category in self._category_index.items():
                expected = (
                    0.0 if reading is None
                    else reading.categories[name].weight
                    if name in reading.categories else 0.0
                )
                actual = self.integrated[node_index, category, S0] / periods
                reference += expected
                difference += abs(actual - expected)
        return 0.0 if reference <= 0.0 else difference / reference


__all__ = ["FlowSettings", "InfluenceFlow", "S0", "S1", "S2", "MOMENTS"]
