"""Colour-traceable influence transport over any compiler IR.

Every representation in this tree already knows its own topology, and the
ProcessGraph knows its own schedule. What none of them carry is the answer to
the question a person actually asks while reading a compiled program: *what
reaches this, from where, and how much of it survived the trip.*

That question has one answer for all of them, so it lives in one place rather
than being re-derived per IR. This module accumulates influence transported
along edges and collapses it into colour. It performs no lowering, owns no
allocation policy, and never mutates the graph it reads -- exactly like
``shell_reference_tables``, and for the same reason: the producing IR stays
authoritative and this stays a view.

Why colour, specifically
------------------------
Because the collapse is lossless in the dimensions that matter and free in the
shader. Each node accumulates *power sums* per source category::

    S0 = sum(w)      S1 = sum(w * hue)      S2 = sum(w * hue^2)

Power sums are linear, so they merge by plain addition. That is the whole
reason for the choice: transports arrive asynchronously and out of order from
a compiler that is still running, and any accumulator that is not commutative
and associative would produce a different picture depending on thread timing.
These produce the same field regardless of arrival order, and they reduce in
parallel.

Mean and dispersion fall out of the sums directly, and map onto colour::

    hue        = S1/S0                 -- where the influence centroid sits
    saturation = 1 - dispersion        -- how concentrated its origin is
    value      = f(S0)                 -- how much arrived at all

Hue is allocated along a *spectral arc* that terminates at violet and never
wraps back to red. This is not decoration. On a full hue circle opposite
sources cancel, so the mean of two opposed hues is numerically meaningless
precisely at the high-mixing nodes worth reading; and two different pairs of
sources collapse to the same grey. On an arc, hue is an ordinary scalar, the
mean is monotone and unambiguous, and "mixed" is carried by dispersion instead
of by cancellation. The cost is that red+violet averages to the same hue as a
pure green source, which is why saturation *must* carry dispersion and is not
available for anything else.

The non-spectral magentas above the arc are unreachable by any transported
colour, by construction. That band is therefore reserved for compiler-semantic
annotation -- loop headers, branch merges, deployment joins -- so an annotation
can never be mistaken for a measurement. Use ``semantic_marker_hue`` for those.

Categories, and why the sums are per-category
---------------------------------------------
A node fed by exactly one runtime input and one baked-in weight is not a
confluence; it is a clean node that happens to be staged. Sharing one arc
between categories would average those two hues together and report a
dispersion that describes nothing. So each category accumulates its own
independent sums, and a consumer renders them separately -- conventionally as
sprite core and rim -- rather than blending across the boundary.

The categories are the binding-time split a partial evaluator computes.
``DYNAMIC`` influence varies at run time; ``BAKED`` influence was frozen when
the program was compiled; ``RECURRENT`` influence arrived through a
loop-carried edge and therefore describes state rather than derivation. A node
reached only by ``BAKED`` sources is constant-foldable. A node with a mixed
staging ratio sits on the specialization frontier.

Termination
-----------
Transport across a back edge multiplies the carried weight by ``decay`` < 1, so
iteration *k* of a loop contributes ``decay**k`` and the series converges. That
is what lets an unbounded loop be traced without a trip count, an unrolling
policy, or a guess: run until the surviving weight falls under ``epsilon``.
Forks at branch arms split weight between the arms, so the cursor population is
bounded rather than exponential. Cursors are advanced heaviest-first, which
makes the field an anytime result -- meaningful from the first transport and
monotonically refining -- and deterministic, because ties break on insertion
ordinal rather than on arrival.

Two outputs, one computation
----------------------------
``InfluenceField.table()`` is the converged diagnostic. ``InfluenceField.trace()``
is the ordered sequence of transports that produced it. A visualiser replays
the trace as moving cursors and arrives at exactly the table, so the animation
is the computation rather than an illustration of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import heapq
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .shell_telemetry import TelemetryChannel
from ..common.tensors.abstract_convolution.node_profile_phase import NodePhaseClock

SCHEMA = "turing-influence-field-v1"

# Source categories. These are binding-time classes, not node kinds.
DYNAMIC = "dynamic"
BAKED = "baked"
RECURRENT = "recurrent"
CATEGORIES = (DYNAMIC, BAKED, RECURRENT)

# Hue allocation. Transported colour occupies [0, SPECTRUM_END] and stops at
# violet; the remainder is reserved for annotation and is never produced here.
SPECTRUM_END = 0.75
RESERVED_BAND = (SPECTRUM_END, 1.0)

# Maximum standard deviation reachable on the arc: half the total weight at
# each endpoint. Dispersion is normalised against this so it lands in [0, 1].
MAX_DISPERSION = SPECTRUM_END / 2.0

# Edge roles, as spelled by the existing IR link vocabulary. Roles outside
# these sets transport normally.
BACK_EDGE_ROLES = frozenset({"loop-latch", "loop-back"})
FORK_ROLES = frozenset({"branch-true", "branch-false"})
BARRIER_ROLES = frozenset({"loop-exit", "branch-merge", "deployment-join"})

# Component kinds that conventionally originate influence, by category. A
# contract may replace this mapping wholesale via ``classifier``.
DYNAMIC_KINDS = frozenset({"feed", "argument", "input", "parameter-feed"})
BAKED_KINDS = frozenset({
    "const", "constant", "literal", "parameter", "weight", "capture",
})


class InfluenceContractError(ValueError):
    """Raised for an influence contract that cannot be honoured as written."""


@dataclass(frozen=True, slots=True)
class InfluenceContract:
    """Compile-time policy for whether and where influence is traced.

    This is the parameter that turns the analysis on. It is declarative and
    exhaustive in the same sense as ``ExtractionContract``: a compilation
    either carries one or does not, and a disabled contract costs nothing
    beyond the branch that skips instrumentation.

    ``scopes`` filters *which* code is instrumented, by fnmatch against the
    path a component reports (``python/Linear/forward``, and so on). An empty
    tuple means every scope. ``stages`` filters which IR stages participate,
    so a caller may trace the ProcessGraph alone, or follow the same influence
    across handoffs into SSA and the backends.
    """

    enabled: bool = False
    categories: tuple[str, ...] = (DYNAMIC, BAKED)
    # What an edge does to the influence crossing it.
    #
    # ``divide`` splits a node's outgoing influence among its successors, so
    # the quantity is conserved the way mass in a pipe network is: a tee
    # divides the dye, it does not clone it. The transport operator is then
    # sub-stochastic and its spectral radius stays below one, so a fixed point
    # exists on any graph, cyclic or not.
    #
    # ``copy`` hands every successor the node's full weight, which is the
    # correct reading for dependency -- a value genuinely does influence all of
    # its consumers. But duplication manufactures weight at every fan-out, and
    # on a cyclic graph weight can return to a node larger than it left. That
    # is not a solver bug and no search strategy fixes it: measured on a Dual
    # IR control shell, out-degree 1.36 against attenuation 0.94 gives a
    # spectral radius of 1.0468, and a series with radius at or above one has
    # no limit to converge to. ``copy`` is well-posed on acyclic graphs only.
    #
    # The default is ``divide`` because the rendering already committed to it:
    # what these solvers draw is dye in pipes, and dye is mass.
    fan_out: str = "divide"

    scopes: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()
    #: Keep the whole distribution per node instead of three moments, so
    #: a location can be analysed rather than only rendered. Costs memory
    #: proportional to distinct contributing sources, so it is opt-in.
    spectral: bool = False
    # Per-hop survival. Below 1.0 the field also decays with distance, which
    # makes locality visible; at exactly 1.0 only ``decay`` and ``epsilon``
    # bound the walk.
    attenuation: float = 0.94
    # Per-back-edge survival. Strictly below 1.0 -- this is what makes an
    # unbounded loop converge.
    decay: float = 0.55
    # Cursors carrying less than this are retired.
    epsilon: float = 1e-4
    # Hard stop, independent of the analytic bound, so a pathological graph
    # cannot spin.
    max_transports: int = 2_000_000
    # Spread hues across the arc within each category as one flat ramp, or
    # subdivide the arc by source group first so an object's members stay
    # tonally adjacent.
    subdivide_by_group: bool = False
    # Saturation shaping. On a large graph most nodes are well mixed, and a
    # linear dispersion ramp collapses the whole field toward grey; the gamma
    # and floor keep it legible.
    saturation_gamma: float = 0.6
    saturation_floor: float = 0.15

    def __post_init__(self) -> None:
        unknown = sorted(set(self.categories) - set(CATEGORIES))
        if unknown:
            raise InfluenceContractError(
                f"unknown influence categories: {unknown}; "
                f"known categories are {list(CATEGORIES)}"
            )
        if not self.categories:
            raise InfluenceContractError(
                "an enabled influence contract must trace at least one category"
            )
        if self.fan_out not in {"divide", "copy"}:
            raise InfluenceContractError(
                f"fan_out must be 'divide' or 'copy'; got {self.fan_out!r}"
            )
        if not 0.0 < self.attenuation <= 1.0:
            raise InfluenceContractError(
                f"attenuation must lie in (0, 1]; got {self.attenuation}"
            )
        if not 0.0 < self.decay < 1.0:
            raise InfluenceContractError(
                "decay must lie in (0, 1) so loop transport converges; "
                f"got {self.decay}"
            )
        if not 0.0 < self.epsilon < 1.0:
            raise InfluenceContractError(
                f"epsilon must lie in (0, 1); got {self.epsilon}"
            )
        if self.max_transports <= 0:
            raise InfluenceContractError("max_transports must be positive")

    @classmethod
    def disabled(cls) -> "InfluenceContract":
        return cls(enabled=False)

    @classmethod
    def from_mapping(cls, raw: Any) -> "InfluenceContract":
        """Build from a YAML/JSON fragment, rejecting unknown keys."""

        if raw is None:
            return cls.disabled()
        if not isinstance(raw, Mapping):
            raise InfluenceContractError("influence must be a mapping")
        known = {slot for slot in cls.__slots__}
        extra = sorted(set(raw) - known)
        if extra:
            raise InfluenceContractError(
                f"influence contract has unknown fields: {extra}"
            )
        tuple_fields = {"categories", "scopes", "stages"}
        prepared = {
            key: tuple(str(item) for item in value)
            if key in tuple_fields else value
            for key, value in raw.items()
        }
        return cls(**prepared)

    def selects_scope(self, path: str) -> bool:
        """Report whether ``path`` is inside the instrumented region."""

        if not self.enabled:
            return False
        if not self.scopes:
            return True
        return any(fnmatchcase(path, pattern) for pattern in self.scopes)

    def selects_stage(self, stage: str) -> bool:
        if not self.enabled:
            return False
        if not self.stages:
            return True
        return any(fnmatchcase(stage, pattern) for pattern in self.stages)


@dataclass(frozen=True, slots=True)
class InfluenceSource:
    """One origin of influence, holding the arc position it was allotted."""

    key: Any
    category: str
    ordinal: int
    hue: float
    label: str = ""
    group: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.hue <= SPECTRUM_END:
            raise InfluenceContractError(
                f"source hue {self.hue} escapes the spectral arc "
                f"[0, {SPECTRUM_END}]; the band above it is reserved for "
                "semantic annotation and must stay unreachable by transport"
            )


@dataclass(frozen=True, slots=True)
class Moments:
    """Weighted power sums over hue, for one category at one node.

    Addition is the merge. That is the entire contract this type exists to
    provide, and it is what allows transports to be applied in any order.
    """

    s0: float = 0.0
    s1: float = 0.0
    s2: float = 0.0

    def __add__(self, other: "Moments") -> "Moments":
        return Moments(self.s0 + other.s0, self.s1 + other.s1, self.s2 + other.s2)

    def deposited(self, hue: float, weight: float) -> "Moments":
        return Moments(
            self.s0 + weight,
            self.s1 + weight * hue,
            self.s2 + weight * hue * hue,
        )

    def scaled(self, factor: float) -> "Moments":
        return Moments(self.s0 * factor, self.s1 * factor, self.s2 * factor)

    @property
    def weight(self) -> float:
        return self.s0

    @property
    def mean(self) -> float:
        return 0.0 if self.s0 <= 0.0 else self.s1 / self.s0

    @property
    def variance(self) -> float:
        if self.s0 <= 0.0:
            return 0.0
        mean = self.s1 / self.s0
        # Floating-point cancellation can drive this fractionally negative on
        # a node whose contributions share a single hue.
        return max(0.0, self.s2 / self.s0 - mean * mean)

    @property
    def dispersion(self) -> float:
        """Normalised spread of contributing hues, in [0, 1]."""

        return min(1.0, math.sqrt(self.variance) / MAX_DISPERSION)


@dataclass(frozen=True, slots=True)
class Spectrum:
    """The full distribution, not a projection of it.

    ``Moments`` keeps three numbers, so by the time anything asks a
    question the spectrum has already been collapsed to a centroid and a
    spread -- and RGB collapses it again to three channels. Both are
    lossy in the same direction: two completely different sets of origins
    can produce identical moments, and at exactly the high-mixing nodes
    worth reading they usually do.

    This keeps every contributing frequency with its weight, so a location
    can be analysed rather than only displayed: which origins reached it,
    in what proportion, and whether its colour is one source or six
    cancelling into the same mean.

    It is a drop-in for ``Moments`` -- same constructor, ``deposited``,
    ``scaled``, ``+``, and the same derived properties -- so the solver
    uses it unchanged. Crucially it keeps the property the whole design
    rests on: merging is addition over a keyed map, which is commutative
    and associative, so transports arriving in any order give the same
    spectrum. Lines are held sorted by frequency so equal spectra have
    equal representations.
    """

    lines: tuple[tuple[float, float], ...] = ()

    def __add__(self, other: "Spectrum") -> "Spectrum":
        merged: dict[float, float] = {}
        for frequency, weight in self.lines:
            merged[frequency] = merged.get(frequency, 0.0) + weight
        for frequency, weight in getattr(other, "lines", ()):
            merged[frequency] = merged.get(frequency, 0.0) + weight
        return Spectrum(tuple(sorted(merged.items())))

    def deposited(self, hue: float, weight: float) -> "Spectrum":
        return self + Spectrum(((float(hue), float(weight)),))

    def scaled(self, factor: float) -> "Spectrum":
        return Spectrum(tuple(
            (frequency, weight * factor) for frequency, weight in self.lines
        ))

    @property
    def s0(self) -> float:
        return sum(weight for _frequency, weight in self.lines)

    @property
    def s1(self) -> float:
        return sum(frequency * weight for frequency, weight in self.lines)

    @property
    def s2(self) -> float:
        return sum(
            frequency * frequency * weight for frequency, weight in self.lines
        )

    @property
    def weight(self) -> float:
        return self.s0

    @property
    def mean(self) -> float:
        total = self.s0
        return 0.0 if total <= 0.0 else self.s1 / total

    @property
    def variance(self) -> float:
        total = self.s0
        if total <= 0.0:
            return 0.0
        mean = self.s1 / total
        return max(0.0, self.s2 / total - mean * mean)

    @property
    def dispersion(self) -> float:
        return min(1.0, math.sqrt(self.variance) / MAX_DISPERSION)

    def at(self, frequency: float, tolerance: float = 1e-12) -> float:
        """Weight arriving from one frequency -- the analysable question."""
        return sum(
            weight for line, weight in self.lines
            if abs(line - float(frequency)) <= tolerance
        )

    # -- normalisation -------------------------------------------------
    #
    # Composition and intensity are different questions and must not be
    # read off the same number. Two nodes can have identical origins in
    # identical proportion and differ a thousandfold in how much arrived;
    # a node deep in a fan-out is dim because weight divided, not because
    # its provenance is any less certain. Reporting raw weight as colour
    # conflates the two and makes depth look like uncertainty.
    #
    # So: `normalised` answers "what is this made of", carrying unit
    # power, and `power` answers "how much got here". A viewer maps the
    # first to chromaticity and the second to luminance, and neither
    # contaminates the other.

    def normalised(self) -> "Spectrum":
        """Unit-power spectrum: composition with intensity divided out."""
        total = self.s0
        if total <= 0.0:
            return Spectrum(())
        return Spectrum(tuple(
            (frequency, weight / total) for frequency, weight in self.lines
        ))

    @property
    def power(self) -> float:
        """Total weight that arrived. The luminance question."""
        return self.s0

    @property
    def support(self) -> int:
        """How many distinct origins contributed at all."""
        return sum(1 for _frequency, weight in self.lines if weight > 0.0)

    @property
    def participation(self) -> float:
        """Effective number of origins: 1 / sum(p^2) over unit power.

        The inverse participation ratio. `support` counts anything above
        zero, so one dominant origin plus nine traces reads as ten; this
        reads as ~1, which is what a person means by "where did this come
        from". Equals `support` exactly when contributions are equal, and
        1 when a single origin carries everything.
        """
        total = self.s0
        if total <= 0.0:
            return 0.0
        concentration = sum(
            (weight / total) ** 2 for _frequency, weight in self.lines
        )
        return 0.0 if concentration <= 0.0 else 1.0 / concentration

    def entropy(self) -> float:
        """Shannon entropy of the composition, in bits.

        Zero when one origin carries everything; log2(n) when n origins
        contribute equally. Reported alongside participation because they
        disagree in the informative direction: entropy is sensitive to the
        tail, participation to the bulk.
        """
        total = self.s0
        if total <= 0.0:
            return 0.0
        accumulated = 0.0
        for _frequency, weight in self.lines:
            share = weight / total
            if share > 0.0:
                accumulated -= share * math.log2(share)
        return accumulated

    def floor(self, epsilon: float) -> "Spectrum":
        """Drop lines under `epsilon` of the total, exactly and visibly.

        A line that survives only as float dust is not evidence of a path;
        it is the residue of one. Filtering is stated rather than implied
        so that "this origin did not reach here" and "it reached here
        immeasurably" stay distinguishable.
        """
        total = self.s0
        if total <= 0.0:
            return Spectrum(())
        cutoff = float(epsilon) * total
        return Spectrum(tuple(
            (frequency, weight) for frequency, weight in self.lines
            if weight >= cutoff
        ))


@dataclass(frozen=True, slots=True)
class CategoryReading:
    """One category's collapsed colour at one node."""

    category: str
    hue: float
    saturation: float
    weight: float
    dispersion: float


@dataclass(frozen=True, slots=True)
class InfluenceReading:
    """Everything a shader needs for one node, already collapsed."""

    key: Any
    categories: Mapping[str, CategoryReading]
    # Normalised total arriving weight across categories, in [0, 1].
    value: float
    # Fraction of arriving weight that was frozen at compile time. 0 means
    # entirely runtime-varying; 1 means constant-foldable.
    staging: float
    # Fraction that arrived through loop-carried edges.
    recurrence: float

    def dominant(self) -> CategoryReading | None:
        """Return the heaviest category, or None if nothing reached here."""

        live = [item for item in self.categories.values() if item.weight > 0.0]
        if not live:
            return None
        return max(live, key=lambda item: item.weight)


@dataclass(frozen=True, slots=True)
class Transport:
    """One cursor hop. Replaying these in order reproduces the table."""

    step: int
    source_key: Any
    target_key: Any
    category: str
    hue: float
    weight: float
    iteration: int
    role: str


def semantic_marker_hue(index: int, count: int) -> float:
    """Allocate an annotation hue inside the reserved non-spectral band.

    Transported colour can never land here, so a marker produced by this
    function is always distinguishable from a measurement.
    """

    if count <= 0:
        raise InfluenceContractError("semantic marker count must be positive")
    low, high = RESERVED_BAND
    # Half-step inset. Placing the first marker flush at ``low`` would put it
    # exactly on SPECTRUM_END, which a source allotted the top of the arc also
    # occupies -- the two bands must not touch at their shared endpoint.
    span = (high - low) * (index % count + 0.5) / count
    return low + span


def allocate_hues(
    sources: Sequence[tuple[Any, str, int, str, str]],
    *,
    subdivide_by_group: bool = False,
) -> tuple[InfluenceSource, ...]:
    """Spread sources across the spectral arc, independently per category.

    Each entry is ``(key, category, ordinal, label, group)``. ``ordinal`` is
    the ordering that gives the mean hue its meaning -- conventionally the
    ProcessGraph schedule level, so hue reads as depth-of-origin and a node's
    mean hue reports where in the computation its influence started. An
    arbitrary ordinal produces a well-defined but uninterpretable centroid.

    Categories are separated by the accumulator rather than by hue, but they
    must not be allotted the *same* hues: a consumer drawing two categories
    together -- as core and rim, conventionally -- renders one flat colour
    wherever they coincide, which is precisely at the staged nodes most worth
    reading.

    Spreading each category independently over the arc collides constantly:
    with three dynamic sources and four baked ones, the second of each lands on
    exactly one third. So every source instead takes a distinct slot on one
    shared grid, ordered by its fractional position within its own category.
    Each category still spans the whole arc and stays monotone in ordinal, and
    two sources sharing a hue is now impossible rather than merely unlikely.
    """

    def spread(
        members: Sequence[tuple[tuple[Any, str, int, str, str], float, int]],
        low: float, high: float,
    ) -> list[InfluenceSource]:
        """Place pre-ranked members on distinct slots of a shared grid."""

        ordered = sorted(members, key=lambda item: (
            item[1], item[2], item[0][2], str(item[0][0])
        ))
        span = high - low
        return [
            InfluenceSource(
                key=entry[0], category=entry[1], ordinal=entry[2],
                hue=low + span * slot / len(ordered),
                label=entry[3], group=entry[4],
            )
            for slot, (entry, _, _) in enumerate(ordered)
        ]

    def ranked(
        entries: Sequence[tuple[Any, str, int, str, str]],
    ) -> list[tuple[tuple[Any, str, int, str, str], float, int]]:
        """Rank every entry by its fractional position inside its category."""

        grouped: dict[str, list[tuple[Any, str, int, str, str]]] = {}
        for entry in entries:
            grouped.setdefault(entry[1], []).append(entry)
        ranks: list[tuple[tuple[Any, str, int, str, str], float, int]] = []
        for category, members in grouped.items():
            members = sorted(members, key=lambda item: (item[2], str(item[0])))
            index = (
                CATEGORIES.index(category) if category in CATEGORIES else len(CATEGORIES)
            )
            for offset, entry in enumerate(members):
                ranks.append((entry, (offset + 0.5) / len(members), index))
        return ranks

    if not sources:
        return ()
    if not subdivide_by_group:
        return tuple(spread(ranked(sources), 0.0, SPECTRUM_END))

    bands: dict[str, list[tuple[Any, str, int, str, str]]] = {}
    for entry in sources:
        bands.setdefault(entry[4], []).append(entry)
    width = SPECTRUM_END / max(1, len(bands))
    allotted: list[InfluenceSource] = []
    for band_index, (_, band_members) in enumerate(sorted(bands.items())):
        allotted.extend(spread(
            ranked(band_members),
            band_index * width,
            (band_index + 1) * width,
        ))
    return tuple(allotted)


def default_classifier(kind: str, attributes: Mapping[str, Any]) -> str | None:
    """Categorise a component, or return None if it originates nothing.

    Anything explicitly marked wins over the kind vocabulary, so a producer
    that knows its own binding time can say so and never be second-guessed.
    """

    declared = attributes.get("influence_category")
    if declared is not None:
        if str(declared) not in CATEGORIES:
            raise InfluenceContractError(
                f"component declares unknown influence category {declared!r}"
            )
        return str(declared)
    if kind in DYNAMIC_KINDS:
        return DYNAMIC
    if kind in BAKED_KINDS:
        return BAKED
    # A component holding a materialised literal is baked whatever it calls
    # itself; this is the case constant folding would collapse.
    if "value" in attributes and not attributes.get("mutable", False):
        return BAKED
    return None


class InfluenceField:
    """Accumulates transported influence over an arbitrary node/edge topology.

    The topology is supplied as opaque keys and roles, so any producer that can
    name its nodes and its edges can be traced: ProcessGraph, SSA, control IR,
    and the C, JavaScript, LLVM, and Fortran shells alike. Nothing here
    inspects an IR type.
    """

    def __init__(
        self,
        contract: InfluenceContract | None = None,
        *,
        profile_channel: "TelemetryChannel | None" = None,
        phase_omega: float = 2.0 * math.pi,
    ) -> None:
        self.contract = contract or InfluenceContract.disabled()
        self._sources: dict[Any, InfluenceSource] = {}
        self._staged: list[tuple[Any, str, int, str, str]] = []
        self._outgoing: dict[Any, list[tuple[Any, str]]] = {}
        self._nodes: set[Any] = set()
        self._barriers: dict[Any, set[str]] = {}
        self._moments: dict[Any, dict[str, Moments]] = {}
        # Spectrum is a drop-in for Moments: same algebra, same derived
        # properties, and it keeps the lines instead of collapsing them.
        self._accumulator = (
            Spectrum if getattr(contract, 'spectral', False) else Moments
        )
        self._transports: list[Transport] = []
        self._converged = False
        # Order in which the compiled program actually activates its values.
        # Empty unless a producer knows it -- a dependency graph carries no
        # such order, but a lowered region does: its steps are linearised.
        self.activation_order: tuple[Any, ...] = ()

        # Opt-in, real-profiled per-node phase (node_profile_phase.py). Off
        # by default -- nothing about propagate()'s existing behaviour or
        # cost changes unless a caller supplies a channel. When enabled,
        # every node's relaxation step (see propagate()) is timed for real
        # via NodePhaseClock, on the SAME shell_telemetry.TelemetryChannel
        # the compiler's own trace/profile instrumentation already flows
        # through -- not a private clock and not a synthetic increment.
        self._phase_channel = profile_channel
        self._phase_omega = float(phase_omega)
        self._phase_clocks: dict[Any, "NodePhaseClock"] = {}
        self._phase_node_ids: dict[Any, int] = {}

    # -- topology intake -------------------------------------------------

    def add_node(self, key: Any) -> None:
        self._nodes.add(key)

    def add_edge(self, source: Any, target: Any, *, role: str = "data") -> None:
        """Record a directed edge. Roles drive fork, decay, and reporting."""

        self._nodes.add(source)
        self._nodes.add(target)
        role = str(role)
        self._outgoing.setdefault(source, []).append((target, role))
        if role in BARRIER_ROLES:
            self._barriers.setdefault(target, set()).add(role)
        self._converged = False

    def stage_source(
        self, key: Any, category: str, ordinal: int,
        label: str = "", group: str = "",
    ) -> None:
        """Declare one origin without allotting its hue yet.

        Hue allocation spreads the whole set across the arc, so it cannot be
        done incrementally: a source's colour depends on how many others share
        its category. Staging lets a live compiler announce origins one at a
        time and pay for allocation once, rather than re-spreading every source
        on each arrival.
        """

        self._staged.append((key, str(category), int(ordinal), str(label), str(group)))
        self._nodes.add(key)
        self._converged = False

    def add_sources(
        self,
        entries: Iterable[tuple[Any, str, int, str, str]],
    ) -> tuple[InfluenceSource, ...]:
        """Stage a batch of origins and allot the arc immediately."""

        self._staged = [
            (key, str(category), int(ordinal), str(label), str(group))
            for key, category, ordinal, label, group in entries
        ]
        for entry in self._staged:
            self._nodes.add(entry[0])
        return self._allocate()

    def _allocate(self) -> tuple[InfluenceSource, ...]:
        allotted = allocate_hues(
            tuple(self._staged),
            subdivide_by_group=self.contract.subdivide_by_group,
        )
        self._sources = {source.key: source for source in allotted}
        self._converged = False
        return allotted

    @property
    def sources(self) -> tuple[InfluenceSource, ...]:
        if len(self._sources) != len(self._staged):
            self._allocate()
        return tuple(self._sources.values())

    def barriers(self) -> Mapping[Any, tuple[str, ...]]:
        """Nodes an IR *declared* to be confluences, and by which roles.

        Worth reading against the measured field: a declared barrier whose
        dispersion stays near zero recombined paths that carried the same
        influence anyway, and a high-dispersion node that appears here in no
        role is a confluence the IR never named.
        """

        return {
            key: tuple(sorted(roles))
            for key, roles in sorted(self._barriers.items(), key=lambda item: str(item[0]))
        }

    # -- transport -------------------------------------------------------

    def propagate(self) -> int:
        """Run transport to convergence. Returns the transport count.

        Influence is relaxed, not enumerated. Every bundle waiting at the same
        node *and the same loop depth* is merged and moved once, rather than
        each distinct path through the graph being walked separately.

        That distinction decides whether this finishes. Enumerating paths is
        fine on a dependency graph, which is acyclic and thin, but a control
        shell is neither: when one numeric region is dispatched from two sites
        and one of them sits inside a loop, its nodes are shared between both,
        every iteration re-enters them, and the path count multiplies. Measured
        on a fourteen-node shell, that was 1,219,159 transports against 2,114
        for the same shell without the shared dispatch -- a 577x blow-up on a
        pattern real programs use constantly.

        Merging is exact, not an approximation: power sums are linear, so
        scaling a merged bundle by an edge factor gives precisely the sum of
        scaling each contribution separately. The fixed point is identical.

        Bundles merge on node and category alone. An earlier attempt kept loop
        depth in the key, on the reasoning that decay bounds depth at
        ``decay**k < epsilon`` and exact per-path iteration counts were worth a
        small constant factor. That reasoning was wrong, and measurably so: a
        recorded control shell contains cycles with *no* back edge -- when one
        numeric region is dispatched from two sites, ``dispatch-result`` feeds
        a consumer that routes back through ``dispatch-feed`` into the same
        region. Those cycles decay only by attenuation and never increment
        depth, but they keep re-injecting into the real loop, so every
        re-injection minted a fresh key and merging never happened. Depth ran
        past 1600 and the run hit ``max_transports``.

        So ``Transport.iteration`` is the back-edge depth at which a bundle was
        moved, not an exact per-path count: bundles arriving at one node from
        different depths are merged and report the depth of the merge. The
        accumulated field is unaffected -- only the per-transport annotation
        loses that resolution.

        Weight strictly decreases along every hop and bundles under ``epsilon``
        retire, so the relaxation terminates. ``max_transports`` is a backstop
        for pathological fan-out, not the primary bound.
        """

        contract = self.contract
        if not contract.enabled:
            return 0
        if len(self._sources) != len(self._staged):
            self._allocate()
        self._moments = {}
        self._transports = []
        selected = {
            key: source for key, source in self._sources.items()
            if source.category in contract.categories
        }

        # Undelivered influence, keyed by node alone.
        pending: dict[Any, dict[str, Moments]] = {}
        depth: dict[Any, int] = {}
        heap: list[tuple[float, int, Any]] = []
        ordinal = 0

        def enqueue(node: Any, weight: float) -> None:
            nonlocal ordinal
            # Heaviest first, ties broken on insertion order, so the result is
            # deterministic and meaningful at any point before convergence.
            heapq.heappush(heap, (-weight, ordinal, node))
            ordinal += 1

        def hold(node: Any, arrival: int, bundle: Mapping[str, Moments]) -> None:
            slot = pending.setdefault(node, {})
            for category, moments in bundle.items():
                slot[category] = slot.get(category, self._accumulator()) + moments
            depth[node] = max(depth.get(node, 0), arrival)

        for source in sorted(selected.values(), key=lambda item: item.ordinal):
            unit = self._accumulator().deposited(source.hue, 1.0)
            self._deposit(source.key, source.category, source.hue, 1.0)
            hold(source.key, 0, {source.category: unit})
            enqueue(source.key, 1.0)

        step = 0
        while heap and step < contract.max_transports:
            _, _, node = heapq.heappop(heap)
            bundle = pending.pop(node, None)
            iteration = depth.pop(node, 0)

            def relax_one_node() -> None:
                # Exactly the per-pop relaxation work; `step`/`node`/`bundle`/
                # `iteration` are the enclosing loop's own locals, closed over
                # (nonlocal `step` since it accumulates). Extracted to a
                # callable so it can be run either directly or through a real
                # NodePhaseClock.tick() -- see below -- without duplicating
                # this logic or changing what it computes either way.
                nonlocal step
                if not bundle:
                    # A stale heap entry: this slot was already drained by an
                    # earlier pop that collected everything waiting there.
                    return
                if sum(item.s0 for item in bundle.values()) < contract.epsilon:
                    return
                edges = self._outgoing.get(node, ())
                if not edges:
                    return
                forks = sum(1 for _, role in edges if role in FORK_ROLES)
                dividing = contract.fan_out == "divide"
                # Under ``divide`` the split across all successors already
                # accounts for branch arms; applying the fork rule as well
                # would halve them twice.
                share = 1.0 / len(edges) if dividing else 1.0

                for target, role in edges:
                    factor = contract.attenuation * share
                    arrival = iteration
                    if role in BACK_EDGE_ROLES:
                        factor *= contract.decay
                        arrival += 1
                    if role in FORK_ROLES and forks and not dividing:
                        # Alternatives, not parallel successors: the arms
                        # divide the weight between them rather than each
                        # taking all.
                        factor /= forks

                    if role in BACK_EDGE_ROLES and RECURRENT in contract.categories:
                        # Influence that crossed a back edge is loop-carried
                        # from here on, whatever binding time it started with.
                        carried = self._accumulator()
                        for moments in bundle.values():
                            carried = carried + moments
                        moved = {RECURRENT: carried.scaled(factor)}
                    else:
                        moved = {
                            category: moments.scaled(factor)
                            for category, moments in bundle.items()
                        }
                    moved = {
                        category: moments for category, moments in moved.items()
                        if moments.s0 > 0.0
                    }
                    delivered = sum(item.s0 for item in moved.values())
                    if delivered < contract.epsilon:
                        continue

                    for category, moments in moved.items():
                        per_category = self._moments.setdefault(target, {})
                        per_category[category] = (
                            per_category.get(category, self._accumulator())
                            + moments
                        )
                        self._transports.append(Transport(
                            step=step, source_key=node, target_key=target,
                            category=category, hue=moments.mean,
                            weight=moments.s0, iteration=arrival, role=role,
                        ))
                        step += 1
                    hold(target, arrival, moved)
                    enqueue(target, delivered)

            if self._phase_channel is None:
                relax_one_node()
            else:
                # Real profiled time, node by node: each pop is one real
                # relaxation operation, timed by wrapping it exactly as it
                # runs -- not a synthetic per-step increment and not a
                # separately-reported duration. See node_profile_phase.py's
                # own module docstring for why the operation is timed by
                # wrapping rather than reported after the fact.
                clock = self._phase_clocks.get(node)
                if clock is None:
                    node_id = self._phase_node_ids.setdefault(
                        node, len(self._phase_node_ids)
                    )
                    clock = NodePhaseClock(
                        node=node_id, omega=self._phase_omega,
                        channel=self._phase_channel,
                    )
                    self._phase_clocks[node] = clock
                clock.tick(relax_one_node)

        self._converged = True
        return len(self._transports)

    def node_phase_clock(self, key: Any) -> "NodePhaseClock | None":
        """This node's real-profiled phase clock, if profiling was enabled
        (``profile_channel=`` at construction) and the node was actually
        visited by ``propagate()``. ``None`` otherwise -- a node propagate()
        never popped has no real operation to have timed, and a field built
        without a channel was never asked to measure anything."""

        return self._phase_clocks.get(key)

    def _deposit(
        self, key: Any, category: str, hue: float, weight: float
    ) -> None:
        per_category = self._moments.setdefault(key, {})
        per_category[category] = per_category.get(
            category, self._accumulator()
        ).deposited(hue, weight)

    # -- readout ---------------------------------------------------------

    def moments(self, key: Any) -> Mapping[str, Moments]:
        return dict(self._moments.get(key, {}))

    def reading(self, key: Any, *, scale: float | None = None) -> InfluenceReading:
        """Collapse one node's sums into shader-ready colour."""

        per_category = self._moments.get(key, {})
        contract = self.contract
        normaliser = scale if scale is not None else self._weight_scale()
        readings: dict[str, CategoryReading] = {}
        for category in contract.categories:
            moments = per_category.get(category, self._accumulator())
            shaped = (1.0 - moments.dispersion) ** contract.saturation_gamma
            readings[category] = CategoryReading(
                category=category,
                hue=moments.mean,
                saturation=(
                    0.0 if moments.weight <= 0.0
                    else contract.saturation_floor
                    + (1.0 - contract.saturation_floor) * shaped
                ),
                weight=moments.weight,
                dispersion=moments.dispersion,
            )
        total = sum(item.weight for item in readings.values())
        baked = readings.get(BAKED)
        recurrent = readings.get(RECURRENT)
        return InfluenceReading(
            key=key,
            categories=readings,
            value=(
                0.0 if total <= 0.0 or normaliser <= 0.0
                else min(1.0, math.log1p(total) / math.log1p(normaliser))
            ),
            staging=(
                0.0 if total <= 0.0 or baked is None else baked.weight / total
            ),
            recurrence=(
                0.0 if total <= 0.0 or recurrent is None
                else recurrent.weight / total
            ),
        )

    def _weight_scale(self) -> float:
        """Heaviest total arrival, used to normalise the value channel."""

        if not self._moments:
            return 0.0
        return max(
            sum(moments.weight for moments in per_category.values())
            for per_category in self._moments.values()
        )

    def table(self) -> tuple[InfluenceReading, ...]:
        """Every reached node's collapsed reading, in deterministic order."""

        if not self._converged:
            self.propagate()
        scale = self._weight_scale()
        return tuple(
            self.reading(key, scale=scale)
            for key in sorted(self._moments, key=str)
        )

    def trace(self) -> tuple[Transport, ...]:
        """The ordered transports whose replay reproduces ``table()``."""

        if not self._converged:
            self.propagate()
        return tuple(self._transports)

    def to_mapping(self) -> Mapping[str, Any]:
        """Serialise for a shell payload, a report, or a renderer handoff."""

        return {
            "schema": SCHEMA,
            "spectrum_end": SPECTRUM_END,
            "reserved_band": list(RESERVED_BAND),
            "categories": list(self.contract.categories),
            "sources": [
                {
                    "key": str(source.key), "category": source.category,
                    "ordinal": source.ordinal, "hue": source.hue,
                    "label": source.label, "group": source.group,
                }
                for source in self.sources
            ],
            "nodes": [
                {
                    "key": str(reading.key),
                    "value": reading.value,
                    "staging": reading.staging,
                    "recurrence": reading.recurrence,
                    "categories": {
                        name: {
                            "hue": item.hue, "saturation": item.saturation,
                            "weight": item.weight, "dispersion": item.dispersion,
                        }
                        for name, item in reading.categories.items()
                    },
                }
                for reading in self.table()
            ],
            "transports": [
                {
                    "step": item.step, "source": str(item.source_key),
                    "target": str(item.target_key), "category": item.category,
                    "hue": item.hue, "weight": item.weight,
                    "iteration": item.iteration, "role": item.role,
                }
                for item in self.trace()
            ],
        }


def attach_to_metagraph(
    metagraph: Any,
    contract: InfluenceContract,
    *,
    classifier: Callable[[str, Mapping[str, Any]], str | None] = default_classifier,
) -> InfluenceField:
    """Subscribe a field to an ``EvolutionMetaGraph`` and return it.

    This is the hook that makes the analysis available to anything already
    recording its evolution -- which is every IR stage, every shell, and every
    backend adapter -- without any of them knowing this module exists. It
    wraps rather than replaces, exactly as ``shell_telemetry.attach_*`` does,
    so existing consumers of the event stream see no change.

    Handoff events are transported like ordinary edges deliberately: influence
    then follows a value across a lowering boundary, and colour authored over
    the ProcessGraph refracts into the SSA that replaced it. Where lowering was
    faithful the colours carry through; where a phi genuinely merged two paths
    a new confluence appears that has no ProcessGraph counterpart, which is the
    one place the topology changed meaningfully.
    """

    field_view = InfluenceField(contract)
    if not contract.enabled:
        return field_view

    counter = [0]
    stages: dict[str, str] = {}

    def ingest(event: Any) -> None:
        if event.kind == "graph-open" and event.graph is not None:
            stages[event.graph.id] = str(event.graph.stage)
            return
        component = getattr(event, "component", None)
        if component is None:
            return
        stage = stages.get(component.graph_id, "")
        if stage and not contract.selects_stage(stage):
            return
        if event.kind in {"component-spawn", "component-update"}:
            detail = event.detail or {}
            attributes = dict(detail.get("attributes") or {})
            path = str(attributes.get("path") or detail.get("label") or "")
            if path and not contract.selects_scope(path):
                return
            field_view.add_node(component)
            category = classifier(str(detail.get("kind") or ""), attributes)
            if category is not None and category in contract.categories:
                raw_level = attributes.get("schedule_level")
                counter[0] += 1
                field_view.stage_source(
                    component, category,
                    int(raw_level) if raw_level is not None else counter[0],
                    str(detail.get("label") or ""),
                    str(attributes.get("influence_group") or stage),
                )
        elif event.kind == "component-link":
            role = str((event.detail or {}).get("role") or "data")
            for source in event.sources:
                field_view.add_edge(source, component, role=role)
        elif event.kind == "component-handoff":
            transformation = str(
                (event.detail or {}).get("transformation") or "handoff"
            )
            for source in event.sources:
                field_view.add_edge(source, component, role=transformation)

    metagraph.subscribe(ingest, replay=True)
    return field_view


def field_from_process_graph(
    process_graph: Any,
    contract: InfluenceContract,
    *,
    classifier: Callable[[str, Mapping[str, Any]], str | None] = default_classifier,
) -> InfluenceField:
    """Build a field directly from a ProcessGraph, without event replay.

    Schedule levels order the hue arc when the graph has been planned, so hue
    reads as depth-of-origin. An unplanned graph falls back to node ordering,
    which still produces a well-defined field but a less interpretable one.
    """

    field_view = InfluenceField(contract)
    if not contract.enabled:
        return field_view

    graph = process_graph.G
    levels = dict(getattr(process_graph, "levels", {}) or {})
    entries: list[tuple[Any, str, int, str, str]] = []
    for index, (node_id, data) in enumerate(graph.nodes(data=True)):
        field_view.add_node(node_id)
        attributes = dict(data.get("attributes") or {})
        kind = str(data.get("kind") or data.get("op") or data.get("label") or "")
        category = classifier(kind, attributes)
        if category is not None and category in contract.categories:
            entries.append((
                node_id, category, int(levels.get(node_id, index)),
                str(data.get("label") or kind),
                str(attributes.get("influence_group") or ""),
            ))
    # A graph with no declared origin still has structural roots; tracing from
    # them beats reporting an empty field.
    if not entries:
        entries = [
            (node_id, DYNAMIC, int(levels.get(node_id, index)), str(node_id), "")
            for index, node_id in enumerate(graph.nodes())
            if graph.in_degree(node_id) == 0
        ]
    field_view.add_sources(entries)
    for source, target, data in graph.edges(data=True):
        field_view.add_edge(
            source, target, role=str(data.get("role") or "data")
        )
    return field_view


def field_from_dual_ir(
    shell: Any,
    contract: InfluenceContract,
    *,
    regions: Mapping[int, Any] | None = None,
    classifier: Callable[[str, Mapping[str, Any]], str | None] = default_classifier,
    label: str = "dual-ir influence",
) -> InfluenceField:
    """Build a field over a ``DualIRShell``: numeric steps plus control shell.

    This is the adapter where the solver's transport semantics actually engage.
    A ProcessGraph carries dependency, and every one of its edges is ``data``,
    so back-edge decay never fires, branch arms never split weight, and the
    barrier roles are never reached -- all of that machinery runs dead. Dual IR
    is the first of the three representations that has loops and branches *as
    structure*, so it is the first one whose edges carry
    ``loop-latch``/``loop-back``, ``branch-true``/``branch-false``, and
    ``loop-exit``/``branch-merge``.

    The control walk is not reimplemented here. ``record_compiled_execution_``
    ``evolution`` already traverses a ``ControlProgram`` and publishes exactly
    that role vocabulary, and ``attach_to_metagraph`` already turns those roled
    links into field edges -- so this records the shell through the existing
    recorder and reads the result. Writing a second walker would mean two
    traversals of the same dataclass tree drifting apart, and the recorder is
    the one the rest of the compiler already trusts.
    """

    field_view = InfluenceField(contract)
    if not contract.enabled:
        return field_view

    from .evolution_metagraph import (
        EvolutionComponentRef,
        EvolutionMetaGraph,
        record_compiled_execution_evolution,
        record_evolution,
        record_fused_program_evolution,
    )

    metagraph = EvolutionMetaGraph()
    field_view = attach_to_metagraph(metagraph, contract, classifier=classifier)

    numeric = getattr(shell, "compiled_shell_program", None)
    control = getattr(shell, "shell_control_program", None)
    # ``DualIRShell`` pairs the numeric and control programs but does not carry
    # the region table -- that lives on the ``AOTCompilation`` that produced it,
    # so a caller holding one passes it here. Reading ``shell.region_programs``
    # silently found nothing and fell back to ``compiled_shell_program``, which
    # on the precompile path is an intentionally empty sentinel: the field then
    # had no sources and transported nothing, while looking like it had worked.
    regions = dict(
        regions
        if regions is not None
        else (getattr(shell, "region_programs", None) or {})
    )
    if not regions and numeric is not None:
        steps = getattr(getattr(numeric, "program", numeric), "steps", ())
        if steps:
            regions = {0: numeric}

    with record_evolution(metagraph):
        region_graphs: dict[int, Any] = {}
        for index, captured in sorted(regions.items()):
            program = getattr(captured, "program", captured)
            region_graphs[int(index)] = record_fused_program_evolution(
                program, label=f"{label} numeric region {int(index)}"
            )
        # The region steps are already linearised, so their order is the order
        # the program brings values into existence. Unrolling does not destroy
        # that -- it makes it explicit, and a body that ran four times appears
        # as four activations, which is exactly how often ink should fire.
        activation: list[Any] = []
        for index, captured in sorted(regions.items()):
            program = getattr(captured, "program", captured)
            graph = region_graphs.get(int(index))
            if graph is None:
                continue
            for step in getattr(program, "steps", ()) or ():
                ref = EvolutionComponentRef(graph.id, str(step.result_id))
                if metagraph.has_component(ref):
                    activation.append(ref)
        field_view.activation_order = tuple(activation)

        # Phi lives in Dual IR as ``carried_aliases`` on the control blocks --
        # a ConditionalBlock carries (true, false, pre-branch, merged) and a
        # LoopBlock carries (updated, initial) -- and SSA only materialises it
        # later. ``record_compiled_execution_evolution`` walks the same tree
        # but never reads those tuples, so the structural edges appear while
        # the value merges they describe do not. Recovering them here keeps
        # the confluence in the field where it belongs.
        _phi_pairs: list[tuple[int, int]] = []

        def _collect_phi(block: Any) -> None:
            name = type(block).__name__
            if name == "SequenceBlock":
                for child in block.blocks:
                    _collect_phi(child)
            elif name == "ConditionalBlock":
                for entry in getattr(block, "carried_aliases", ()) or ():
                    if len(entry) == 4:
                        true_value, false_value, _pre, merged = entry
                        _phi_pairs.append((int(true_value), int(merged)))
                        _phi_pairs.append((int(false_value), int(merged)))
                _collect_phi(block.body)
                if getattr(block, "orelse", None) is not None:
                    _collect_phi(block.orelse)
            elif name in {"LoopBlock", "WhileBlock"}:
                for entry in getattr(block, "carried_aliases", ()) or ():
                    if len(entry) == 2:
                        updated, initial = entry
                        # The header phi takes the entry value and the value
                        # the latch carries back around.
                        _phi_pairs.append((int(initial), int(updated)))
                _collect_phi(getattr(block, "body", None) or SequenceBlockStub())
                condition = getattr(block, "condition", None)
                if condition is not None:
                    _collect_phi(condition)

        class SequenceBlockStub:
            blocks: tuple = ()

        if control is not None:
            try:
                _collect_phi(control.root)
            except Exception:
                _phi_pairs = []

        if control is not None:
            record_compiled_execution_evolution(
                control,
                region_graphs=region_graphs,
                region_programs={
                    int(index): getattr(captured, "program", captured)
                    for index, captured in regions.items()
                },
                label=label,
            )

        # ``carried_aliases`` names values in the compiler's cross-region
        # value-id space -- the same space ``precompile_to_ssa`` resolves
        # through ``external_value`` when it emits the Phi. A loop-carried
        # value lives *across* regions by definition, so looking it up in one
        # region's feeds and results finds the wrong table and misses most of
        # them. Components are keyed by value id wherever they were recorded,
        # so resolve against every component the run produced.
        owner: dict[int, Any] = {}
        for component in metagraph.snapshot().components:
            try:
                value_id = int(component.ref.local_id)
            except (TypeError, ValueError):
                continue
            owner.setdefault(value_id, component.ref)
        for produced, merged in _phi_pairs:
            source = owner.get(produced)
            target = owner.get(merged)
            if source is not None and target is not None and source != target:
                field_view.add_edge(source, target, role="phi")

    return field_view


def field_from_sympy(
    equations: Sequence[Any],
    contract: InfluenceContract,
    *,
    classifier: Callable[[str, Mapping[str, Any]], str | None] = default_classifier,
) -> InfluenceField:
    """Build a field over authored SymPy equations: the stage before compiling.

    This is the origin of the correlation. Every later stage is a
    transformation of these expressions, so tracing the same influence here
    is what lets a reader see one quantity across all four representations
    rather than four unrelated pictures.

    Nodes are keyed by the expression itself. SymPy expressions are hashable
    and structurally equal, so a subexpression appearing in two equations is
    ONE node without any CSE pass being run -- which is the honest shape of
    authored mathematics, where a repeated wave-speed term genuinely is the
    same term rather than a copy that happens to match.

    Free symbols are the sources, ordered by name so the hue arc is stable
    across runs; ordinal is expression depth, keeping the convention the
    other builders use, where hue reads as depth-of-origin.
    """

    field_view = InfluenceField(contract)
    if not contract.enabled:
        return field_view

    depths: dict[Any, int] = {}

    def visit(expression: Any, depth: int) -> None:
        previous = depths.get(expression)
        if previous is not None and previous <= depth:
            return
        depths[expression] = depth
        field_view.add_node(expression)
        for operand in getattr(expression, "args", ()) or ():
            visit(operand, depth + 1)
            field_view.add_edge(operand, expression, role="data")

    for equation in equations:
        right = getattr(equation, "rhs", equation)
        visit(right, 0)
        left = getattr(equation, "lhs", None)
        if left is not None:
            field_view.add_node(left)
            field_view.add_edge(right, left, role="data")
            depths.setdefault(left, 0)

    symbols = sorted(
        (node for node in depths if not (getattr(node, "args", ()) or ())),
        key=str,
    )
    entries: list[tuple[Any, str, int, str, str]] = []
    for ordinal, symbol in enumerate(symbols):
        attributes = {"authored_symbol": str(symbol)}
        category = classifier("symbol", attributes) or DYNAMIC
        if category in contract.categories:
            entries.append((symbol, category, ordinal, str(symbol), ""))
    field_view.add_sources(entries)
    return field_view


def field_from_ssa(
    module: Any,
    contract: InfluenceContract,
    *,
    functions: Sequence[str] | None = None,
    classifier: Callable[[str, Mapping[str, Any]], str | None] = default_classifier,
) -> InfluenceField:
    """Build a field over lowered SSA: def-use plus real block control flow.

    This is the representation that actually carries the control structure. A
    dependency graph has none, and a control shell that has been unrolled no
    longer has a loop to describe -- but SSA keeps blocks, so a ``CondBr`` is a
    fork, a ``Br`` back to an already-seen block is a back edge, and a ``Phi``
    is where two paths' values recombine.

    Roles are assigned so the solver's existing transport applies unchanged:
    branch arms divide, back edges decay and reclassify as loop-carried, and a
    phi merges by adding power sums exactly as any other confluence does.
    """

    field_view = InfluenceField(contract)
    if not contract.enabled:
        return field_view

    table = dict(getattr(module, "functions", None) or {})
    selected = [
        name for name in (functions if functions is not None else table)
        if name in table
    ]

    staged: list[tuple[Any, str, int, str, str]] = []
    activation: list[Any] = []
    ordinal = 0

    for name in selected:
        function = table[name]
        blocks = dict(getattr(function, "blocks", None) or {})
        order = list(blocks)
        position = {label: index for index, label in enumerate(order)}
        # A value's defining instruction is its node; an argument is a node
        # with no definition inside this function.
        defined: dict[Any, Any] = {}
        for label in order:
            for instruction in getattr(blocks[label], "instrs", ()) or ():
                result = getattr(instruction, "res", None)
                if result is not None:
                    defined[(name, result.id)] = (name, label, result.id)

        first_of_block: dict[str, Any] = {}
        for label in order:
            instructions = list(getattr(blocks[label], "instrs", ()) or ())
            if instructions:
                result = getattr(instructions[0], "res", None)
                first_of_block[label] = (
                    (name, label, result.id) if result is not None
                    else (name, label, "entry")
                )

        for label in order:
            block = blocks[label]
            successors = list(getattr(block, "successors", ()) or ())
            instructions = list(getattr(block, "instrs", ()) or ())
            for index, instruction in enumerate(instructions):
                result = getattr(instruction, "res", None)
                key = ((name, label, result.id) if result is not None
                       else (name, label, f"op{index}"))
                field_view.add_node(key)
                activation.append(key)

                operation = str(getattr(instruction, "op", ""))
                for argument in getattr(instruction, "args", ()) or ():
                    source = defined.get((name, getattr(argument, "id", None)))
                    if source is None:
                        # An operand with no definition here is an entry value:
                        # a parameter or an incoming constant, and therefore an
                        # origin of influence rather than a derivation.
                        source = (name, "entry", getattr(argument, "id", None))
                        field_view.add_node(source)
                        staged.append((
                            source, DYNAMIC, ordinal, f"{name}:arg", name,
                        ))
                        ordinal += 1
                        defined[(name, getattr(argument, "id", None))] = source
                    role = "phi" if operation == "Phi" else "data"
                    field_view.add_edge(source, key, role=role)

                if operation in {"Const"}:
                    staged.append((key, BAKED, ordinal, f"{name}:const", name))
                    ordinal += 1

                # Control edges leave from the terminator of the block.
                if index == len(instructions) - 1 and successors:
                    forking = operation == "CondBr" and len(successors) > 1
                    for arm, target in enumerate(successors):
                        entry = first_of_block.get(target)
                        if entry is None:
                            continue
                        backward = position.get(target, 0) <= position[label]
                        if backward:
                            role = "loop-back"
                        elif forking:
                            role = "branch-true" if arm == 0 else "branch-false"
                        else:
                            role = "control-next"
                        field_view.add_edge(key, entry, role=role)

    field_view.add_sources(staged)
    field_view.activation_order = tuple(activation)
    return field_view


__all__ = [
    "SCHEMA",
    "DYNAMIC", "BAKED", "RECURRENT", "CATEGORIES",
    "SPECTRUM_END", "RESERVED_BAND", "MAX_DISPERSION",
    "BACK_EDGE_ROLES", "FORK_ROLES", "BARRIER_ROLES",
    "InfluenceContractError", "InfluenceContract", "InfluenceSource",
    "Moments", "Spectrum", "CategoryReading", "InfluenceReading", "Transport",
    "InfluenceField",
    "semantic_marker_hue", "allocate_hues", "default_classifier",
    "attach_to_metagraph", "field_from_sympy", "field_from_process_graph",
    "field_from_dual_ir", "field_from_ssa",
]
