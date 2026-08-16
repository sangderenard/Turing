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
    scopes: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()
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


@dataclass(frozen=True, slots=True)
class _Cursor:
    """Heap entry. Ordered heaviest-first, ties broken on insertion order."""

    negated_weight: float
    ordinal: int
    key: Any
    category: str
    hue: float
    weight: float
    iteration: int

    def __lt__(self, other: "_Cursor") -> bool:
        if self.negated_weight != other.negated_weight:
            return self.negated_weight < other.negated_weight
        return self.ordinal < other.ordinal


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

    def __init__(self, contract: InfluenceContract | None = None) -> None:
        self.contract = contract or InfluenceContract.disabled()
        self._sources: dict[Any, InfluenceSource] = {}
        self._staged: list[tuple[Any, str, int, str, str]] = []
        self._outgoing: dict[Any, list[tuple[Any, str]]] = {}
        self._nodes: set[Any] = set()
        self._barriers: dict[Any, set[str]] = {}
        self._moments: dict[Any, dict[str, Moments]] = {}
        self._transports: list[Transport] = []
        self._converged = False

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

        Weight strictly decreases along every hop, so the walk terminates:
        each traversal multiplies by at most ``max(attenuation, decay) <= 1``
        with at least one factor strictly below 1 on any cycle, and cursors
        under ``epsilon`` retire. ``max_transports`` is a backstop for
        pathological fan-out, not the primary bound.
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

        heap: list[_Cursor] = []
        ordinal = 0
        for source in sorted(selected.values(), key=lambda item: item.ordinal):
            self._deposit(source.key, source.category, source.hue, 1.0)
            heapq.heappush(heap, _Cursor(
                negated_weight=-1.0, ordinal=ordinal, key=source.key,
                category=source.category, hue=source.hue, weight=1.0,
                iteration=0,
            ))
            ordinal += 1

        step = 0
        while heap and step < contract.max_transports:
            cursor = heapq.heappop(heap)
            edges = self._outgoing.get(cursor.key, ())
            if not edges:
                continue
            forks = sum(1 for _, role in edges if role in FORK_ROLES)
            for target, role in edges:
                weight = cursor.weight * contract.attenuation
                iteration = cursor.iteration
                if role in BACK_EDGE_ROLES:
                    weight *= contract.decay
                    iteration += 1
                if role in FORK_ROLES and forks:
                    # Alternatives, not parallel successors: the arms divide
                    # the weight between them rather than each taking all of
                    # it, which is also what bounds the cursor population.
                    weight /= forks
                if weight < contract.epsilon:
                    continue
                category = (
                    RECURRENT
                    if role in BACK_EDGE_ROLES
                    and RECURRENT in contract.categories
                    else cursor.category
                )
                self._deposit(target, category, cursor.hue, weight)
                self._transports.append(Transport(
                    step=step, source_key=cursor.key, target_key=target,
                    category=category, hue=cursor.hue, weight=weight,
                    iteration=iteration, role=role,
                ))
                heapq.heappush(heap, _Cursor(
                    negated_weight=-weight, ordinal=ordinal, key=target,
                    category=category, hue=cursor.hue, weight=weight,
                    iteration=iteration,
                ))
                ordinal += 1
                step += 1
        self._converged = True
        return len(self._transports)

    def _deposit(
        self, key: Any, category: str, hue: float, weight: float
    ) -> None:
        per_category = self._moments.setdefault(key, {})
        per_category[category] = per_category.get(
            category, Moments()
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
            moments = per_category.get(category, Moments())
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


__all__ = [
    "SCHEMA",
    "DYNAMIC", "BAKED", "RECURRENT", "CATEGORIES",
    "SPECTRUM_END", "RESERVED_BAND", "MAX_DISPERSION",
    "BACK_EDGE_ROLES", "FORK_ROLES", "BARRIER_ROLES",
    "InfluenceContractError", "InfluenceContract", "InfluenceSource",
    "Moments", "CategoryReading", "InfluenceReading", "Transport",
    "InfluenceField",
    "semantic_marker_hue", "allocate_hues", "default_classifier",
    "attach_to_metagraph", "field_from_process_graph",
]
