"""THE TIME FIELD: local time velocity as a scalar potential on a graph.

Not an engine feature. Nothing here knows what an engine is. This is the
form the dt system needs in order to be time-velocity aware. It was staged
in `engine_toy/time_field.py` and exercised against a real machine graph
there; it now lives in `src/common/dt_system`, which is universal and
serves every sim (moved 2026-09-22; the engine_toy module re-exports it).

VOCABULARY. `tau` here is TIME VELOCITY -- the rate a node's clock runs
relative to the reference -- which is what the dt system now means by tau
everywhere. It is measured (world time advanced over world time asked)
and never scheduled on; the dt system schedules on each participant's
EXCHANGE TIME instead (`time_contracts`, `participants`).

WHAT IT IS

Every node carries a local time velocity: the rate its own clock runs
relative to a reference. The reference is a gauge choice -- "realtime",
or whatever cap we set -- and the ABSOLUTE LEVEL IS UNOBSERVABLE. Inside
a zone every law is already written against that zone's own clock, so
the physics is internally exact and nothing needs to know its own rate.
Time drops out.

It does not drop out in three places, and they are the whole reason this
module exists:

  1. crossing a boundary -- a product sampled in one zone and read in
     another must be resampled, which needs the rate it was sampled at
  2. comparing across zones -- dyno figures, a parity check against a
     compiled core, anything scored; both lie silently otherwise
  3. WHEN THE RATE IS CHANGING

STORED AS A LOG, AND WITH ITS DERIVATIVE

The field is `log_tau` per node. A joint's time ratio is then a
DIFFERENCE across it, and any difference field has zero circulation
around a closed loop -- so cycle consistency is structural rather than
solved. (Measured on a real machine graph: 306 nodes, 259 coupling
edges, 57 components, 10 independent cycles. Not a tree. The constraint
is real; this formulation is what makes it vanish.)

`dlog_tau_dt` is carried alongside, and it is the one with physical
consequences. A CONSTANT time gradient is unobservable -- an ideal
differential passes it losslessly, forever, felt by nobody. A CHANGING
one is a force: the adaptor's own freedom must accelerate, which takes
torque. The field is therefore dynamical, like position and velocity,
not a static map.

WHAT CAN CARRY A GRADIENT

Not a flag we invent -- a property of the mechanism. A joint can carry a
sustained time gradient if and only if it has a RATE-DIFFERENCE DEGREE
OF FREEDOM. A differential has one (the spiders). A slipping clutch has
one. A converter has one (fluid slip). A rigid shaft, a splined hub or a
gear mesh does not, and across one of those a rate mismatch accumulates
twist without bound: stall, then shatter. That is not a rule imposed
from outside, it is the same equation at infinite stiffness, and it is
what bounds the field PHYSICALLY instead of by a tuning constant.

Phase is a separate question from rate. A differential absorbs rate
difference but lets accumulated angle drift without limit -- correct for
wheels, catastrophic for a timing drive. So a phase-locked joint admits
no gradient at ANY stiffness, which is why an engine's crank, cams and
valvetrain are necessarily one zone and the boundary is at the coupling.

WE CHOOSE WHAT STORES TIME FORCE

The transient reaction from `dlog_tau_dt` has to go somewhere, and that
is a design choice rather than a fact -- see `TimeForceStore`. Whatever
we pick sets the ramp limit, because the store's inertia is what decides
how fast the field may legally change. Load shedding must RAMP, and how
fast it may ramp is a property of the machine, not of the scheduler.
Slamming the time field is dumping the clutch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

__all__ = (
    "ADAPTORS", "StoreLedger", "TimeAdaptor", "TimeField", "TimeFieldConfig",
    "TimeForceStore", "adaptor_for", "time_angle",
)


# ---------------------------------------------------------------------
# what a joint is able to do when the rates across it do not match
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TimeAdaptor:
    """A mechanism's capability to sit across a time gradient.

    `rate_freedom`   can it carry a SUSTAINED rate difference at all
    `lossless`       does a sustained difference cost anything. An ideal
                     open differential is lossless -- equal torque T on
                     both outputs, carrier torque 2T, omega_c the mean:
                         P_in  = 2T * (w1 + w2)/2 = T(w1 + w2)
                         P_out = T*w1 + T*w2      = T(w1 + w2)
                     equal, for ANY rate difference. A clutch or a
                     converter absorbs the same difference as heat.
                     Dissipation is a property of the ADAPTOR, not of
                     time gradients.
    `phase_locked`   does the joint carry phase as well as rate. A timing
                     drive does; no adaptor can sit inside one.
    `store_inertia_kg_m2`
                     the freedom's own inertia -- what must be
                     accelerated when the gradient CHANGES. This is the
                     ramp limit. 0.0 means "not chosen yet".
    """

    kind: str
    rate_freedom: bool
    lossless: bool
    phase_locked: bool = False
    store_inertia_kg_m2: float = 0.0
    note: str = ""

    @property
    def admits_gradient(self) -> bool:
        """Whether a sustained time gradient may exist across this joint."""
        return self.rate_freedom and not self.phase_locked


#: The mechanisms, classified by what they can actually do. The slipping
#: family already exists in couplings.py and already models slip; the
#: differential is the only lossless member and is the reason a time
#: gradient can be free rather than merely affordable.
ADAPTORS: dict[str, TimeAdaptor] = {
    "differential": TimeAdaptor(
        "differential", rate_freedom=True, lossless=True,
        note="spiders absorb unbounded rate difference at zero power cost"),
    "fluid-coupling": TimeAdaptor(
        "fluid-coupling", rate_freedom=True, lossless=False,
        note="slip becomes heat in the fluid"),
    "torque-converter": TimeAdaptor(
        "torque-converter", rate_freedom=True, lossless=False,
        note="slip becomes heat; locking up removes the freedom entirely"),
    "dry-friction": TimeAdaptor(
        "dry-friction", rate_freedom=True, lossless=False,
        note="slips, wears, and heats"),
    "wet-multi-plate": TimeAdaptor(
        "wet-multi-plate", rate_freedom=True, lossless=False),
    # --- no freedom: these SHEAR across a gradient ---
    "rigid-bolted-joint": TimeAdaptor("rigid-bolted-joint", False, True),
    "rigid-keyed-hub": TimeAdaptor("rigid-keyed-hub", False, True),
    "torque-shaft": TimeAdaptor("torque-shaft", False, True),
    "rotational-bearing": TimeAdaptor("rotational-bearing", False, True),
    # --- phase matters, so no adaptor is admissible at any stiffness ---
    "camshaft-timing-drive": TimeAdaptor(
        "camshaft-timing-drive", rate_freedom=False, lossless=True,
        phase_locked=True,
        note="losing accumulated angle here is losing valve timing"),
}


def adaptor_for(kind: "str | TimeAdaptor") -> TimeAdaptor:
    """The capability of a joint of this kind. Unknown kinds are treated
    as RIGID -- the conservative answer, because assuming a freedom that
    is not there is how a gradient silently shears something.

    A declared ``TimeAdaptor`` is taken as itself: that is how a coupling
    states a store inertia the table leaves "not chosen yet", e.g.
    ``dataclasses.replace(ADAPTORS["differential"], store_inertia_kg_m2=0.2)``.
    """
    if isinstance(kind, TimeAdaptor):
        return kind
    return ADAPTORS.get(str(kind), ADAPTORS["torque-shaft"])


# ---------------------------------------------------------------------
# where the transient reaction goes -- a choice, not a fact
# ---------------------------------------------------------------------

class TimeForceStore:
    """What absorbs the reaction when the field CHANGES.

    A sustained gradient is free through a differential. Changing one is
    not: the freedom must angularly accelerate at d/dt[(w1-w2)/2], which
    costs torque equal to the store's inertia times that acceleration.
    Whatever we choose here is what sets how fast the field may move.
    """

    #: The adaptor's own rotating freedom -- real hardware, real inertia,
    #: reaction goes where it physically goes. The honest default.
    ADAPTOR_INERTIA = "adaptor-inertia"

    #: A declared capacitance in the field itself: the field absorbs its
    #: own transients and the machine never feels the ramp. Cheap and
    #: forgiving, but it is a fiction the parts do not know about.
    FIELD_CAPACITANCE = "field-capacitance"

    #: No store at all: any derivative is refused outright rather than
    #: absorbed. The field may only be changed between frames where the
    #: joint is already at zero transmitted torque.
    NONE = "none"


@dataclass
class TimeFieldConfig:
    store: str = TimeForceStore.ADAPTOR_INERTIA
    #: Reference rate. A gauge choice -- "realtime", or whatever cap is
    #: set. The absolute level is unobservable; only differences and
    #: derivatives are physical.
    reference_hz: float = 60.0
    #: Hard bound on |d(log tau)/dt| when the store cannot supply one.
    #: Only consulted for FIELD_CAPACITANCE / NONE; with ADAPTOR_INERTIA
    #: the limit is derived per joint from real inertia instead.
    fallback_max_ramp_per_s: float = 2.0


# ---------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------

@dataclass
class TimeField:
    """Log time velocity, and its derivative, per node.

    `log_tau[n] == 0.0` means node n runs at the reference rate.
    Negative means its clock runs slow.
    """

    nodes: tuple[str, ...]
    log_tau: list[float] = field(default_factory=list)
    dlog_tau_dt: list[float] = field(default_factory=list)
    config: TimeFieldConfig = field(default_factory=TimeFieldConfig)
    _index: dict[str, int] = field(default_factory=dict, repr=False)
    #: scope -> the scope it is nested inside, mirroring the dt graph's
    #: RoundNode tree. Absent means "a root scope, directly in the frame".
    _parent: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def flat(cls, nodes: Iterable[str], config: TimeFieldConfig | None = None) -> "TimeField":
        """A field at the reference everywhere -- no dilation anywhere."""
        names = tuple(str(n) for n in nodes)
        return cls(
            nodes=names,
            log_tau=[0.0] * len(names),
            dlog_tau_dt=[0.0] * len(names),
            config=config or TimeFieldConfig(),
            _index={n: i for i, n in enumerate(names)},
        )

    # -- reading -------------------------------------------------------
    def velocity(self, node: str) -> float:
        """This node's clock rate as a plain ratio (1.0 = reference)."""
        return math.exp(self.log_tau[self._index[node]])

    def ratio(self, a: str, b: str) -> float:
        """How much faster a runs than b. The only physical reading of
        two nodes -- a DIFFERENCE, which is why the level is free."""
        return math.exp(self.log_tau[self._index[a]] - self.log_tau[self._index[b]])

    def gradient(self, a: str, b: str) -> float:
        """log-space gradient across a joint. Zero circulation around any
        cycle, by construction."""
        return self.log_tau[self._index[a]] - self.log_tau[self._index[b]]

    def gradient_rate(self, a: str, b: str) -> float:
        """d/dt of the gradient -- the quantity with consequences."""
        return self.dlog_tau_dt[self._index[a]] - self.dlog_tau_dt[self._index[b]]

    # -- the physical questions ----------------------------------------
    def admits(self, a: str, b: str, joint_kind: str) -> bool:
        """Whether the gradient currently standing across this joint is
        one the joint can actually carry."""
        if abs(self.gradient(a, b)) <= 1e-12:
            return True          # no gradient: anything carries it
        return adaptor_for(joint_kind).admits_gradient

    def shear_reaction_nm(self, a: str, b: str, joint_kind: str,
                          omega_ref_rad_s: float) -> float:
        """Torque the joint must supply because the gradient is CHANGING.

        The freedom accelerates at d/dt[(w_a - w_b)/2]; with w scaling as
        the local rate, that derivative is omega * d(gradient)/dt / 2.
        Times the store's inertia. Returns 0.0 for a steady gradient --
        which is the whole point: constant dilation is free, transients
        are not.
        """
        adaptor = adaptor_for(joint_kind)
        rate = self.gradient_rate(a, b)
        if rate == 0.0:
            return 0.0
        if not adaptor.admits_gradient:
            # No freedom to accelerate: the mismatch goes into the
            # structure instead. Reported as infinite so the caller sees
            # a shear, not a small number it might round away.
            return float("inf")
        inertia = adaptor.store_inertia_kg_m2
        if self.config.store == TimeForceStore.FIELD_CAPACITANCE:
            return 0.0           # the field eats its own transient
        if self.config.store == TimeForceStore.NONE:
            return float("inf")  # refuse rather than absorb
        return inertia * abs(omega_ref_rad_s * rate * 0.5)

    def max_ramp_per_s(self, joint_kind: str, omega_ref_rad_s: float,
                       torque_budget_nm: float) -> float:
        """How fast the field may legally change across this joint.

        NOT a tuning constant: inverted from the store's real inertia and
        whatever torque the joint can stand. This is why load shedding
        has to ramp, and how fast it is allowed to.
        """
        adaptor = adaptor_for(joint_kind)
        if not adaptor.admits_gradient:
            return 0.0
        if self.config.store == TimeForceStore.FIELD_CAPACITANCE:
            return self.config.fallback_max_ramp_per_s
        inertia = adaptor.store_inertia_kg_m2
        denom = abs(inertia * omega_ref_rad_s * 0.5)
        if denom <= 1e-30:
            return self.config.fallback_max_ramp_per_s
        return abs(torque_budget_nm) / denom

    # -- writing -------------------------------------------------------
    def set_target(self, node: str, log_tau_target: float, dt: float,
                   max_ramp_per_s: float) -> float:
        """Move one node toward a target, rate-limited. Returns the
        realised log_tau.

        The rate limit is the physics, not smoothing: the derivative is
        what the machine feels."""
        i = self._index[node]
        if dt <= 0.0:
            return self.log_tau[i]
        want = float(log_tau_target) - self.log_tau[i]
        cap = abs(max_ramp_per_s) * dt
        step = max(-cap, min(cap, want))
        self.log_tau[i] += step
        self.dlog_tau_dt[i] = step / dt
        return self.log_tau[i]

    def settle(self) -> None:
        """Declare the field steady: derivatives to zero, levels kept. A
        steady gradient costs nothing, so this is the free state."""
        self.dlog_tau_dt = [0.0] * len(self.nodes)

    # -- hierarchy: the dt graph's RoundNode tree, in the field ---------
    def set_parent(self, node: str, parent: str) -> None:
        """Nest one scope inside another.

        `dt_system`'s RoundNode already nests -- an outer round delegates
        its window to an inner controller that may subdivide it -- so a
        beam step inside a rigid-body round inside a frame round is
        already expressible there. Nested windows MULTIPLY, and this
        field is stored as a log, so they simply ADD. The RoundNode tree
        and the field's tree are the same tree, which is why one scalar
        per node is enough for a heterogeneous stack (beam, rigid body,
        actuators, circuits, engines) rather than needing a scheme per
        kind.
        """
        if node == parent:
            raise ValueError("a scope cannot be its own parent")
        self._parent[node] = parent

    def effective_log_tau(self, node: str) -> float:
        """This scope's rate INCLUDING everything it is nested inside.

        Walks to the root adding logs -- the composition that makes a
        hierarchy work. Cycles are refused rather than looped."""
        seen: set[str] = set()
        total = 0.0
        cursor = node
        while cursor is not None:
            if cursor in seen:
                raise ValueError(f"time scope cycle at {cursor}")
            seen.add(cursor)
            total += self.log_tau[self._index[cursor]]
            cursor = self._parent.get(cursor)
        return total

    def effective_velocity(self, node: str) -> float:
        return math.exp(self.effective_log_tau(node))

    # -- crossing a boundary -------------------------------------------
    def resample(self, value_per_local_second: float, frm: str, to: str) -> float:
        """Convert a RATE sampled in one zone into another zone's rate.

        This is what "products carry their time angle" buys. A quantity
        measured per local second in `frm` is not the same number per
        local second in `to`; dropping this is how coupled things drift
        apart silently.
        """
        return float(value_per_local_second) * self.ratio(to, frm)


# ---------------------------------------------------------------------
# what the store has had to absorb: the energy build-up
# ---------------------------------------------------------------------

@dataclass
class StoreLedger:
    """What the time force store is holding, per zone.

    Charged by ramping the field, discharged when the field is steady.
    A store that never discharges is a zone under sustained time
    pressure -- which is the condition worth selling.
    """

    stored_j: float = 0.0
    #: frames in a row the store has failed to return to rest
    persistence: int = 0
    #: exponential mean of (asked - got) / asked, 0..1
    shortfall_ema: float = 0.0
    rest_j: float = 1e-9

    def observe(self, *, reaction_nm: float, slip_rad_s: float, dt_s: float,
                asked_s: float, got_s: float, alpha: float = 0.25) -> None:
        """Fold one frame in. `reaction_nm * slip_rad_s` is the power the
        store had to supply; `asked` vs `got` is the leading edge."""
        power_w = abs(float(reaction_nm)) * abs(float(slip_rad_s))
        self.stored_j = max(0.0, self.stored_j + power_w * float(dt_s))
        shortfall = 0.0 if asked_s <= 0.0 else max(0.0, (asked_s - got_s) / asked_s)
        self.shortfall_ema = (1.0 - alpha) * self.shortfall_ema + alpha * shortfall
        if self.stored_j > self.rest_j or shortfall > 0.0:
            self.persistence += 1
        else:
            self.persistence = 0

    def relax(self, dt_s: float, relaxation_s: float = 0.5) -> None:
        """A steady field costs nothing, so the store bleeds back to
        rest. This is what stops a single transient looking like chronic
        trouble."""
        if relaxation_s <= 0.0:
            self.stored_j = 0.0
            return
        self.stored_j *= max(0.0, 1.0 - float(dt_s) / relaxation_s)


def time_angle(field_: TimeField, node: str) -> float:
    """The rate a product from this node was sampled at, to be carried
    with it. Plain ratio, so 1.0 is the reference and needs no
    conversion."""
    return field_.velocity(node)
