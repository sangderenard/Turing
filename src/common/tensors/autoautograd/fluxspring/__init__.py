from .fs_types import (
    LearnCtrl, NodeCtrl, EdgeTransportLearn, EdgeTransport, EdgeCtrl,
    NodeSpec, EdgeSpec, FaceLearn, FaceSpec,
    DirichletCfg, DECSpec, RegCfg, FluxSpringSpec, SpectralCfg
)
from .fs_io import load_fluxspring, save_fluxspring, validate_fluxspring
from .fs_dec import (
    incidence_tensors_AT,
    validate_boundary_of_boundary_AT,
    edge_vectors_AT,
    edge_strain_AT,
    face_flux_AT,
    curvature_activation_AT,
    edge_energy_AT,
    face_energy_from_strain_AT,
    total_energy_AT,
    dec_energy_and_gradP_AT,
    path_edge_energy_AT,
    transport_tick,
    pump_tick,
)
# Spectral utilities
from .spectral_readout import compute_metrics
from .fs_harness import RingHarness
from typing import Callable, Sequence
import logging
# Torch bridge is optional import to keep AT-only usage clean.

from ...abstraction import AbstractTensor as AT

logger = logging.getLogger(__name__)


def spiral_slot(tick: int, row_idx: int, spiral_len: int) -> int:
    """Return the slot index for ``row_idx`` at ``tick`` following the
    spiral pattern.

    Parameters
    ----------
    tick:
        Global tick counter.
    row_idx:
        Row index within the parameter tensor.
    spiral_len:
        Total number of slots in the spiral.
    """

    if spiral_len <= 0:
        return 0
    return int((tick - row_idx) % spiral_len)


def required_spiral_len(spec: FluxSpringSpec, extra_delay: int = 0) -> int:
    """Return the minimal spiral length for a spec.

    Parameters
    ----------
    spec:
        FluxSpring configuration providing the spectral window length.
    extra_delay:
        Additional delay slots beyond ``spec.spectral.win_len``.  Exposed so
        future stages can reserve space for effects such as transport lag.

    Notes
    -----
    The spiral length is the maximal delay supported by the system.  It is
    computed as ``spec.spectral.win_len + extra_delay``.
    """

    return int(spec.spectral.win_len) + int(extra_delay)
def _tape():
    # Autograd is monkey-patched onto AT in your stack.
    try:
        return AT.autograd.tape
    except Exception:
        # Fallback if needed
        from ...autograd import autograd as _ag
        return _ag.tape

def _rebind_param(
    param: AT | None,
    learn: bool,
    out: list[AT],
    *,
    label: str | None = None,
    mark_structural_if_frozen: bool = True,
) -> AT | None:
    """Toggle requires_grad, attach to tape, label, and collect trainables."""
    if param is None:
        return None

    # Re-leaf on the active tape, preserving object identity in the spec.
    t = param.detach()
    t.requires_grad_(learn)

    tape = _tape()
    if label is not None:
        tape.annotate(t, label=label)

    if learn:
        # Drop any stale gradients/caches to make 'first tick' checks clean.
        # we can't zero gradients until we have gradients to zero
        # so for diagnostic reasons and because it takes many ticks
        # to accumulate gradients, we will only zero them when we are sure
        # they exist.
        #if hasattr(t, "zero_grad"):
        #    t.zero_grad(clear_cache=True)
        out.append(t)
    elif False and mark_structural_if_frozen:
        # Make frozen fields explicit so they don't pollute param lists.
        # this is disabled for reasons explained above
        tape.mark_structural(t, label=label)

    return t


class ParamWheel:
    """Manage multiple in-flight versions of a single parameter.

    Each wheel stores ``W`` parameter versions alongside matching gradient
    buffers.  Slots are rotated in a ring; gradients accumulate until the slot
    is evicted, at which point an update function is applied.
    """

    def __init__(
        self,
        base: AT,
        setter: Callable[[AT], None],
        *,
        slots: int = 2,
        label: str | None = None,
    ):
        self.setter = setter
        self.label = label

        self._versions: list[AT] = [base]
        for _ in range(slots - 1):
            t = base.detach().clone()
            t.requires_grad_(True)
            self._versions.append(t)

        # Gradient ring buffers parallel to ``_versions``
        self._grads: list[AT | None] = [None for _ in range(slots)]
        self._frozen = False

        tape = _tape()
        if label is not None:
            for p in self._versions:
                tape.annotate(p, label=label)

        # ``idx`` tracks the next slot to receive new data; the value returned
        # from :meth:`rotate` is the slot to be evicted and updated.  Start at
        # ``-1`` so that an initial ``rotate(); bind_slot()`` sequence activates
        # slot ``0`` to remain compatible with legacy call patterns.
        self.idx = -1

    # ------------------------------------------------------------------
    @property
    def params(self) -> list[AT]:
        """Alias for ``versions`` to preserve legacy callers."""
        return self._versions

    def versions(self) -> list[AT]:
        return self._versions

    def grads(self) -> list[AT | None]:
        return self._grads

    # ------------------------------------------------------------------
    def grow(self, slots: int) -> None:
        """Grow the wheel to at least ``slots`` entries.

        Growth is disallowed after :meth:`freeze` has been called.
        """

        if self._frozen or slots <= len(self._versions):
            if slots > len(self._versions) and self._frozen:
                raise RuntimeError("ParamWheel is frozen")
            return

        base = self._versions[0]
        tape = _tape()
        for _ in range(len(self._versions), slots):
            t = base.detach().clone()
            t.requires_grad_(True)
            self._versions.append(t)
            self._grads.append(None)
            if self.label is not None:
                tape.annotate(t, label=self.label)

    # ------------------------------------------------------------------
    def freeze(self) -> None:
        """Prevent further growth of the wheel."""

        self._frozen = True

    # ------------------------------------------------------------------
    def rotate(self) -> int:
        evicted = self.idx
        self.idx = (self.idx + 1) % len(self._versions)
        return evicted

    # ------------------------------------------------------------------
    def bind_slot(self) -> AT:
        p = self._versions[self.idx]
        self.setter(p)
        return p

    # ------------------------------------------------------------------
    def slots_for_tick(self, tick: int) -> list[int]:
        """Return slot indices for each row at ``tick``."""

        W = len(self._versions)
        base = self._versions[0]
        shape = getattr(base, "shape", ())
        rows = int(shape[0]) if len(shape) > 0 else 1
        return [spiral_slot(tick, r, W) for r in range(rows)]

    # ------------------------------------------------------------------
    def bind_for_tick(self, tick: int) -> set[int]:
        """Bind row-wise parameter versions based on ``tick``.

        Returns the set of slot indices touched during this binding so that
        their gradients can later be stashed.
        """

        row_slots = self.slots_for_tick(tick)
        used = set(row_slots)
        for r, s in enumerate(row_slots):
            v = self._versions[s]
            logger.debug(
                "bind_for_tick row=%d slot=%d id=%d requires_grad=%s",
                r,
                s,
                id(v),
                bool(getattr(v, "requires_grad", False)),
            )

        if len(row_slots) == 1:
            self.setter(self._versions[row_slots[0]])
            return used

        rows_out = [self._versions[s][r] for r, s in enumerate(row_slots)]
        stacked = AT.stack(rows_out, dim=0)
        self.setter(stacked)
        return used

    # ------------------------------------------------------------------
    def value_for_slots(self, row_slots: Sequence[int]) -> AT:
        """Reconstruct the parameter tensor from ``row_slots``."""

        if len(row_slots) == 1:
            return self._versions[row_slots[0]]

        rows_out = [self._versions[s][r] for r, s in enumerate(row_slots)]
        return AT.stack(rows_out, dim=0)

    # ------------------------------------------------------------------
    def stash_grads(self, slots: set[int]) -> None:
        """Accumulate gradients from ``slots`` into the wheel buffers."""

        for s in slots:
            p = self._versions[s]
            g = getattr(p, "grad", None)
            if g is None:
                continue
            g = AT.get_tensor(g)
            prev = self._grads[s]
            self._grads[s] = g if prev is None else prev + g
            if hasattr(p, "_grad"):
                p._grad = None

    # ------------------------------------------------------------------
    def apply_slot(self, idx: int, update_fn: Callable[[AT, AT], AT]) -> None:
        if idx < 0 or idx >= len(self._versions):
            return
        p = self._versions[idx]
        g = self._grads[idx]
        if g is None:
            g = getattr(p, "grad", None)
        if g is None:
            return
        new_p = update_fn(p, g)
        p.data = AT.get_tensor(new_p)
        self._grads[idx] = None
        if hasattr(p, "_grad"):
            p._grad = None


def register_param_wheels(
    spec: FluxSpringSpec, *, slots: int | None = None, extra_delay: int = 0
) -> list[ParamWheel]:
    """Instantiate :class:`ParamWheel` objects for all learnable parameters.

    When ``spec.spectral.enabled`` is ``True`` and ``slots`` is not provided,
    the number of slots defaults to the FFT window length so that every
    parameter wheel maintains a full window of versions.  Otherwise two slots
    are used as a minimal ring.
    """

    if slots is None:
        slots = required_spiral_len(spec, extra_delay) if spec.spectral.enabled else 2

    wheels: list[ParamWheel] = []
    tmp: list[AT] = []

    cnt = 0
    # Nodes
    for n in spec.nodes:
        lc = n.ctrl.learn
        for attr in ("alpha", "w", "b"):
            learn = getattr(lc, attr)
            p = _rebind_param(getattr(n.ctrl, attr), learn, tmp, label=f"node[{n.id}].ctrl.{attr}")
            setattr(n.ctrl, attr, p)
            if learn and p is not None:
                wheels.append(
                    ParamWheel(p, lambda t, n=n, attr=attr: setattr(n.ctrl, attr, t), slots=slots, label=f"node[{n.id}].ctrl.{attr}")
                )

    # Edges
    for e in spec.edges:
        lc = e.ctrl.learn
        for attr in ("alpha", "w", "b"):
            learn = getattr(lc, attr)
            p = _rebind_param(getattr(e.ctrl, attr), learn, tmp, label=f"edge[{e.src}->{e.dst}].ctrl.{attr}")
            setattr(e.ctrl, attr, p)
            if learn and p is not None:
                wheels.append(
                    ParamWheel(p, lambda t, e=e, attr=attr: setattr(e.ctrl, attr, t), slots=slots, label=f"edge[{e.src}->{e.dst}].ctrl.{attr}")
                )

        lt = e.transport.learn
        for attr in ("kappa", "k", "l0", "lambda_s", "x"):
            learn = getattr(lt, attr)
            p = _rebind_param(getattr(e.transport, attr), learn, tmp, label=f"edge[{e.src}->{e.dst}].tr.{attr}")
            setattr(e.transport, attr, p)
            if learn and p is not None:
                wheels.append(
                    ParamWheel(p, lambda t, e=e, attr=attr: setattr(e.transport, attr, t), slots=slots, label=f"edge[{e.src}->{e.dst}].tr.{attr}")
                )

    # Faces
    for f in spec.faces:
        lf = f.learn
        fid = getattr(f, "id", "?")
        for attr in ("alpha", "c"):
            learn = getattr(lf, attr)
            p = _rebind_param(getattr(f, attr, None), learn, tmp, label=f"face[{fid}].{attr}")
            setattr(f, attr, p)
            if learn and p is not None:
                wheels.append(
                    ParamWheel(p, lambda t, f=f, attr=attr: setattr(f, attr, t), slots=slots, label=f"face[{fid}].{attr}")
                )
    logger.debug(
        "register_param_wheels: created %d wheels slots=%d spectral=%s", 
        len(wheels),
        int(slots),
        bool(spec.spectral.enabled),
    )
    return wheels


def wheel_tick(
    psi: AT,
    spec: FluxSpringSpec,
    *,
    wheels: Sequence[ParamWheel],
    tick: int,
    update_fn: Callable[[AT, AT], AT] = lambda p, g: p,
    **pump_kw,
) -> tuple[AT, dict[str, AT]]:
    """Run a single :func:`pump_tick` with parameters sourced from wheels.

    Parameters
    ----------
    psi:
        State vector passed directly to :func:`pump_tick`.
    spec:
        FluxSpring specification mutated in-place with the assembled parameters.
    wheels:
        Sequence of :class:`ParamWheel` objects controlling learnable tensors.
    tick:
        Global tick counter used when selecting slots for each row via
        ``spiral_slot(tick, row_idx, W)``.
    update_fn:
        Callable applied to the parameter in the evicted slot using the stored
        gradient.  Defaults to a no-op.
    **pump_kw:
        Additional keyword arguments forwarded to :func:`pump_tick`.
    """

    # Bind rows to slot-specific leaves for this tick
    used_per_wheel: list[set[int]] = []
    for w in wheels:
        used_per_wheel.append(w.bind_for_tick(tick))

    # Single forward/backward pass
    psi, stats = pump_tick(psi, spec, **pump_kw)

    # Stash gradients from touched slots
    for w, used in zip(wheels, used_per_wheel):
        w.stash_grads(used)

    # Rotate wheels and update only the evicted slot
    for w in wheels:
        ev = w.rotate()
        w.apply_slot(ev, update_fn)

    return psi, stats


def register_learnable_params(spec: FluxSpringSpec) -> list[AT]:
    """Legacy helper returning a flat list of learnable parameter tensors."""

    params: list[AT] = []

    # Nodes
    for n in spec.nodes:
        lc = n.ctrl.learn
        n.ctrl.alpha = _rebind_param(n.ctrl.alpha, lc.alpha, params, label=f"node[{n.id}].ctrl.alpha")
        n.ctrl.w = _rebind_param(n.ctrl.w, lc.w, params, label=f"node[{n.id}].ctrl.w")
        n.ctrl.b = _rebind_param(n.ctrl.b, lc.b, params, label=f"node[{n.id}].ctrl.b")

    # Edges
    for e in spec.edges:
        lc = e.ctrl.learn
        e.ctrl.alpha = _rebind_param(e.ctrl.alpha, lc.alpha, params, label=f"edge[{e.src}->{e.dst}].ctrl.alpha")
        e.ctrl.w = _rebind_param(e.ctrl.w, lc.w, params, label=f"edge[{e.src}->{e.dst}].ctrl.w")
        e.ctrl.b = _rebind_param(e.ctrl.b, lc.b, params, label=f"edge[{e.src}->{e.dst}].ctrl.b")

        lt = e.transport.learn
        e.transport.kappa = _rebind_param(e.transport.kappa, lt.kappa, params, label=f"edge[{e.src}->{e.dst}].tr.kappa")
        e.transport.k = _rebind_param(e.transport.k, lt.k, params, label=f"edge[{e.src}->{e.dst}].tr.k")
        e.transport.l0 = _rebind_param(e.transport.l0, lt.l0, params, label=f"edge[{e.src}->{e.dst}].tr.l0")
        e.transport.lambda_s = _rebind_param(e.transport.lambda_s, lt.lambda_s, params, label=f"edge[{e.src}->{e.dst}].tr.lambda_s")
        e.transport.x = _rebind_param(e.transport.x, lt.x, params, label=f"edge[{e.src}->{e.dst}].tr.x")

    # Faces
    for f in spec.faces:
        lf = f.learn
        f.alpha = _rebind_param(f.alpha, lf.alpha, params, label=f"face[{getattr(f, 'id', '?')}].alpha")
        f.c = _rebind_param(f.c, lf.c, params, label=f"face[{getattr(f, 'id', '?')}].c")

    return params
