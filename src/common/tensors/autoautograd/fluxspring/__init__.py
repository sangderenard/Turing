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
from typing import Callable, Sequence, Iterable
import logging

from ...abstraction import AbstractTensor as AT

logger = logging.getLogger(__name__)


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
    """Toggle requires_grad, attach to tape, label, and collect trainables.

    Keeps the same Python object identity but detaches so it becomes a fresh
    leaf on the current tape. Parameters intended to ride a ParamWheel should
    be rebound once here, after which wheel slices are assigned directly.
    """
    if param is None:
        return None

    t = param.detach()
    t.requires_grad_(learn)

    tape = _tape()
    if label is not None:
        tape.annotate(t, label=label)

    if learn:
        out.append(t)
    elif False and mark_structural_if_frozen:
        tape.mark_structural(t, label=label)

    return t


class ParamWheel:
    """Simple slice-based parameter wheel.

    Stores independent parameter versions in a fixed list. Callers bind the
    appropriate slice for a tick via :meth:`bind_for_tick`, which simply assigns
    the chosen slot tensor to the owning spec attribute through the provided
    ``setter``. No copying, stashing or rotation side-effects are performed.
    """

    def __init__(
        self,
        base: AT,
        setter: Callable[[AT], None],
        *,
        slots: int = 2,
        rings: int = 1,
        label: str | None = None,
        initialization: AT | str = "ones",
    ) -> None:
        self.setter = setter
        self.label = label
        self.slots: int = int(max(1, slots))
        self.rings: int = int(max(1, rings))
        self.idx: int = -1  # next slot to bind with bind_slot()

        # Initialize per-slot tensors
        self._params: list[AT] = []
        base_t = AT.get_tensor(base)
        if isinstance(initialization, str):
            if initialization == "ones":
                init_val = base_t * 0 + 1
            elif initialization == "zeros":
                init_val = base_t * 0
            else:
                init_val = base_t * 0
        else:
            init_val = AT.get_tensor(initialization)
            # If provided scalar/shape doesn't match, try a reshape/broadcast
            try:
                init_val = AT.reshape(init_val, base_t.shape)
            except Exception:
                pass
        for s in range(self.slots):
            p = init_val.clone() if hasattr(init_val, "clone") else AT.get_tensor(init_val)
            p.requires_grad_(True)
            # Annotate each leaf so whiteboard/debuggers can locate them
            lbl = f"{label or 'ParamWheel'}[slot={s}]"
            try:
                _tape().annotate(p, label=lbl)
            except Exception:
                pass
            self._params.append(p)

        # Optional external grad stash used by some callers
        self._grads: list[AT | None] = [None] * self.slots

    # Back-compat property used throughout the repo
    @property
    def params(self) -> list[AT]:
        return self._params

    def versions(self) -> list[AT]:
        return list(self._params)

    def grads(self) -> list[AT | None]:
        return list(self._grads)

    # Lightweight helpers -------------------------------------------------
    def rotate(self) -> int:
        evicted = self.idx
        self.idx = (self.idx + 1) % self.slots if self.slots > 0 else -1
        return evicted

    def bind_slot(self, slot: int | None = None) -> int:
        if self.slots <= 0:
            raise RuntimeError("ParamWheel has no slots")
        if slot is None:
            if self.idx < 0:
                self.idx = 0
            slot = self.idx
        self.setter(self._params[int(slot)])
        return int(slot)

    def bind_for_tick(self, tick: int, row_idx: int = 0) -> set[int]:
        slot = int((tick + row_idx) % self.slots)
        self.setter(self._params[slot])
        return {slot}

    def stash_grads(self, used: Iterable[int]) -> None:
        for s in used:
            p = self._params[int(s)]
            g = getattr(p, "grad", None)
            # Allow whiteboard to write `_grad` as a side channel
            if g is None:
                g = getattr(p, "_grad", None)
            self._grads[int(s)] = AT.get_tensor(g) if g is not None else None

    def apply_slot(self, slot: int, update_fn: Callable[[AT, AT], AT]) -> None:
        p = self._params[int(slot)]
        g = getattr(p, "_grad", None) or self._grads[int(slot)] or getattr(p, "grad", None)
        if g is None:
            return
        new_p = update_fn(p, g)
        # Re-leaf to keep future grads clean
        new_leaf = AT.get_tensor(new_p).detach() if hasattr(new_p, "detach") else AT.get_tensor(new_p)
        try:
            new_leaf.requires_grad_(True)
        except Exception:
            pass
        self._params[int(slot)] = new_leaf
        # If the most recent bind targeted this slot, refresh the binding
        if self.idx >= 0 and (self.idx % self.slots) == int(slot):
            self.setter(self._params[int(slot)])

    # Convenience used by whiteboard demo to expose per-slot scalars/vectors
    def value_for_slots(self, slots: Iterable[int], attr: str | None = None) -> AT:
        vals = []
        for s in slots:
            v = AT.get_tensor(self._params[int(s)])
            v = v.reshape(-1) if hasattr(v, "reshape") else v
            vals.append(v)
        return AT.stack(vals)



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
        slots = spec.spectral.win_len if getattr(spec, "spectral", None) and getattr(spec.spectral, "enabled", False) else max(2, getattr(spec, "stages", 2))
    rings = 1 + (int(extra_delay) + int(slots) - 1) // int(slots)

    wheels: list[ParamWheel] = []
    tmp: list[AT] = []
    # Nodes
    for n in spec.nodes:
        lc = n.ctrl.learn
        for attr in ("alpha", "w", "b"):
            learn = getattr(lc, attr)
            p = _rebind_param(getattr(n.ctrl, attr), learn, tmp, label=f"node[{n.id}].ctrl.{attr}")
            setattr(n.ctrl, attr, p)
            if learn and p is not None:
                wheels.append(
                    ParamWheel(
                        p,
                        lambda t, n=n, attr=attr: setattr(n.ctrl, attr, t),
                        slots=int(slots),
                        rings=int(rings),
                        label=f"node[{n.id}].ctrl.{attr}",
                    )
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
                    ParamWheel(
                        p,
                        lambda t, e=e, attr=attr: setattr(e.ctrl, attr, t),
                        slots=int(slots),
                        rings=int(rings),
                        label=f"edge[{e.src}->{e.dst}].ctrl.{attr}",
                    )
                )

        lt = e.transport.learn
        for attr in ("kappa", "k", "l0", "lambda_s", "x"):
            learn = getattr(lt, attr)
            if not learn:
                continue
            p = _rebind_param(getattr(e.transport, attr), learn, tmp, label=f"edge[{e.src}->{e.dst}].tr.{attr}")
            setattr(e.transport, attr, p)
            if learn and p is not None:
                wheels.append(
                    ParamWheel(
                        p,
                        lambda t, e=e, attr=attr: setattr(e.transport, attr, t),
                        slots=int(slots),
                        rings=int(rings),
                        label=f"edge[{e.src}->{e.dst}].tr.{attr}",
                    )
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
                    ParamWheel(
                        p,
                        lambda t, f=f, attr=attr: setattr(f, attr, t),
                        slots=int(slots),
                        rings=int(rings),
                        label=f"face[{fid}].{attr}",
                    )
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

    # Bind params to the correct slot slices for this tick
    for w in wheels:
        w.bind_for_tick(tick)

    # Single forward pass (any gradient handling is external)
    psi, stats = pump_tick(psi, spec, **pump_kw)
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
