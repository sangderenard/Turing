from .fs_types import (
    LearnCtrl, NodeCtrl, EdgeTransportLearn, EdgeTransport, EdgeCtrl,
    NodeSpec, EdgeSpec, FaceLearn, FaceSpec,
    DirichletCfg, DECSpec, RegCfg, FluxSpringSpec
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
from typing import Callable
# Torch bridge is optional import to keep AT-only usage clean.

from ...abstraction import AbstractTensor as AT

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
    """Manage multiple in-flight versions of a single parameter."""

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
        self.params: list[AT] = [base]
        for _ in range(slots - 1):
            t = base.detach().clone()
            t.requires_grad_(True)
            self.params.append(t)
        tape = _tape()
        if label is not None:
            for p in self.params:
                tape.annotate(p, label=label)
        self.idx = -1  # first rotate brings index to 0

    def rotate(self) -> int:
        evicted = self.idx
        self.idx = (self.idx + 1) % len(self.params)
        return evicted

    def bind_slot(self) -> AT:
        p = self.params[self.idx]
        self.setter(p)
        return p

    def versions(self) -> list[AT]:
        return self.params

    def grads(self) -> list[AT | None]:
        return [getattr(p, "grad", None) for p in self.params]

    def apply_slot(self, idx: int, update_fn: Callable[[AT, AT], AT]) -> None:
        if idx < 0 or idx >= len(self.params):
            return
        p = self.params[idx]
        g = getattr(p, "grad", None)
        if g is None:
            return
        new_p = update_fn(p, g)
        p.data = AT.get_tensor(new_p)
        p.grad = None


def register_param_wheels(spec: FluxSpringSpec, *, slots: int = 2) -> list[ParamWheel]:
    """Instantiate :class:`ParamWheel` objects for all learnable parameters."""

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

    return wheels


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
