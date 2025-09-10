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
# Torch bridge is optional import to keep AT-only usage clean.

from ...abstraction import AbstractTensor as AT


def _rebind_param(param: AT | None, learn: bool, out: list[AT]) -> AT | None:
    """Toggle ``requires_grad`` on ``param`` and collect trainables.

    The previous implementation detached every parameter before enabling
    gradients.  This broke the connection between the ``FluxSpring`` spec
    and the autograd tape, leaving ``p.grad`` as ``None`` for all learnable
    fields.  Simply toggling ``requires_grad`` in place preserves the
    original tensor objects so gradients propagate correctly.
    """

    if param is None:
        return None

    # Ensure the tensor is registered on the current tape with the desired
    # gradient flag.  ``requires_grad_(False)`` is a no-op for most backends
    # but keeps intent explicit.
    param.requires_grad_(learn)
    if learn:
        out.append(param)
    return param


def register_learnable_params(spec: FluxSpringSpec) -> list[AT]:
    """Re-register parameters on the global grad tape based on learn flags.

    ``FluxSpringSpec`` instances may carry tensors that were created before the
    final graph was assembled.  This helper walks the spec, detaches any prior
    graph history and sets ``requires_grad`` on fields whose ``learn`` boolean is
    ``True``.  The returned list contains the tensors that will accumulate
    gradients during optimisation.
    """

    params: list[AT] = []

    for n in spec.nodes:
        lc = n.ctrl.learn
        n.ctrl.alpha = _rebind_param(n.ctrl.alpha, lc.alpha, params)
        n.ctrl.w = _rebind_param(n.ctrl.w, lc.w, params)
        n.ctrl.b = _rebind_param(n.ctrl.b, lc.b, params)

    for e in spec.edges:
        lc = e.ctrl.learn
        e.ctrl.alpha = _rebind_param(e.ctrl.alpha, lc.alpha, params)
        e.ctrl.w = _rebind_param(e.ctrl.w, lc.w, params)
        e.ctrl.b = _rebind_param(e.ctrl.b, lc.b, params)

        lt = e.transport.learn
        e.transport.kappa = _rebind_param(e.transport.kappa, lt.kappa, params)
        e.transport.k = _rebind_param(e.transport.k, lt.k, params)
        e.transport.l0 = _rebind_param(e.transport.l0, lt.l0, params)
        e.transport.lambda_s = _rebind_param(e.transport.lambda_s, lt.lambda_s, params)
        e.transport.x = _rebind_param(e.transport.x, lt.x, params)

    for f in spec.faces:
        lf = f.learn
        f.alpha = _rebind_param(f.alpha, lf.alpha, params)
        f.c = _rebind_param(f.c, lf.c, params)

    return params
