import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring import (
    FluxSpringSpec,
    NodeSpec,
    NodeCtrl,
    LearnCtrl,
    EdgeSpec,
    EdgeCtrl,
    EdgeTransport,
    EdgeTransportLearn,
    DECSpec,
    register_learnable_params,
)
from src.common.tensors.autoautograd.fluxspring.fs_harness import (
    LineageLedger,
    RingHarness,
)
from src.common.tensors.autoautograd.fluxspring.fs_dec import pump_tick


def test_param_version_ring_snapshots():
    param = AT.tensor(1.0)
    param.requires_grad_(True)

    node = NodeSpec(
        id=0,
        p0=AT.get_tensor([0.0]),
        v0=AT.get_tensor([0.0]),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(w=param, learn=LearnCtrl(alpha=False, w=True, b=False)),
        scripted_axes=[0],
    )
    edge = EdgeSpec(
        src=0,
        dst=0,
        transport=EdgeTransport(learn=EdgeTransportLearn(False, False, False, False, False)),
        ctrl=EdgeCtrl(learn=LearnCtrl(False, False, False)),
    )
    spec = FluxSpringSpec(
        version="t",
        D=1,
        nodes=[node],
        edges=[edge],
        faces=[],
        dec=DECSpec(D0=[[0.0]], D1=[]),
    )
    register_learnable_params(spec)

    harness = RingHarness(default_size=5)
    ledger = LineageLedger()
    psi = AT.tensor([0.0])
    lids = []
    for _ in range(3):
        lid = ledger.ingest()
        lids.append(lid)
        psi, _ = pump_tick(psi, spec, eta=0.0, harness=harness, lineage_id=lid)

    mat = harness.get_params_for_lineages(lids, ledger)
    assert mat.shape == (3, 1)
    assert float(AT.get_tensor(mat[0, 0])) == pytest.approx(1.0)
