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
    register_param_wheels,
    SpectralCfg,
)
from src.common.tensors.autoautograd.fluxspring.fs_harness import (
    LineageLedger,
    RingHarness,
)
from src.common.tensors.autoautograd.fluxspring.fs_dec import pump_tick
from src.common.tensors.autograd import autograd


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
    wheels = register_param_wheels(spec)
    for w in wheels:
        w.rotate(); w.bind_slot()

    harness = RingHarness(default_size=5)
    ledger = LineageLedger()
    psi = AT.tensor([0.0])
    lids = []
    for _ in range(3):
        lid = ledger.ingest()
        lids.append(lid)
        psi, _ = pump_tick(psi, spec, eta=0.0, harness=harness, lineage_id=lid)
        for w in wheels:
            ev = w.rotate()
            w.apply_slot(ev, lambda p, g: p)
            w.bind_slot()

    mat = harness.get_params_for_lineages(lids, ledger)
    assert mat.shape == (3, 1)
    assert float(AT.get_tensor(mat[0, 0])) == pytest.approx(1.0)

    assert ledger.lineages() == tuple(lids)
    ledger.purge_through_lid(lids[1])
    assert ledger.lineages() == (lids[2],)



def test_param_version_ring_respects_stage_depth():
    param = AT.tensor(1.0)
    param.requires_grad_(True)

    node0 = NodeSpec(
        id=0,
        p0=AT.get_tensor([0.0]),
        v0=AT.get_tensor([0.0]),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(),
        scripted_axes=[0],
    )
    node1 = NodeSpec(
        id=1,
        p0=AT.get_tensor([0.0]),
        v0=AT.get_tensor([0.0]),
        mass=AT.tensor(1.0),
        ctrl=NodeCtrl(w=param, learn=LearnCtrl(alpha=False, w=True, b=False)),
        scripted_axes=[0],
    )
    edge = EdgeSpec(
        src=0,
        dst=1,
        transport=EdgeTransport(learn=EdgeTransportLearn(False, False, False, False, False)),
        ctrl=EdgeCtrl(learn=LearnCtrl(False, False, False)),
    )
    spec = FluxSpringSpec(
        version="t",
        D=1,
        nodes=[node0, node1],
        edges=[edge],
        faces=[],
        dec=DECSpec(D0=[[-1.0, 1.0]], D1=[]),
    )
    wheels = register_param_wheels(spec)
    for w in wheels:
        w.rotate(); w.bind_slot()
    harness = RingHarness(default_size=10)
    ledger = LineageLedger()
    psi = AT.tensor([0.0, 0.0])
    lids: list[int] = []
    for i in range(4):
        lid = ledger.ingest()
        lids.append(lid)
        param = spec.nodes[1].ctrl.w
        with autograd.no_grad():
            param.data[0] = float(i + 1)
        psi, _ = pump_tick(psi, spec, eta=0.0, harness=harness, lineage_id=lid)
        for w in wheels:
            ev = w.rotate()
            w.apply_slot(ev, lambda p, g: p)
            w.bind_slot()

    mat = harness.get_params_for_lineages(lids[:2], ledger)
    idx = harness.param_labels.index("node[1].ctrl.w")
    vals = AT.get_tensor(mat)[:, idx]
    assert float(vals[0]) == pytest.approx(1.0)
    assert float(vals[1]) == pytest.approx(2.0)


def test_param_wheels_sized_to_spectral_window():
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
        spectral=SpectralCfg(enabled=True, win_len=7),
    )

    wheels = register_param_wheels(spec)
    assert wheels
    assert all(len(w.versions()) == spec.spectral.win_len for w in wheels)

