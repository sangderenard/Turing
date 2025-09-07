from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autoautograd.residual_store import ResidualStore, Space


def test_residual_store_put_axis():
    rs = ResidualStore()
    rs.put(1, AbstractTensor.get_tensor(1.0), space=Space.G, width=1, axis=0)
    rs.put(1, AbstractTensor.get_tensor(2.0), space=Space.G, width=1, axis=1)

    bucket = rs.get_bucket(Space.G)
    assert 1 in bucket
    items = bucket[1]
    assert len(items) == 2
    axes = sorted(items.keys())
    vals = sorted(
        float(getattr(it.value, "item", lambda: it.value)()) for it in items.values()
    )
    assert axes == [0, 1]
    assert vals == [1.0, 2.0]

