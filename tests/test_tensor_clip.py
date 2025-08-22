from src.common.tensors.abstraction import AbstractTensor


def test_clip_alias_min_max():
    t = AbstractTensor.tensor([-1.0, 0.0, 1.0])
    clipped = t.clip(min=-0.5, max=0.5)
    assert clipped.tolist() == [-0.5, 0.0, 0.5]


def test_clip_alias_numpy_names():
    t = AbstractTensor.tensor([-1.0, 0.0, 1.0])
    clipped = t.clip(a_min=-0.5, a_max=0.5)
    assert clipped.tolist() == [-0.5, 0.0, 0.5]
