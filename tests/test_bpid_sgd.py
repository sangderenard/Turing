from src.common.tensors.abstract_nn.optimizer import Adam, BPIDSGD, SGD
from src.common.tensors.abstraction import AbstractTensor as AT, tensor_identity


def test_bpid_sgd_integral_accumulation():
    p = AT.get_tensor([0.0])
    g = AT.get_tensor([1.0])
    opt = BPIDSGD([p], lr=1.0, kp=0.0, ki=1.0, kd=0.0)
    new_p = opt.step([p], [g])[0]
    AT.copyto(p, new_p)
    assert p.tolist() == [-1.0]
    new_p = opt.step([p], [g])[0]
    AT.copyto(p, new_p)
    assert p.tolist() == [-3.0]


def test_sgd_step():
    p = AT.get_tensor([0.0])
    g = AT.get_tensor([1.0])
    opt = SGD([p], lr=0.1)
    new_p = opt.step([p], [g])[0]
    assert new_p.tolist() == [-0.1]


def test_optimizer_state_uses_monotonic_tensor_identity():
    parameter = AT.get_tensor([1.0])
    gradient = AT.get_tensor([0.25])
    optimizer = Adam([parameter])

    optimizer.step([parameter], [gradient])

    token = tensor_identity(parameter)
    assert token in optimizer.m
    assert token in optimizer.v
    assert id(parameter) not in optimizer.m


def test_bpid_state_uses_monotonic_tensor_identity():
    parameter = AT.get_tensor([1.0])
    gradient = AT.get_tensor([0.25])
    optimizer = BPIDSGD([parameter])

    optimizer.step([parameter], [gradient])

    token = tensor_identity(parameter)
    assert token in optimizer.pid._integral
    assert token in optimizer.pid._prev_error
    assert id(parameter) not in optimizer.pid._integral
