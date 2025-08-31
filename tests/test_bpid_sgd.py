from src.common.tensors.abstract_nn.optimizer import BPIDSGD
from src.common.tensors.abstraction import AbstractTensor as AT


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
