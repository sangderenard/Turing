from src.common.tensors.autograd import autograd, GradTape
from src.common.tensors.abstraction import AbstractTensor


def test_orphan_node_detection():
    autograd.tape = GradTape()

    a = AbstractTensor.get_tensor([1.0, 2.0], tape=autograd.tape).requires_grad_()
    b = a * 2
    c = b + 1  # <- will be orphaned
    d = a * 3
    loss = d.sum()

    loss.backward(retain_graph=True)

    orphans = autograd.tape.orphan_data()
    ids = {info["id"] for info in orphans}
    assert id(c) in ids
