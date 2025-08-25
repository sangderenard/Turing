from src.common.tensors import AbstractTensor
from src.common.tensors.autograd_process import AutogradProcess
from src.common.tensors.process_diagram import build_training_diagram, render_training_diagram


def test_process_diagram_build_and_render(tmp_path):
    autograd = AbstractTensor.autograd
    tape = autograd.tape
    tape._nodes.clear()
    tape.graph.clear()

    x = AbstractTensor.tensor([1.0, 2.0, 3.0])
    y = AbstractTensor.tensor([2.0, 4.0, 6.0])
    w = AbstractTensor.tensor([0.0])
    w.requires_grad = True

    def forward_fn():
        pred = x * w
        err = pred - y
        sq = err * err
        loss_val = sq.sum().item()
        return sq, loss_val

    proc = AutogradProcess(tape)
    proc.training_loop(forward_fn, [w], steps=1, lr=0.1)

    diagram = build_training_diagram(proc)
    assert "loss" in diagram
    assert any(n.startswith("cache_") for n in diagram.nodes)

    out_file = tmp_path / "diagram.png"
    render_training_diagram(proc, out_file)
    assert out_file.exists() and out_file.stat().st_size > 0
