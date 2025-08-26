import networkx as nx

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
    # Forward graph nodes should carry layered execution metadata so that the
    # diagram arranges operations sequentially.
    f_layers = {data.get("layer") for _, data in proc.forward_graph.nodes(data=True)}
    assert len(f_layers) > 1
    d_layers = {data.get("layer") for _, data in diagram.nodes(data=True)}
    assert min(d_layers) == 0 and max(d_layers) > 0

    out_file = tmp_path / "diagram.png"
    rendered = render_training_diagram(proc, out_file)
    assert isinstance(rendered, nx.DiGraph)
    assert "loss" in rendered
    assert out_file.exists()


def test_capture_all_records_without_requires_grad():
    autograd = AbstractTensor.autograd
    tape = autograd.tape
    tape._nodes.clear()
    tape.graph.clear()

    autograd.capture_all = True
    x = AbstractTensor.tensor([1.0, 2.0])
    y = AbstractTensor.tensor([3.0, 4.0])
    z = x + y
    loss = z.sum()
    autograd.capture_all = False

    tape.mark_loss(loss)
    proc = AutogradProcess(tape)
    proc.build(loss)

    assert proc.forward_graph.number_of_edges() > 0

    tape._nodes.clear()
    tape.graph.clear()
