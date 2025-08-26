"""Generate a process diagram for an NDPCA3Conv3d classifier.

This demo builds a tiny classifier using the metric-driven
:class:`~src.common.tensors.abstract_convolution.ndpca3conv.NDPCA3Conv3d`
layer and renders the recorded autograd process to an image. The
network is the same Riemannian convolution used by ``AsciiKernelClassifier``
so the resulting diagram reflects the data flow through that classifier.
The output format can be selected at runtime (PNG, SVG, or PDF).
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np


def _compare_tensors(name: str, expected: AbstractTensor, actual: AbstractTensor) -> None:
    """Print diagnostic information when tensors diverge."""

    if not np.allclose(expected.data, actual.data, atol=1e-6):
        diff = np.abs(expected.data - actual.data)
        print(f"{name} mismatch: max={diff.max()} mean={diff.mean()}")

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autograd import autograd, GradTape
from src.common.tensors.autograd_process import AutogradProcess
from src.common.tensors.process_diagram import render_training_diagram
from src.common.tensors.abstract_nn.core import Linear, Model
from src.common.tensors.abstract_convolution.ndpca3conv import NDPCA3Conv3d
from src.common.tensors.abstract_nn.losses import MSELoss
from src.common.tensors.abstract_nn.optimizer import Adam

# --- configuration kept intentionally small for a compact diagram ---
BATCH_SIZE = 1
IN_CHANNELS = 3
IMG_D, IMG_H, IMG_W = 1, 8, 8
NUM_CLASSES = 4
EPOCHS = 1
LEARNING_RATE = 1e-2


class DemoModel(Model):
    """Simple classifier combining NDPCA3Conv3d and a linear head."""

    def __init__(self, like: AbstractTensor, grid_shape: tuple[int, int, int]):
        conv = NDPCA3Conv3d(
            in_channels=IN_CHANNELS,
            out_channels=4,
            like=like,
            grid_shape=grid_shape,
            boundary_conditions=("neumann",) * 6,
            k=3,
            eig_from="g",
            pointwise=True,
        )
        flatten = lambda x: x.reshape(x.shape[0], -1)
        fc = Linear(4 * IMG_D * IMG_H * IMG_W, NUM_CLASSES, like=like, bias=True)
        super().__init__(layers=[conv, fc], activations=[None, None])
        self.flatten = flatten
        self.conv = conv
        self.fc = fc
        self.package = None

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        x = self.conv.forward(x, package=self.package)
        x = self.flatten(x)
        x = self.fc.forward(x)
        return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Render training diagram for NDPCA3Conv3d demo")
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png", help="Output image format")
    parser.add_argument("--dpi", type=float, default=None, help="Raster resolution for PNG output")
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.5,
        help="Distance between nodes to give edges more room",
    )

    args = parser.parse_args()

    np.random.seed(0)
    img_np = np.random.rand(BATCH_SIZE, IN_CHANNELS, IMG_D, IMG_H, IMG_W).astype(np.float32)
    img = AbstractTensor.get_tensor(img_np)

    metric_np = np.tile(np.eye(3, dtype=np.float32), (IMG_D, IMG_H, IMG_W, 1, 1))
    metric = AbstractTensor.get_tensor(metric_np)
    package = {"metric": {"g": metric, "inv_g": metric}}

    model = DemoModel(like=img, grid_shape=(IMG_D, IMG_H, IMG_W))
    model.package = package
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    target_np = np.zeros((BATCH_SIZE, NUM_CLASSES), dtype=np.float32)
    target_np[0, 0] = 1.0
    target = AbstractTensor.get_tensor(target_np)

    # Preserve initial parameters for later validation
    initial_params = [p.clone() for p in model.parameters()]

    # --- single training epoch ---
    rng_before = np.random.get_state()
    logits = model.forward(img)
    loss = loss_fn.forward(logits, target)
    grad_pred = loss_fn.backward(logits, target)
    model.backward(grad_pred)
    params = model.parameters()
    grads = model.grads()
    with autograd.no_grad():
        new_params = optimizer.step(params, grads)
        i = 0
        for layer in model.layers:
            layer_params = layer.parameters()
            for j in range(len(layer_params)):
                AbstractTensor.copyto(layer_params[j], new_params[i])
                i += 1
    model.zero_grad()
    rng_after = np.random.get_state()
    if repr(rng_before) != repr(rng_after):
        print("Warning: RNG state changed during training step")

    diagnostics = {
        "logits": logits.clone(),
        "loss": loss.clone(),
        "grads": [g.clone() for g in grads],
        "updated_params": [p.clone() for p in model.parameters()],
        "rng_state": rng_after,
    }

    # --- capture autograd process on a fresh tape ---
    img_id, target_id = id(img), id(target)
    autograd.tape = GradTape()
    autograd.capture_all = True
    # Register all root tensors so their identities are preserved when replaying
    for tensor in [img, target, metric, *model.parameters()]:
        autograd.tape.create_tensor_node(tensor)
    logits = model.forward(img)
    loss = loss_fn.forward(logits, target)
    autograd.capture_all = False
    autograd.tape.mark_loss(loss)

    # Add metadata for better visualization
    for node_id, node_data in autograd.tape.graph.nodes(data=True):
        node_data['metadata'] = {
            'operation': node_data.get('op', 'unknown'),
            'description': f"Node {node_id} performing {node_data.get('op', 'unknown')}"
        }

    proc = AutogradProcess(autograd.tape)
    proc.build(loss)
    loss_id = proc.tape._loss_id
    autograd.tape = GradTape()

    out_file = Path(__file__).with_name(f"ndpca3conv3d_training.{args.format}")
    render_training_diagram(
        proc, out_file, format=args.format, dpi=args.dpi, node_spacing=args.spacing
    )

    print(f"Process diagram written to {out_file}")

    # --- replay and validate ---

    def replay_forward(proc: AutogradProcess, feed: dict[int, AbstractTensor]):
        """Replay a forward pass using the schedule from ``proc``.

        ``GradTape`` only records ``GradNode`` entries for tensors that are the
        result of an operation.  Leaf constants appear in the exported forward
        graph (and therefore in ``forward_schedule``) but have no corresponding
        entry in ``tape._nodes``.  The original implementation attempted to look
        up every scheduled ID in ``_nodes`` which raised a ``KeyError`` when such
        constants were encountered.  During replay we lazily inject these
        constant values from the recorded operation context instead.
        """

        values = dict(feed)
        for tid in proc.forward_schedule:
            # Pre-supplied inputs and parameters already have values.
            if tid in values:
                continue

            node = proc.tape._nodes.get(tid)
            if node is None:
                # Leaf tensor – pull original reference from the tape
                tensor = proc.tape._tensor_refs.get(tid)
                if tensor is None:
                    raise KeyError(tid)
                values[tid] = tensor
                continue
            args = [values[parent] for parent, _ in node.parents]
            result = getattr(args[0], node.op)(*args[1:], **node.ctx.get('params', {}))

            values[tid] = result

        return values

    def replay_training_step(
        img_tensor: AbstractTensor,
        target_tensor: AbstractTensor,
        params: list[AbstractTensor],
        reference: dict | None = None,
    ):
        rng_before = np.random.get_state()
        # Seed feed with any tensors recorded on the tape (e.g. metrics)
        feed = dict(proc.tape._tensor_refs)
        feed.update({img_id: img_tensor, target_id: target_tensor})
        for tid, idx in proc.tape._parameters.items():
            feed[tid] = params[idx]
        values = replay_forward(proc, feed)
        preds = list(proc.forward_graph.predecessors(loss_id))
        if preds:
            logits_tid = preds[0]
            logits_val = values[logits_tid]
        else:
            logits_val = None
            if reference:
                print("Loss node has no predecessor; cannot compare logits")
        loss_val = values[loss_id]
        if reference and logits_val is not None:
            _compare_tensors("logits", reference["logits"], logits_val)
            _compare_tensors("loss", reference["loss"], loss_val)
        grads = AbstractTensor.autograd.grad(
            loss_val,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        if reference:
            for idx, (g_ref, g_val) in enumerate(zip(reference["grads"], grads)):
                if g_val is None:
                    continue
                _compare_tensors(f"grad_{idx}", g_ref, g_val)
        with AbstractTensor.autograd.no_grad():
            optimizer = Adam(params, lr=LEARNING_RATE)
            grads_for_opt = [
                g if g is not None else AbstractTensor.zeros_like(p)
                for p, g in zip(params, grads)
            ]
            new_params = optimizer.step(params, grads_for_opt)
            for idx, (p, new_p) in enumerate(zip(params, new_params)):
                AbstractTensor.copyto(p, new_p)
                if reference:
                    _compare_tensors(
                        f"param_{idx}", reference["updated_params"][idx], p
                    )
        rng_after = np.random.get_state()
        if reference and repr(reference.get("rng_state")) != repr(rng_after):
            print("Replay consumed RNG differently")
        return loss_val

    # Validate that replayed training reaches the same weights
    model_replay = DemoModel(like=img, grid_shape=(IMG_D, IMG_H, IMG_W))
    model_replay.package = package
    for p_new, p_old in zip(model_replay.parameters(), initial_params):
        AbstractTensor.copyto(p_new, p_old)
    replay_training_step(img, target, model_replay.parameters(), diagnostics)

    expected = model.parameters()
    obtained = model_replay.parameters()
    assert all(np.allclose(e.data, o.data, atol=1e-6) for e, o in zip(expected, obtained)), "Replay weight update mismatch"

    # Validate new input path
    np.random.seed(1)
    new_img_np = np.random.rand(BATCH_SIZE, IN_CHANNELS, IMG_D, IMG_H, IMG_W).astype(np.float32)
    new_img = AbstractTensor.get_tensor(new_img_np)
    logits_model = model.forward(new_img)
    loss_model = loss_fn.forward(logits_model, target)
    feed = {img_id: new_img, target_id: target}
    for tid, idx in proc.tape._parameters.items():
        feed[tid] = model.parameters()[idx]
    vals = replay_forward(proc, feed)
    preds = list(proc.forward_graph.predecessors(loss_id))
    if preds:
        logits_tid = preds[0]
        logits_replay = vals[logits_tid]
        assert np.allclose(logits_model.data, logits_replay.data, atol=1e-6)
    else:
        print("Loss node has no predecessor; skipping logits comparison")
    loss_replay = vals[loss_id]
    assert np.allclose(loss_model.data, loss_replay.data, atol=1e-6)

    print("Replay validation succeeded")


if __name__ == "__main__":
    main()
