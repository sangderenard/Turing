"""Generate a process diagram for an NDPCA3Conv3d classifier.

This demo builds a tiny classifier using the metric-driven
:class:`~src.common.tensors.abstract_convolution.ndpca3conv.NDPCA3Conv3d`
layer and renders the recorded autograd process to a PNG image.  The
network is the same Riemannian convolution used by ``AsciiKernelClassifier``
so the resulting diagram reflects the data flow through that classifier.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

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
EPOCHS = 5
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

    # --- train the model using the manual backward helpers ---
    for _ in range(EPOCHS):
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

    # --- capture autograd process on a fresh tape ---
    autograd.tape = GradTape()
    autograd.capture_all = True
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

    out_file = Path(__file__).with_name("ndpca3conv3d_training.png")
    render_training_diagram(proc, out_file)
    print(f"Process diagram written to {out_file}")


if __name__ == "__main__":
    main()
