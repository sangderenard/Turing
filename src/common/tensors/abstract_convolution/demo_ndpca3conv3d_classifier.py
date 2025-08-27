"""
Demo: Image Classifier using NDPCA3Conv3d and Linear layers (AbstractTensor framework)

- Replace the image loading section with your own data pipeline.
- This demo uses random weights and random input for illustration.
"""

import numpy as np
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_nn.core import Linear, Model
from src.common.tensors.abstract_convolution.ndpca3conv import NDPCA3Conv3d
from src.common.tensors.abstract_nn.losses import MSELoss
from src.common.tensors.abstract_nn.optimizer import Adam

# --- Config ---
BATCH_SIZE = 1
IN_CHANNELS = 3   # e.g., RGB
IMG_D, IMG_H, IMG_W = 1, 32, 32  # For 2D images, set D=1
NUM_CLASSES = 100

# --- Placeholder: Load your image as a numpy array of shape (B, C, D, H, W) ---
# Replace this with your own image loading logic
np.random.seed(42)
img_np = np.random.rand(BATCH_SIZE, IN_CHANNELS, IMG_D, IMG_H, IMG_W).astype(np.float32)
img = AbstractTensor.get_tensor(img_np)

# --- Metric placeholder (identity for demo) ---
metric_np = np.tile(np.eye(3, dtype=np.float32), (IMG_D, IMG_H, IMG_W, 1, 1))
metric = AbstractTensor.get_tensor(metric_np)
package = {"metric": {"g": metric, "inv_g": metric}}


# --- Model definition using Model wrapper ---
class SimpleClassifierModel(Model):
    def __init__(self, in_channels, num_classes, like, grid_shape):
        conv = NDPCA3Conv3d(
            in_channels=in_channels,
            out_channels=16,
            like=like,
            grid_shape=grid_shape,
            boundary_conditions=("neumann",)*6,
            k=3,
            eig_from="g",
            pointwise=True,
        )
        flatten = lambda x: x.reshape(x.shape[0], -1)
        fc = Linear(16 * IMG_D * IMG_H * IMG_W, num_classes, like=like, bias=True)
        # Model expects a list of layers and activations (None for custom/functional)
        super().__init__(layers=[conv, fc], activations=[None, None])
        self.flatten = flatten
        self.conv = conv
        self.fc = fc
        self.package = None

    def forward(self, x):
        # Pass package to conv, then flatten, then fc
        x = self.conv.forward(x, package=self.package)
        x = self.flatten(x)
        x = self.fc.forward(x)
        return x

# --- Training demo ---
EPOCHS = 400
LEARNING_RATE = 1e-2

# Dummy target: random one-hot for classification (for demo)
target_np = np.zeros((BATCH_SIZE, NUM_CLASSES), dtype=np.float32)
target_class = np.random.randint(0, NUM_CLASSES, size=(BATCH_SIZE,))
for i, c in enumerate(target_class):
    target_np[i, c] = 1.0
target = AbstractTensor.get_tensor(target_np)

model = SimpleClassifierModel(IN_CHANNELS, NUM_CLASSES, like=img, grid_shape=(IMG_D, IMG_H, IMG_W))
model.package = package
loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

print("Training...")

for epoch in range(1, EPOCHS + 1):
    # Forward
    logits = model.forward(img)
    loss = loss_fn.forward(logits, target)
    # Backward
    loss.backward()
    # Optimizer step
    params = model.parameters()
    grads = [p.grad for p in params]
    new_params = optimizer.step(params, grads)
    # Update model params
    i = 0
    for layer in model.layers:
        layer_params = layer.parameters()
        for j in range(len(layer_params)):
            layer_params[j].data[...] = new_params[i].data
            i += 1
    model.zero_grad()
    # Print loss
    if epoch == 1 or epoch == EPOCHS or epoch % 5 == 0:
        print(f"Epoch {epoch:2d}: loss = {float(loss.data):.20f}")

# Final prediction
logits = model.forward(img)
preds = np.argmax(logits.data, axis=1)
print(f"Predicted class indices: {preds}")
print(f"True class indices: {target_class}")

# --- For your own data, replace the img loading section above ---
