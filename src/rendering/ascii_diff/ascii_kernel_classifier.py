"""ASCII classifier using tensor backends for parallel evaluation."""
from __future__ import annotations

from typing import Any
from pathlib import Path
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ...common.tensors import AbstractTensor, Faculty
from ...common.tensors.numpy_backend import NumPyTensorOperations
from ...common.tensors.abstract_nn.core import Model, Linear
from ...common.tensors.abstract_convolution.ndpca3conv import NDPCA3Conv3d
from ...common.tensors.abstract_nn.losses import MSELoss
from ...common.tensors.abstract_nn.optimizer import Adam
from ...common.tensors.abstract_nn.utils import set_seed
try:  # optional torch backend
    from ...common.tensors.torch_backend import PyTorchTensorOperations
except Exception:  # pragma: no cover - torch is optional
    PyTorchTensorOperations = None  # type: ignore[misc]

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

from .charset_ops import obtain_charset


def _backend_numpy(ops: AbstractTensor) -> bool:
    """Return True if ``ops`` uses a NumPy-based backend."""
    return isinstance(ops, NumPyTensorOperations)


def _backend_torch(ops: AbstractTensor) -> bool:
    """Return True if ``ops`` uses a PyTorch backend."""
    return PyTorchTensorOperations is not None and isinstance(ops, PyTorchTensorOperations)

DEFAULT_FONT_PATH = Path(__file__).with_name("consola.ttf")


class AsciiKernelClassifier:
    def __init__(
        self,
        ramp: str,
        font_path: str | Path = DEFAULT_FONT_PATH,
        font_size: int = 16,
        char_size: tuple[int, int] = (16, 16),
        loss_mode: str = "sad",
        *,
        use_nn: bool = True,
        epsilon: float = 1e-4,
        max_epochs: int = 1,
    ) -> None:
        self.ramp = ramp
        self.vocab_size = len(ramp)
        self.font_path = str(font_path)
        self.font_size = font_size
        self.char_size = char_size
        self.loss_mode = loss_mode  # "sad" or "ssim"
        self.charset: list[str] | None = None
        self.charBitmasks: list[AbstractTensor] | None = None
        self.use_nn = use_nn
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.nn_model: Model | None = None
        self.nn_trained = False
        self.nn_metric = None
        self.nn_grid_shape = None
        # Profiling support toggled via the TURING_PROFILE env var
        self.profile = bool(int(os.getenv("TURING_PROFILE", "0")))
        self.profile_stats: dict[str, float] = {"train_ms": 0.0, "classify_ms": 0.0}
        # Store per-call classification durations when profiling
        self.classify_durations: list[float] = []
        self.set_font(font_path=self.font_path, font_size=self.font_size, char_size=self.char_size)

    def set_font(self, font_path=None, font_size=None, char_size=None):
        """Set font parameters and regenerate reference bitmasks."""
        if font_path is not None:
            self.font_path = font_path
        if font_size is not None:
            self.font_size = font_size
        if char_size is not None:
            self.char_size = char_size
        self._prepare_reference_bitmasks()
        self._prepare_nn_data()
        self._train_nn()

    def _prepare_reference_bitmasks(self) -> None:
        # Always use the ramp as the preset_charset so all ramp characters are attempted
        fonts, charset, charBitmasks, _max_w, _max_h = obtain_charset(
            font_files=[self.font_path], font_size=self.font_size, complexity_level=0, preset_charset=self.ramp
        )
        filtered = [(c, bm) for c, bm in zip(charset, charBitmasks) if bm is not None]
        self.charset = [c for c, _ in filtered] # type: ignore
        # self.char_size is (W, H), interpolate expects (H, W) for size
        self.charBitmasks = [AbstractTensor.F.interpolate(AbstractTensor.get_tensor(bm).to_dtype("float") / 255.0, size=(self.char_size[1], self.char_size[0])) for _, bm in filtered] # type: ignore

    def _prepare_nn_data(self):
        bitmasks = self.charBitmasks or []
        n_classes = len(bitmasks)
        h, w = self.char_size[1], self.char_size[0]
        inputs: list[AbstractTensor] = []
        targets: list[np.ndarray] = []
        for idx, bm in enumerate(bitmasks):
            arr = bm.reshape(1, h, w)
            arr = arr.unsqueeze(0)
            inputs.append(arr)
            target_row = np.zeros((n_classes,), dtype=np.float32)
            target_row[idx] = 1.0
            targets.append(target_row)
        x = AbstractTensor.stack(inputs, dim=0)
        y = AbstractTensor.get_tensor(np.stack(targets, axis=0))
        return x, y

    def _train_nn(self) -> None:
        
        start = time.perf_counter() if self.profile else None
        set_seed(0)
        train_x, train_y = self._prepare_nn_data()
        n_classes = train_y.shape[1]
        like = train_x[0]
        h, w = self.char_size[1], self.char_size[0]
        self.nn_grid_shape = (1, h, w)
        metric_np = np.tile(np.eye(3, dtype=np.float32), (1, h, w, 1, 1))
        metric = AbstractTensor.get_tensor(metric_np)
        package = {"metric": {"g": metric, "inv_g": metric}}

        class CharClassifier(Model):
            def __init__(self, like, grid_shape):
                conv = NDPCA3Conv3d(
                    in_channels=1,
                    out_channels=16,
                    like=like,
                    grid_shape=grid_shape,
                    boundary_conditions=("neumann",) * 6,
                    k=3,
                    eig_from="g",
                    pointwise=True,
                )
                flatten = lambda t: t.reshape(t.shape[0], -1)
                fc = Linear(16 * grid_shape[0] * grid_shape[1] * grid_shape[2], n_classes, like=like, bias=True)
                super().__init__(layers=[conv, fc], activations=[None, None])
                self.flatten = flatten
                self.conv = conv
                self.fc = fc
                self.package = None

            def forward(self, x: AbstractTensor):
                x = self.conv.forward(x, package=self.package)
                x = self.flatten(x)
                return self.fc.forward(x)

        model = CharClassifier(like, self.nn_grid_shape)
        model.package = package
        loss_fn = MSELoss()
        optimizer = Adam(model.parameters(), lr=1e-2)

        for epoch in range(1, self.max_epochs + 1):
            logits = model.forward(train_x)
            loss = loss_fn.forward(logits, train_y)
            grad_pred = loss_fn.backward(logits, train_y)
            model.backward(grad_pred)
            params = model.parameters()
            grads = model.grads()
            new_params = optimizer.step(params, grads)
            i = 0
            for layer in model.layers:
                layer_params = layer.parameters()
                for j in range(len(layer_params)):
                    layer_params[j].data[...] = new_params[i].data
                    i += 1
            model.zero_grad()
            # ascii_kernel_classifier.py, inside training loop
            import sys
            print(f"Epoch {epoch}: loss={float(loss.data):.6f}", file=sys.__stderr__, flush=True)
            if float(loss.data) < self.epsilon:
                break

        self.nn_model = model
        self.nn_metric = metric
        self.nn_trained = True
        if self.profile and start is not None:
            self.profile_stats["train_ms"] += (time.perf_counter() - start) * 1000.0

    def _resize_tensor_to_char(self, tensor: AbstractTensor) -> AbstractTensor:
        # self.char_size is (W, H), interpolate expects (H, W) for size
        return AbstractTensor.F.interpolate(tensor, size=(self.char_size[1], self.char_size[0]))

    def sad_loss(self, candidate: AbstractTensor, reference: AbstractTensor) -> float:
        """Sum of absolute differences between ``candidate`` and ``reference``."""
        diff = candidate - reference
        abs_diff = (diff ** 2) ** 0.5
        total = abs_diff.mean() * abs_diff.numel()
        return float(total.item())

    def ssim_loss(self, candidate: AbstractTensor, reference: AbstractTensor) -> float:
        if not SSIM_AVAILABLE:
            raise RuntimeError("SSIM loss requires scikit-image")
        np_backend = AbstractTensor.get_tensor(cls=NumPyTensorOperations)
        arr1 = candidate.to_backend(np_backend)
        arr2 = reference.to_backend(np_backend)
        return 1.0 - ssim(
            arr1.numpy(),
            arr2.numpy(),
            data_range=1.0,
        )

    def classify_batch(self, subunit_batch: np.ndarray) -> dict:
        start = time.perf_counter() if self.profile else None
        batch = AbstractTensor.get_tensor(subunit_batch).to_dtype("float")
        batch_shape = tuple(batch.shape)
        N = batch_shape[0]
        if len(batch_shape) == 4 and batch_shape[3] == 3:
            luminance_tensor = AbstractTensor.get_tensor(batch.mean(dim=3)) / 255.0
        elif len(batch_shape) == 3:
            luminance_tensor = batch / 255.0
        else:
            luminance_tensor = AbstractTensor.get_tensor().zeros((N, self.char_size[1], self.char_size[0]), dtype=batch.float_dtype)

        expected_hw_shape = (self.char_size[1], self.char_size[0])
        luminance_tensor = AbstractTensor.get_tensor(luminance_tensor)
        if luminance_tensor.shape[1:] != expected_hw_shape:
            resized = [AbstractTensor.F.interpolate(luminance_tensor[i], size=expected_hw_shape) for i in range(N)]
            luminance_tensor = AbstractTensor.get_tensor().stack(resized, dim=0)

        if self.use_nn:
            if not self.nn_trained:
                self._train_nn()
            inputs = luminance_tensor.reshape(N, 1, 1, expected_hw_shape[0], expected_hw_shape[1])
            self.nn_model.package = {"metric": {"g": self.nn_metric, "inv_g": self.nn_metric}}
            logits = self.nn_model.forward(inputs)
            idxs = logits.argmax(dim=1)
            chars = [self.charset[int(i)] for i in idxs.tolist()]
            result = {
                "indices": idxs,
                "chars": chars,
                "losses": None,
                "logits": logits,
            }
            if self.profile and start is not None:
                elapsed = (time.perf_counter() - start) * 1000.0
                self.profile_stats["classify_ms"] += elapsed
                self.classify_durations.append(elapsed)
            return result

        refs = AbstractTensor.get_tensor().stack(self.charBitmasks, dim=0)
        expanded_inputs = luminance_tensor[:, None, :, :].repeat_interleave(repeats=refs.shape[0], dim=1)
        expanded_refs = refs[None, :, :, :].repeat_interleave(repeats=N, dim=0)
        diff = expanded_inputs - expanded_refs
        abs_diff = (diff ** 2) ** 0.5
        losses = AbstractTensor.get_tensor(abs_diff.mean(dim=(2, 3)))
        idxs = losses.argmin(dim=1)
        row_indices = AbstractTensor.get_tensor(np.arange(N, dtype=np.int64))
        selected_losses = losses[row_indices, idxs]
        chars = [self.charset[int(i)] for i in idxs.tolist()]
        result = {
            "indices": idxs,
            "chars": chars,
            "losses": selected_losses,
            "logits": None,
        }
        if self.profile and start is not None:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.profile_stats["classify_ms"] += elapsed
            self.classify_durations.append(elapsed)
        return result
if __name__ == "__main__":
    import argparse
    from .charset_ops import obtain_charset
    fontfile = Path(__file__).with_name("consola.ttf")
    parser = argparse.ArgumentParser(description="Test AsciiKernelClassifier on its own font bitmaps at 1:1 scale.")
    parser.add_argument("--font", type=str, default=str(fontfile), help="Path to a TTF/OTF font file")
    parser.add_argument("--size", type=int, default=16, help="Font size (default: 16)")
    parser.add_argument("--ramp", type=str, default=" .:░▒▓█", help="ASCII ramp to use (default: block ramp)")
    args = parser.parse_args()

    # Generate bitmaps for the ramp characters at the requested size
    fonts, charset, bitmasks, max_w, max_h = obtain_charset([args.font], args.size, 0, preset_charset=args.ramp)
    print(f"Font: {args.font}\nSize: {args.size}\nRamp: {args.ramp}\n")
    print("Testing classifier on its own reference bitmaps (should be perfect match):\n")

    # Stack bitmasks into a batch (N, H, W)
    batch = np.stack(bitmasks, axis=0)
    classifier = AsciiKernelClassifier(args.ramp, font_path=args.font, font_size=args.size, char_size=(max_w, max_h))
    result = classifier.classify_batch(batch)
    for i, (expected, predicted, loss) in enumerate(zip(charset, result["chars"], result["losses"])):
        status = "OK" if expected == predicted else "FAIL"
        print(f"[{status}] idx={i} expected='{expected}' classified='{predicted}' loss={loss:.4f}")
