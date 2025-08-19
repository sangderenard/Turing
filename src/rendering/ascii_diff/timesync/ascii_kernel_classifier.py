#!/usr/bin/env python3
"""ASCII classifier using tensor backends for parallel evaluation."""
from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensors import (
    AbstractTensor,
    Faculty,
)
# --- END HEADER ---

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

from fontmapper.FM16.modules.charset_ops import obtain_charset


def _backend_numpy(ops: AbstractTensor) -> bool:
    """Return True if ``ops`` uses a NumPy-based backend."""
    return isinstance(ops, NumPyTensorOperations)


def _backend_torch(ops: AbstractTensor) -> bool:
    """Return True if ``ops`` uses a PyTorch backend."""
    return isinstance(ops, PyTorchTensorOperations)

class AsciiKernelClassifier:
    def __init__(
        self,
        ramp: str,
        font_path: str = "fontmapper/FM16/consola.ttf",
        font_size: int = 16,
        char_size: tuple[int, int] = (16, 16),
        loss_mode: str = "sad",
    ) -> None:
        self.ramp = ramp
        self.vocab_size = len(ramp)
        self.font_path = font_path
        self.font_size = font_size
        self.char_size = char_size
        self.loss_mode = loss_mode  # "sad" or "ssim"
        self.charset: list[str] | None = None
        self.charBitmasks: list[AbstractTensor] | None = None
        self._prepare_reference_bitmasks()

    def set_font(self, font_path=None, font_size=None, char_size=None):
        """Set font parameters and regenerate reference bitmasks."""
        if font_path is not None:
            self.font_path = font_path
        if font_size is not None:
            self.font_size = font_size
        if char_size is not None:
            self.char_size = char_size
        self._prepare_reference_bitmasks()

    def _prepare_reference_bitmasks(self) -> None:
        fonts, charset, charBitmasks, _max_w, _max_h = obtain_charset(
            font_files=[self.font_path], font_size=self.font_size, complexity_level=0
        )
        filtered = [(c, bm) for c, bm in zip(charset, charBitmasks) if c in self.ramp and bm is not None]
        self.charset = [c for c, _ in filtered] # type: ignore
        # self.char_size is (W, H), interpolate expects (H, W) for size
        self.charBitmasks = [AbstractTensor.F.interpolate(AbstractTensor.get_tensor(bm), size=(self.char_size[1], self.char_size[0])) for _, bm in filtered] # type: ignore

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
        np_backend = AbstractTensor.get_tensor(faculty=Faculty.NUMPY)
        arr1 = candidate.to_backend(np_backend)
        arr2 = reference.to_backend(np_backend)
        return 1.0 - ssim(
            arr1.numpy(),
            arr2.numpy(),
            data_range=1.0,
        )

    def classify_batch(self, subunit_batch: np.ndarray) -> dict:
        batch = AbstractTensor.get_tensor(subunit_batch).to_dtype("float")
        batch_shape = batch.shape()
        N = batch_shape[0]
        if len(batch_shape) == 4 and batch_shape[3] == 3:
            luminance_tensor = batch.mean(dim=3) / 255.0
        elif len(batch_shape) == 3:
            luminance_tensor = batch / 255.0
        else:
            # self.char_size is (W,H), tensor shape should be (N,H,W)
            luminance_tensor = AbstractTensor.get_tensor().zeros((N, self.char_size[1], self.char_size[0]), dtype=batch.float_dtype)
        
        # Compare tensor's (H,W) with classifier's (target_H, target_W)
        expected_hw_shape = (self.char_size[1], self.char_size[0])
        if luminance_tensor.shape()[1:] != expected_hw_shape:
            resized = [AbstractTensor.F.interpolate(luminance_tensor[i], size=expected_hw_shape) for i in range(N)]
            luminance_tensor = AbstractTensor.get_tensor().stack(resized, dim=0)
        refs = AbstractTensor.get_tensor().stack(self.charBitmasks, dim=0)
        expanded_inputs = luminance_tensor[:, None, :, :].repeat_interleave(repeats=refs.shape[0], dim=1)
        expanded_refs = refs[None, :, :, :].repeat_interleave(repeats=N, dim=0)
        diff = expanded_inputs - expanded_refs
        abs_diff = (diff ** 2) ** 0.5
        losses = abs_diff.mean(dim=(2, 3))
        idxs = losses.argmin(dim=1)
        row_indices = AbstractTensor.get_tensor().arange(N, dtype=losses.long_dtype)
        selected_losses = losses[row_indices, idxs]
        chars = [self.charset[int(i)] for i in idxs.tolist()]
        return {
            "indices": idxs,
            "chars": chars,
            "losses": selected_losses,
            "logits": None,
        }
