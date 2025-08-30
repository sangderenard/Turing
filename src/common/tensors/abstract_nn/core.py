from __future__ import annotations
from typing import List
from pathlib import Path
import json
from ..abstraction import AbstractTensor
import random, math
from .utils import from_list_like, zeros_like, transpose2d
from .activations import Identity
from ..logger import get_tensors_logger
from ..autograd import autograd

logger = get_tensors_logger()

def _randn_matrix(rows: int, cols: int, like: AbstractTensor, scale: float = 0.02, requires_grad=True, tape=None) -> AbstractTensor:
    data = [[random.gauss(0.0, 1.0) * scale for _ in range(cols)] for _ in range(rows)]
    return from_list_like(data, requires_grad=requires_grad, like=like, tape=tape)

def _to_tuple2(x):
    return (x, x) if isinstance(x, int) else x

def _to_tuple3(x):
    return (x, x, x) if isinstance(x, int) else x


def _ensure_batch_dim(x: AbstractTensor, target_ndim: int = 2) -> tuple[AbstractTensor, bool]:
    """Add a leading batch dimension if ``x`` is missing one.

    Parameters
    ----------
    x:
        Input tensor.
    target_ndim:
        Expected dimensionality including the batch dimension. If ``x`` has
        ``target_ndim - 1`` dimensions, a new leading dimension of size 1 is
        added.

    Returns
    -------
    (tensor, bool)
        The possibly reshaped tensor and a flag indicating whether a batch
        dimension was added.
    """
    added = False
    try:
        if x.ndim == target_ndim - 1:
            x = x.reshape(1, *x.shape())
            added = True
    except Exception:
        # If ``x`` lacks ndim/shape metadata, leave it unchanged.
        pass
    return x, added

class Linear:
    def __init__(self, in_dim: int, out_dim: int, like: AbstractTensor, bias: bool = True, init: str = "auto_relu", _label_prefix=None):
        self.like = like
        if init == "he" or init == "auto_relu":
            scale = math.sqrt(2.0 / float(in_dim))
        elif init == "xavier":
            scale = math.sqrt(1.0 / float(in_dim + out_dim))
        else:
            scale = 0.02
        logger.debug(
            f"Linear layer init: in_dim={in_dim}, out_dim={out_dim}, bias={bias}, init={init}, scale={scale}"
        )
        self.W = _randn_matrix(in_dim, out_dim, like=like, scale=scale)
        autograd.tape.create_tensor_node(self.W)
        self.W._label = f"{_label_prefix+'.' if _label_prefix else ''}Linear.W"
        autograd.tape.annotate(self.W, label=self.W._label)
        # Bias should be a trainable parameter registered on the current tape
        self.b = (
            from_list_like([[0.01] * out_dim], like=like, requires_grad=True, tape=autograd.tape)
            if bias
            else None
        )
        if self.b is not None:
            autograd.tape.create_tensor_node(self.b)
            self.b._label = f"{_label_prefix+'.' if _label_prefix else ''}Linear.b"
            autograd.tape.annotate(self.b, label=self.b._label)
        logger.debug(
            f"Linear layer weights shape: {getattr(self.W, 'shape', None)}; bias shape: {getattr(self.b, 'shape', None) if self.b is not None else None}"
        )

    def parameters(self) -> List[AbstractTensor]:
        return [p for p in (self.W, self.b) if p is not None]

    def zero_grad(self):
        self.W.zero_grad()
        if self.b is not None:
            self.b.zero_grad()

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        # Ensure parameters are registered on the current tape so loss.backward()
        # can discover them via the tape's parameter registry even after tape resets.
        try:
            tape = autograd.tape
            for p in (self.W, self.b) if self.b is not None else (self.W,):
                if p is None:
                    continue
                try:
                    p._tape = tape  # type: ignore[attr-defined]
                except Exception:
                    pass
                tape.create_tensor_node(p)
        except Exception:
            pass
        logger.debug(f"Linear.forward called with input shape: {getattr(x, 'shape', None)}")
        x, added = _ensure_batch_dim(x, target_ndim=2)
        out = x @ self.W
        self._x = x
        autograd.tape.annotate(out, label="Linear.forward.matmul")
        logger.debug(f"Linear matmul output shape: {getattr(out, 'shape', None)}")
        if self.b is not None:
            # Rely on backend broadcasting for (N,D) + (1,D). The add backward
            # rule will unbroadcast gradients back to the bias shape directly.
            out = out + self.b
        autograd.tape.annotate(out, label="Linear.forward.output")
        if added:
            shape = out.shape()
            out = out.reshape(shape[1]) if len(shape) == 2 else out.reshape(*shape[1:])
        return out

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if getattr(self, "_x", None) is None:
            raise RuntimeError("Linear.backward called before forward")
        grad_out, added = _ensure_batch_dim(grad_out, target_ndim=2)
        x = self._x
        xT = x.swapaxes(0, 1)
        self.gW = xT @ grad_out
        self.W._grad = self.gW
        if self.b is not None:
            self.gb = grad_out.sum(dim=0, keepdim=True)
            self.b._grad = self.gb
        WT = self.W.swapaxes(0, 1)
        dx = grad_out @ WT
        self._x = None
        if added:
            shape = dx.shape()
            dx = dx.reshape(shape[1]) if len(shape) == 2 else dx.reshape(*shape[1:])
        return dx


class Flatten:
    def __init__(self, like: AbstractTensor):
        self.like = like
        self._shape = None

    def parameters(self) -> List[AbstractTensor]:
        return []

    def zero_grad(self):
        self._shape = None
        self._added = False

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        x, added = _ensure_batch_dim(x, target_ndim=2)
        self._shape = x.shape()
        self._added = added
        out = x.reshape(self._shape[0], -1)
        if added:
            shape = out.shape()
            out = out.reshape(shape[1]) if len(shape) == 2 else out.reshape(*shape[1:])
        return out

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if self._shape is None:
            raise RuntimeError("Flatten.backward called before forward")
        grad = grad_out.reshape(*self._shape)
        if getattr(self, "_added", False):
            shape = grad.shape()
            grad = grad.reshape(shape[1]) if len(shape) == 2 else grad.reshape(*shape[1:])
        return grad


class RectConv2d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        like: AbstractTensor,
        bias: bool = True,
    ):
        self.like = like
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_tuple2(kernel_size)
        self.stride = _to_tuple2(stride)
        self.padding = _to_tuple2(padding)
        self.dilation = _to_tuple2(dilation)
        kH, kW = self.kernel_size
        scale = math.sqrt(2.0 / (in_channels * kH * kW))
        w_data = [
            [
                [
                    [random.gauss(0.0, 1.0) * scale for _ in range(kW)]
                    for _ in range(kH)
                ]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        # Create parameters directly with requires_grad on the current global tape
        self.W = from_list_like(w_data, like=like, requires_grad=True, tape=autograd.tape)
        self.b = from_list_like([0.0] * out_channels, like=like, requires_grad=True, tape=autograd.tape) if bias else None
        self.gW = zeros_like(self.W)
        self.gb = zeros_like(self.b) if self.b is not None else None
        self._x = None
        self._cols = None
        self._x_shape = None
        self._added = False

    def parameters(self) -> List[AbstractTensor]:
        return [p for p in (self.W, self.b) if p is not None]

    def zero_grad(self):
        self.gW = zeros_like(self.W)
        if self.b is not None:
            self.gb = zeros_like(self.b)
        # Clear autograd gradients on parameters
        self.W.zero_grad()
        if self.b is not None:
            self.b.zero_grad()
        self._x = None
        self._cols = None
        self._x_shape = None
        self._added = False

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        # Re-register parameters on the current tape for this forward pass
        try:
            tape = autograd.tape
            for p in (self.W, self.b) if self.b is not None else (self.W,):
                if p is None:
                    continue
                try:
                    p._tape = tape  # type: ignore[attr-defined]
                except Exception:
                    pass
                tape.create_tensor_node(p)
        except Exception:
            pass
        x, added = _ensure_batch_dim(x, target_ndim=4)
        self._added = added
        self._x = x
        self._x_shape = x.shape
        cols = x.unfold2d(
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self._cols = cols
        Wm = self.W.reshape(self.out_channels, -1)
        out = Wm @ cols
        if self.b is not None:
            out = out + self.b.reshape(1, -1, 1)
        N = self._x_shape[0]
        pH, pW = self.padding
        sH, sW = self.stride
        dH, dW = self.dilation
        kH, kW = self.kernel_size
        H, W = self._x_shape[2], self._x_shape[3]
        Hout = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        Wout = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        out = out.reshape(N, self.out_channels, Hout, Wout)
        if added:
            out = out.reshape(*out.shape()[1:])
        return out

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if self._x is None or self._cols is None or self._x_shape is None:
            raise RuntimeError("RectConv2d.backward called before forward")
        if getattr(self, "_added", False):
            grad_out = grad_out.reshape(1, *grad_out.shape())
        N, _, Hout, Wout = grad_out.shape()
        L = Hout * Wout
        grad_mat = grad_out.reshape(N, self.out_channels, L)
        cols_T = self._cols.transpose(1, 2)
        gW = grad_mat @ cols_T
        self.gW = gW.sum(dim=0).reshape(*self.W.shape)
        if self.b is not None:
            self.gb = grad_mat.sum(dim=(0, 2)).reshape(*self.b.shape)
        # Mirror gradients to parameters
        self.W._grad = self.gW
        if self.b is not None:
            self.b._grad = self.gb
        # Mirror gradients to parameters
        self.W._grad = self.gW
        if self.b is not None:
            self.b._grad = self.gb
        Wm = self.W.reshape(self.out_channels, -1)
        WT = Wm.transpose(0, 1)
        dcols = WT @ grad_mat
        dx = AbstractTensor.fold2d(
            dcols,
            output_size=self._x_shape,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self._x = None
        self._cols = None
        self._x_shape = None
        if getattr(self, "_added", False):
            dx = dx.reshape(*dx.shape()[1:])
        return dx
class RectConv3d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        like: AbstractTensor,
        bias: bool = True,
    ):
        self.like = like
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_tuple3(kernel_size)
        self.stride = _to_tuple3(stride)
        self.padding = _to_tuple3(padding)
        self.dilation = _to_tuple3(dilation)
        kD, kH, kW = self.kernel_size
        scale = math.sqrt(2.0 / (in_channels * kD * kH * kW))
        w_data = [
            [
                [
                    [
                        [random.gauss(0.0, 1.0) * scale for _ in range(kW)]
                        for _ in range(kH)
                    ]
                    for _ in range(kD)
                ]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.W = from_list_like(w_data, like=like)
        self.W.requires_grad_(True)
        self.W._tape = autograd.tape
        autograd.tape.create_tensor_node(self.W)
        self.b = from_list_like([0.0] * out_channels, like=like) if bias else None
        if self.b is not None:
            self.b.requires_grad_(True)
            self.b._tape = autograd.tape
            autograd.tape.create_tensor_node(self.b)
        self.gW = zeros_like(self.W)
        self.gb = zeros_like(self.b) if self.b is not None else None
        self._x = None
        self._cols = None
        self._x_shape = None
        self._added = False

    def parameters(self) -> List[AbstractTensor]:
        return [p for p in (self.W, self.b) if p is not None]

    def zero_grad(self):
        self.gW = zeros_like(self.W)
        if self.b is not None:
            self.gb = zeros_like(self.b)
        # Clear autograd gradients on parameters
        self.W.zero_grad()
        if self.b is not None:
            self.b.zero_grad()
        self._x = None
        self._cols = None
        self._x_shape = None
        self._added = False

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        # Re-register parameters on the current tape for this forward pass
        try:
            tape = autograd.tape
            for p in (self.W, self.b) if self.b is not None else (self.W,):
                if p is None:
                    continue
                try:
                    p._tape = tape  # type: ignore[attr-defined]
                except Exception:
                    pass
                tape.create_tensor_node(p)
        except Exception:
            pass
        x, added = _ensure_batch_dim(x, target_ndim=5)
        self._added = added
        self._x = x
        self._x_shape = x.shape
        # Unfold using AbstractTensor op
        cols = x.unfold3d(
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self._cols = cols
        N = self._x_shape[0]
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        dD, dH, dW = self.dilation
        D, H, W = self._x_shape[2], self._x_shape[3], self._x_shape[4]
        Dpad, Hpad, Wpad = D + 2 * self.padding[0], H + 2 * self.padding[1], W + 2 * self.padding[2]
        eKD = dD * (kD - 1) + 1
        eKH = dH * (kH - 1) + 1
        eKW = dW * (kW - 1) + 1
        Dout = (Dpad - eKD) // sD + 1
        Hout = (Hpad - eKH) // sH + 1
        Wout = (Wpad - eKW) // sW + 1
        Wm = self.W.reshape(self.out_channels, -1)
        out = Wm @ cols
        if self.b is not None:
            out = out + self.b.reshape(1, -1, 1)
        out = out.reshape(N, self.out_channels, Dout, Hout, Wout)
        if added:
            out = out.reshape(*out.shape()[1:])
        return out

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if self._x is None or self._cols is None or self._x_shape is None:
            raise RuntimeError("RectConv3d.backward called before forward")
        if getattr(self, "_added", False):
            grad_out = grad_out.reshape(1, *grad_out.shape())
        N, _, Dout, Hout, Wout = grad_out.shape
        L = Dout * Hout * Wout
        grad_mat = grad_out.reshape(N, self.out_channels, L)
        cols_T = self._cols.transpose(1, 2)
        gW = grad_mat @ cols_T
        self.gW = gW.sum(dim=0).reshape(*self.W.shape)
        if self.b is not None:
            self.gb = grad_mat.sum(dim=(0, 2)).reshape(*self.b.shape)
        Wm = self.W.reshape(self.out_channels, -1)
        WT = Wm.transpose(0, 1)
        dcols = WT @ grad_mat
        dx = AbstractTensor.fold3d(
            dcols,
            output_size=self._x_shape,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self._x = None
        self._cols = None
        self._x_shape = None
        if getattr(self, "_added", False):
            dx = dx.reshape(*dx.shape()[1:])
        return dx


class MaxPool2d:
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = None,
        padding: int | tuple[int, int] = 0,
        like: AbstractTensor,
    ):
        self.like = like
        self.kernel_size = _to_tuple2(kernel_size)
        self.stride = _to_tuple2(stride or kernel_size)
        self.padding = _to_tuple2(padding)
        self._idxs = None
        self._x_shape = None
        self._L = None
        self._kHW = None
        self._added = False

    def parameters(self) -> List[AbstractTensor]:
        return []

    def zero_grad(self):
        self._idxs = None
        self._x_shape = None
        self._L = None
        self._kHW = None
        self._added = False

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        x, added = _ensure_batch_dim(x, target_ndim=4)
        self._added = added
        self._x_shape = x.shape()
        kH, kW = self.kernel_size
        patches = x.unfold2d(
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        N, CK, L = patches.shape()
        C = self._x_shape[1]
        patches = patches.reshape(N, C, kH * kW, L)
        self._kHW = kH * kW
        self._L = L
        values = patches.max(dim=2)
        self._idxs = patches.argmax(dim=2)
        pH, pW = self.padding
        sH, sW = self.stride
        H, W = self._x_shape[2], self._x_shape[3]
        Hout = (H + 2 * pH - kH) // sH + 1
        Wout = (W + 2 * pW - kW) // sW + 1
        out = values.reshape(N, C, Hout, Wout)
        if added:
            out = out.reshape(*out.shape()[1:])
        return out

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if self._idxs is None or self._x_shape is None:
            raise RuntimeError("MaxPool2d.backward called before forward")
        if getattr(self, "_added", False):
            grad_out = grad_out.reshape(1, *grad_out.shape())
        N, C, Hout, Wout = grad_out.shape()
        grad_cols = grad_out.reshape(N, C, 1, self._L)
        ar = self.like.arange(0, self._kHW).reshape(1, 1, self._kHW, 1)
        mask = (ar == self._idxs.reshape(N, C, 1, self._L))
        grad_cols = mask * grad_cols
        grad_cols = grad_cols.reshape(N, C * self._kHW, self._L)
        dx = AbstractTensor.fold2d(
            grad_cols,
            output_size=self._x_shape,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
        )
        if getattr(self, "_added", False):
            dx = dx.reshape(*dx.shape()[1:])
        return dx

class Model:
    def __init__(self, layers: List[Linear], activations) -> None:
        logger.debug(f"Model init with {len(layers)} layers and activations {activations}")
        self.layers = layers
        if isinstance(activations, list):
            assert len(activations) == len(layers), "activations list must match layers"
            self.activations = activations
        else:
            self.activations = [activations] * len(layers)
        self._pre = [None] * len(layers)
        self._post = [None] * len(layers)

    def parameters(self) -> List[AbstractTensor]:
        ps: List[AbstractTensor] = []
        for layer in self.layers:
            ps.extend(layer.parameters())
        return ps

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> list:
        """Return a JSON-serializable representation of model parameters."""
        state: list[list] = []
        for layer in self.layers:
            layer_state = [p.tolist() for p in layer.parameters()]
            state.append(layer_state)
        return state

    def load_state_dict(self, state: list) -> None:
        """Load parameters from ``state`` produced by :meth:`state_dict`."""
        for layer, layer_state in zip(self.layers, state):
            params = layer.parameters()
            for p, data in zip(params, layer_state):
                tensor = from_list_like(data, like=p)
                p.data[...] = tensor.data

    def save_state(self, path: str | Path) -> None:
        """Serialize model parameters to ``path`` as JSON."""
        with open(Path(path), "w", encoding="utf-8") as fh:
            json.dump(self.state_dict(), fh)

    def load_state(self, path: str | Path) -> None:
        """Restore model parameters from a JSON file at ``path``."""
        with open(Path(path), "r", encoding="utf-8") as fh:
            state = json.load(fh)
        self.load_state_dict(state)

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        logger.debug(f"Model.forward called with input shape: {getattr(x, 'shape', None)}")
        x, added = _ensure_batch_dim(x, target_ndim=2)
        for i, layer in enumerate(self.layers):
            logger.debug(f"Model.forward: passing through layer {i} ({layer})")
            z = layer.forward(x)
            self._pre[i] = z
            act = self.activations[i]
            x = act.forward(z) if act is not None else z
            self._post[i] = x
            logger.debug(f"Model.forward: after activation, shape: {getattr(x, 'shape', None)}")
        if added:
            shape = x.shape()
            x = x.reshape(shape[1]) if len(shape) == 2 else x.reshape(*shape[1:])
        return x


class Sequential(Model):
    pass
