from __future__ import annotations
from typing import List
from pathlib import Path
import json
from ..abstraction import AbstractTensor
import random, math
from .utils import from_list_like, zeros_like, transpose2d
from .activations import Identity
from ..logger import get_tensors_logger

logger = get_tensors_logger()

def _randn_matrix(rows: int, cols: int, like: AbstractTensor, scale: float = 0.02) -> AbstractTensor:
    data = [[random.gauss(0.0, 1.0) * scale for _ in range(cols)] for _ in range(rows)]
    return from_list_like(data, like=like)

def _to_tuple2(x):
    return (x, x) if isinstance(x, int) else x

def _to_tuple3(x):
    return (x, x, x) if isinstance(x, int) else x

class Linear:
    def __init__(self, in_dim: int, out_dim: int, like: AbstractTensor, bias: bool = True, init: str = "auto_relu"):
        self.like = like
        if init == "he" or init == "auto_relu":
            scale = math.sqrt(2.0 / float(in_dim))
        elif init == "xavier":
            # Use a tanh-friendly Xavier gain by default
            scale = math.sqrt(1.0 / float(in_dim + out_dim))
        else:
            scale = 0.02
        logger.debug(
            f"Linear layer init: in_dim={in_dim}, out_dim={out_dim}, bias={bias}, init={init}, scale={scale}"
        )
        self.W = _randn_matrix(in_dim, out_dim, like=like, scale=scale)
        self.W.requires_grad_(True)
        # Seed a small positive bias to avoid symmetric stall at init
        self.b = from_list_like([[0.01] * out_dim], like=like) if bias else None
        if self.b is not None:
            self.b.requires_grad_(True)
        logger.debug(
            f"Linear layer weights shape: {getattr(self.W, 'shape', None)}; bias shape: {getattr(self.b, 'shape', None) if self.b is not None else None}"
        )
        self.gW = zeros_like(self.W)
        self.gb = zeros_like(self.b) if self.b is not None else None
        self._x = None

    def parameters(self) -> List[AbstractTensor]:
        return [p for p in (self.W, self.b) if p is not None]

    def zero_grad(self):
        self.gW = zeros_like(self.W)
        if self.b is not None:
            self.gb = zeros_like(self.b)
        self._x = None

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        logger.debug(f"Linear.forward called with input shape: {getattr(x, 'shape', None)}")
        self._x = x
        out = x @ self.W
        logger.debug(f"Linear matmul output shape: {getattr(out, 'shape', None)}")
        if self.b is not None:
            b = self.b.broadcast_rows(out.shape[0], label="Linear.forward(bias)")
            logger.debug(f"Linear bias broadcasted shape: {getattr(b, 'shape', None)}")
            out = out + b
        return out


    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if self._x is None:
            raise RuntimeError("Linear.backward called before forward; self._x is None")
        xT = self._x.transpose(0, 1)
        self.gW = xT @ grad_out
        if self.b is not None:
            gb_raw = grad_out.sum(dim=0, keepdim=True)
            self.gb = grad_out.ensure_tensor(gb_raw)
        WT = self.W.transpose(0, 1)
        grad_in = grad_out @ WT
        self._x = None
        return grad_in


class Flatten:
    def __init__(self, like: AbstractTensor):
        self.like = like
        self._shape = None

    def parameters(self) -> List[AbstractTensor]:
        return []

    def zero_grad(self):
        self._shape = None

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        self._shape = x.shape()
        return x.reshape(self._shape[0], -1)

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if self._shape is None:
            raise RuntimeError("Flatten.backward called before forward")
        return grad_out.reshape(*self._shape)


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
        self.W = from_list_like(w_data, like=like)
        self.b = from_list_like([0.0] * out_channels, like=like) if bias else None
        self.gW = zeros_like(self.W)
        self.gb = zeros_like(self.b) if self.b is not None else None
        self._x = None
        self._cols = None
        self._x_shape = None
        self._cols = None
        self._x_shape = None

    def parameters(self) -> List[AbstractTensor]:
        return [p for p in (self.W, self.b) if p is not None]

    def zero_grad(self):
        self.gW = zeros_like(self.W)
        if self.b is not None:
            self.gb = zeros_like(self.b)
        self._x = None
        self._cols = None
        self._x_shape = None

    def forward(self, x: AbstractTensor) -> AbstractTensor:
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
        return out.reshape(N, self.out_channels, Hout, Wout)

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if self._x is None or self._cols is None or self._x_shape is None:
            raise RuntimeError("RectConv2d.backward called before forward")
        N, _, Hout, Wout = grad_out.shape()
        L = Hout * Wout
        grad_mat = grad_out.reshape(N, self.out_channels, L)
        cols_T = self._cols.transpose(1, 2)
        gW = grad_mat @ cols_T
        self.gW = gW.sum(dim=0).reshape(*self.W.shape)
        if self.b is not None:
            self.gb = grad_mat.sum(dim=(0, 2)).reshape(*self.b.shape)
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
        self.b = from_list_like([0.0] * out_channels, like=like) if bias else None
        self.gW = zeros_like(self.W)
        self.gb = zeros_like(self.b) if self.b is not None else None
        self._x = None

    def parameters(self) -> List[AbstractTensor]:
        return [p for p in (self.W, self.b) if p is not None]

    def zero_grad(self):
        self.gW = zeros_like(self.W)
        if self.b is not None:
            self.gb = zeros_like(self.b)
        self._x = None
        self._cols = None
        self._x_shape = None

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view

        self._x = x
        self._x_shape = x.shape
        arr = x.data
        pD, pH, pW = self.padding
        arr_p = np.pad(arr, ((0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)))
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        dD, dH, dW = self.dilation
        eKD = dD * (kD - 1) + 1
        eKH = dH * (kH - 1) + 1
        eKW = dW * (kW - 1) + 1
        win = sliding_window_view(arr_p, (eKD, eKH, eKW), axis=(2, 3, 4))
        win = win[:, :, ::sD, ::sH, ::sW, ::dD, ::dH, ::dW]
        N = self._x_shape[0]
        Dout, Hout, Wout = win.shape[2], win.shape[3], win.shape[4]
        cols = np.transpose(win, (0, 1, 5, 6, 7, 2, 3, 4)).reshape(
            N, self.in_channels * kD * kH * kW, Dout * Hout * Wout
        )
        cols_t = x.ensure_tensor(cols)
        self._cols = cols_t
        Wm = self.W.reshape(self.out_channels, -1)
        out = Wm @ cols_t
        if self.b is not None:
            out = out + self.b.reshape(1, -1, 1)
        return out.reshape(N, self.out_channels, Dout, Hout, Wout)

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        import numpy as np
        if self._x is None or self._cols is None or self._x_shape is None:
            raise RuntimeError("RectConv3d.backward called before forward")
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
        dcols_np = dcols.data
        C = self.in_channels
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding
        dD, dH, dW = self.dilation
        D, H, W = self._x_shape[2], self._x_shape[3], self._x_shape[4]
        dcols_np = dcols_np.reshape(N, C, kD, kH, kW, Dout, Hout, Wout)
        grad_win = np.transpose(dcols_np, (0, 1, 5, 6, 7, 2, 3, 4))
        dx_p = np.zeros((N, C, D + 2 * pD, H + 2 * pH, W + 2 * pW), dtype=dcols_np.dtype)
        for kd in range(kD):
            d_slice = slice(kd * dD, kd * dD + sD * Dout, sD)
            for kh in range(kH):
                h_slice = slice(kh * dH, kh * dH + sH * Hout, sH)
                for kw in range(kW):
                    w_slice = slice(kw * dW, kw * dW + sW * Wout, sW)
                    dx_p[:, :, d_slice, h_slice, w_slice] += grad_win[
                        :, :, :, :, :, kd, kh, kw
                    ]
        dx = dx_p[:, :, pD : pD + D, pH : pH + H, pW : pW + W]
        result = self._x.ensure_tensor(dx)
        self._x = None
        self._cols = None
        self._x_shape = None
        return result


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

    def parameters(self) -> List[AbstractTensor]:
        return []

    def zero_grad(self):
        self._idxs = None
        self._x_shape = None
        self._L = None
        self._kHW = None

    def forward(self, x: AbstractTensor) -> AbstractTensor:
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
        return values.reshape(N, C, Hout, Wout)

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if self._idxs is None or self._x_shape is None:
            raise RuntimeError("MaxPool2d.backward called before forward")
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

    def grads(self) -> List[AbstractTensor]:
        gs: List[AbstractTensor] = []
        for l in self.layers:
            layer_grads = getattr(l, "grads", None)
            if callable(layer_grads):
                gs.extend(layer_grads())
            else:
                gs.append(l.gW)
                if l.b is not None:
                    gs.append(l.gb)
        assert len(gs) == len(self.parameters()), "grads count must match parameters count"
        return gs

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
        for i, layer in enumerate(self.layers):
            logger.debug(f"Model.forward: passing through layer {i} ({layer})")
            z = layer.forward(x)
            self._pre[i] = z
            act = self.activations[i]
            x = act.forward(z) if act is not None else z
            self._post[i] = x
            logger.debug(f"Model.forward: after activation, shape: {getattr(x, 'shape', None)}")
        return x

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        g = grad_out
        for i in reversed(range(len(self.layers))):
            act = self.activations[i]
            if act is not None:
                g = act.backward(self._pre[i], g)
            g = self.layers[i].backward(g)
        return g

class Sequential(Model):
    pass
