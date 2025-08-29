from __future__ import annotations
from ..abstraction import AbstractTensor
from ..autograd import autograd
from ..backward import BACKWARD_REGISTRY
import sys

# -------- helpers (pure-tensor, numerically stable) --------

def _sigmoid_stable(x: AbstractTensor) -> AbstractTensor:
    # piecewise-stable sigmoid:
    # if x >= 0: 1/(1+exp(-x)) ; else: exp(x)/(1+exp(x))
    pos = x.greater_equal(0.0)
    neg = x.less(0.0)
    z_pos = (-x).exp()
    y_pos = 1.0 / (1.0 + z_pos)
    z_neg = x.exp()
    y_neg = z_neg / (1.0 + z_neg)
    return pos * y_pos + neg * y_neg

def _tanh_stable(x: AbstractTensor) -> AbstractTensor:
    return x.tanh()

def _hard_sigmoid(x: AbstractTensor) -> AbstractTensor:
    lo = x.less(-2.5)
    hi = x.greater(2.5)
    mid = (x.greater_equal(-2.5)) * (x.less_equal(2.5))
    return lo * 0.0 + hi * 1.0 + mid * (0.2 * x + 0.5)

# -------- base --------

class Activation:
    """Stateless activation with explicit backward; no internal caching."""
    def __call__(self, x: AbstractTensor) -> AbstractTensor:
        return self.forward(x)
    def forward(self, x: AbstractTensor) -> AbstractTensor: ...
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor: ...
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

# -------- core set --------

class Identity(Activation):
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        return x
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return grad_out

class ReLU(Activation):
    """y = max(x, 0)"""
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        with autograd.no_grad():
            pos = x.greater(0.0)
            y = x * pos
        autograd.record("relu", [x], y)
        return y
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return bw_relu(grad_out, x)

class LeakyReLU(Activation):
    """y = x if x>0 else alpha*x"""
    def __init__(self, alpha: float = 0.01):
        self.alpha = float(alpha)
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        with autograd.no_grad():
            pos = x.greater(0.0)
            neg = x.less_equal(0.0)
            y = x * pos + (self.alpha * x) * neg
        autograd.record("leaky_relu", [x], y, params={"alpha": self.alpha})
        return y
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return bw_leaky_relu(grad_out, x, self.alpha)
    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"

class ELU(Activation):
    """y = x if x>0 else alpha*(exp(x)-1)"""
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        with autograd.no_grad():
            pos = x.greater(0.0)
            neg = x.less_equal(0.0)
            y = x * pos + (self.alpha * (x.exp() - 1.0)) * neg
        autograd.record("elu", [x], y, params={"alpha": self.alpha})
        return y
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return bw_elu(grad_out, x, self.alpha)
    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha})"

class Sigmoid(Activation):
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        with autograd.no_grad():
            y = _sigmoid_stable(x)
        autograd.record("sigmoid", [x], y)
        return y
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return bw_sigmoid(grad_out, x)

class Tanh(Activation):
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        with autograd.no_grad():
            y = _tanh_stable(x)
        autograd.record("tanh", [x], y)
        return y
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return bw_tanh(grad_out, x)

class SiLU(Activation):  # aka Swish
    """y = x * sigmoid(x)"""
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        with autograd.no_grad():
            s = _sigmoid_stable(x)
            y = x * s
        autograd.record("silu", [x], y)
        return y
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return bw_silu(grad_out, x)

class GELU(Activation):
    """
    Gaussian Error Linear Unit (approximation, tanh-based)
    y ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x^3)))
    """
    _K = 0.7978845608028654  # sqrt(2/pi)
    _C = 0.044715
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        with autograd.no_grad():
            x3 = x * x * x
            inner = self._K * (x + self._C * x3)
            t = _tanh_stable(inner)
            y = 0.5 * x * (1.0 + t)
        autograd.record("gelu", [x], y)
        return y
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return bw_gelu(grad_out, x)

# -------- hard/lightweight variants (no exp/log) --------

class HardSigmoid(Activation):
    """y = clamp(0, 1, 0.2*x + 0.5) implemented piecewise to avoid clamp dependency"""
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        with autograd.no_grad():
            y = _hard_sigmoid(x)
        autograd.record("hard_sigmoid", [x], y)
        return y
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return bw_hard_sigmoid(grad_out, x)

class HardSwish(Activation):
    """y = x * hard_sigmoid(x)"""
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        with autograd.no_grad():
            h = _hard_sigmoid(x)
            y = x * h
        autograd.record("hard_swish", [x], y)
        return y
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return bw_hard_swish(grad_out, x)
    def __repr__(self) -> str:
        return "HardSwish()"

class ReLU6(Activation):
    """y = min(max(x,0), 6) implemented piecewise"""
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        with autograd.no_grad():
            neg = x.less_equal(0.0)
            sat = x.greater_equal(6.0)
            mid = (x.greater(0.0)) * (x.less(6.0))
            y = neg * 0.0 + sat * 6.0 + mid * x
        autograd.record("relu6", [x], y)
        return y
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        return bw_relu6(grad_out, x)

# -------- backward registry bindings --------

def bw_relu(g, x):
    with autograd.no_grad():
        pos = x.greater(0.0)
        return g * pos

def bw_leaky_relu(g, x, alpha):
    with autograd.no_grad():
        pos = x.greater(0.0)
        neg = x.less_equal(0.0)
        return g * (pos + (alpha * neg))

def bw_elu(g, x, alpha):
    with autograd.no_grad():
        pos = x.greater(0.0)
        neg = x.less_equal(0.0)
        return g * (pos + (alpha * x.exp()) * neg)

def bw_sigmoid(g, x):
    with autograd.no_grad():
        y = _sigmoid_stable(x)
        return g * y * (1.0 - y)

def bw_tanh(g, x):
    with autograd.no_grad():
        y = _tanh_stable(x)
        return g * (1.0 - (y * y))

def bw_silu(g, x):
    with autograd.no_grad():
        s = _sigmoid_stable(x)
        return g * (s * (1.0 + x * (1.0 - s)))

def bw_gelu(g, x):
    with autograd.no_grad():
        x2 = x * x
        x3 = x2 * x
        inner = GELU._K * (x + GELU._C * x3)
        t = _tanh_stable(inner)
        din_dx = GELU._K * (1.0 + 3.0 * GELU._C * x2)
        dt_dx = (1.0 - t * t) * din_dx
        dy_dx = 0.5 * (1.0 + t) + 0.5 * x * dt_dx
        return g * dy_dx

def bw_hard_sigmoid(g, x):
    with autograd.no_grad():
        mid = (x.greater_equal(-2.5)) * (x.less_equal(2.5))
        return g * (0.2 * mid)

def bw_hard_swish(g, x):
    with autograd.no_grad():
        h = _hard_sigmoid(x)
        mid = (x.greater_equal(-2.5)) * (x.less_equal(2.5))
        dh = 0.2 * mid
        return g * (h + x * dh)

def bw_relu6(g, x):
    with autograd.no_grad():
        mid = (x.greater(0.0)) * (x.less(6.0))
        return g * mid

BACKWARD_REGISTRY.register_from_module(sys.modules[__name__])

# -------- registry (nice for configs) --------

ACTIVATIONS = {
    "identity": Identity,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "silu": SiLU,
    "swish": SiLU,
    "gelu": GELU,
    "hard_sigmoid": HardSigmoid,
    "hard_swish": HardSwish,
    "relu6": ReLU6,
}
