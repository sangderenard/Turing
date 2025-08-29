from __future__ import annotations
from ..abstraction import AbstractTensor

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
        pos = x.greater(0.0)
        return x * pos
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        pos = x.greater(0.0)
        return grad_out * pos

class LeakyReLU(Activation):
    """y = x if x>0 else alpha*x"""
    def __init__(self, alpha: float = 0.01):
        self.alpha = float(alpha)
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        pos = x.greater(0.0)
        neg = x.less_equal(0.0)
        return x * pos + (self.alpha * x) * neg
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        pos = x.greater(0.0)
        neg = x.less_equal(0.0)
        return grad_out * (pos + (self.alpha * neg))
    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"

class ELU(Activation):
    """y = x if x>0 else alpha*(exp(x)-1)"""
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        pos = x.greater(0.0)
        neg = x.less_equal(0.0)
        return x * pos + (self.alpha * (x.exp() - 1.0)) * neg
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        pos = x.greater(0.0)
        neg = x.less_equal(0.0)
        return grad_out * (pos + (self.alpha * x.exp()) * neg)
    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha})"

class Sigmoid(Activation):
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        return _sigmoid_stable(x)
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        y = _sigmoid_stable(x)
        return grad_out * y * (1.0 - y)

class Tanh(Activation):
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        return _tanh_stable(x)
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        y = _tanh_stable(x)
        return grad_out * (1.0 - (y * y))

class SiLU(Activation):  # aka Swish
    """y = x * sigmoid(x)"""
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        s = _sigmoid_stable(x)
        return x * s
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        s = _sigmoid_stable(x)
        return grad_out * (s * (1.0 + x * (1.0 - s)))

class GELU(Activation):
    """
    Gaussian Error Linear Unit (approximation, tanh-based)
    y ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x^3)))
    """
    _K = 0.7978845608028654  # sqrt(2/pi)
    _C = 0.044715
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        x3 = x * x * x
        inner = self._K * (x + self._C * x3)
        t = _tanh_stable(inner)
        return 0.5 * x * (1.0 + t)
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        x2 = x * x
        x3 = x2 * x
        inner = self._K * (x + self._C * x3)
        t = _tanh_stable(inner)
        # dt/dx = (1 - t^2) * d(inner)/dx
        din_dx = self._K * (1.0 + 3.0 * self._C * x2)
        dt_dx = (1.0 - t * t) * din_dx
        dy_dx = 0.5 * (1.0 + t) + 0.5 * x * dt_dx
        return grad_out * dy_dx

# -------- hard/lightweight variants (no exp/log) --------

class HardSigmoid(Activation):
    """y = clamp(0, 1, 0.2*x + 0.5) implemented piecewise to avoid clamp dependency"""
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        lo = x.less(-2.5)            # y=0
        hi = x.greater(2.5)          # y=1
        mid = (x.greater_equal(-2.5)) * (x.less_equal(2.5))
        return lo * 0.0 + hi * 1.0 + mid * (0.2 * x + 0.5)
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        mid = (x.greater_equal(-2.5)) * (x.less_equal(2.5))
        return grad_out * (0.2 * mid)

class HardSwish(Activation):
    """y = x * hard_sigmoid(x)"""
    def __init__(self):
        self._hsig = HardSigmoid()
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        return x * self._hsig.forward(x)
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        h = self._hsig.forward(x)
        dh = self._hsig.backward(x, grad_out=AbstractTensor.get_tensor(1.0))  # derivative skeleton
        # d(x*h)/dx = h + x * h'
        return grad_out * (h + x * (dh / grad_out))  # avoid constructing new constants
    def __repr__(self) -> str:
        return "HardSwish()"

class ReLU6(Activation):
    """y = min(max(x,0), 6) implemented piecewise"""
    def forward(self, x: AbstractTensor) -> AbstractTensor:
        neg = x.less_equal(0.0)
        sat = x.greater_equal(6.0)
        mid = (x.greater(0.0)) * (x.less(6.0))
        return neg * 0.0 + sat * 6.0 + mid * x
    def backward(self, x: AbstractTensor, grad_out: AbstractTensor) -> AbstractTensor:
        mid = (x.greater(0.0)) * (x.less(6.0))
        return grad_out * mid

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
