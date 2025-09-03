"""
Backward Rule Registry
======================

This module is an **easy-to-read, static registry** of known backward rules
for common tensor operations. Each rule stores a small snippet of Python code
that can be turned into an executable function at runtime. The registry is
intended to serve simultaneously as:

1) **Hard data backing** for a `backward.py` implementation (machine-readable).
2) **A computer-science + pure-math reference**, readable by humans.
3) **A learning tool** with precise domain assumptions and broadcasting notes.

Design principles
-----------------
- Each entry declares:
  - `arity`: "unary" | "binary" | "n-ary"
  - `signature`: canonical forward signature (names of inputs and parameters)
  - `latex`: math definition of the forward and its derivative(s)
  - `backward`: Python code for the backward pass (as string). The snippet
           uses `g` for the upstream gradient and may reference helper symbols
           defined in `helpers_spec` below.
  - `domain`: assumptions on inputs (e.g., positivity for `log`)
  - `notes`: shape/broadcasting remarks, corner cases, and subgradient choices.
  - `tags`: optional topical tags.

- **Broadcasting**: For elementwise rules, broadcasting is handled by
  `unbroadcast(g_like, shape)` in the pseudocode, meaning: sum-reduce `g_like`
  along axes that were broadcast in the forward so the result matches `shape`.

- **Reduce ops**: For reductions (`sum`, `mean`, ...), use `expand_to(x.shape)`
  to broadcast a reduced gradient back to the input's shape.

- **Soft subgradients**: Where a function is nondifferentiable at isolated
  points (e.g., `abs` at 0, `max` ties), we specify a standard subgradient
  (e.g., 0 at 0 for `abs`, tie-breaking by 0.5 for `maximum/minimum`).


Backward code helpers
---------------------
The backward code snippets are designed to be trivially translatable to your
tensor API. The following helper identifiers are assumed to exist (as math
concepts):

- `unbroadcast(G, shape)`
    Sum-reduce `G` over the axes that were broadcast relative to `shape` and
    reshape to `shape`. (Right-inverse of broadcasting.)

- `sum(x, axis=None, keepdim=False)`
    Summation along `axis` (or all axes if None).

- `expand_to(G, shape)`
    Broadcast `G` to `shape` by inserting/trailing singleton dims as needed.

- `where(cond, a, b)`
    Elementwise select.

- `indicator(cond)`
    1 where cond is true, else 0. (Use exact dtype semantics of your backend.)

- `eps`
    A small positive constant, used where divisions/logs need guarding.

- `ones_like(x)` / `zeros_like(x)`
    Standard helpers.

- `T(x)`
    Transpose (or last-two-dims transpose) – context dependent per rule.

- `permute(x, perm)`
    General axis permutation.

- `reshape(x, shape)`
    Shape change with shared storage.

- `stack([...], dim)` / `concat([...], dim)` / `split(x, sizes, dim)` / `unstack(x, dim)`
    Usual tensor layout ops.

- `softmax(x, dim)` / `log_softmax(x, dim)` / `sigmoid(x)` / `tanh(x)`
    Standard nonlinearities.

- `matmul(A, B)`
    Batched matrix multiply with broadcasting semantics.

- `dot(a, b, dim)`
    Inner product along the specified `dim` (vectors).

All ops used in the snippets should map to your `AbstractTensor` surface
(using `+ - * / **`, `.sum`, `.reshape`, `.permute`, `.transpose`, etc.).


Registry structure
------------------
Each entry is a dictionary with keys:
    {
      "arity": str,
      "signature": str,
      "latex": str or list[str],
      "backward": dict[str, str],     # map input_name -> backward code snippet
      "domain": str,
      "notes": str,
      "tags": list[str]
    }

The entire registry is exposed as `BACKWARD_RULES`.
"""

from __future__ import annotations
from typing import Dict, Any, List
from .abstraction import AbstractTensor

helpers_spec: Dict[str, str] = {
    "unbroadcast":
        "unbroadcast(G, shape): reduce-sum G over axes that were broadcast to match shape; reshape to shape.",
    "expand_to":
        "expand_to(G, shape): broadcast G to target shape by (re)inserting singleton dims.",
    "indicator":
        "indicator(cond): 1 where cond, else 0 (cast to appropriate dtype).",
    "eps":
        "A small positive constant to guard divisions/logs/square-roots (e.g., 1e-12).",
    "T":
        "T(X): transpose last two dims of X (matrix transpose); vectors are treated as row matrices.",
}

def unbroadcast(G, target_shape):
    g_shape = list(getattr(G, "shape", ()))
    t_shape = list(target_shape)

    # If gradient is a scalar, broadcast directly to target shape
    if not g_shape and t_shape:
        return AbstractTensor.ones(tuple(t_shape), dtype=getattr(G, "dtype", None), device=getattr(G, "device", None)) * G

    # Reduce extra leading dims
    if len(g_shape) > len(t_shape):
        reduce_axes = tuple(range(len(g_shape) - len(t_shape)))
        G = AbstractTensor.sum(G, dim=reduce_axes)

    # Now same rank: sum where target has size 1 but grad has >1
    g_shape = list(getattr(G, "shape", ()))
    for ax, (gs, ts) in enumerate(zip(g_shape, t_shape)):
        if ts == 1 and gs != 1:
            G = AbstractTensor.sum(G, dim=ax, keepdim=True)

    return G.reshape(tuple(t_shape))

def expand_to(G, shape):
    # Prefer a native broadcast_to if available
    try:
        return AbstractTensor.broadcast_to(G, shape)
    except Exception:
        # Fallback: multiply by ones of target shape to force broadcast
        ones = AbstractTensor.ones(shape, dtype=getattr(G, "dtype", None), device=getattr(G, "device", None))
        return ones * G

def indicator(cond):
    return AbstractTensor.where(cond, 1, 0)

def I_like(X):
    *batch, m, n = X.shape
    dtype = getattr(X, "dtype", None)
    device = getattr(X, "device", None)
    k = m if m < n else n

    # 1) Best case: backend provides an eye()
    try:
        I = AbstractTensor.eye(m, n, dtype=dtype, device=device)  # shape (m, n)
    except Exception:
        # 2) Fallback: build via arange equality if available
        try:
            i = AbstractTensor.arange(m, dtype="int64", device=device).reshape(m, 1)
            j = AbstractTensor.arange(n, dtype="int64", device=device).reshape(1, n)
            ones = AbstractTensor.ones((m, n), dtype=dtype, device=device)
            zeros = AbstractTensor.zeros((m, n), dtype=dtype, device=device)
            I = AbstractTensor.where(i == j, ones, zeros)
        except Exception:
            # 3) Last resort: explicit scatter along the diagonal
            I = AbstractTensor.zeros((m, n), dtype=dtype, device=device)
            # Note: uses setitem semantics your tensors already support in slice backward
            for t in range(k):
                I[(slice(None),) * 0 + (t, t)] = 1  # i.e., I[t, t] = 1

    # Broadcast over batch dims if needed
    if batch:
        I = AbstractTensor.broadcast_to(I, tuple(batch) + (m, n))

    return I


def eps():
    return 1e-12

def T(X):
    ndim = getattr(X, "ndim", len(getattr(X, "shape", ())))
    if ndim < 2:
        if ndim == 0:
            X = X.reshape((1, 1))
        else:
            X = X.reshape((1, X.shape[0]))
    return X.transpose(-2, -1)

BACKWARD_RULES: Dict[str, Dict[str, Any]] = {
    # ----------------------------------------------------------------------
    # Elementwise unary
    # ----------------------------------------------------------------------
    "neg": {
        "arity": "unary",
        "signature": "y = -x",
        "latex": r"y = -x, \quad \frac{\partial y}{\partial x} = -1",
        "backward": {
            "x": "gx = unbroadcast(-g, x.shape)"
        },
        "python": {
            "parameters" : ["g", "x"],
            "body" : "return unbroadcast(-g, x.shape)"
        },
        "domain": "x: any real",
        "notes": "Pure sign flip.",
        "tags": ["elementwise", "unary"],
    },
    "exp": {
        "arity": "unary",
        "signature": "y = exp(x)",
        "latex": r"y = e^x, \quad \frac{\partial y}{\partial x} = e^x = y",
        "backward": {
            "x": "gx = unbroadcast(g * y, x.shape)"
        },
        "python":  {"parameters": ["g", "x", "y"], "body": "return unbroadcast(g * y, x.shape)"},
        "domain": "x: any real",
        "notes": "Use forward output `y` if available for efficiency.",
        "tags": ["elementwise", "unary", "smooth"],
    },
    "log": {
        "arity": "unary",
        "signature": "y = log(x)",
        "latex": r"y = \log x, \quad \frac{\partial y}{\partial x} = 1/x",
        "backward": {
            "x": "gx = unbroadcast(g / (x + eps), x.shape)"
        },
        "python": {"parameters": ["g", "x"], "body": "return unbroadcast(g / (x + eps()), x.shape)"},
        "domain": "x > 0 (strict); practical: x >= eps",
        "notes": "Guard at 0 with eps to avoid NaNs.",
        "tags": ["elementwise", "unary", "smooth", "domain"],
    },
    "sqrt": {
        "arity": "unary",
        "signature": "y = sqrt(x)",
        "latex": r"y = \sqrt{x}, \quad \frac{\partial y}{\partial x} = \frac{1}{2\sqrt{x}}",
        "backward": {
            "x": "gx = unbroadcast(0.5 * g / (y + eps), x.shape)"
        },
        "python": {"parameters": ["g", "x", "y"], "body": "return unbroadcast(0.5 * g / (y + eps()), x.shape)"},
        "domain": "x >= 0; practical: x >= 0 with eps guard",
        "notes": "Prefer using forward output `y` for stability.",
        "tags": ["elementwise", "unary", "smooth", "domain"],
    },
    "abs": {
        "arity": "unary",
        "signature": "y = |x|",
        "latex": r"y = |x|, \quad \frac{\partial y}{\partial x} = \mathrm{sign}(x) \text{ for } x \neq 0",
        "backward": {
            "x": "gx = unbroadcast(g * where(x>0, 1, where(x<0, -1, 0)), x.shape)"
        },
        "python": {
            "parameters": ["g", "x"],
            "body": "return unbroadcast(g * AbstractTensor.where(x>0, 1, AbstractTensor.where(x<0, -1, 0)), x.shape)",
        },
        "domain": "x: any real",
        "notes": "Subgradient at x=0 set to 0. Other choices are valid but less common.",
        "tags": ["elementwise", "unary", "nonsmooth"],
    },
    "sin": {
        "arity": "unary",
        "signature": "y = sin(x)",
        "latex": r"y = \sin x, \quad \frac{\partial y}{\partial x} = \cos x",
        "backward": {
            "x": "gx = unbroadcast(g * cos(x), x.shape)"
        },
        "python": {
                    "parameters": ["g", "x"],
                    "body": "return unbroadcast(g * AbstractTensor.cos(x), x.shape)"
                },
        "domain": "x: any real",
        "notes": "",
        "tags": ["elementwise", "unary", "smooth", "trig"],
    },
    "cos": {
        "arity": "unary",
        "signature": "y = cos(x)",
        "latex": r"y = \cos x, \quad \frac{\partial y}{\partial x} = -\sin x",
        "backward": {
            "x": "gx = unbroadcast(-g * sin(x), x.shape)"
        },
        "python": {
            "parameters": ["g", "x"],
            "body": "return unbroadcast(-g * AbstractTensor.sin(x), x.shape)"
        },
        "domain": "x: any real",
        "notes": "",
        "tags": ["elementwise", "unary", "smooth", "trig"],
    },
    "tan": {
        "arity": "unary",
        "signature": "y = tan(x)",
        "latex": r"y = \tan x, \quad \frac{\partial y}{\partial x} = \sec^2 x = 1 + \tan^2 x",
        "backward": {
            "x": "gx = unbroadcast(g * (1 + y*y), x.shape)"
        },
        "python": {
                    "parameters": ["g", "x", "y"],
                    "body": "return unbroadcast(g * (1 + y*y), x.shape)"
                },
        "domain": "x != (pi/2 + k*pi)",
        "notes": "Use forward output `y` to compute 1 + tan^2(x).",
        "tags": ["elementwise", "unary", "smooth", "trig"],
    },
    "tanh": {
        "arity": "unary",
        "signature": "y = tanh(x)",
        "latex": r"y = \tanh x, \quad \frac{\partial y}{\partial x} = 1 - \tanh^2 x = 1 - y^2",
        "backward": {
            "x": "gx = unbroadcast(g * (1 - (x.detach().tanh()*x.detach().tanh())), x.shape)"
        },
        "python": {
            "parameters": ["g", "x"],
            "body": "y = x.detach().tanh(); return unbroadcast(g * (1 - y*y), x.shape)"
        },
        "domain": "x: any real",
        "notes": "",
        "tags": ["elementwise", "unary", "smooth", "nn"],
    },
    "sigmoid": {
        "arity": "unary",
        "signature": "y = sigmoid(x)",
        "latex": r"y = \sigma(x) = \frac{1}{1+e^{-x}}, \quad \frac{\partial y}{\partial x} = y(1-y)",
        "backward": {
            "x": "gx = unbroadcast(g * y * (1 - y), x.shape)"
        },
        "python":{
            "parameters": ["g", "x", "y"],
            "body": "return unbroadcast(g * y * (1 - y), x.shape)"
        },
        "domain": "x: any real",
        "notes": "Use numerically stable sigmoid implementation in forward.",
        "tags": ["elementwise", "unary", "smooth", "nn"],
    },
    "relu": {
        "arity": "unary",
        "signature": "y = relu(x) = max(x, 0)",
        "latex": r"y = \max(x,0), \quad \frac{\partial y}{\partial x} = \mathbf{1}_{x>0}",
        "backward": {
            "x": "gx = unbroadcast(g * indicator(x>0), x.shape)"
        },
        "python": {
            "parameters": ["g", "x"],
            "body": "return unbroadcast(g * AbstractTensor.where(x>0, 1, 0), x.shape)"
        },
        "domain": "x: any real",
        "notes": "At x=0 use subgradient 0 by convention.",
        "tags": ["elementwise", "unary", "nonsmooth", "nn"],
    },
    "leaky_relu": {
        "arity": "unary",
        "signature": "y = leaky_relu(x; alpha) = max(x, alpha*x)",
        "latex": r"y=\begin{cases}x & x>0\\ \alpha x & x\le0\end{cases},\quad \frac{\partial y}{\partial x}=\begin{cases}1 & x>0\\ \alpha & x\le0\end{cases}",
        "backward": {
            "x": "gx = unbroadcast(g * where(x>0, 1, alpha), x.shape)"
        },
        "python": {
            "parameters": ["g", "x", "alpha"],
            "body": "return unbroadcast(g * AbstractTensor.where(x>0, 1, alpha), x.shape)"
        },
        "domain": "alpha in (0,1) typically",
        "notes": "Alpha passed as parameter to forward. Same value used in backward.",
        "tags": ["elementwise", "unary", "nonsmooth", "nn"],
    },
    "softplus": {
        "arity": "unary",
        "signature": "y = softplus(x) = log(1 + exp(x))",
        "latex": r"y = \log(1+e^{x}),\quad \frac{\partial y}{\partial x} = \sigma(x)",
        "backward": {
            "x": "gx = unbroadcast(g * sigmoid(x), x.shape)"
        },
        "python": {
            "parameters": ["g", "x"],
            "body": "return unbroadcast(g * AbstractTensor.sigmoid(x), x.shape)"
        },
        "domain": "x: any real",
        "notes": "Use numerically stable softplus in forward.",
        "tags": ["elementwise", "unary", "smooth", "nn"],
    },

    # ----------------------------------------------------------------------
    # Elementwise binary
    # ----------------------------------------------------------------------
    "add": {
        "arity": "binary",
        "signature": "z = x + y",
        "latex": r"z = x+y,\quad \frac{\partial z}{\partial x}=1,\ \frac{\partial z}{\partial y}=1",
        "backward": {
            "x": "gx = unbroadcast(g, x.shape)",
            "y": "gy = unbroadcast(g, y.shape)",
        },
        "python": {"parameters": ["g", "x", "y"], "body": "gx=unbroadcast(g, x.shape); gy=unbroadcast(g, y.shape); return gx, gy"},
        "domain": "x,y: any real; broadcasting allowed",
        "notes": "Pure passthrough with broadcasting unrolled.",
        "tags": ["elementwise", "binary"],
    },
    "sub": {
        "arity": "binary",
        "signature": "z = x - y",
        "latex": r"z = x-y,\quad \frac{\partial z}{\partial x}=1,\ \frac{\partial z}{\partial y}=-1",
        "backward": {
            "x": "gx = unbroadcast(g, x.shape)",
            "y": "gy = unbroadcast(-g, y.shape)",
        },
        "python": {"parameters": ["g", "x", "y"], "body": "gx=unbroadcast(g, x.shape); gy=unbroadcast(-g, y.shape); return gx, gy"},
        "domain": "x,y: any real; broadcasting allowed",
        "notes": "",
        "tags": ["elementwise", "binary"],
    },
    "mul": {
        "arity": "binary",
        "signature": "z = x * y",
        "latex": r"z = xy,\quad \frac{\partial z}{\partial x}=y,\ \frac{\partial z}{\partial y}=x",
        "backward": {
            "x": "gx = unbroadcast(g * y, x.shape)",
            "y": "gy = unbroadcast(g * x, y.shape)",
        },
        "python": {"parameters": ["g", "x", "y"], "body": "return unbroadcast(g*y, x.shape), unbroadcast(g*x, y.shape)"},
        "domain": "x,y: any real; broadcasting allowed",
        "notes": "",
        "tags": ["elementwise", "binary"],
    },
    "div": {
        "arity": "binary",
        "signature": "z = x / y",
        "latex": r"z = x/y,\quad \frac{\partial z}{\partial x}=1/y,\ \frac{\partial z}{\partial y}=-x/y^2",
        "backward": {
            "x": "gx = unbroadcast(g / (y + eps), x.shape)",
            "y": "gy = unbroadcast(-g * x / ((y + eps)*(y + eps)), y.shape)",
        },
        "python": {"parameters": ["g", "x", "y"], "body": "ys=y+eps(); return unbroadcast(g/ys, x.shape), unbroadcast(-g*x/(ys*ys), y.shape)"},
        "domain": "y != 0; practical: |y| >= eps",
        "notes": "Guard division by zero with eps.",
        "tags": ["elementwise", "binary"],
    },
    "pow": {
        "arity": "binary",
        "signature": "z = x ** p   # exponent p may be tensor or scalar",
        "latex": [
            r"z = x^p, \quad \frac{\partial z}{\partial x} = p x^{p-1}",
            r"\frac{\partial z}{\partial p} = x^p \log x \quad (x>0)"
        ],
        "backward": {
            "x": "gx = unbroadcast(g * p * (x)**(p - 1), x.shape)",
            "p": "gp = unbroadcast(g * (x)**p * log(abs(x) + eps), p.shape)",
        },
        "python": {
            "parameters": ["g", "x", "p"],
            "body": "absx=abs(x); xs=absx+eps(); ys=x**p; gx=unbroadcast(g*p*(x**(p-1)), x.shape); gp=unbroadcast(g*ys*AbstractTensor.log(xs), p.shape); return gx, gp"
        },
        "domain": "x>0 if p varies; if p is integer constant, extend by continuity.",
        "notes": "When p is constant, only x-branch is needed.",
        "tags": ["elementwise", "binary", "domain"],
    },
    "maximum": {
        "arity": "binary",
        "signature": "z = maximum(x, y)",
        "latex": r"z_i = \max(x_i, y_i)",
        "backward": {
            "x": "gx = unbroadcast(g * where(x>y, 1, where(x<y, 0, 0.5)), x.shape)",
            "y": "gy = unbroadcast(g * where(y>x, 1, where(y<x, 0, 0.5)), y.shape)",
        },
        "python": {
            "parameters": ["g", "x", "y"],
            "body": (
                "mx0=AbstractTensor.where(x>y,1,AbstractTensor.where(x<y,0,0.5)); "
                "mx1=AbstractTensor.where(y>x,1,AbstractTensor.where(y<x,0,0.5)); "
                "return unbroadcast(g*mx0, x.shape), unbroadcast(g*mx1, y.shape)"
            )
        },
        "domain": "x,y: any real",
        "notes": "At ties (x==y) we choose to split gradient equally (0.5/0.5).",
        "tags": ["elementwise", "binary", "nonsmooth"],
    },
    "minimum": {
        "arity": "binary",
        "signature": "z = minimum(x, y)",
        "latex": r"z_i = \min(x_i, y_i)",
        "backward": {
            "x": "gx = unbroadcast(g * where(x<y, 1, where(x>y, 0, 0.5)), x.shape)",
            "y": "gy = unbroadcast(g * where(y<x, 1, where(y>x, 0, 0.5)), y.shape)",
        },
        "python": {
            "parameters": ["g", "x", "y"],
            "body": (
                "mn0=AbstractTensor.where(x<y,1,AbstractTensor.where(x>y,0,0.5)); "
                "mn1=AbstractTensor.where(y<x,1,AbstractTensor.where(y>x,0,0.5)); "
                "return unbroadcast(g*mn0, x.shape), unbroadcast(g*mn1, y.shape)"
            )
        },
        "domain": "x,y: any real",
        "notes": "At ties (x==y) split the gradient 0.5/0.5.",
        "tags": ["elementwise", "binary", "nonsmooth"],
    },

    # ----------------------------------------------------------------------
    # Reductions
    # ----------------------------------------------------------------------
    "sum": {
        "arity": "unary",
        "signature": "y = sum(x, axis=None, keepdim=False)",
        "latex": r"y = \sum_{i \in \mathcal{I}} x_i",
        "backward": {
            "x": "gx = expand_to(g, x.shape)"
        },
        "python": {
            "parameters": ["g", "x"],
            "body": "return expand_to(g, x.shape)"
        },
        "domain": "x: any real",
        "notes": "If axis is specified and keepdim=False, conceptually unsqueeze `g` on the reduced axes before expand.",
        "tags": ["reduction", "linear"],
    },
    "mean": {
        "arity": "unary",
        "signature": "y = mean(x, axis=None, keepdim=False)",
        "latex": r"y = \frac{1}{N}\sum_{i \in \mathcal{I}} x_i",
        "backward": {
            "x": "N = number_of_elements_reduced; gx = expand_to(g, x.shape) / N"
        },
        "python": {
            "parameters": ["g", "x", "axis=None", "keepdim=False"],
            "body": (
                "N = max(1, x.numel() // max(1, g.numel())); "
                "return expand_to(g, x.shape) / N"
            )
        },
        "domain": "x: any real",
        "notes": "Same broadcasting mechanics as `sum`, scaled by 1/N.",
        "tags": ["reduction", "linear"],
    },
    "matmul": {
        "arity": "binary",
        "signature": "Y = matmul(A, B)  # ... x m x k  @  ... x k x n",
        "latex": [
            r"Y = A B,\quad dA = G B^\top,\quad dB = A^\top G",
            r"(Broadcasted batch dims require summing over broadcast axes.)"
        ],
        "backward": {
            "A": "s=g.shape() if callable(getattr(g,'shape',None)) else g.shape; g2 = g.reshape((1, *s)) if getattr(g, 'ndim', len(s)) == 1 else g; gA = unbroadcast(matmul(g2, T(B)), A.shape)",
            "B": "s=g.shape() if callable(getattr(g,'shape',None)) else g.shape; g2 = g.reshape((1, *s)) if getattr(g, 'ndim', len(s)) == 1 else g; gB = unbroadcast(matmul(T(A), g2), B.shape)"
        },
        "python": {
            "parameters": ["g", "A", "B"],
            "body": "s=g.shape() if callable(getattr(g,'shape',None)) else g.shape; g=g.reshape((1, *s)) if getattr(g, 'ndim', len(s))==1 else g; gA=unbroadcast(AbstractTensor.matmul(g, T(B)), A.shape); gB=unbroadcast(AbstractTensor.matmul(T(A), g), B.shape); return gA, gB"
        },
        "domain": "Inner dims match; batch dims broadcastable.",
        "notes": "Use T() as last-two-dims transpose. Apply unbroadcast to fold batch broadcasting.",
        "tags": ["linear-algebra"],
    },

    "var": {
        "arity": "unary",
        "signature": "y = var(x, axis=None, keepdim=False, unbiased=False)",
        "latex": r"y = \frac{1}{N}\sum_i (x_i - \bar{x})^2 \text{ (population)}",
        "backward": {
            "x": "mu = mean(x, axis, keepdim=True); gx = expand_to(g, x.shape) * 2*(x - mu)/N"
        },
        "python": {
            "parameters": ["g", "x", "axis", "keepdim", "unbiased"],
            "body": (
                "mu = AbstractTensor.mean(x, axis=axis, keepdim=True); "
                "N  = max(1, x.numel() // max(1, g.numel())); "
                "return expand_to(g, x.shape) * 2*(x - mu) / N"
            )
        },
        "domain": "x: any real",
        "notes": "For unbiased=True replace N by (N-1) in forward; backward uses population scaling in many libs; match your forward exactly.",
        "tags": ["reduction", "statistics"],
    },
    "std": {
        "arity": "unary",
        "signature": "y = std(x, axis=None, keepdim=False, unbiased=False) = sqrt(var(x))",
        "latex": r"y = \sqrt{\mathrm{var}(x)}",
        "backward": {
            "x": "mu = mean(x, axis, keepdim=True); v = mean((x - mu)**2, axis, keepdim=True); gx = expand_to(g, x.shape) * (x - mu) / (sqrt(v) * N + eps)"
        },
        "python": {
            "parameters": ["g", "x", "axis", "keepdim", "unbiased"],
            "body": (
                "mu = AbstractTensor.mean(x, axis=axis, keepdim=True); "
                "v  = AbstractTensor.mean((x - mu)**2, axis=axis, keepdim=True); "
                "N  = max(1, x.numel() // max(1, g.numel())); "
                "return expand_to(g, x.shape) * (x - mu) / (AbstractTensor.sqrt(v) * N + eps())"
            )
        },
        "domain": "x: any real",
        "notes": "Derives via chain rule from var -> sqrt.",
        "tags": ["reduction", "statistics"],
    },

    # ----------------------------------------------------------------------
    # Shape / layout ops
    # ----------------------------------------------------------------------
    "clone": {
        "arity": "unary",
        "signature": "y = x.clone()",
        "latex": r"y = \mathrm{clone}(x),\quad \frac{\partial y}{\partial x} = 1",
        "backward": {
            "x": "gx = g.clone()"
        },
        "python": {
            "parameters": ["g", "x"],
            "body": "return g.clone() if hasattr(g, 'clone') else g"
        },
        "domain": "x: any real",
        "notes": "Clone is an identity operation with independent storage; gradient passes through unchanged.",
        "tags": ["shape"],
    },
    
    "reshape": {
        "arity": "unary",
        "signature": "y = reshape(x, new_shape)",
        "latex": r"y = \mathrm{Reshape}(x)",
        "backward": {
            "x": "gx = reshape(g, x.shape)"
        },
        "python": {
            "parameters": ["g", "x", "new_shape=None"],
            "body": "return AbstractTensor.reshape(g, getattr(x, 'shape', new_shape))"
        },
        "domain": "Same number of elements.",
        "notes": "Gradient reshapes back.",
        "tags": ["shape"],
    },
    "transpose_last2": {
        "arity": "unary",
        "signature": "y = T(x)  # last-two-dims transpose",
        "latex": r"y = x^\top,\quad \mathrm{(last\ two\ dims)}",
        "backward": {
            "x": "gx = T(g)"
        },
        "python": {
            "parameters": ["g", "x"],
            "body": "return T(g)"
        },
        "domain": "Tensor with rank >=2",
        "notes": "For general permutes use `permute` rule.",
        "tags": ["shape", "linear"],
    },
    "permute": {
        "arity": "unary",
        "signature": "y = permute(x, perm)",
        "latex": r"y_{i_{\pi(1)},\ldots,i_{\pi(n)}} = x_{i_1,\ldots,i_n}",
        "backward": {
            "x": "inv = inverse_permutation(perm); gx = permute(g, inv)"
        },
        "python": {
            "parameters": ["g", "x", "perm"],
            "body": "inv=[0]*len(perm);for i,pv in enumerate(perm): inv[pv]=i;return AbstractTensor.permute(g, inv)"
        },
        "domain": "Any rank",
        "notes": "Pure reindexing; inverse permutation for backward.",
        "tags": ["shape", "linear"],
    },
    "broadcast_to": {
        "arity": "unary",
        "signature": "y = broadcast_to(x, shape)",
        "latex": r"y = \mathrm{Broadcast}(x)",
        "backward": {
            "x": "gx = unbroadcast(g, x.shape)"
        },
        "python": {
            "parameters": ["g", "x", "shape"],
            "body": "return unbroadcast(g, x.shape)"
        },
        "domain": "Target shape is broadcast-compatible.",
        "notes": "Right-inverse of broadcasting via summation.",
        "tags": ["shape"],
    },
    "slice": {
        "arity": "unary",
        "signature": "y = x[slices]",
        "latex": r"y = S(x) \text{ (slicing operator)}",
        "backward": {
            "x": "gx = zeros_like(x); gx[slices] = g"
        },
        "python": {
            "parameters": ["g", "x", "slices"],
            "body": "gx=AbstractTensor.zeros_like(x); gx[slices]=g; return gx"
        },
        "domain": "Any real; slices valid.",
        "notes": "Implements a simple scatter into the sliced region.",
        "tags": ["shape", "indexing"],
    },
    "index_set": {
        "arity": "binary",
        "signature": "y = x; y[idx] = v",
        "latex": r"y = x; y[\\text{idx}] = v",
        "backward": {
            "x": "gx = g.clone()",
            "v": "gv = g[idx]"
        },
        "python": {
            "parameters": ["g", "x", "v", "idx"],
            "body": "gx=g.clone(); gv=g[idx]; return gx, gv"
        },
        "domain": "Any real; idx valid.",
        "notes": "Scatter assignment; gradient flows from selected region to the value.",
        "tags": ["indexing"],
    },
    "gather": {
        "arity": "binary",
        "signature": "y = gather(x, index, dim)",
        "latex": r"y_i = x_{index_i}",
        "backward": {
            "x": "gx = zeros_like(x); gx[index] = g"
        },
        "python": {
            "parameters": ["g", "x", "index"],
            "body": "gx=AbstractTensor.zeros_like(x); gx[index]=g; return gx"
        },
        "domain": "Any real; index valid.",
        "notes": "Backward of gather: scatter gradient back to x; no gradient w.r.t index or dim.",
        "tags": ["indexing"],
    },
    "scatter": {
        "arity": "ternary",
        "signature": "y = scatter(x, index, src, dim)",
        "latex": r"y_i = x_i + src_j \text{ for } i = index_j, \; y_i = x_i \text{ otherwise}",
        "backward": {
            "x": "gx = g.clone()",
            "src": "gsrc = g[index]"
        },
        "python": {
            "parameters": ["g", "x", "src", "index"],
            "body": "gx=g.clone(); gsrc=g[index]; return gx, gsrc"
        },
        "domain": "Any real; index valid.",
        "notes": "Backward of scatter: pass gradient through x; gather gradient for src at index positions.",
        "tags": ["indexing"],
    },
    "concat": {
        "arity": "n-ary",
        "signature": "y = concat([x1, x2, ...], dim)",
        "latex": r"y = \mathrm{Concat}(x_1,x_2,\ldots; \mathrm{dim})",
        "backward": {
            "x*": "gs = split(g, [xi.shape[dim] for xi in xs], dim)"
        },
        "python": {
            "parameters": ["g", "*xs", "dim"],
            "body": "xs=list(xs); sizes=[x.shape[dim] for x in xs]; parts=AbstractTensor.split(g, sizes, dim); return tuple(unbroadcast(parts[i], xs[i].shape) for i in range(len(xs)))"
        },
        "domain": "Shapes must align on all dims except `dim`.",
        "notes": "Backward is a split of `g` into per-input gradients.",
        "tags": ["shape", "layout"],
    },
    "stack": {
        "arity": "n-ary",
        "signature": "y = stack([x1, x2, ...], dim)",
        "latex": r"y = \mathrm{Stack}(x_1,x_2,\ldots; \mathrm{dim})",
        "backward": {
            "x*": "gx_k = unstack(g, dim)[k]"
        },
        "python": {
            "parameters": ["g", "*xs", "dim"],
            "body": "xs=list(xs); parts=AbstractTensor.unstack(g, dim); return tuple(unbroadcast(parts[i], xs[i].shape) for i in range(len(xs)))"
        },
        "domain": "All inputs same shape.",
        "notes": "Backward is unstack along the stacking dim.",
        "tags": ["shape", "layout"],
    },

    # ----------------------------------------------------------------------
    # Selection / logic
    # ----------------------------------------------------------------------
    "where": {
        "arity": "n-ary",
        "signature": "y = where(cond, a, b)",
        "latex": r"y_i = \begin{cases} a_i & \text{if } cond_i \\ b_i & \text{otherwise}\end{cases}",
        "backward": {
            "a": "ga = unbroadcast(g * indicator(cond), a.shape)",
            "b": "gb = unbroadcast(g * (1 - indicator(cond)), b.shape)"
        },
        "python":{
            "parameters": ["g", "cond", "a", "b"],
            "body": (
                "mask = AbstractTensor.where(cond, 1, 0); "
                "ga = unbroadcast(g * mask, a.shape); "
                "gb = unbroadcast(g * (1 - mask), b.shape); "
                "return None, ga, gb"
            )
        },
        "domain": "cond is boolean/tensor; a,b broadcast-compatible.",
        "notes": "No gradient w.r.t. cond (discrete).",
        "tags": ["selection"],
    },
    "clamp": {
        "arity": "unary",
        "signature": "y = clamp(x, min=None, max=None)",
        "latex": r"y = \min(\max(x, m), M)",
        "backward": {
            "x": "gx = unbroadcast(g * indicator((min is None or x>min) and (max is None or x<max)), x.shape)"
        },
        "python": {
            "parameters": ["g", "x", "min", "max"],
            "body": (
                "left  = AbstractTensor.ones_like(x) if (min is None) else AbstractTensor.where(x>min, 1, 0); "
                "right = AbstractTensor.ones_like(x) if (max is None) else AbstractTensor.where(x<max, 1, 0); "
                "mask = left * right; "
                "return unbroadcast(g * mask, x.shape)"
            )
        },
        "domain": "x real; min<=max if provided.",
        "notes": "Zero gradient outside the open interval; subgradient at endpoints set to 0.",
        "tags": ["elementwise", "nonsmooth"],
    },

    # ----------------------------------------------------------------------
    # Linear algebra
    # ----------------------------------------------------------------------
    "matmul": {
        "arity": "binary",
        "signature": "Y = matmul(A, B)  # ... x m x k  @  ... x k x n",
        "latex": [
            r"Y = A B,\quad dA = G B^\top,\quad dB = A^\top G",
            r"(Broadcasted batch dims require summing over broadcast axes.)"
        ],
        "backward": {
            "A": "s=g.shape() if callable(getattr(g,'shape',None)) else g.shape; g2 = g.reshape((1, *s)) if getattr(g, 'ndim', len(s)) == 1 else g; gA = unbroadcast(matmul(g2, T(B)), A.shape)",
            "B": "s=g.shape() if callable(getattr(g,'shape',None)) else g.shape; g2 = g.reshape((1, *s)) if getattr(g, 'ndim', len(s)) == 1 else g; gB = unbroadcast(matmul(T(A), g2), B.shape)"
        },
        "python": {
            "parameters": ["g", "A", "B"],
            "body": "s=g.shape() if callable(getattr(g,'shape',None)) else g.shape; g=g.reshape((1, *s)) if getattr(g, 'ndim', len(s))==1 else g; gA=unbroadcast(AbstractTensor.matmul(g, T(B)), A.shape); gB=unbroadcast(AbstractTensor.matmul(T(A), g), B.shape); return gA, gB"
        },
        "domain": "Inner dims match; batch dims broadcastable.",
        "notes": "Use T() as last-two-dims transpose. Apply unbroadcast to fold batch broadcasting.",
        "tags": ["linear-algebra"],
    },
    "dot": {
        "arity": "binary",
        "signature": "y = dot(a, b, dim=-1)  # inner product vectors",
        "latex": r"y = \sum_i a_i b_i,\quad da = g b,\ db = g a",
        "backward": {
            "a": "ga = unbroadcast(g * b, a.shape)",
            "b": "gb = unbroadcast(g * a, b.shape)"
        },
        "python": {
            "parameters": ["g", "a", "b"],
            "body": "return unbroadcast(g*b, a.shape), unbroadcast(g*a, b.shape)"
        },
        "domain": "a,b same shape along `dim`.",
        "notes": "Equivalent to reduce(sum(a*b, dim)).",
        "tags": ["linear-algebra"],
    },
    "norm2": {
        "arity": "unary",
        "signature": "y = ||x||_2 = sqrt(sum(x^2, axis, keepdim=False))",
        "latex": r"y = \sqrt{\sum_i x_i^2},\quad \frac{\partial y}{\partial x} = \frac{x}{\|x\|_2}",
        "backward": {
            "x": "gx = expand_to(g, x.shape) * x / (norm2(x, axis, keepdim=True) + eps)"
        },
        "python": {
            "parameters": ["g", "x", "axis"],
            "body": "n = AbstractTensor.sqrt(AbstractTensor.sum(x*x, axis=axis, keepdim=True)) + eps(); return expand_to(g, x.shape) * x / n"
        },
        "domain": "x real; handle zero norm via eps.",
        "notes": "For axis=None treat as scalar norm; else vectorized by axis.",
        "tags": ["linear-algebra", "smooth"],
    },
    "inverse": {
        "arity": "unary",
        "signature": "Y = inverse(X)",
        "latex": r"dX = -X^{-\top} \, G \, X^{-\top}",
        "backward": {
            "X": "gX = - matmul(T(inverse(X)), matmul(g, T(inverse(X))))"
        },
        "python": {
            "parameters": ["g", "X"],
            "body": "iXT = T(AbstractTensor.inverse(X)); return - AbstractTensor.matmul(iXT, AbstractTensor.matmul(g, iXT))"
        },
        "domain": "X square and nonsingular.",
        "notes": "Uses the identity d(X^{-1}) = -X^{-1}(dX)X^{-1}.",
        "tags": ["linear-algebra", "matrix-calculus", "advanced"],
    },
    "det": {
        "arity": "unary",
        "signature": "y = det(X)",
        "latex": r"dy = \det(X)\,\mathrm{tr}(X^{-1} dX)\quad \Rightarrow\quad \frac{\partial y}{\partial X} = y \, X^{-\top}",
        "backward": {
            "X": "gX = g * det(X) * T(inverse(X))"
        },
        "python": {
            "parameters": ["g", "X"],
            "body": "return g * AbstractTensor.det(X) * T(AbstractTensor.inverse(X))"
        },
        "domain": "X square and nonsingular.",
        "notes": "When X is singular, gradient is undefined; practical handling may return NaNs or zeros.",
        "tags": ["linear-algebra", "matrix-calculus", "advanced"],
    },
    "trace": {
        "arity": "unary",
        "signature": "y = trace(X)",
        "latex": r"dy = \mathrm{tr}(dX) \Rightarrow \frac{\partial y}{\partial X} = I",
        "backward": {
            "X": "gX = g * I_like(X)"
        },
        "python": {
            "parameters": ["g", "X"],
            "body": "return g * I_like(X)"
        },
        "domain": "Square last-two-dims assumed; otherwise trace over min(n,m).",
        "notes": "Implement I_like(X) as an identity on the last two dims, broadcast over batch dims.",
        "tags": ["linear-algebra"],
    },

    # ----------------------------------------------------------------------
    # Softmax family
    # ----------------------------------------------------------------------
    "softmax": {
        "arity": "unary",
        "signature": "y = softmax(x, dim)",
        "latex": r"\frac{\partial y}{\partial x} = \mathrm{diag}(y) - y y^\top",
        "backward": {
            "x": "s = softmax(x, dim); dot = sum(g * s, dim, keepdim=True); gx = (g - dot) * s"
        },
        "python":{
            "parameters": ["g", "x", "dim"],
            "body": "s=AbstractTensor.softmax(x, dim); dot=AbstractTensor.sum(g*s, axis=dim, keepdim=True); return (g - dot) * s"
        },
        "domain": "x real; `dim` valid.",
        "notes": "Classic VJP form avoids an explicit Jacobian.",
        "tags": ["nn", "softmax"],
    },
    "log_softmax": {
        "arity": "unary",
        "signature": "y = log_softmax(x, dim)",
        "latex": r"\frac{\partial y}{\partial x} = I - \mathbf{1} y^\top_{\exp} \quad\text{with } y_{\exp}=\mathrm{softmax}(x)",
        "backward": {
            "x": "s = softmax(x, dim); dot = sum(g, dim, keepdim=True); gx = g - dot * s"
        },
        "python":{
            "parameters": ["g", "x", "dim"],
            "body": "s=AbstractTensor.softmax(x, dim); dot=AbstractTensor.sum(g, axis=dim, keepdim=True); return g - dot * s"
        },
        "domain": "x real; `dim` valid.",
        "notes": "Derived from softmax and the chain rule of log.",
        "tags": ["nn", "softmax"],
    },
}
# End of BACKWARD_RULES
