"""The authored trigonometric surface of ``AbstractTensor``.

Two implementations live behind one set of names.

``operator``
    ``self._apply_operator("sin", ...)``, which lowers to the repository
    kernel ``unary_double`` with opcode 29, whose body is
    ``call double @sin(double)`` -- an extern resolved by the platform's libm.
    This is what the surface has always been, and what every packed
    trigonometry artifact in this tree computed with, forward and reverse.

``signal_math``
    ``common.tensors.signal_math``: baked cores evaluated with AbstractTensor
    arithmetic. It captures into repository SSA like any other authored
    source, so one definition reaches native, shader and wasm products alike
    and the constants travel inside the graph.

Which one is live is a DELIBERATE switch, not a default that drifted into
place. On the eager path the signal-math route measured ~280x slower than the
libm operator, because every Horner step is a separate dispatched tensor op --
that cost is the interpreter, not the mathematics, and it is the cost the
compiler exists to remove. Until a compiled path is the one being exercised,
the operator route stays live so nothing in the tree slows down by accident.

Flip it with :func:`use_signal_math` / :func:`use_backend_operator`, or read
:func:`active_implementation`. The functions below are the SOURCE the
trigonometry catalogue ingests (``mathematical_library._trigonometry_methods``
calls ``inspect.getsource`` on each), so whichever route is live is also what
a standard object bakes.
"""

from __future__ import annotations


#: ``"operator"`` or ``"signal_math"``. See the module docstring.
_IMPLEMENTATION = "operator"

#: Which prebaked quality the signal-math route resolves to.
_QUALITY = "audio"


def use_signal_math(quality: str = "audio") -> str:
    """Route the whole surface through the baked signal-math cores.

    Bakes the cores HERE rather than letting the first call trigger it.
    Baking measures each core against a reference, which needs a numeric
    backend; if the first call happens inside an SSA capture -- exactly what a
    standard object does -- the bake lands under the SSA backend and dies with
    "No tensor backend available for tensor creation". Warming at the switch
    keeps the cores a property of the choice rather than of whoever calls
    first.
    """

    from ..signal_math import signal_math

    global _IMPLEMENTATION, _QUALITY
    signal_math(str(quality))
    _IMPLEMENTATION, _QUALITY = "signal_math", str(quality)
    return _IMPLEMENTATION


def use_backend_operator() -> str:
    """Route the whole surface back through the backend's own operator."""

    global _IMPLEMENTATION
    _IMPLEMENTATION = "operator"
    return _IMPLEMENTATION


def active_implementation() -> tuple[str, str]:
    return _IMPLEMENTATION, _QUALITY


def _surface():
    from ..signal_math import signal_math

    return signal_math(_QUALITY)


def _dispatch(name: str, value):
    """One place where the choice is made, so every method reads the same."""

    if _IMPLEMENTATION == "signal_math":
        return getattr(_surface(), name)(value)
    return value._apply_operator(name, value, None)


def sin(self) -> "AbstractTensor":
    return _dispatch("sin", self)


def cos(self) -> "AbstractTensor":
    return _dispatch("cos", self)


def tan(self) -> "AbstractTensor":
    return _dispatch("tan", self)


def asin(self) -> "AbstractTensor":
    return _dispatch("asin", self)


def acos(self) -> "AbstractTensor":
    return _dispatch("acos", self)


def atan(self) -> "AbstractTensor":
    return _dispatch("atan", self)


def sinh(self) -> "AbstractTensor":
    return _dispatch("sinh", self)


def cosh(self) -> "AbstractTensor":
    return _dispatch("cosh", self)


def tanh(self) -> "AbstractTensor":
    return _dispatch("tanh", self)


def asinh(self) -> "AbstractTensor":
    return _dispatch("asinh", self)


def acosh(self) -> "AbstractTensor":
    return _dispatch("acosh", self)


def atanh(self) -> "AbstractTensor":
    return _dispatch("atanh", self)


# Derived via identities. The signal-math surface derives these the same way
# from its own cores, so the identity is stated once here and the difference
# between the routes stays confined to the primitives above.

def sec(self) -> "AbstractTensor":
    return self.cos() ** -1


def csc(self) -> "AbstractTensor":
    return self.sin() ** -1


def cot(self) -> "AbstractTensor":
    return self.cos() / self.sin()


def sech(self) -> "AbstractTensor":
    return self.cosh() ** -1


def csch(self) -> "AbstractTensor":
    return self.sinh() ** -1


def coth(self) -> "AbstractTensor":
    return self.cosh() / self.sinh()


def sinc(self) -> "AbstractTensor":
    return self.sin() / self

def deg2rad(degrees: float | "AbstractTensor") -> "AbstractTensor":
    if type(degrees) in (float, int):
        from ..abstraction import AbstractTensor
        degrees = AbstractTensor.get_tensor(degrees)
    return degrees * (degrees.pi() / 180.0)

def rad2deg(radians: float | "AbstractTensor") -> "AbstractTensor":
    if type(radians) in (float, int):
        from ..abstraction import AbstractTensor
        radians = AbstractTensor.get_tensor(radians)
    return radians * (180.0 / radians.pi())
