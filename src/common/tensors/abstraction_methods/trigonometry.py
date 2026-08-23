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


#: The coordinator serving compiled kernels, once one has been installed.
_LAUNCHER = None


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


def use_signal_kernels(quality: str = "draft") -> str:
    """Route the surface through COMPILED kernels the bank has admitted.

    ``use_signal_math`` selects the eager cores: the authoring surface, which
    is the same mathematics interpreted one tensor operation at a time. This
    selects the COMPILED form of that same mathematics. The bank builds each
    kernel from the baked cores, verifies it against the kernel's own Python
    reference, and refuses to serve a variant that did not match -- so an
    admitted kernel is one that demonstrated agreement, not one that merely
    emitted.

    A name the bank has no kernel for falls back to the EAGER surface rather
    than to the backend operator. The choice stays "our mathematics, compiled
    or interpreted"; it never silently reverts to a borrowed libm, which is
    the failure this pack exists to end.

    Gradients: the launched forward returns a fresh tensor, so the tape does
    not see the chain that produced it. The bank carries the VJP kernels
    (``sin_vjp`` and friends) but nothing binds them to autograd yet, so this
    route is forward-only. ``use_signal_math`` remains the differentiable one.
    """

    from ....compiler.kernel_bank import LaunchCoordinator, open_signal_bank

    global _IMPLEMENTATION, _QUALITY, _LAUNCHER
    _LAUNCHER = LaunchCoordinator(open_signal_bank(str(quality)))
    _KERNEL_KEYS.clear()
    _IMPLEMENTATION, _QUALITY = "signal_kernels", str(quality)
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


#: Method name -> the bank's spec key, resolved once per installed launcher.
_KERNEL_KEYS: dict = {}


def _kernel_key(name: str):
    """The bank's key for a surface method, or ``None``.

    The bank keys a kernel by what it IS -- ``sin_d32``, the sine core at
    thirty-two digits -- while the surface asks by what the caller SAID,
    ``sin``. Looking the bare name up directly matched nothing, so the
    ``signal_kernels`` route silently served every call from the eager
    surface while claiming to be the compiled one. When several digit
    variants exist the highest wins: the caller chose this route for
    fidelity, and the profiler, not this table, is where a cheaper variant
    would earn its place.
    """

    if name in _KERNEL_KEYS:
        return _KERNEL_KEYS[name]
    specs = _LAUNCHER.bank.specs
    if name in specs:
        _KERNEL_KEYS[name] = name
        return name
    prefix = name + "_d"
    digits = [
        int(key[len(prefix):]) for key in specs
        if key.startswith(prefix) and key[len(prefix):].isdigit()
    ]
    _KERNEL_KEYS[name] = f"{prefix}{max(digits)}" if digits else None
    return _KERNEL_KEYS[name]


def _routed(name: str, value):
    """One routed launch, or ``None`` when the bank has no kernel by that name.

    Returning ``None`` rather than raising is what lets a partial pack be
    useful: the bank carries kernels for a handful of the surface today, and
    the rest must keep working through the eager cores.
    """

    if _LAUNCHER is None:
        return None
    name = _kernel_key(name)
    if name is None:
        return None

    import numpy as np

    from ..abstraction import AbstractTensor

    # ``__array__`` hands back the buffer. ``tolist()`` would materialise one
    # Python float per element first, which costs more than the kernel it is
    # feeding by two orders of magnitude -- measured at 136 ns/element against
    # the kernel's 2.5, on a route whose whole purpose is to be the fast one.
    source = np.asarray(value, dtype=np.float64)
    flat = np.ascontiguousarray(source.reshape(-1))
    produced = _LAUNCHER.launch(
        name, x=flat, y=np.zeros_like(flat), n=int(flat.size),
    )
    settled = np.asarray(produced, dtype=np.float64).reshape(source.shape)
    return AbstractTensor.get_tensor(settled, like=value)


def _dispatch(name: str, value):
    """One place where the choice is made, so every method reads the same."""

    if _IMPLEMENTATION == "signal_kernels":
        routed = _routed(name, value)
        if routed is not None:
            return routed
        return getattr(_surface(), name)(value)
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
