"""N-double arithmetic owned by a precision-aware tensor wrapper.

An extended value stores an ordinary tensor whose last dimension carries its
limbs, interleaved at stride ``limbs``: the value it denotes is the exact sum
of them, and every limb is a plain slice of the base dtype. ``Precision`` owns
that representation so an unaware tensor operation cannot accidentally treat
limbs as ordinary channels. Every numerical step is still built from the base
tensor's ``+``, ``-`` and ``*`` operations, so operator dispatch and recording
remain backend-neutral.

Limbs as CHANNELS is what makes the storage fit existing tensor backends. The
wrapper remains visible until an explicit collapse, while its payload is an
ordinary wider tensor. A four-channel pixel at two limbs is eight in the last
dimension read with stride two, so RGBA keeps striding and the precision rides
alongside it. Widening is a shape change and collapsing is a strided sum, both
ordinary tensor work.

``n`` is a parameter, not a fixed choice of two. One limb is ordinary
arithmetic and the shim declines; two is double-double; four is quad-double;
larger is an expansion in Shewchuk's sense. Each limb buys roughly 53 more
bits, so ``n`` limbs is about ``16*n`` decimal digits.

Three properties make this practical:

* the limbs are DATA-PARALLEL. Each element's limb chain is dependent, but
  elements are independent, so it vectorises and the cost is flop count rather
  than latency;
* the expansion is ordinary tensor arithmetic, so it RECORDS on the tape and
  the gradient flows through the chain that actually computed the value. There
  is no second derivative to maintain;
* every step preserves the exact sum. ``two_sum`` and ``two_product`` are
  error-free transformations, and the renormalisation below only ever replaces
  a pair by another pair with the same total, so the represented value is
  never silently damaged -- only the final truncation to ``n`` limbs discards
  anything, and it discards exactly what falls below the requested precision.

WHERE THIS STOPS BEING THE RIGHT TOOL. Multiplication is O(n**2) limb
products, and the renormalisation is O(n**2) sequential dependent sums. Around
eight limbs (~128 digits) that is already worse than carrying an integer
significand, and for the millions-of-digits regime the answer is fixed-point
limbs with FFT multiplication, not floating expansions. This is the right tool
from 2 to roughly 8.

The error-free transformations depend on the arithmetic NOT being
reassociated: an optimiser that "simplifies" ``a - (s - a)`` to zero deletes
the correction and silently collapses everything here to plain double. The
compiler must treat these as strict-FP, and recognising ``two_product`` to
fold it onto a fused multiply-add is a compiler identity -- both belong there,
not in workarounds here.
"""

from __future__ import annotations

from typing import Any, Sequence

#: Everything about a limb that depends on which float carries it, stated
#: once. Two elements exist because two worlds do: the CPU lanes hold limbs
#: in binary64, while WGSL and GLSL have no float64 at all, so their limbs
#: are binary32 and there are roughly twice as many of them for the same
#: precision.
#:
#: * ``split`` is Veltkamp's constant ``2**ceil(t/2) + 1`` for a t-bit
#:   significand -- splitting with the wrong one does not degrade Dekker's
#:   product, it invalidates the theorem.
#: * ``digits_per_limb`` is ``t * log10(2)``, used to size orders and the
#:   exact-rational working precision.
#: * ``max_limbs`` is what the repository lowering accepts; ``ladder`` is
#:   the widths a lane actually offers. binary64 walks 2/3/4; binary32
#:   walks 2/4/6/8 -- even steps, because each rung is priced against the
#:   binary64 rung it replaces (two f32 limbs per f64 limb) and odd rungs
#:   would be tiers nothing on the other side corresponds to.
#: * ``sensible_limit`` is where the expansion stops being the right
#:   representation at all (see the module note); it scales with limb
#:   count, not with carried bits, because the O(n**2) cost does.
LIMB_ELEMENTS: dict[str, dict] = {
    "float64": {
        "split": 134217729.0,  # 2**27 + 1
        "digits_per_limb": 15.95,
        "max_limbs": 4,
        "ladder": (2, 3, 4),
        "sensible_limit": 8,
    },
    "float32": {
        "split": 4097.0,  # 2**12 + 1
        "digits_per_limb": 7.22,
        "max_limbs": 8,
        "ladder": (2, 4, 6, 8),
        "sensible_limit": 16,
    },
}

_ELEMENT_SPELLINGS = {
    "float64": "float64", "f64": "float64", "double": "float64",
    "float32": "float32", "f32": "float32", "single": "float32",
}


def limb_element_facts(element: Any = None) -> dict:
    """The facts row for a limb element.

    ``None`` means the value never had a dtype stamped, which today is
    always the binary64 default the scalar path assumes. A PRESENT but
    unrecognised element is refused: guessing a splitting constant produces
    plausible numbers whose residuals are quietly wrong.
    """

    if element is None:
        return LIMB_ELEMENTS["float64"]
    name = _ELEMENT_SPELLINGS.get(str(element).casefold())
    if name is None:
        raise ValueError(
            f"no limb arithmetic facts for element {element!r}; "
            "supported elements are float64 and float32"
        )
    return LIMB_ELEMENTS[name]


# Dekker's splitting constant for binary64: 2**27 + 1. The binary64 alias
# survives because the eager path below IS binary64; element-aware callers
# read ``limb_element_facts`` instead.
_SPLIT = LIMB_ELEMENTS["float64"]["split"]

# Beyond this the expansion is the wrong representation; see the module note.
SENSIBLE_LIMIT = LIMB_ELEMENTS["float64"]["sensible_limit"]

def limbs_for_digits(digits: int, element: Any = None) -> int:
    """Limbs needed to carry ``digits`` decimal digits."""

    import math

    facts = limb_element_facts(element)
    needed = int(math.ceil(float(digits) / facts["digits_per_limb"]))
    return int(min(max(needed, 1), facts["sensible_limit"]))


# --------------------------------------------------------------------------
# the representation: limbs are CHANNELS, in the last dimension


def limb(value: Any, index: int, limbs: int) -> Any:
    """Limb ``index`` of an interleaved value -- stride ``limbs``, last axis.

    Interleaved rather than planar: every step of an expansion uses one
    value's limbs together, so they belong adjacent. A four-channel pixel at
    two limbs is eight in the last dimension read with stride two, which is
    why RGBA keeps working -- the channels are still there, just striding.
    """

    if int(limbs) <= 1:
        return value
    return value[..., int(index)::int(limbs)]


def interleave(parts: Sequence) -> Any:
    """``k`` parts of shape ``(..., C)`` into one ``(..., C*k)``."""

    from .abstraction import AbstractTensor

    parts = list(parts)
    if len(parts) == 1:
        return parts[0]
    stacked = AbstractTensor.stack(parts, dim=-1)
    shape = list(stacked.shape)
    return stacked.reshape(*shape[:-2], int(shape[-2]) * int(shape[-1]))


def widen(value: Any, limbs: int) -> Any:
    """Promote to ``limbs`` limbs: the value, then zeros beside it."""

    limbs = int(limbs)
    if limbs <= 1:
        return value
    zero = plain(value, "mul", 0.0)
    wide = interleave([value] + [zero] * (limbs - 1))
    wide.limbs = limbs
    return wide


def narrow(value: Any, limbs: int) -> Any:
    """Collapse the limbs back to one channel per value."""

    limbs = int(limbs)
    if limbs <= 1:
        return value
    total = limb(value, 0, limbs)
    for index in range(1, limbs):
        total = plain(total, "add", limb(value, index, limbs))
    total.limbs = 1
    return total


# --------------------------------------------------------------------------
# error-free transformations


def plain(left: Any, op: str, right: Any) -> Any:
    """One operator at a SINGLE limb -- the calculator's primitive layer.

    The transformations below are the IMPLEMENTATION of extended precision, so
    they cannot themselves be extended. They operate on limb slices, which are
    ordinary tensors, and dispatch only diverts for a ``Precision`` -- so this
    is a plain operator call and cannot re-enter. Spelling it out says the
    arithmetic here is deliberately primitive rather than incidentally so.
    """

    from .abstraction import AbstractTensor

    return AbstractTensor._apply_operator(left, op, left, right)


def two_sum(a, b):
    """Knuth: ``a + b == s + e`` exactly, for any a and b."""

    s = plain(a, "add", b)
    shifted = plain(s, "sub", a)
    return s, plain(plain(a, "sub", plain(s, "sub", shifted)), "add",
                    plain(b, "sub", shifted))


def _split(a):
    """Dekker: halve a significand into non-overlapping pieces."""

    c = plain(a, "mul", _SPLIT)
    high = plain(c, "sub", plain(c, "sub", a))
    return high, plain(a, "sub", high)


def two_product(a, b):
    """Dekker: ``a * b == p + e`` exactly.

    The splitting form, because it needs no operation the base dtype lacks.
    A backend with a fused multiply-add can do this in two instructions; that
    substitution is the compiler's job.
    """

    p = plain(a, "mul", b)
    ah, al = _split(a)
    bh, bl = _split(b)
    error = plain(plain(ah, "mul", bh), "sub", p)
    error = plain(error, "add", plain(ah, "mul", bl))
    error = plain(error, "add", plain(al, "mul", bh))
    return p, plain(error, "add", plain(al, "mul", bl))


# --------------------------------------------------------------------------
# expansions


def renormalise(terms: Sequence, limbs: int) -> list:
    """Reduce a pile of terms to ``limbs`` non-overlapping ones.

    Sweeping ``two_sum`` along the sequence pushes each rounding error one
    place down while leaving the TOTAL untouched, so repeating the sweep sorts
    magnitude into place without a single branch -- which is what makes this
    usable on a tensor, where the classic zero-skipping compress cannot go.

    The tail below ``limbs`` is folded into the last kept limb rather than
    dropped, so the result is the closest representable value rather than a
    truncation.
    """

    working = list(terms)
    if not working:
        return []

    # Distillation, branch-free. A FORWARD sweep is the wrong primitive: it
    # only ever lets a term see its neighbour, so m terms need m sweeps and
    # the cost goes quadratic. A BACKWARD accumulation lets the head see every
    # term in one pass, and its tail holds the exact remainder -- so repeating
    # it peels off one correctly-rounded limb at a time, O(m) per limb.
    kept = []
    rest = working
    for _ in range(limbs):
        if not rest:
            break
        carry = rest[-1]
        tail = [None] * (len(rest) - 1)
        for index in range(len(rest) - 2, -1, -1):
            carry, tail[index] = two_sum(rest[index], carry)
        kept.append(carry)
        rest = tail
    while len(kept) < limbs:
        kept.append(plain(kept[-1], "mul", 0.0))
    # Whatever is left lies below the requested precision; folding it into the
    # last limb keeps the result the nearest representable value.
    for leftover in rest:
        kept[-1] = plain(kept[-1], "add", leftover)
    return kept


def add_expansions(left: Sequence, right: Sequence, limbs: int) -> list:
    """Exact sum of two expansions, renormalised to ``limbs``."""

    return renormalise(list(left) + list(right), limbs)


def negate(terms: Sequence) -> list:
    return [plain(term, "mul", -1.0) for term in terms]


def multiply_expansions(left: Sequence, right: Sequence, limbs: int) -> list:
    """Every limb against every limb, exactly, then renormalised.

    Each partial product contributes BOTH halves of its error-free form, so no
    information is lost before the renormalisation decides what fits.
    """

    pieces = []
    for a in left:
        for b in right:
            high, low = two_product(a, b)
            pieces.append(high)
            pieces.append(low)
    return renormalise(pieces, limbs)


def _lead(terms: Sequence):
    """An approximation of an expansion's value from its top limbs.

    Two are enough: the distillation guarantees the rest are below the second
    limb's rounding, so this is accurate to about a limb -- which is exactly
    what a long-division digit needs.
    """

    if len(terms) == 1:
        return terms[0]
    return plain(terms[0], "add", terms[1])


def divide_expansions(left: Sequence, right: Sequence, limbs: int) -> list:
    """Long division: take a leading quotient digit, subtract, repeat.

    Each pass gains one limb of the quotient, so ``limbs + 1`` passes give the
    requested precision with one to spare for the final renormalisation.
    """

    remainder = list(left)
    quotient = []
    for _ in range(limbs + 1):
        # The estimate must read the top TWO limbs, not just the first. The
        # branch-free distillation cannot skip zeros the way the classic
        # compress does, so an exact cancellation leaves a zero in the leading
        # slot with the real value one place down. Reading only slot zero then
        # yields a zero quotient digit and the pass is wasted -- which is why
        # division gained a limb only every other width.
        digit = plain(_lead(remainder), "truediv", _lead(right))
        quotient.append(digit)
        product = multiply_expansions([digit], right, limbs + 2)
        remainder = add_expansions(remainder, negate(product), limbs + 2)
    return renormalise(quotient, limbs)


# --------------------------------------------------------------------------
# carrying limbs on a tensor



def _flatten(nested):
    """Every leaf of a nested list, in order."""

    if isinstance(nested, list):
        for item in nested:
            yield from _flatten(item)
    else:
        yield nested


def limbs_of(value: Any, width: int, like: Any) -> list:
    """An operand as ``width`` ordinary tensors, one per limb.

    A tensor operand is assumed ALREADY widened -- the promotion happens once
    at the boundary and the width travels with the value after that, so an
    operator does not re-promote and cannot double-count. A scalar has no
    channels to stride, so it becomes its own leading limb and zeros.

    What comes back are ordinary tensors with nothing attached, which is why
    the arithmetic below cannot re-enter this: each one asks its operators for
    the default single limb.
    """

    if hasattr(value, "shape"):
        # A tensor says how wide it already is. Slicing a plain tensor at
        # stride ``width`` would hand back an EMPTY limb for every channel
        # past the first, which is the shape error that makes this the one
        # thing a tensor cannot be left to guess about itself.
        held = int(getattr(value, "limbs", 1) or 1)
        if held < width:
            value = widen(narrow(value, held) if held > 1 else value, width)
        return [limb(value, index, width) for index in range(width)]
    seed = limb(like, 0, width)
    zero = plain(seed, "mul", 0.0)
    return ([plain(zero, "add", value)]
            + [zero for _ in range(width - 1)])


def to_float_list(value: Any, limbs: int) -> list:
    """Every limb as plain Python floats, for handing to exact arithmetic."""

    return [limb(value, index, limbs).tolist() for index in range(int(limbs))]


# --------------------------------------------------------------------------
# the dispatch shim


_DIRECT = {"add", "iadd", "sub", "isub", "mul", "imul", "truediv", "itruediv"}
_REVERSED = {"radd", "rsub", "rmul", "rtruediv"}
_HANDLED = _DIRECT | _REVERSED | {"neg"}


def apply(op: str, left: Any, right: Any, *, limbs: int = 1,
          accumulator: Any = None, accumulate_output: bool = False):
    """The limb work, driven by ARGUMENTS, returning an ordinary tensor.

    ``limbs`` is what the caller will ACCEPT, which is not what the operands
    carry. Two 2-limb values have a 4-limb exact product, and down a chain
    that grows without bound, so renormalising to ``limbs`` IS the precision
    choice: the operands say what is available, this says what is kept.

    It is a parameter rather than ambient state because ambient state does not
    survive compilation -- whatever sets it is a store nothing in the program
    reads, so it is deleted as dead and every operator lowers single-limb while
    appearing to have honoured the request.

    The result is a TENSOR, not a pair or a wrapper. Limbs live in the last
    dimension at stride ``limbs``, so widening is a shape change and every
    return type stays what it was. That is what lets this sit in the operator
    dispatch at all.

    ``accumulator`` takes the exact intermediate instead of it being
    renormalised away, so a chain pays the truncation once at the end rather
    than at every step; ``accumulate_output`` hands the accumulator back so
    the caller can keep chaining exactly.
    """

    if op not in _HANDLED:
        return None
    width = int(limbs or 1)
    if width <= 1 and accumulator is None:
        return None
    if width > SENSIBLE_LIMIT:
        raise ValueError(
            f"{width} limbs asked of an expansion; past about {SENSIBLE_LIMIT} "
            f"this representation is the wrong one -- carry an integer "
            f"significand in fixed point instead"
        )
    like = left if hasattr(left, "shape") else right
    if not hasattr(like, "shape"):
        return None

    if op == "neg":
        pieces = negate(limbs_of(left, width, like))
    else:
        first = limbs_of(left, width, like)
        second = limbs_of(right, width, like)
        if op in _REVERSED:
            first, second = second, first
        base = op[1:] if op in _REVERSED else op.lstrip("i")
        if base == "add":
            pieces = add_expansions(first, second, width)
        elif base == "sub":
            pieces = add_expansions(first, negate(second), width)
        elif base == "mul":
            pieces = multiply_expansions(first, second, width)
        else:
            pieces = divide_expansions(first, second, width)

    if accumulator is not None:
        for piece in pieces:
            accumulator.absorb(piece)
        if accumulate_output:
            return accumulator
        pieces = accumulator.to_expansion(width)
    result = interleave(pieces)
    result.limbs = width
    return result




def constant(like: Any, parts: Sequence, limbs: int) -> Any:
    """A constant whose limbs the caller derived, laid out as channels."""

    from .abstraction import AbstractTensor

    # ``like`` is ONE channel wide already -- a caller hands a limb slice, not
    # an interleaved value -- so slicing here would halve it again.
    seed = plain(like, "mul", 0.0)
    return interleave([plain(seed, "add", part) for part in parts[:limbs]])


# --------------------------------------------------------------------------
# The type that owns the width


class Precision:
    """An AbstractTensor plus the limb channels that give it its width.

    WHY THIS IS A TYPE and not a flag on every tensor. The limbs live in extra
    channels, and a general tensor operation cannot be taught to see past
    channels it does not know about. Measured on a two-limb value: ``mean``
    returned half of it, because it divided by the widened element count.
    ``sum``, ``max``, ``abs`` and ``reshape`` happened to be right, which is
    worse -- it means the failures are scattered and silent rather than
    systematic and loud.

    So the width is owned by a type that also owns the operators that
    understand it. A `Precision` supports the basic unary and binary
    arithmetic and nothing else; anything outside that either collapses at the
    boundary or raises. Being unsupported is a fine answer, being quietly
    halved is not.

    Anything that wants enduring precision therefore does two things: make the
    operands `Precision`, and stay within the basic operators. That is the
    whole contract.
    """

    #: How two widths combine. KEEP-WIDEST: an operation touching a 3-limb
    #: operand yields 3 limbs, so precision is never silently discarded by
    #: meeting a narrower value. The alternative rules -- take the narrower,
    #: or track significant digits -- are policies this deliberately does not
    #: choose for the caller, because dropping precision is the kind of thing
    #: that should be asked for rather than inherited.
    COMBINE = "widest"

    __slots__ = ("_terms", "limbs")

    @classmethod
    def __class_getitem__(cls, width):
        """Make authored ``Precision[n]`` annotations valid Python objects."""

        import types

        return types.GenericAlias(cls, width)

    def __init__(self, value: Any, limbs: int):
        # PLANAR STORAGE. The limbs are held as one contiguous tensor per
        # limb, not as channels strided through a single payload.
        #
        # Interleaving was never chosen on its merits: the compiler's own
        # arithmetic has always been one SSA value per limb, and the only
        # thing that needed a single object was an ARRAY, which a planar
        # stack provides just as well. What interleaving cost was paid on
        # every access -- a limb was a strided view, so every operation on
        # it read memory with a gap -- and measured on the eager path,
        # planar limbs run about 1.35x faster at every size from a
        # thousand elements to a million.
        #
        # It also stops the shape from lying. An interleaved payload is a
        # perfectly ordinary tensor of n*limbs elements, so an operation
        # that does not know about limbs computes confidently on the wrong
        # count -- which is exactly how ``mean`` came to return half of a
        # two-limb value. A planar stack keeps every limb the shape the
        # caller declared.
        #
        # A sequence is taken as the terms themselves; a tensor is taken
        # as an interleaved payload and split once, so every existing
        # caller and every stored artifact keeps working unchanged.
        self.limbs = int(limbs)
        if isinstance(value, (list, tuple)):
            self._terms = tuple(value)
        else:
            self._terms = tuple(
                limb(value, index, self.limbs) for index in range(self.limbs)
            )

    # -- crossing the boundary --------------------------------------------

    @classmethod
    def constant(cls, like: "Precision", parts: Sequence) -> "Precision":
        """A constant the caller derived to more digits than a double holds.

        The whole point of a wide chain is defeated if the values entering it
        are single doubles. An exact coefficient is not representable -- the
        sine core's second term is -1/6, whose remainder is -9.25e-18 -- so
        adding ``float(-1/6)`` at every Horner step reintroduces exactly the
        error the width was bought to avoid.
        """

        seed = like.term(0)
        return cls(constant(seed, parts, like.limbs), like.limbs)

    @classmethod
    def of(cls, value: Any, limbs: int = 2) -> "Precision":
        """Promote an ordinary tensor. This is where width is decided."""

        width = max(int(limbs), 1)
        zero = plain(value, "mul", 0.0)
        return cls([value] + [zero] * (width - 1), width)

    def collapse(self) -> Any:
        """Back to an ordinary tensor, paying the rounding once."""

        total = self._terms[0]
        for term in self._terms[1:]:
            total = plain(total, "add", term)
        return total

    @property
    def value(self) -> Any:
        """The ordinary tensor this denotes -- COLLAPSED, never the wide one.

        Reaching for ``.value`` is what a caller naturally does, so it gives
        back something that is safe to hand anywhere: a plain tensor of the
        expected shape, rounded once. The interleaved form stays internal,
        because passing THAT to a general operation is the failure being
        contained -- it sees channels it cannot know are limbs, and divides by
        the wrong count.

        Losing the extra digits here is the point. It happens at a named
        boundary, in one place, rather than silently in whichever operation
        happened not to understand the layout.
        """

        return self.collapse()

    # -- layout: a boundary fact, declared per destination ---------------
    #
    # There is no universally best arrangement of limbs in a buffer, and
    # the measurements say so plainly. A compiled per-element kernel wants
    # ELEMENT-MAJOR ("interleaved"): element i's limbs land on one cache
    # line, and on the CPU lanes that measured 1.2x to 1.7x faster than
    # the alternative. A GPU wants LIMB-MAJOR ("blocked"): adjacent
    # invocations then read adjacent addresses instead of addresses a
    # stride apart, which is what coalescing rewards. The eager surface
    # wants neither, because it holds its limbs planar and never packs
    # them at all.
    #
    # So the layout is not a property of the value. It is a property of
    # the DESTINATION, chosen where the value crosses into one, and the
    # only thing this type owes is an exact conversion in both directions.

    def interleaved(self):
        """Element-major: element i, limb k at flat index ``i * limbs + k``.

        What every compiled kernel in this tree addresses today, and what
        the artifacts and feeds already on disk contain.
        """

        return interleave(list(self._terms))

    def blocked(self):
        """Limb-major: element i, limb k at flat index ``k * count + i``.

        Each limb contiguous. Offered because a dispatch-parallel
        destination reads it better, not because anything here prefers it.
        """

        from .abstraction import AbstractTensor

        return AbstractTensor.concat(
            [term.reshape(-1) for term in self._terms], dim=0,
        )

    @classmethod
    def from_interleaved(cls, value, limbs: int) -> "Precision":
        """Adopt an element-major buffer without copying its meaning."""

        return cls(value, int(limbs))

    @classmethod
    def from_blocked(cls, value, limbs: int) -> "Precision":
        """Adopt a limb-major buffer, splitting it back into terms."""

        from .abstraction import AbstractTensor

        flat = AbstractTensor.get_tensor(value).reshape(-1)
        width = max(int(limbs), 1)
        count = int(flat.shape[0]) // width
        return cls(
            [flat[position * count:(position + 1) * count]
             for position in range(width)],
            width,
        )

    @property
    def _value(self):
        """The interleaved payload, built on demand.

        Kept because the compiled ABI is interleaved -- a per-element
        kernel wants one element's limbs adjacent -- and because artifacts
        and feeds already on disk are written that way. It is a boundary
        format now, not the storage: nothing inside this type reads it.
        """

        return interleave(list(self._terms))

    def to_float_lists(self) -> list:
        return [term.tolist() for term in self._terms]

    # -- the representation ------------------------------------------------

    def term(self, index: int) -> Any:
        return self._terms[int(index)]

    def terms(self, width: int | None = None) -> list:
        width = self.limbs if width is None else int(width)
        if width == self.limbs:
            return list(self._terms)
        if width < self.limbs:
            return list(self._terms[:width])
        zero = plain(self._terms[0], "mul", 0.0)
        return list(self._terms) + [zero] * (width - self.limbs)

    def __repr__(self) -> str:
        return f"Precision(limbs={self.limbs})"

    def __getattr__(self, name):
        """Refuse every tensor method this type has not endorsed.

        The interleaved value is deliberately not reachable. Handing it to a
        general tensor operation is the whole failure being contained -- that
        operation sees channels it does not know are limbs, and ``mean``
        divides by the wrong count while ``sum`` happens to survive. A caller
        that wants ordinary tensor work should ``collapse()`` first and mean
        it.
        """

        raise AttributeError(
            f"Precision does not endorse {name!r}. Use the basic arithmetic "
            f"operators, or collapse() for an ordinary tensor -- reaching "
            f"past this hands limb channels to something that cannot see them."
        )

    # -- arithmetic --------------------------------------------------------

    @staticmethod
    def width_of(operand: Any) -> int:
        return operand.limbs if isinstance(operand, Precision) else 1

    @classmethod
    def dispatch(cls, op: str, left: Any, right: Any, *, limbs: int = 1,
                 accumulator: Any = None, accumulate_output: bool = False):
        """One operator at a stated width, on operands of any width."""

        if op not in _HANDLED:
            raise TypeError(
                f"Precision supports the basic arithmetic operators; {op!r} "
                f"is not one of them. Collapse first if that is what you mean "
                f"-- a wide value put through an operation that cannot see "
                f"its limbs comes back wrong rather than refused."
            )
        width = max(int(limbs or 1), cls.width_of(left), cls.width_of(right))
        # ``term(0)`` is ONE channel -- element-shaped, not interleaved --
        # so it must never be handed to ``limbs_of``, whose scalar branch
        # slices its template again and halves it (measured: a three-element
        # value against a scalar broadcast (3,) into (2,); a two-element one
        # broadcast by coincidence, which is why the defect survived its own
        # test). The scalar expansion is built here from the element-shaped
        # template directly: the leading limb carries the value, the rest
        # are exact zeros. ``limbs_of`` keeps its interleaved contract for
        # the callers that honour it.
        template = (left.term(0) if isinstance(left, Precision)
                    else right.term(0) if isinstance(right, Precision)
                    else (left if hasattr(left, "shape") else right))
        template_zero = plain(template, "mul", 0.0)

        def operand(value):
            if isinstance(value, Precision):
                return value.terms(width)
            if hasattr(value, "shape"):
                return limbs_of(value, width, value)
            return ([plain(template_zero, "add", value)]
                    + [template_zero for _index in range(width - 1)])

        if op == "neg":
            pieces = negate(operand(left))
        else:
            first, second = operand(left), operand(right)
            if op in _REVERSED:
                first, second = second, first
            base = op[1:] if op in _REVERSED else op.lstrip("i")
            if base == "add":
                pieces = add_expansions(first, second, width)
            elif base == "sub":
                pieces = add_expansions(first, negate(second), width)
            elif base == "mul":
                pieces = multiply_expansions(first, second, width)
            else:
                pieces = divide_expansions(first, second, width)
        if accumulator is not None:
            for piece in pieces:
                accumulator.absorb(piece)
            if accumulate_output:
                return accumulator
            pieces = accumulator.to_expansion(width)
        return cls(interleave(pieces), width)

    def __add__(self, other): return Precision.dispatch("add", self, other)
    def __radd__(self, other): return Precision.dispatch("add", other, self)
    def __sub__(self, other): return Precision.dispatch("sub", self, other)
    def __rsub__(self, other): return Precision.dispatch("rsub", self, other)
    def __mul__(self, other): return Precision.dispatch("mul", self, other)
    def __rmul__(self, other): return Precision.dispatch("mul", other, self)
    def __truediv__(self, other): return Precision.dispatch("truediv", self, other)
    def __rtruediv__(self, other): return Precision.dispatch("rtruediv", self, other)
    def __neg__(self): return Precision.dispatch("neg", self, None)

    # -- the rest of the basic surface, limb-correct ---------------------
    #
    # Everything below is built from the operators above and from ordinary
    # tensor methods applied to individual LIMBS, which are ordinary
    # tensors. What is never done is hand the interleaved payload to a
    # tensor operation that does not know about limbs: that reads the
    # channels as data and answers confidently from a fraction of the
    # value, which is the failure this whole type exists to prevent.

    def element_count(self) -> int:
        """How many VALUES this holds, never how many limbs."""

        from .abstraction import AbstractTensor

        return int(
            AbstractTensor.get_tensor(self.term(0)).reshape(-1).shape[0]
        )

    def _map_limbs(self, function) -> "Precision":
        """Apply an EXACT per-limb map: one that cannot round.

        Multiplying every limb by the same sign is exact and distributes
        over the expansion. Nothing that ROUNDS may use this -- rounding a
        limb alone loses its relationship to the ones below it, which is
        the entire content of the representation.
        """

        return Precision(
            interleave([function(term) for term in self.terms()]), self.limbs
        )

    def _slice_limbs(self, start: int, stop: int) -> "Precision":
        """Elements ``[start:stop]``, with each one's limbs kept together."""

        return Precision(
            interleave([
                term.reshape(-1)[start:stop] for term in self.terms()
            ]),
            self.limbs,
        )

    def sign(self):
        """The sign of the whole expansion, as an ordinary tensor.

        The leading limb decides: a distilled expansion keeps every tail
        below an ulp of the head, so no tail can outvote it.
        """

        return self.term(0).sign()

    def __abs__(self) -> "Precision":
        """Magnitude, by negating the WHOLE expansion where it is negative.

        Taking the absolute value of each limb separately is a different
        number: ``|a| + |b|`` is not ``|a + b|`` when the tail opposes the
        head, which for a distilled expansion is the usual case.
        """

        sign = self.sign()
        return self._map_limbs(lambda term: term * sign)

    def floor(self):
        """The integer part, as an ordinary tensor.

        Deliberately not a ``Precision``: an integer needs no limbs, and
        returning one would invite a caller to carry a tail that is
        exactly zero. The collapsed sum decides and the residual the
        collapse discarded corrects it -- a value whose head sits just
        above an integer while its tail pulls it below belongs to the
        lower one, and the head alone cannot see that.
        """

        candidate = self.collapse().floor()
        residual = (self - candidate).collapse()
        return candidate - (residual < 0.0) * 1.0

    def sqrt(self) -> "Precision":
        """Newton's iteration, which is a fixed point of the answer.

        Needs only multiply, add and divide, so it is exactly as wide as
        the expansion it runs on -- no core, no interval, no table. The
        double seed carries about sixteen digits and every step doubles
        them, so the count is read from the width rather than fixed, plus
        one for the seed's own last bit.
        """

        import math

        seed = Precision.of(self.collapse().sqrt(), self.limbs)
        steps = max(1, int(math.ceil(math.log2(max(self.limbs, 1))))) + 1
        root = seed
        for _step in range(steps):
            root = (root + self / root) * 0.5
        return root

    def __pow__(self, exponent) -> "Precision":
        """Integer powers by repeated squaring, and nothing else.

        A fractional power is a transcendental and belongs to the signal
        cores, which state the interval they are valid on and measure
        their own error. Answering one here would mean collapsing to a
        double and discarding every limb the caller asked for.
        """

        if isinstance(exponent, float) and exponent.is_integer():
            exponent = int(exponent)
        if not isinstance(exponent, int):
            raise TypeError(
                f"Precision supports integer powers; {exponent!r} is a "
                "transcendental one -- take it through the signal cores, "
                "which state the interval they are valid on"
            )
        one = Precision.of(self.collapse() * 0.0 + 1.0, self.limbs)
        if exponent == 0:
            return one
        if exponent < 0:
            return one / (self ** -exponent)
        result, base, remaining = None, self, exponent
        while remaining:
            if remaining & 1:
                result = base if result is None else result * base
            remaining >>= 1
            if remaining:
                base = base * base
        return result

    # -- transcendentals: the cores, at this width -----------------------
    #
    # These are not arithmetic and cannot be built from it, so they are
    # not implemented here: they are ROUTED to the materialised proof
    # cores, which are the same programs the compiler lowers and which
    # measure their own error against an exact oracle. The width travels
    # with the argument, so a wide value gets a wide Horner chain rather
    # than a double's worth of answer widened afterwards.
    #
    # A core is valid on ITS OWN INTERVAL and carries no range reduction,
    # so an argument outside that interval is refused by name rather than
    # pushed through a polynomial that approximates nothing there. That is
    # the same rule the compiled kernels follow -- their router falls back
    # to the eager surface beyond the radius -- and it is deliberate: a
    # silently extrapolated core returns a plausible number, which is the
    # one outcome this type exists to prevent. Reduction belongs to the
    # caller, or to the signal surface that owns turns and binades.

    def _core(self, name: str) -> "Precision":
        from .signal_symbolic import CORE_RADII, evaluate_proof

        radius = CORE_RADII.get(name)
        if radius is None:
            raise AttributeError(
                f"no proof core is registered for {name!r}"
            )
        magnitude = abs(self).collapse()
        try:
            worst = max(
                abs(float(each))
                for each in _flatten(magnitude.tolist())
            )
        except (TypeError, ValueError):
            worst = float("inf")
        if worst > float(radius):
            raise ValueError(
                f"{name}: argument reaches {worst!r}, outside the core's "
                f"proven interval +-{radius}. Reduce the range first -- a "
                "core evaluated beyond its interval approximates nothing "
                "and would return a plausible wrong answer at every limb"
            )
        return evaluate_proof(name, self, self.limbs)

    def sin(self) -> "Precision":
        return self._core("sin")

    def cos(self) -> "Precision":
        return self._core("cos")

    def tan(self) -> "Precision":
        return self._core("tan")

    def exp(self) -> "Precision":
        return self._core("exp")

    def expm1(self) -> "Precision":
        return self._core("expm1")

    def log1p(self) -> "Precision":
        return self._core("log1p")

    def sinh(self) -> "Precision":
        return self._core("sinh")

    def cosh(self) -> "Precision":
        return self._core("cosh")

    def tanh(self) -> "Precision":
        return self._core("tanh")

    def atan(self) -> "Precision":
        return self._core("atan")

    def sum(self) -> "Precision":
        """Add every element WITHOUT collapsing on the way.

        The reduction that made this type necessary. Summing a collapsed
        expansion rounds each element to one double before adding, so a
        two-limb field reports a one-limb total and the width bought
        nothing -- measured at eight parts in ten to the seventeenth on a
        three-element field, which is one whole double of the answer
        discarded.

        Done by PAIRWISE HALVING: fold the upper half onto the lower with
        one wide add and repeat. Every add is the exact expansion add, so
        the total is exact; there are log2(n) of them and each is
        vectorised, so it costs a handful of passes rather than one
        sequential step per element. An odd element is carried into the
        next round untouched rather than dropped or counted twice.
        """

        from .abstraction import AbstractTensor

        held = self
        count = held.element_count()
        while count > 1:
            half = count // 2
            folded = (
                held._slice_limbs(0, half)
                + held._slice_limbs(half, 2 * half)
            )
            if count % 2:
                tail = held._slice_limbs(count - 1, count)
                held = Precision(
                    interleave([
                        AbstractTensor.concat([left, right], dim=0)
                        for left, right in zip(folded.terms(), tail.terms())
                    ]),
                    held.limbs,
                )
                count = half + 1
            else:
                held = folded
                count = half
        return held

    def mean(self) -> "Precision":
        """The average, as wide as the values averaged.

        Divides by the ELEMENT count. Dividing by the widened count --
        elements times limbs -- is what returned half of a two-limb value
        in the measurement this class's docstring opens with, and it
        looked like an ordinary answer.
        """

        return self.sum() / float(max(self.element_count(), 1))

