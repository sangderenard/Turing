# Rational tensor composition design

Status: proposed design, no implementation yet  
Date: 2026-09-24

## Decision

A simple rational type is feasible for `AbstractTensor`.

Its honest guarantee is a **structural quotient**: numerator and denominator
remain separate through supported arithmetic, and division occurs only at an
explicit collapse boundary. It is not an arbitrary-precision rational-number
system. Fixed-width tensor elements can overflow, and the repository has no
backend-neutral big-integer tensor or tensor-GCD operation. Those facts must
remain visible in the contract.

The primary use case is repeated division whose quotient cannot be resolved
inside a fixed precision expansion. Rational is therefore a lazy division
carrier, not principally a convenience spelling for small integer fractions.
Once a value enters a rational-containing type, every later division remains
in numerator/denominator form. `Precision` protects the component arithmetic;
rational structure preserves the unresolved quotient beyond the limb budget.

For example, repeated division does not execute a sequence of rounded
quotients:

```text
x = a / b
x = x / (c / d)
x = x / (e / f)
```

It carries:

```text
x = (a*d*f) / (b*c*e)
```

and pays for one division only when `quotient()` or `collapse()` is explicitly
requested. This can retain information that no fixed sum of limbs can hold as
an already-evaluated quotient. It does not remove rounding or overflow from
the numerator and denominator multiplications themselves, which is why the
coefficient domain and cancellation rules remain part of the contract.

The implementation will use one concrete type for each unordered combination
of rational, complex, and precision features. Promotion is the union of the
operands' features, so operand order cannot change the result type.

## Existing systems that govern the design

`Precision` establishes the numerical-substrate rules:

- a wrapper owns representation-specific operators;
- promotion and collapse are explicit boundaries;
- the widest precision survives a mixed operation;
- unsupported operations refuse rather than misread the representation; and
- compiler lowering expands the wrapper into ordinary backend values before
  emission.

`ComplexPrecision` establishes the composition rule: complex algebra is
written over two already-correct real coefficient values. It does not
duplicate precision arithmetic.

Rational follows both rules. It owns quotient algebra, while its numerator
and denominator use the existing coefficient type unchanged.

## Canonical feature sets

The three features form seven non-empty unordered sets. The concrete runtime
representations are:

| Features | Canonical representation |
|---|---|
| complex | native complex `AbstractTensor` |
| precision | `Precision` |
| rational | `Rational` |
| complex + precision | `ComplexPrecision` |
| rational + precision | `RationalPrecision` |
| complex + rational | `ComplexRational` |
| complex + rational + precision | `ComplexRationalPrecision` |

An ordinary real `AbstractTensor` is the empty feature set.

For every binary operation, the result feature set is:

```text
features(result) = features(left) union features(right)
```

The result class is then selected from the table above. Precision width is an
orthogonal fact: any result containing precision uses the maximum width of
its operands.

This realizes rational precedence without discarding other features. A
rational mixed with a non-rational becomes a rational-containing composite;
the non-rational operand is embedded with denominator one. A complex value is
not reduced to a real value, and precision is not collapsed.

## Representations

### Rational

```text
Rational
    numerator: AbstractTensor
    denominator: AbstractTensor
```

### RationalPrecision

```text
RationalPrecision
    numerator: Precision
    denominator: Precision
    limbs: widest coefficient width
```

### ComplexRational

```text
ComplexRational
    real: Rational
    imag: Rational
```

### ComplexRationalPrecision

```text
ComplexRationalPrecision
    real: RationalPrecision
    imag: RationalPrecision
    limbs: widest coefficient width
```

Complex rational values are pairs of real rational coefficients, not one
ratio with a complex denominator. This matches `ComplexPrecision`, preserves
a meaningful real-denominator invariant, and avoids multiple structurally
different representations of the same complex number.

Each class is explicit. There is no dynamic class generation and no ordering
of wrapper nesting for callers to choose.

## Invariants

Every rational coefficient maintains these invariants:

1. Numerator and denominator are values in the same coefficient domain.
2. The denominator is nonzero at every represented element.
3. For ordered real coefficients, denominator signs are normalized positive
   elementwise by negating both components where needed.
4. Numerator and denominator shapes are broadcast-compatible, and `shape`
   reports their broadcast result.
5. A plain value promoted to rational is `value / 1` in the target coefficient
   domain.
6. Objects are immutable at the wrapper level. In-place dunders return a new
   value rather than mutating component identity.
7. Rational wrappers are unhashable. Structural pairs are not canonical until
   a supported normalization operation proves that they are.
8. Numerator and denominator components are finite. `NaN` and infinity are
   never valid stored rational components.
9. Every operation either proves its component arithmetic remains inside the
   coefficient domain's admitted limits or refuses before performing the
   unsafe arithmetic.

Denominator validation must never use an epsilon. Zero is an invalid rational
denominator, not a small floating-point condition. Eager construction checks
it. Compiled construction must carry a nonzero obligation or an explicit
runtime assertion; it must not silently weaken the invariant.

## Rational limits

Rational structure exists specifically to avoid asking IEEE `NaN` and
infinity to describe a mathematical condition after information has already
been lost. Limits are therefore part of the rational value's contract, not a
diagnostic performed after an operation.

### Three different limit questions

A rational value answers three separate questions:

1. **Structurally valid:** are numerator and denominator finite, and is the
   denominator proven nonzero?
2. **Operation-safe:** can the next numerator/denominator additions and
   multiplications execute in the coefficient domain without overflow,
   destructive underflow, or invalid precision transforms?
3. **Collapsible:** is the represented quotient inside the finite range and
   requested accuracy of the destination tensor type?

These questions must not be conflated. `1e300 / 1e-300` is a valid finite
rational pair even though its quotient cannot collapse to binary64. It may
continue through rational algebra. Conversely, two individually valid pairs
may not be safe to multiply if their component products exceed the available
exponent range.

### Limit record

Each rational coefficient carries a non-tensor `RationalLimits` fact record.
The value still owns exactly two numerical tensors; limits are scalar proof
metadata. At minimum the record contains conservative global enclosures over
all tensor elements:

```text
RationalLimits
    numerator_interval: [low, high]
    denominator_interval: [low, high], excluding zero
    numerator_min_nonzero_abs: optional lower magnitude proof
    denominator_min_abs: positive lower magnitude proof
    numerator_max_abs: finite upper magnitude proof
    denominator_max_abs: finite upper magnitude proof
    coefficient_element: int width, float32, float64, or Precision element
    precision_limbs: one unless the coefficient domain contains Precision
    component_finite: proven boolean
    denominator_nonzero: proven boolean
    quotient_interval: optional outward-rounded enclosure
    provenance: the boundary or operation that derived each fact
```

Intervals are enclosures, never sampled estimates. Integer endpoints use
exact Python integers. Floating endpoints are rounded outward. A missing fact
is unknown, not zero and not permission.

The interval may be global rather than elementwise. This is conservative but
keeps the promised two-tensor representation. A future specialized system may
carry tiled or elementwise limits as external analysis data; it must not
quietly add hidden numerical channels to the rational value.

### Sources of facts

Limit facts may come only from:

- exact literals and denominator-one promotion;
- declared input/extraction contracts;
- an explicit eager boundary scan using finite/min/max reductions;
- exact propagation from already-proven operand intervals; or
- a named compiler identity whose prerequisites are recorded.

They may not come from observing that a prior result happened not to be
`NaN`, from an epsilon clamp, from a few sample points, or from reconstructing
value identity by name or position.

Eager construction establishes missing facts once at the boundary and raises
before adopting a non-finite component or zero denominator. Compiled rational
sections require sufficient declared or derived facts before emission. If a
dynamic section cannot be proven safe without a runtime check, that check must
be an explicit section admission guard with a status channel owned by the
calling contract. A backend that cannot represent the guard reports a
shortfall; it does not emit an operation whose failure convention is
`NaN`/infinity.

### Propagation

Every rational operator derives result limits before doing component work.
Ordinary outward-rounded interval arithmetic supplies conservative bounds:

- negation reverses interval endpoints;
- addition/subtraction combine endpoint sums/differences;
- multiplication considers all endpoint products;
- reciprocal swaps numerator and denominator bounds after a numerator-nonzero
  proof; and
- quotient bounds are derived only when the denominator interval excludes
  zero.

For composite rational types, limits are carried independently for real and
imaginary rational coefficients. Complex values have component bounds, not an
ordered complex interval. Norm-squared bounds may be derived when division
needs them.

### Checked component arithmetic

Before forming a product or sum, the operation compares its derived bound
with the coefficient domain's admitted range:

- fixed-width integers use their exact minimum and maximum;
- float32/float64 use their finite exponent ranges and distinguish normal,
  subnormal, and zero;
- `Precision` uses its limb element's exponent range—the limbs extend
  significand precision, not exponent range—and its existing strict-FP
  obligations; and
- exact zero is permitted only when proven algebraically, not when a nonzero
  lower bound fell below the representable range.

An unsafe operation follows this order:

1. apply an exact, licensed cancellation;
2. attempt an exact common power-of-two rebalance when an exponent-range fact
   proves that scaling both numerator and denominator preserves every
   component;
3. recompute the operation bounds; then
4. raise `RationalLimitError` eagerly or report a compiler rational-limit
   shortfall before emission.

It never executes first and consults `isfinite` afterward. It never replaces
the result with a clamp, zero, `NaN`, or infinity.

### Exact common scaling

Numerator and denominator are homogeneous coordinates: multiplying both by
the same nonzero factor leaves the represented quotient unchanged. The only
automatic magnitude rebalance admitted initially is a shared power of two,
because binary scaling is exact when the existing precision pipeline's normal
exponent-range obligations are satisfied.

The limit record derives an integer shift window in which both components
remain finite and every proven nonzero component remains representable. A
shift may occur only inside that window. If the window is empty, the operation
refuses. Decimal scaling or division by a measured maximum is not a substitute
because either would round the components merely to avoid reporting the real
limit.

Every rebalance records its common exponent and the range facts that licensed
it. This uses the same discipline as the existing precision identity
`scaling_by_power_of_two`: without the exponent-range fact, the supposedly
exact scaling must not fire.

### Thoughtful cancellation

Cancellation needs both identity and admissibility:

- the two factors must be the same authoritative component identity;
- the cancelled factor must be proven nonzero;
- removing it must not discard a signed-zero, exceptional, or domain fact;
  rational components exclude the exceptions by invariant; and
- the receipt names the identity and nonzero proof that licensed the rewrite.

Limit facts make this stronger than relying on IEEE behavior. `x/x -> 1` is
not inferred because a hardware division happened to return something finite;
it fires because `x` is the same value and its interval excludes zero. If
either proof is missing, cancellation does not occur.

### Public limit surface

The rational family exposes read-only limit information:

```text
value.limits
value.is_structurally_valid
value.can_apply(operation, other=None)
value.can_collapse(dtype=None)
value.require(operation, other=None)
```

`can_apply` and `can_collapse` return proof results with reasons, not bare
optimistic booleans. `require` raises `RationalLimitError` with the failed
bound, coefficient domain, operation, and provenance. None of these evaluates
the rational quotient.

## Normalization policy

Scalar rational libraries commonly reduce by GCD after construction and
operations. That policy does not transfer directly to this tensor system:

- `AbstractTensor` has no sanctioned `gcd` operation;
- elementwise Euclidean loops have data-dependent iteration counts;
- hidden reduction would add substantial and backend-dependent work to every
  operator;
- it could force eager host synchronization; and
- fixed-width intermediate products can overflow before a late GCD repairs
  their size.

Therefore version one does **not** perform automatic GCD reduction and does
not claim a unique lowest-terms representation. Sign normalization and
denominator validation are still mandatory because operator semantics depend
on them.

The absence of general GCD reduction does not permit known factors to grow
needlessly. Exact structural cancellation is part of version one whenever
identity is proven rather than guessed. A numerator and denominator carrying
the same source/component identity may cancel subject to the nonzero
invariant. The compiler performs this through catalogued exact identities and
records why each cancellation fired. Eager code may use object identity or an
explicitly preserved component identity; it must not scan values and infer
equality.

Reciprocal is an O(1) structural operation: it swaps numerator and denominator
after establishing that the old numerator is nonzero. Division by a rational
uses that reciprocal and multiplication. No quotient is evaluated during
either operation.

A later integer-only `normalize()` may be added after a backend-neutral tensor
GCD exists. At that point multiplication and division should cross-cancel
before multiplying, which is the standard defense against avoidable integer
growth. No float or precision-expansion coefficient will pretend to support
GCD normalization.

## Arithmetic surface

The initial endorsed surface is deliberately closed and small:

- `+`, reflected `+`
- `-`, reflected `-`
- unary `-`
- `*`, reflected `*`
- `/`, reflected `/`
- integer `**`
- `reciprocal()`
- `collapse()`
- structural component access

For `a/b` and `c/d`, rational arithmetic is:

```text
add: (a*d + b*c) / (b*d)
sub: (a*d - b*c) / (b*d)
mul: (a*c)         / (b*d)
div: (a*d)         / (b*c)
neg: (-a)          / b
```

Division additionally requires the divisor numerator to be nonzero.

These formulas are evaluated with exact identity cancellation before new
products are formed. This matters most for the intended long division chains:
the system must not manufacture a large `x/x` factor and wait for a future
normalizer to notice it.

Integer powers use exponentiation by squaring on numerator and denominator;
a negative exponent swaps them after checking the numerator. Non-integer
powers refuse because they are not closed over rational values.

Complex composites use the same algebra already present in
`ComplexPrecision`, with rational coefficients:

```text
(a + b*i) * (c + d*i) = (a*c - b*d) + (a*d + b*c)*i

(a + b*i) / (c + d*i)
    real = (a*c + b*d) / (c*c + d*d)
    imag = (b*c - a*d) / (c*c + d*d)
```

Here each letter is itself `Rational` or `RationalPrecision`. Complex types
have no ordering comparisons.

Operations not explicitly endorsed raise with an instruction to collapse.
Reduction, indexing, reshape, comparison, and transcendental behavior are not
implicitly inherited from `AbstractTensor`.

## Promotion

Promotion must be centralized even though the concrete classes are manual.
Each dunder asks the same composition function for a target feature set and
target precision width before doing algebra. This prevents the existing
`Precision <op> ComplexPrecision` failure, where the left-hand wrapper commits
to its own dispatch before the more expressive right-hand type can absorb it.

Conceptually:

```text
describe(value) -> (features, limbs)
join(left, right) -> (features(left) union features(right), max(limbs))
promote(value, target_description) -> canonical concrete value
```

The helper selects types and performs representation-preserving promotion. It
does not perform arithmetic. Every result is the same canonical class whether
the richer operand appears on the left or right.

Required embeddings include:

- real to complex: `(x, 0)`;
- non-rational to rational: `x / 1`;
- narrow to precision: `Precision.of(x, width)`;
- rational to complex-rational: `(x, 0/1)`; and
- any combination of the above to the union feature set.

### Entry into the rational domain

Ordinary `AbstractTensor / AbstractTensor` retains its established meaning and
returns an ordinary tensor. Changing every division in the repository would
be an incompatible semantic change.

Automatic preservation begins at a declared rational boundary:

- `Rational.of(value)` means `value / 1`;
- `Rational.ratio(numerator, denominator)` constructs an unevaluated ratio;
- rational-containing source annotations promote arguments at function entry;
  and
- any operation with one rational-containing operand promotes the other
  operand and remains rational-containing thereafter.

Thus a division-heavy system opts into the domain once, at its inputs or its
first ratio, and every subsequent mixed operation preserves the quotient
automatically. No caller must re-wrap intermediate results.

## Boundaries

`components()` exposes wrapper-owned structure for compiler lowering. It does
not collapse values.

`quotient()` on `Rational` returns the ordinary tensor division
`numerator / denominator`. On `RationalPrecision`, it returns a `Precision`
division and therefore retains limb width.

Neither method is called implicitly by arithmetic, shape inspection,
serialization planning, compiler lowering, or promotion. The only automatic
behavior is preservation of quotient structure. Evaluation is always a named
boundary.

`collapse()` always returns an ordinary `AbstractTensor`:

- `Rational`: evaluate the quotient;
- `RationalPrecision`: evaluate the wide quotient, then collapse once;
- `ComplexRational`: collapse real and imaginary rational coefficients, then
  assemble a native complex tensor;
- `ComplexRationalPrecision`: evaluate each wide rational coefficient,
  collapse each once, then assemble a native complex tensor.

The distinction between `quotient()` and `collapse()` prevents precision from
being discarded merely because rational structure was removed.

## Exactness statement

The type preserves fraction **structure** exactly. Numerical exactness depends
on its coefficient domain:

- integer `Rational` operations are exact only while all fixed-width integer
  intermediates remain in range;
- floating `Rational` operations retain a deferred quotient but their
  numerator and denominator arithmetic still rounds;
- `RationalPrecision` performs coefficient arithmetic at its owned limb width
  and delays the final division, but it remains a finite expansion; and
- none of these is a substitute for an arbitrary-precision integer rational.

The useful precision gain is nevertheless real: a numerator/denominator pair
can denote a quotient that cannot be stored as a finite limb expansion. The
claim is not that the pair has infinite-precision coefficients; it is that the
division rounding has not happened yet. Repeated division continues extending
that structural representation until the explicit evaluation boundary.

No documentation, repr, or compiler receipt should call the type
arbitrary-precision or universally exact.

## Autograd

Rational arithmetic is expressed only through existing component operators.
The gradient tape therefore records the arithmetic that actually constructs
the numerator and denominator. A final `quotient()` or `collapse()` records
the division boundary normally.

Integer rational tensors do not claim useful gradients. Floating and
precision composites inherit the behavior of their coefficient operations;
there is no parallel derivative implementation.

## Compiler design

No backend receives a rational object or a new machine-level rational opcode.
As with precision, wrappers are source and repository-SSA facts that lower to
ordinary component values before emission.

The lowering order is outer algebra to inner substrate:

```text
complex-rational-precision
    -> complex algebra over rational-precision coefficients
    -> rational algebra over precision numerator/denominator coefficients
    -> existing precision_* operations
    -> ordinary backend SSA values
```

The compiler must record component identities, not rediscover them by name or
position. Suggested durable records parallel the existing precision receipt:

- rational value id -> numerator id, denominator id;
- complex value id -> real id, imaginary id;
- precision value id -> ordered limb ids; and
- one feature-set descriptor and limb width on each authored boundary.

Each rational section contract also records its admitted limits:

- source and result component intervals;
- denominator-nonzero and component-finite proofs;
- coefficient dtype limits;
- every common power-of-two rebalance and its allowed shift window;
- each cancelled identity and its nonzero proof;
- collapse destination and quotient-range proof, when collapse is present;
  and
- unresolved/unsafe bounds as explicit backend shortfalls.

This follows the existing precision catalogue, where exponent- and
operand-range facts license exact rewrites and their absence prevents the
rewrite. Rational lowering extends that fact discipline; it does not infer
safety from a backend's eventual `isfinite` result.

Direct wrapper tests are required. A test that manually spells component
algebra proves the inner substrate only; it does not prove wrapper ingestion,
promotion, call ABI, return publication, or collapse.

## Initial exclusions

The first version does not include:

- automatic GCD reduction;
- arbitrary-precision integers;
- hashability;
- non-integer powers;
- transcendental functions that pretend to return rationals;
- implicit conversion to a native tensor;
- complex ordering;
- silent zero-denominator repair;
- epsilon denominator clamps;
- post-operation `isfinite` recovery;
- `NaN` or infinity as a rational limit/status representation;
- unchecked component overflow or destructive underflow;
- backend-specific rational implementations; or
- a generic nesting API that lets callers choose competing wrapper orders.

## Test contract

### Promotion

- Exhaust every ordered pair of the seven feature sets for `+`, `-`, `*`,
  and `/`.
- Assert the canonical union-feature result class.
- Repeat with operands reversed and verify the same class and value.
- Pin widest-limb retention for every precision-containing result.
- Preserve the existing tensor/precision and tensor/complex behavior.

### Rational algebra

- Compare small integer cases against Python `Fraction` without approaching
  dtype limits.
- Cover negative denominators, zero numerators, scalar/tensor promotion,
  broadcasting, reverse operators, negative integer powers, and refusal of
  zero denominators and non-integer powers.
- Run long repeated-division chains against a Python `Fraction` oracle and
  compare with the same chain that evaluates every intermediate quotient.
  The rational path must collapse once and retain information the eager
  quotient chain loses.
- Pin reciprocal as a component swap and prove that no component division is
  recorded before explicit evaluation.
- Pin every exact identity cancellation with the component identities and
  nonzero fact that licensed it.
- Cover known-safe, exactly rebalanced, structurally-valid-but-not-collapsible,
  and operation-unsafe limit cases for every coefficient dtype.
- Prove that unsafe eager operations raise before component arithmetic and
  that compiled unsafe sections report a pre-emission shortfall.
- Exercise intervals that approach zero from one side without crossing it;
  denominator validity must follow the proved interval, never an epsilon.
- Verify that exact common power-of-two scaling leaves the collapsed value and
  every component identity receipt consistent.
- Demonstrate that unreduced representations compare numerically when
  comparison support is introduced, while remaining unhashable.

### Composite algebra

- Compare complex-rational operations against pairs of Python `Fraction`.
- Show that `RationalPrecision` retains information lost by an early ordinary
  division.
- Cover all four rational-containing classes through `components`,
  `quotient` where applicable, and `collapse`.

### Compiler

- Compile each concrete wrapper directly through the sanctioned
  `lower_ast_source_to_ssa` entry.
- Verify exact component identity records across parameters, calls, returns,
  and collapse boundaries.
- Verify that rational lowering feeds the existing precision pipeline for
  precision-containing composites.
- Require zero surviving abstract rational or precision operators before
  backend emission.

## Implementation sequence

1. Add promotion-matrix tests and repair the existing
   `Precision <op> ComplexPrecision` asymmetry.
2. Add the shared feature descriptor and canonical promotion helper without
   changing ordinary `AbstractTensor` arithmetic.
3. Add `RationalLimits`, outward-rounded interval propagation, dtype limits,
   and pre-operation refusal tests.
4. Add `Rational` and its focused eager algebra tests.
5. Add reciprocal, proven structural cancellation, exact power-of-two
   rebalancing, and repeated-division
   regressions before expanding the surface.
6. Add `RationalPrecision` by delegating coefficient work to `Precision`.
7. Add `ComplexRational` and `ComplexRationalPrecision` by composing their
   real coefficient types exactly as `ComplexPrecision` does.
8. Add direct source descriptors, limit facts, and rational component identity
   records.
9. Lower complex composition, then rational structure, then reuse the existing
   precision transaction.
10. Add call/return/compiler execution regressions for every composite type.

Each step leaves unsupported combinations refusing loudly until its canonical
type and tests exist. No temporary collapse fallback is permitted.

## Feasibility conclusion

The eager type family is straightforward because all required arithmetic is
already expressible through `AbstractTensor` and `Precision`. The intended
repeated-division use case is exactly where structural rationals add something
limbs cannot: they postpone quotient rounding instead of buying a finite
number of additional quotient bits. The difficult part is not the fraction
formulas; it is preserving canonical promotion, proven cancellation, and
component identity through the compiler.

The design is feasible if the first milestone is defined as structural
rationals with explicit collapse and without automatic GCD. A promise of
canonical, overflow-free, arbitrary-precision rationals would require a new
integer substrate and is outside this simple design.
