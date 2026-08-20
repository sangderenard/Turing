"""Loop interchange for reduction nests: the planner's answer to stride.

The measured problem (docs/KERNEL_BANK_DESIGN.md, the vectorization arc):
a gemm-shaped nest carries its reduction INNERMOST --

    for j in range(n):
        total = 0.0
        for p in range(k):
            total = total + A[i * k + p] * B[p * n + j]
        C[i * n + j] = alpha * total + beta * C[i * n + j]

-- so the hot loop walks ``B`` at stride ``n``. No vectorizer profits from
that ordering regardless of aliasing facts (measured: datalayout + derived
noalias moved the 64^3 core only ~0.85 -> 0.97 GF/s). The interchanged,
accumulator-promoted form puts the unit-stride axis innermost:

    for j in range(n):
        C[i * n + j] = beta * C[i * n + j]
    for p in range(k):
        for j in range(n):
            C[i * n + j] = C[i * n + j] + alpha * (A[i * k + p] * B[p * n + j])

making every inner-loop access unit-stride in ``j`` -- the shape SIMD
wants, and the shape real BLAS uses.

Two facts make this a COMPILER decision rather than a rewrite users do:

* **Legality is contract law.** Accumulating into ``C`` per ``p`` reorders
  the floating-point sum -- a REASSOCIATION, licensed by the work
  contract's ``inexact_identities`` axis (deploy/fast) and refused under
  the exact presets (prove/develop). The gate is consulted at transform
  time; an exact contract gets the authored order untouched.
* **Profitability is stride evidence.** Strides are read from the authored
  index arithmetic itself (the same authority
  ``derive_extents_from_source`` reads) -- the transform fires only when
  the reduction variable carries a non-unit stride on some load while the
  parallel loop variable is unit-stride on the store, and the decision
  records that evidence, deployment-classification style.

The recognized shape is deliberately narrow (v1): a parallel loop whose
body is exactly [scalar accumulator init to 0.0; a reduction loop whose
single statement is ``acc = acc + <term>``; a store
``target[index] = <c1> * acc + <rest>``] with ``c1`` a name or constant
free of the loop variables and ``<rest>`` free of the reduction variable
and the accumulator. Anything else is left untouched, with the reason
recorded. The transformed output is two sequential loops storing to one
array -- the exact shape the loop-carried storage aliasing fix
(test_compiled_linalg.py's promoted pin) made safe; this pass MUST NOT be
backported to a tree without that fix.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class InterchangeDecision:
    """One nest's verdict, with the evidence either way."""

    function: str
    line: int
    interchanged: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class InterchangeResult:
    source: str
    decisions: tuple[InterchangeDecision, ...]


def _stride_of(index: ast.AST, variable: str) -> object:
    """The coefficient of ``variable`` in a linear index, or None.

    Recognizes the flat row-major forms this tree compiles: ``v``,
    ``v * s + w``, ``w + v * s`` and their commuted products. Returns 1,
    a stride NAME, 0 (variable absent), or None (unrecognized)."""

    if isinstance(index, ast.Name):
        return 1 if index.id == variable else 0
    if isinstance(index, ast.BinOp) and isinstance(index.op, ast.Add):
        left = _stride_of(index.left, variable)
        right = _stride_of(index.right, variable)
        if left is None or right is None:
            return None
        if left == 0:
            return right
        if right == 0:
            return left
        return None
    if isinstance(index, ast.BinOp) and isinstance(index.op, ast.Mult):
        for factor, other in (
            (index.left, index.right), (index.right, index.left),
        ):
            if isinstance(factor, ast.Name) and factor.id == variable:
                if isinstance(other, ast.Name):
                    return other.id
                if isinstance(other, ast.Constant):
                    return other.value
                return None
        if not any(
            isinstance(name, ast.Name) and name.id == variable
            for name in ast.walk(index)
        ):
            return 0
        return None
    if not any(
        isinstance(name, ast.Name) and name.id == variable
        for name in ast.walk(index)
    ):
        return 0
    return None


def _names_in(node: ast.AST) -> set[str]:
    return {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    }


def _is_pure_expression(node: ast.AST) -> bool:
    """Whether evaluating ``node`` has no user-observable effect.

    Interchange changes evaluation order.  V1 therefore admits only the
    expression vocabulary used by the flat BLAS kernels; calls, attributes,
    comprehensions, named expressions, and dynamic subscript bases are all
    refused rather than guessed pure.
    """

    return all(isinstance(child, (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
        ast.Constant, ast.Subscript, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    )) for child in ast.walk(node)) and all(
        isinstance(child.value, ast.Name)
        for child in ast.walk(node)
        if isinstance(child, ast.Subscript)
    )


def _is_pure_range_loop(node: ast.For) -> bool:
    iterator = node.iter
    return (
        not node.orelse
        and isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == "range"
        and not iterator.keywords
        and 1 <= len(iterator.args) <= 3
        and all(_is_pure_expression(argument) for argument in iterator.args)
    )


def _subscript_bases(node: ast.AST) -> tuple[str, ...] | None:
    bases: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        if not isinstance(child.value, ast.Name):
            return None
        bases.append(child.value.id)
    return tuple(bases)


def _match_reduction_body(
    parallel: ast.For,
) -> tuple[ast.Assign, ast.For, ast.Assign] | None:
    """[acc = 0.0; for R: acc = acc + term; store[idx] = c1*acc + rest]"""

    if len(parallel.body) != 3:
        return None
    init, reduction, store = parallel.body
    if not (
        isinstance(init, ast.Assign)
        and len(init.targets) == 1
        and isinstance(init.targets[0], ast.Name)
        and isinstance(init.value, ast.Constant)
        and init.value.value == 0.0
    ):
        return None
    accumulator = init.targets[0].id
    if not (
        isinstance(reduction, ast.For)
        and isinstance(reduction.target, ast.Name)
        and len(reduction.body) == 1
        and isinstance(reduction.body[0], ast.Assign)
    ):
        return None
    update = reduction.body[0]
    if not (
        len(update.targets) == 1
        and isinstance(update.targets[0], ast.Name)
        and update.targets[0].id == accumulator
        and isinstance(update.value, ast.BinOp)
        and isinstance(update.value.op, ast.Add)
        and isinstance(update.value.left, ast.Name)
        and update.value.left.id == accumulator
    ):
        return None
    if not (
        isinstance(store, ast.Assign)
        and len(store.targets) == 1
        and isinstance(store.targets[0], ast.Subscript)
    ):
        return None
    return init, reduction, store


def _split_store(
    store: ast.Assign, accumulator: str,
) -> tuple[ast.expr, ast.expr | None] | None:
    """``c1 * acc + rest`` -> (c1, rest); ``acc`` alone -> (1.0, None)."""

    value = store.value
    if isinstance(value, ast.Name) and value.id == accumulator:
        return ast.Constant(value=1.0), None
    if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)):
        return None
    for scaled, rest in ((value.left, value.right),
                         (value.right, value.left)):
        if (
            isinstance(scaled, ast.BinOp)
            and isinstance(scaled.op, ast.Mult)
        ):
            for factor, other in (
                (scaled.left, scaled.right), (scaled.right, scaled.left),
            ):
                if (
                    isinstance(other, ast.Name)
                    and other.id == accumulator
                    and isinstance(factor, (ast.Name, ast.Constant))
                ):
                    return factor, rest
        if isinstance(scaled, ast.Name) and scaled.id == accumulator:
            return ast.Constant(value=1.0), rest
    return None


def interchange_reduction_loops(
    source: str, *, licensed: bool | None = None,
) -> InterchangeResult:
    """Interchange every recognized reduction nest the contract licenses.

    ``licensed`` overrides the contract gate for testing; ``None`` asks
    the active work contract's ``inexact_identities`` axis, because the
    transform reassociates a floating-point sum and an exact contract
    forbids exactly that.
    """

    if licensed is None:
        from .work_contract import active_contract

        licensed = bool(active_contract().inexact_identities)

    tree = ast.parse(source)
    decisions: list[InterchangeDecision] = []
    changed = False

    for function in [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]:
        parameters = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }

        class _Rewriter(ast.NodeTransformer):
            def visit_For(self, node: ast.For):
                self.generic_visit(node)
                matched = _match_reduction_body(node)
                if matched is None or not isinstance(
                    node.target, ast.Name
                ):
                    return node
                init, reduction, store = matched
                parallel_var = node.target.id
                reduction_var = reduction.target.id
                accumulator = init.targets[0].id
                reasons: list[str] = []

                term = reduction.body[0].value.right
                store_index = store.targets[0].slice
                store_stride = _stride_of(store_index, parallel_var)
                reduction_store_stride = _stride_of(
                    store_index, reduction_var
                )
                term_strides = [
                    (
                        ast.unparse(load.value),
                        _stride_of(load.slice, parallel_var),
                        _stride_of(load.slice, reduction_var),
                    )
                    for load in ast.walk(term)
                    if isinstance(load, ast.Subscript)
                ]
                split = _split_store(store, accumulator)
                store_base = store.targets[0].value
                term_bases = _subscript_bases(term)

                verdict = True
                if split is None:
                    reasons.append(
                        "store is not c1*acc + rest; not recognized"
                    )
                    verdict = False
                if not (
                    _is_pure_range_loop(node)
                    and _is_pure_range_loop(reduction)
                ):
                    reasons.append(
                        "parallel and reduction loops must be side-effect-free "
                        "range loops without else clauses"
                    )
                    verdict = False
                if not (
                    _is_pure_expression(term)
                    and _is_pure_expression(store.targets[0].slice)
                    and (
                        split is None
                        or _is_pure_expression(split[0])
                        and (
                            split[1] is None
                            or _is_pure_expression(split[1])
                        )
                    )
                ):
                    reasons.append(
                        "term, scale, remainder, and store index must use the "
                        "side-effect-free v1 arithmetic/subscript vocabulary"
                    )
                    verdict = False
                if not isinstance(store_base, ast.Name):
                    reasons.append("store target must be a named formal buffer")
                    verdict = False
                elif store_base.id not in parameters:
                    reasons.append(
                        f"store buffer {store_base.id!r} is not a formal with "
                        "compiler-owned storage identity"
                    )
                    verdict = False
                if term_bases is None or any(
                    base not in parameters for base in (term_bases or ())
                ):
                    reasons.append(
                        "every reduction load must use a named formal buffer"
                    )
                    verdict = False
                elif (
                    isinstance(store_base, ast.Name)
                    and store_base.id in term_bases
                ):
                    reasons.append(
                        "reduction term reads the destination buffer; promoted "
                        "stores would change later reduction inputs"
                    )
                    verdict = False
                if store_stride != 1 or reduction_store_stride != 0:
                    reasons.append(
                        "store must be unit-stride in the parallel "
                        "variable and independent of the reduction "
                        f"variable (got {store_stride!r}, "
                        f"{reduction_store_stride!r})"
                    )
                    verdict = False
                if not any(
                    parallel_stride == 1 and reduction_stride not in (0, 1)
                    for _name, parallel_stride, reduction_stride
                    in term_strides
                ):
                    reasons.append(
                        "no load is unit-stride in the parallel variable "
                        "while strided in the reduction variable; "
                        "interchange buys nothing: "
                        + repr(term_strides)
                    )
                    verdict = False
                if accumulator in _names_in(term) or (
                    split is not None and split[1] is not None
                    and (
                        reduction_var in _names_in(split[1])
                        or accumulator in _names_in(split[1])
                    )
                ):
                    reasons.append(
                        "rest/term reference the reduction variable or "
                        "accumulator; promotion is not equivalence-safe"
                    )
                    verdict = False
                if split is not None and isinstance(split[0], ast.Name) and (
                    split[0].id in {parallel_var, reduction_var, accumulator}
                    or split[0].id not in parameters
                ):
                    reasons.append(
                        "scale factor depends on the loop variables"
                    )
                    verdict = False
                if verdict and not licensed:
                    reasons.append(
                        "recognized and profitable, but the active work "
                        "contract forbids inexact identities and this "
                        "promotion reassociates the sum -- authored order "
                        "kept"
                    )
                    verdict = False
                elif verdict:
                    reasons.append(
                        "reduction variable strided where the parallel "
                        "variable is unit-stride "
                        f"({term_strides!r}); contract licenses "
                        "reassociation; interchanged with accumulator "
                        "promoted into the store target"
                    )

                decisions.append(InterchangeDecision(
                    function=function.name,
                    line=node.lineno,
                    interchanged=verdict,
                    reasons=tuple(reasons),
                ))
                if not verdict:
                    return node

                nonlocal_changed()
                c1, rest = split
                prologue_value = (
                    copy.deepcopy(rest) if rest is not None
                    else ast.Constant(value=0.0)
                )
                prologue = ast.For(
                    target=copy.deepcopy(node.target),
                    iter=copy.deepcopy(node.iter),
                    body=[ast.Assign(
                        targets=[copy.deepcopy(store.targets[0])],
                        value=prologue_value,
                    )],
                    orelse=[],
                )
                accumulate = ast.For(
                    target=copy.deepcopy(reduction.target),
                    iter=copy.deepcopy(reduction.iter),
                    body=[ast.For(
                        target=copy.deepcopy(node.target),
                        iter=copy.deepcopy(node.iter),
                        body=[ast.Assign(
                            targets=[copy.deepcopy(store.targets[0])],
                            value=ast.BinOp(
                                left=ast.Subscript(
                                    value=copy.deepcopy(
                                        store.targets[0].value
                                    ),
                                    slice=copy.deepcopy(
                                        store.targets[0].slice
                                    ),
                                    ctx=ast.Load(),
                                ),
                                op=ast.Add(),
                                right=ast.BinOp(
                                    left=copy.deepcopy(c1),
                                    op=ast.Mult(),
                                    right=copy.deepcopy(term),
                                ),
                            ),
                        )],
                        orelse=[],
                    )],
                    orelse=[],
                )
                return [prologue, accumulate]

        state = {"changed": False}

        def nonlocal_changed() -> None:
            state["changed"] = True

        function.body = [
            statement
            for item in (
                _Rewriter().visit(statement) for statement in function.body
            )
            for statement in (item if isinstance(item, list) else [item])
        ]
        if state["changed"]:
            changed = True

    if not changed:
        return InterchangeResult(source, tuple(decisions))
    ast.fix_missing_locations(tree)
    return InterchangeResult(ast.unparse(tree) + "\n", tuple(decisions))


__all__ = [
    "InterchangeDecision",
    "InterchangeResult",
    "interchange_reduction_loops",
]
