"""Hierarchical, printable planning objects.

Text is a view, never the authority.  Each logical line is a typed item and
each nested scope is an explicit closure with an explicit capture set.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping

from ..transmogrifier.ssa import Instr, SSAValue


@dataclass(frozen=True)
class PlanLine:
    opcode: str
    inputs: tuple[int, ...] = ()
    outputs: tuple[int, ...] = ()
    attributes: tuple[tuple[str, Any], ...] = ()
    input_roles: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        opcode: str,
        *,
        inputs=(),
        outputs=(),
        attributes: Mapping[str, Any] | None = None,
        input_roles=(),
    ) -> "PlanLine":
        return cls(
            str(opcode),
            tuple(int(value) for value in inputs),
            tuple(int(value) for value in outputs),
            tuple(sorted((attributes or {}).items())),
            tuple(str(role) for role in input_roles),
        )


@dataclass(frozen=True)
class PlanClosure:
    name: str
    captures: tuple[int, ...]
    items: tuple["PlanItem", ...]
    closure_id: int = -1
    # Shape (and dtype) of the region's values, carried from the process graph's
    # per-node domain so the lowered SSA values are the arrays they are, not
    # shapeless scalars. ``(value_id, shape, dtype)`` per value.
    value_shapes: tuple[tuple[int, tuple[int, ...], str], ...] = ()


@dataclass(frozen=True)
class PlanCall:
    """A typed call edge between two planned closures.

    The call owns its argument/result correlation.  Backend lowering may
    inline the callee, retain a native call, or split deployment, but it must
    never rediscover this relationship from source names or runtime values.
    """

    callsite_id: int
    callee: PlanClosure
    argument_value_ids: tuple[int, ...] = ()
    result_value_ids: tuple[int, ...] = ()
    argument_bindings: tuple[tuple[int, int], ...] = ()
    result_bindings: tuple[tuple[int, int], ...] = ()
    enclosing_loop_ids: tuple[int, ...] = ()


PlanItem = PlanLine | PlanClosure | PlanCall


#: How a tensor operation is spelled once it is known to act on scalars.
#:
#: The planner applies this ONLY when the result and every operand have an
#: empty shape; a genuinely tensor-shaped operation keeps its lowercase
#: name and is resolved through the tensor likeness table instead. The two
#: spellings are therefore the same operation at different ranks, which is
#: why anything interpreting repository SSA -- a backend, or the reference
#: evaluator -- must read them through this one table rather than keep a
#: private copy that can drift from it.
TENSOR_OPERATION_SCALAR_SPELLING: dict[str, str] = {
    "add": "Add", "sub": "Sub", "mul": "Mul",
    "truediv": "Div", "div": "Div", "floordiv": "FloorDiv",
    "mod": "Mod", "pow": "Pow", "neg": "Neg", "abs": "Abs",
    "equal": "Eq", "not_equal": "Ne", "less": "Lt",
    "less_equal": "Le", "greater": "Gt", "greater_equal": "Ge",
    "logical_and": "LAnd", "logical_or": "LOr",
    "logical_not": "LNot", "maximum": "Max", "minimum": "Min",
    "sqrt": "Sqrt", "exp": "Exp", "log": "Log",
}


def plan_region_to_ssa_instrs(region: PlanClosure) -> tuple[Instr, ...]:
    """Lower one planner-owned flat region to repository SSA instructions.

    Each SSA value carries the shape and dtype recorded for it on the region
    (``value_shapes``, from the process graph's per-node domain), so a value is
    the array it is and array ops lower as array ops rather than scalars.
    """

    shape_of = {
        int(value_id): tuple(int(dimension) for dimension in shape)
        for value_id, shape, _dtype in region.value_shapes
    }
    dtype_of = {
        int(value_id): dtype for value_id, _shape, dtype in region.value_shapes
    }
    # The graph domain is deliberately permissive and often records scalar
    # control values with its default numerical dtype.  Operator semantics are
    # authoritative where they are stricter: comparisons/logical operations
    # produce predicates, and address indices/extents are integers.  Retain
    # these contracts in repository SSA rather than asking a target emitter to
    # reverse-engineer them from syntax.
    predicate_ops = {
        "eq", "equal", "ne", "not_equal", "lt", "less", "le",
        "less_equal", "gt", "greater", "ge", "greater_equal",
        "land", "logical_and", "lor", "logical_or", "lnot",
        "logical_not", "is", "is_not", "contains", "not_contains",
    }
    integer_result_ops = {"len", "length", "extent"}
    scalar_cast_dtypes = {
        "float": "float64",
        "int": "int",
        "bool": "bool",
    }
    for item in region.items:
        if not isinstance(item, PlanLine):
            continue
        opcode = str(item.opcode).casefold()
        if item.outputs and opcode in {"const", "constant"}:
            literal = dict(item.attributes).get("value")
            if isinstance(literal, bool):
                dtype_of[int(item.outputs[0])] = "bool"
            elif isinstance(literal, int):
                dtype_of[int(item.outputs[0])] = "int"
            elif isinstance(literal, float):
                dtype_of[int(item.outputs[0])] = "float64"
        if item.outputs and opcode in predicate_ops:
            dtype_of[int(item.outputs[0])] = "bool"
        elif item.outputs and opcode in integer_result_ops:
            dtype_of[int(item.outputs[0])] = "int"
        elif item.outputs and opcode in scalar_cast_dtypes:
            dtype_of[int(item.outputs[0])] = scalar_cast_dtypes[opcode]
        if opcode == "getelementptr":
            # Only repository address arithmetic requires integer indices.
            # High-level Indexed/IndexedStore may be a dictionary lookup whose
            # key retains any authored type and is lowered through a table.
            index_inputs = item.inputs[1:]
            for value_id in index_inputs:
                dtype_of[int(value_id)] = "int"

    def value(value_id: int) -> SSAValue:
        value_id = int(value_id)
        existing = values.get(value_id)
        if existing is not None:
            return existing
        made = SSAValue(
            value_id,
            dtype=dtype_of.get(value_id, "float64"),
            shape=shape_of.get(value_id, ()),
        )
        values[value_id] = made
        return made

    values: dict[int, SSAValue] = {}
    next_value_id = max((
        int(value_id)
        for item in region.items
        if isinstance(item, PlanLine)
        for value_id in (*item.inputs, *item.outputs)
    ), default=-1) + 1

    def fresh_like(result: SSAValue) -> SSAValue:
        nonlocal next_value_id
        made = SSAValue(
            next_value_id,
            dtype=result.dtype,
            shape=tuple(result.shape),
        )
        values[next_value_id] = made
        next_value_id += 1
        return made

    instructions = []
    for item in region.items:
        if not isinstance(item, PlanLine):
            raise ValueError(
                f"{region.name!r} is not a flat operator region"
            )
        if len(item.outputs) > 1:
            raise ValueError(
                f"{item.opcode!r} may publish at most one SSA result"
            )
        result = (
            value(int(item.outputs[0])) if item.outputs else None
        )
        # A Python identity replacement is already the call's graph-native
        # meaning.  Its callable/definition input is provenance, not runtime
        # data.  Keep this distinction explicit through argument roles instead
        # of teaching each backend that ``float`` (or every future identity)
        # happens to carry a CPython function object in operand zero.
        paired_inputs = tuple(zip(item.inputs, item.input_roles))
        if len(item.input_roles) == len(item.inputs):
            semantic_inputs = tuple(
                int(value_id)
                for value_id, role in paired_inputs
                if str(role).casefold() not in {
                    "callee", "func", "function", "definition",
                }
            )
            semantic_roles = tuple(
                str(role)
                for _value_id, role in paired_inputs
                if str(role).casefold() not in {
                    "callee", "func", "function", "definition",
                }
            )
        else:
            semantic_inputs = tuple(int(value_id) for value_id in item.inputs)
            semantic_roles = tuple(str(role) for role in item.input_roles)

        opcode = str(item.opcode)
        attributes = dict(item.attributes)
        if opcode.casefold() in scalar_cast_dtypes:
            attributes.setdefault("source_operator", opcode)
            attributes.setdefault(
                "target_dtype", scalar_cast_dtypes[opcode.casefold()]
            )
            opcode = "Cast"
        elif opcode.casefold() == "tensor":
            # AbstractTensor.tensor(x) is the general ensure-type idiom.  Once
            # x reaches typed repository SSA, normalization is represented by
            # a same-value cast carrying the schema promise; target code does
            # not reconstruct or invoke a Python object.
            attributes.setdefault("source_operator", opcode)
            attributes.setdefault("target_dtype", dtype_of.get(
                int(item.outputs[0]) if item.outputs else -1, "float64"
            ))
            opcode = "Cast"

        scalar_spelling = TENSOR_OPERATION_SCALAR_SPELLING
        is_scalar = (
            result is not None
            and not tuple(result.shape)
            and all(not tuple(value(value_id).shape) for value_id in semantic_inputs)
        )
        if is_scalar and opcode.casefold() in scalar_spelling:
            opcode = scalar_spelling[opcode.casefold()]

        # Python's variadic min/max and the tensor clamp convenience are
        # ordinary binary SSA folds.  Decomposing them here keeps evaluation
        # order and data dependencies visible, and gives every backend the
        # same primitive program instead of four bespoke builtin handlers.
        fold_opcode = {"max": "Max", "min": "Min"}.get(opcode.casefold())
        if is_scalar and fold_opcode is not None and len(semantic_inputs) >= 2:
            operands = [value(value_id) for value_id in semantic_inputs]
            accumulator = operands[0]
            for position, operand in enumerate(operands[1:], 1):
                fold_result = (
                    result if position == len(operands) - 1
                    else fresh_like(result)
                )
                instructions.append(Instr(
                    fold_opcode,
                    [accumulator, operand],
                    fold_result,
                    arg_roles=["left", "right"],
                    attributes={**attributes, "source_operator": opcode},
                ))
                accumulator = fold_result
            continue
        if is_scalar and opcode.casefold() == "clamp" and len(semantic_inputs) == 3:
            operand, lower, upper = (
                value(value_id) for value_id in semantic_inputs
            )
            bounded_below = fresh_like(result)
            instructions.append(Instr(
                "Max", [operand, lower], bounded_below,
                arg_roles=["operand", "lower"],
                attributes={**attributes, "source_operator": opcode},
            ))
            instructions.append(Instr(
                "Min", [bounded_below, upper], result,
                arg_roles=["operand", "upper"],
                attributes={**attributes, "source_operator": opcode},
            ))
            continue

        instructions.append(Instr(
            opcode,
            [value(value_id) for value_id in semantic_inputs],
            result,
            arg_roles=list(semantic_roles),
            attributes=attributes,
        ))
    return tuple(instructions)


@dataclass(frozen=True)
class HierarchyValueTable:
    """Collision-free IDs for values whose local IDs live in many shells."""

    correlations: tuple[tuple[int, int, int], ...]

    @cached_property
    def _global_ids(self) -> dict[tuple[int, int], int]:
        """Index the immutable correlation table once.

        Hierarchical composition asks for the same endpoint mappings many
        times while nesting calls, control blocks, bindings and shader
        storage.  Scanning ``correlations`` for every request makes that
        otherwise-linear compiler stage quadratic in the number of scoped
        values.  The tuple remains the canonical, printable representation;
        this index is only its exact lookup form and does not derive IDs from
        names, observed values, or runtime state.
        """

        return {
            (int(scope), int(local)): int(global_id)
            for scope, local, global_id in self.correlations
        }

    def global_id(self, closure_id: int, local_value_id: int) -> int:
        return self._global_ids[(int(closure_id), int(local_value_id))]


@dataclass(frozen=True)
class HierarchyIdentityReduction:
    """Result of removing semantically transparent call closures."""

    root: PlanClosure
    collapsed_callsites: tuple[int, ...]
    rounds: int


def reduce_hierarchy_identities(
    root: PlanClosure,
    identity_closure_ids: set[int] | frozenset[int],
) -> HierarchyIdentityReduction:
    """Remove post-planning call boundaries proven to be SSA identities.

    Identity discovery happens only after hierarchy construction, when argument
    and result bindings are explicit.  Value unification therefore remains the
    authority: this pass removes the now-redundant closure/control boundary but
    never invents an alias from source names or observed runtime values.

    The rewrite is a fixed point so future identities that erase enclosing
    structural reasons can expose further collapses without changing callers.
    """

    identities = frozenset(int(value) for value in identity_closure_ids)
    collapsed: list[int] = []
    rounds = 0
    current = root
    while True:
        changed = False

        def rewrite(closure: PlanClosure) -> PlanClosure:
            nonlocal changed
            items = []
            for item in closure.items:
                if isinstance(item, PlanCall):
                    callee = rewrite(item.callee)
                    if int(callee.closure_id) in identities:
                        collapsed.append(int(item.callsite_id))
                        changed = True
                        continue
                    items.append(PlanCall(
                        item.callsite_id,
                        callee,
                        item.argument_value_ids,
                        item.result_value_ids,
                        item.argument_bindings,
                        item.result_bindings,
                        item.enclosing_loop_ids,
                    ))
                elif isinstance(item, PlanClosure):
                    items.append(rewrite(item))
                else:
                    items.append(item)
            return PlanClosure(
                closure.name,
                closure.captures,
                tuple(items),
                closure.closure_id,
                closure.value_shapes,
            )

        updated = rewrite(current)
        if not changed:
            break
        current = updated
        rounds += 1
    return HierarchyIdentityReduction(
        current,
        tuple(dict.fromkeys(collapsed)),
        rounds,
    )


def assign_hierarchy_ids(
    root: PlanClosure,
    previous: HierarchyValueTable | None = None,
) -> tuple[PlanClosure, HierarchyValueTable]:
    """Assign canonical IDs once and preserve them when a plan is extended.

    ``(closure_id, local_id)`` is a scoped source address, not a second
    runtime identity.  The returned global ID is the one semantic identity
    used by every later compiler stage.  A refresh may append newly exposed
    control endpoints, but it must never renumber an endpoint that was
    already assigned.
    """

    next_closure = 0

    def number(closure: PlanClosure) -> PlanClosure:
        nonlocal next_closure
        closure_id = next_closure
        next_closure += 1
        items = tuple(
            PlanCall(
                item.callsite_id,
                number(item.callee),
                item.argument_value_ids,
                item.result_value_ids,
                item.argument_bindings,
                item.result_bindings,
                item.enclosing_loop_ids,
            )
            if isinstance(item, PlanCall)
            else number(item)
            if isinstance(item, PlanClosure)
            else item
            for item in closure.items
        )
        return PlanClosure(
            closure.name,
            closure.captures,
            items,
            closure_id,
            closure.value_shapes,
        )

    planned = number(root)
    keys: set[tuple[int, int]] = set()
    unions: list[
        tuple[tuple[int, int], tuple[int, int]]
    ] = []

    def collect(closure: PlanClosure) -> None:
        closure_id = int(closure.closure_id)
        for local_id in closure.captures:
            keys.add((closure_id, int(local_id)))
        for item in closure.items:
            if isinstance(item, PlanLine):
                for local_id in (*item.inputs, *item.outputs):
                    keys.add((closure_id, int(local_id)))
            elif isinstance(item, PlanCall):
                child_id = int(item.callee.closure_id)
                for local_id in (
                    *item.argument_value_ids,
                    *item.result_value_ids,
                ):
                    keys.add((closure_id, int(local_id)))
                for caller, callee in item.argument_bindings:
                    left = (closure_id, int(caller))
                    right = (child_id, int(callee))
                    keys.update((left, right))
                    unions.append((left, right))
                for callee, caller in item.result_bindings:
                    left = (child_id, int(callee))
                    right = (closure_id, int(caller))
                    keys.update((left, right))
                    unions.append((left, right))
                collect(item.callee)
            elif isinstance(item, PlanClosure):
                collect(item)

    collect(planned)
    parents = {key: key for key in keys}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for left, right in unions:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    previous_ids = (
        {}
        if previous is None
        else {
            (int(scope), int(local)): int(global_id)
            for scope, local, global_id in previous.correlations
        }
    )
    class_previous_ids: dict[tuple[int, int], set[int]] = {}
    for key in keys:
        if key in previous_ids:
            class_previous_ids.setdefault(find(key), set()).add(
                previous_ids[key]
            )
    conflicting = {
        root_key: tuple(sorted(ids))
        for root_key, ids in class_previous_ids.items()
        if len(ids) > 1
    }
    if conflicting:
        raise ValueError(
            "hierarchy refresh attempted to merge previously distinct "
            f"canonical IDs: {conflicting!r}"
        )

    root_ids: dict[tuple[int, int], int] = {
        root_key: next(iter(ids))
        for root_key, ids in class_previous_ids.items()
    }
    next_global_id = 1 + max(previous_ids.values(), default=-1)
    correlations = []
    for closure_id, local_id in sorted(keys):
        root_key = find((closure_id, local_id))
        if root_key not in root_ids:
            root_ids[root_key] = next_global_id
            next_global_id += 1
        global_id = root_ids[root_key]
        correlations.append((closure_id, local_id, global_id))
    return planned, HierarchyValueTable(tuple(correlations))


def render_plan_ascii(root: PlanClosure) -> str:
    """Render a stable tree view without changing the planning object."""

    lines: list[str] = []

    def visit(item: PlanItem, prefix: str, last: bool) -> None:
        branch = "`- " if last else "|- "
        if isinstance(item, PlanCall):
            arguments = ",".join(map(str, item.argument_value_ids)) or "-"
            results = ",".join(map(str, item.result_value_ids)) or "-"
            argument_bindings = ",".join(
                f"{caller}->{callee}"
                for caller, callee in item.argument_bindings
            ) or "-"
            result_bindings = ",".join(
                f"{callee}->{caller}"
                for callee, caller in item.result_bindings
            ) or "-"
            lines.append(
                f"{prefix}{branch}call #{item.callsite_id} "
                f"args=[{arguments}] results=[{results}] "
                f"arg-bind=[{argument_bindings}] "
                f"result-bind=[{result_bindings}]"
            )
            child_prefix = prefix + ("   " if last else "|  ")
            visit(item.callee, child_prefix, True)
            return
        if isinstance(item, PlanClosure):
            captures = ",".join(map(str, item.captures)) or "-"
            lines.append(
                f"{prefix}{branch}closure {item.name} "
                f"id={item.closure_id} captures=[{captures}]"
            )
            child_prefix = prefix + ("   " if last else "|  ")
            for index, child in enumerate(item.items):
                visit(child, child_prefix, index == len(item.items) - 1)
            return
        inputs = ",".join(map(str, item.inputs)) or "-"
        outputs = ",".join(map(str, item.outputs)) or "-"
        roles = ",".join(item.input_roles) or "-"
        attributes = " ".join(
            f"{name}={value!r}" for name, value in item.attributes
        )
        suffix = f" {attributes}" if attributes else ""
        lines.append(
            f"{prefix}{branch}{item.opcode} in=[{inputs}] roles=[{roles}] "
            f"out=[{outputs}]"
            f"{suffix}"
        )

    visit(root, "", True)
    return "\n".join(lines)


__all__ = [
    "PlanCall",
    "PlanClosure",
    "PlanItem",
    "PlanLine",
    "HierarchyValueTable",
    "HierarchyIdentityReduction",
    "assign_hierarchy_ids",
    "reduce_hierarchy_identities",
    "render_plan_ascii",
    "plan_region_to_ssa_instrs",
]
