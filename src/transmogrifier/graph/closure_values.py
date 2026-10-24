"""Closures returned as values, lowered to capture records.

A definition that returns one of its own nested functions (or a lambda)
hands its caller a Python closure: code plus the enclosing values that
code reads.  The process graph has no function-valued data, so the
closure is converted here, before graph construction, into the two
things it is made of:

* the defining function returns a record ``(tag, slot, ...)`` -- the
  variant tag and the captured values, read at the return, which is
  exactly when Python's late-bound cells stop changing;
* a generated sibling ``<name>__call(record, *args)`` holds every variant
  body; it unpacks the variant's captures from the record and runs it.

Every call through a returned closure becomes a call of that sibling with
the record as its first argument.  ``g = make(n); g(3)`` is
``g = make(n); make__call(g, 3)``, and ``make(n)(3)`` is
``make__call(make(n), 3)``.

The conversion is whole-tree: the record changes the defining function's
return, so it applies only when every reference to the definition in the
tree is such a call site.  Anything else (the closure stored, passed,
returned again, rebound, captured by a nested scope, or a variant that
uses ``nonlocal``, defaults, decorators or a different arity) leaves the
definition untouched, and the compiler's ordinary handling stands.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import Any


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _own_scope_nodes(scope: ast.AST):
    """Nodes evaluated in ``scope`` itself, not in a nested scope.

    A nested definition's name, decorators, defaults and annotations are
    evaluated in the enclosing scope; its body is not.  A comprehension's
    first iterable is evaluated in the enclosing scope; the rest is not.
    """

    if isinstance(scope, ast.Lambda):
        pending: list[ast.AST] = [scope.body]
    elif isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        pending = list(scope.body)
    else:
        pending = list(getattr(scope, "body", ()))
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pending.extend(node.decorator_list)
            pending.extend(
                default for default in (
                    *node.args.defaults, *node.args.kw_defaults,
                ) if default is not None
            )
            continue
        if isinstance(node, ast.Lambda):
            pending.extend(
                default for default in (
                    *node.args.defaults, *node.args.kw_defaults,
                ) if default is not None
            )
            continue
        if isinstance(node, ast.ClassDef):
            pending.extend(node.decorator_list)
            pending.extend(node.bases)
            continue
        if isinstance(node, _COMPREHENSIONS):
            pending.append(node.generators[0].iter)
            continue
        pending.extend(ast.iter_child_nodes(node))


def _child_scopes(scope: ast.AST):
    for node in _own_scope_nodes(scope):
        if isinstance(node, _SCOPES) or isinstance(node, _COMPREHENSIONS):
            yield node


def _parameter_names(arguments: ast.arguments) -> set[str]:
    return {
        argument.arg for argument in (
            *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
            *filter(None, (arguments.vararg, arguments.kwarg)),
        )
    }


def _bound_names(scope: ast.AST) -> set[str]:
    """Names local to ``scope`` by Python's binding rules."""

    if isinstance(scope, _COMPREHENSIONS):
        return {
            node.id
            for generator in scope.generators
            for node in ast.walk(generator.target)
            if isinstance(node, ast.Name)
        }
    bound: set[str] = set()
    declared: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        bound |= _parameter_names(scope.args)
    for node in _own_scope_nodes(scope):
        if isinstance(node, ast.Name) and isinstance(
            node.ctx, (ast.Store, ast.Del),
        ):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        )):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
    return bound - declared


def _free_names(scope: ast.AST) -> set[str]:
    """Names ``scope`` (or anything nested in it) reads from outside it."""

    reads = {
        node.id for node in _own_scope_nodes(scope)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    if isinstance(scope, _COMPREHENSIONS):
        reads |= {
            node.id
            for node in ast.walk(scope)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
    for child in _child_scopes(scope):
        reads |= _free_names(child)
    return reads - _bound_names(scope)


def _scope_declares(scope: ast.AST, kinds: tuple[type, ...]) -> bool:
    return any(
        isinstance(node, kinds)
        for node in ast.walk(scope)
    )


@dataclass
class _Variant:
    tag: int
    node: ast.FunctionDef | ast.Lambda
    parameters: tuple[str, ...]
    captures: tuple[str, ...]


@dataclass
class _ClosurePlan:
    definition: ast.FunctionDef
    owner: ast.Module | ast.ClassDef
    variants: list[_Variant]
    variant_of_return: dict[int, _Variant]
    nested_definitions: dict[str, ast.FunctionDef]
    slots: tuple[str, ...]
    call_name: str
    record_name: str
    sites: list[tuple[str, Any]] = field(default_factory=list)


def _variant_shape(node: ast.AST) -> tuple[str, ...] | None:
    """Positional parameter names, or None for a shape not converted."""

    arguments = node.args
    if (
        arguments.posonlyargs or arguments.vararg or arguments.kwonlyargs
        or arguments.kwarg or arguments.defaults
    ):
        return None
    if isinstance(node, ast.FunctionDef):
        if node.decorator_list:
            return None
        if any(
            isinstance(inner, (
                ast.Yield, ast.YieldFrom, ast.Await, ast.Nonlocal,
                ast.Global,
            ))
            for inner in _own_scope_nodes(node)
        ):
            return None
    elif not isinstance(node, ast.Lambda):
        return None
    if any(
        isinstance(inner, ast.Nonlocal) for inner in ast.walk(node)
    ):
        return None
    if "super" in _free_names(node) or "__class__" in _free_names(node):
        return None
    return tuple(argument.arg for argument in arguments.args)


def _closure_plan(
    definition: ast.FunctionDef,
    owner: ast.Module | ast.ClassDef,
    taken_names: set[str],
) -> _ClosurePlan | None:
    if any(
        isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await))
        for node in _own_scope_nodes(definition)
    ):
        return None
    returns = [
        node for node in _own_scope_nodes(definition)
        if isinstance(node, ast.Return)
    ]
    if not returns:
        return None
    nested: dict[str, list[ast.FunctionDef]] = {}
    for node in _own_scope_nodes(definition):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested.setdefault(node.name, []).append(node)
    if not any(
        isinstance(node.value, ast.Lambda)
        or (isinstance(node.value, ast.Name) and node.value.id in nested)
        for node in returns
    ):
        return None
    returned_names = {
        id(node.value)
        for node in returns
        if isinstance(node.value, ast.Name)
    }
    variants: list[_Variant] = []
    by_name: dict[str, _Variant] = {}
    variant_of_return: dict[int, _Variant] = {}
    for node in sorted(
        returns, key=lambda item: (item.lineno, item.col_offset),
    ):
        value = node.value
        if isinstance(value, ast.Lambda):
            target: ast.AST = value
        elif isinstance(value, ast.Name) and value.id in nested:
            if len(nested[value.id]) != 1:
                return None
            if value.id in by_name:
                variant_of_return[id(node)] = by_name[value.id]
                continue
            target = nested[value.id][0]
        else:
            return None
        parameters = _variant_shape(target)
        if parameters is None:
            return None
        variant = _Variant(len(variants), target, parameters, ())
        variants.append(variant)
        variant_of_return[id(node)] = variant
        if isinstance(target, ast.FunctionDef):
            by_name[target.name] = variant
    if len({len(variant.parameters) for variant in variants}) != 1:
        return None
    # A returned definition is used only as the value returned: called or
    # stored in the defining scope it would still be a function there.
    for node in _own_scope_nodes(definition):
        if (
            isinstance(node, ast.Name)
            and node.id in by_name
            and id(node) not in returned_names
        ):
            return None
    for child in _child_scopes(definition):
        if child in [variant.node for variant in variants]:
            continue
        if _free_names(child) & set(by_name):
            return None
    enclosing = _bound_names(definition)
    for variant in variants:
        free = _free_names(variant.node)
        captures = tuple(sorted(free & enclosing))
        if set(captures) & set(nested):
            return None
        variant.captures = captures
    # The variant bodies share one generated function: a name local to one
    # variant must not be a global read by another.
    globals_read = [
        _free_names(variant.node) - enclosing for variant in variants
    ]
    locals_bound = [
        _bound_names(variant.node) | set(variant.captures)
        | set(variants[0].parameters)
        for variant in variants
    ]
    for index, names in enumerate(locals_bound):
        for other, reads in enumerate(globals_read):
            if other != index and names & reads:
                return None
    slots = tuple(dict.fromkeys(
        name for variant in variants for name in variant.captures
    ))
    call_name = f"{definition.name}__call"
    record_name = f"{definition.name}__closure"
    if call_name in taken_names or record_name in taken_names:
        return None
    return _ClosurePlan(
        definition=definition,
        owner=owner,
        variants=variants,
        variant_of_return=variant_of_return,
        nested_definitions={name: nodes[0] for name, nodes in nested.items()},
        slots=slots,
        call_name=call_name,
        record_name=record_name,
    )


def _definitions(tree: ast.Module):
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef):
            yield statement, tree
        elif isinstance(statement, ast.ClassDef):
            for member in statement.body:
                if isinstance(member, ast.FunctionDef):
                    yield member, statement


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    return {
        id(child): node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }


def _enclosing_function(node: ast.AST, parents: dict[int, ast.AST]):
    current = parents.get(id(node))
    while current is not None and not isinstance(current, _SCOPES):
        current = parents.get(id(current))
    return current


def _references(
    tree: ast.Module, plan: _ClosurePlan, parents: dict[int, ast.AST],
) -> list[ast.AST] | None:
    """Every expression in ``tree`` that names the definition."""

    name = plan.definition.name
    references: list[ast.AST] = []
    if isinstance(plan.owner, ast.Module):
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == name:
                if not isinstance(node.ctx, ast.Load):
                    return None
                references.append(node)
            elif (
                isinstance(node, ast.Attribute) and node.attr == name
                and not (
                    isinstance(node.value, ast.Name)
                    and any(
                        isinstance(statement, ast.ClassDef)
                        and statement.name == node.value.id
                        for statement in tree.body
                    )
                )
            ):
                # ``module.make`` -- an imported spelling this pass does
                # not resolve.
                return None
        return references
    class_name = plan.owner.name
    methods = {
        id(member) for member in plan.owner.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr == name):
            continue
        receiver = node.value
        if isinstance(receiver, ast.Name) and receiver.id == class_name:
            references.append(node)
            continue
        if isinstance(receiver, ast.Name) and receiver.id in {"self", "cls"}:
            function = _enclosing_function(node, parents)
            while function is not None and id(function) not in methods:
                function = _enclosing_function(function, parents)
            if function is not None:
                references.append(node)
                continue
        return None
    return references


def _call_receiver(reference: ast.AST, plan: _ClosurePlan) -> ast.expr:
    if isinstance(reference, ast.Attribute):
        return ast.Attribute(
            value=copy.deepcopy(reference.value),
            attr=plan.call_name,
            ctx=ast.Load(),
        )
    return ast.Name(id=plan.call_name, ctx=ast.Load())


def _plan_sites(
    tree: ast.Module, plan: _ClosurePlan, parents: dict[int, ast.AST],
) -> bool:
    references = _references(tree, plan, parents)
    if not references:
        return False
    arity = len(plan.variants[0].parameters)
    names_done: set[tuple[int, str]] = set()
    for reference in references:
        call = parents.get(id(reference))
        if not (isinstance(call, ast.Call) and call.func is reference):
            return False
        holder = parents.get(id(call))
        if isinstance(holder, ast.Call) and holder.func is call:
            if (
                holder.keywords or len(holder.args) != arity
                or any(isinstance(arg, ast.Starred) for arg in holder.args)
            ):
                return False
            plan.sites.append(("direct", (holder, reference)))
            continue
        if not (
            isinstance(holder, ast.Assign)
            and len(holder.targets) == 1
            and isinstance(holder.targets[0], ast.Name)
        ):
            return False
        bound_name = holder.targets[0].id
        function = _enclosing_function(holder, parents)
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        key = (id(function), bound_name)
        if key in names_done:
            continue
        if bound_name in _parameter_names(function.args):
            return False
        if any(
            isinstance(node, (ast.Global, ast.Nonlocal))
            and bound_name in node.names
            for node in _own_scope_nodes(function)
        ):
            return False
        if any(
            bound_name in _free_names(child)
            for child in _child_scopes(function)
        ):
            return False
        reference_spellings = {ast.dump(reference)}
        uses: list[ast.Call] = []
        for node in _own_scope_nodes(function):
            if not (isinstance(node, ast.Name) and node.id == bound_name):
                continue
            if isinstance(node.ctx, ast.Store):
                assign = parents.get(id(node))
                if not (
                    isinstance(assign, ast.Assign)
                    and len(assign.targets) == 1
                    and assign.targets[0] is node
                    and isinstance(assign.value, ast.Call)
                    and assign.value.func is not None
                    and any(
                        assign.value.func is other for other in references
                    )
                ):
                    return False
                reference_spellings.add(ast.dump(assign.value.func))
                continue
            if not isinstance(node.ctx, ast.Load):
                return False
            use = parents.get(id(node))
            if not (
                isinstance(use, ast.Call) and use.func is node
                and not use.keywords and len(use.args) == arity
                and not any(isinstance(arg, ast.Starred) for arg in use.args)
            ):
                return False
            uses.append(use)
        if len(reference_spellings) != 1:
            return False
        names_done.add(key)
        plan.sites.extend(("bound", (use, reference)) for use in uses)
    return True


def _definitely_bound_before(
    definition: ast.FunctionDef,
) -> dict[int, set[str]]:
    """Names bound on every path to each ``return`` of ``definition``."""

    result: dict[int, set[str]] = {}

    def binds(statement: ast.stmt) -> set[str]:
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(statement, ast.AnnAssign) and statement.value is None:
                return set()
            targets = (
                statement.targets if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            return {
                node.id for target in targets for node in ast.walk(target)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
            }
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            return {
                (alias.asname or alias.name).split(".")[0]
                for alias in statement.names
            }
        return set()

    def walk(statements: list[ast.stmt], bound: set[str]) -> None:
        bound = set(bound)
        for statement in statements:
            if isinstance(statement, ast.Return):
                result[id(statement)] = set(bound)
            for block_name in ("body", "orelse", "finalbody"):
                block = getattr(statement, block_name, None)
                if (
                    isinstance(block, list) and block
                    and isinstance(block[0], ast.stmt)
                    and not isinstance(statement, (
                        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                    ))
                ):
                    walk(block, bound)
            for handler in getattr(statement, "handlers", ()) or ():
                walk(handler.body, bound)
            bound |= binds(statement)

    walk(definition.body, _parameter_names(definition.args))
    return result


def _copy_location(new: ast.AST, old: ast.AST) -> ast.AST:
    """Give generated nodes the authored location they stand for.

    Nodes copied from authored source keep their own locations."""

    for node in ast.walk(new):
        if (
            isinstance(node, (ast.expr, ast.stmt, ast.arg))
            and getattr(node, "lineno", None) is None
        ):
            ast.copy_location(node, old)
    return new


def _apply(plan: _ClosurePlan) -> None:
    definition = plan.definition
    bound_at = _definitely_bound_before(definition)
    # Returns become capture records.
    for node in list(_own_scope_nodes(definition)):
        if not isinstance(node, ast.Return):
            continue
        variant = plan.variant_of_return[id(node)]
        available = bound_at.get(id(node), set())
        elements: list[ast.expr] = [ast.Constant(variant.tag)]
        for slot in plan.slots:
            if slot in variant.captures or slot in available:
                elements.append(ast.Name(id=slot, ctx=ast.Load()))
            else:
                elements.append(ast.Constant(None))
        node.value = _copy_location(
            ast.Tuple(elts=elements, ctx=ast.Load()), node,
        )
    # The returned nested definitions no longer exist as functions.
    returned = {
        id(variant.node) for variant in plan.variants
        if isinstance(variant.node, ast.FunctionDef)
    }
    for node in list(ast.walk(definition)):
        for block_name in ("body", "orelse", "finalbody"):
            block = getattr(node, block_name, None)
            if not isinstance(block, list):
                continue
            kept = [
                statement for statement in block
                if id(statement) not in returned
            ]
            if len(kept) != len(block):
                if not kept:
                    kept = [_copy_location(ast.Pass(), block[0])]
                setattr(node, block_name, kept)
    definition.returns = None

    # The generated dispatcher holds every variant body.
    parameters = plan.variants[0].parameters
    record = plan.record_name
    branches: list[tuple[_Variant, list[ast.stmt]]] = []
    for variant in plan.variants:
        body: list[ast.stmt] = []
        if variant.parameters != parameters:
            renamed = [
                (mine, shared)
                for mine, shared in zip(variant.parameters, parameters)
                if mine != shared
            ]
            body.append(ast.Assign(
                targets=[ast.Tuple(
                    elts=[ast.Name(id=mine, ctx=ast.Store())
                          for mine, _ in renamed],
                    ctx=ast.Store(),
                )] if len(renamed) > 1 else [
                    ast.Name(id=renamed[0][0], ctx=ast.Store())
                ],
                value=ast.Tuple(
                    elts=[ast.Name(id=shared, ctx=ast.Load())
                          for _, shared in renamed],
                    ctx=ast.Load(),
                ) if len(renamed) > 1 else ast.Name(
                    id=renamed[0][1], ctx=ast.Load(),
                ),
            ))
        for capture in variant.captures:
            body.append(ast.Assign(
                targets=[ast.Name(id=capture, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=record, ctx=ast.Load()),
                    slice=ast.Constant(plan.slots.index(capture) + 1),
                    ctx=ast.Load(),
                ),
            ))
        if isinstance(variant.node, ast.Lambda):
            body.append(ast.Return(value=copy.deepcopy(variant.node.body)))
        else:
            body.extend(copy.deepcopy(variant.node.body))
        branches.append((variant, body))
    if len(branches) == 1:
        dispatch = branches[0][1]
    else:
        dispatch = branches[-1][1]
        for variant, body in reversed(branches[:-1]):
            dispatch = [ast.If(
                test=ast.Compare(
                    left=ast.Subscript(
                        value=ast.Name(id=record, ctx=ast.Load()),
                        slice=ast.Constant(0),
                        ctx=ast.Load(),
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(variant.tag)],
                ),
                body=body,
                orelse=dispatch,
            )]
    annotations: list[ast.expr | None] = []
    for position in range(len(parameters)):
        annotation = next((
            copy.deepcopy(variant.node.args.args[position].annotation)
            for variant in plan.variants
            if variant.node.args.args[position].annotation is not None
        ), None)
        annotations.append(annotation)
    returns = next((
        copy.deepcopy(variant.node.returns)
        for variant in plan.variants
        if isinstance(variant.node, ast.FunctionDef)
        and variant.node.returns is not None
    ), None)
    call_definition = ast.FunctionDef(
        name=plan.call_name,
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg=record),
                *(
                    ast.arg(arg=name, annotation=annotation)
                    for name, annotation in zip(parameters, annotations)
                ),
            ],
            vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None,
            defaults=[],
        ),
        body=dispatch,
        decorator_list=(
            [ast.Name(id="staticmethod", ctx=ast.Load())]
            if isinstance(plan.owner, ast.ClassDef) else []
        ),
        returns=returns,
        type_params=[],
    )
    _copy_location(call_definition, definition)
    position = plan.owner.body.index(definition)
    plan.owner.body.insert(position + 1, call_definition)
    if hasattr(definition, "_turing_source_module"):
        call_definition._turing_source_module = (
            definition._turing_source_module
        )

    # Calls through the closure call the dispatcher with the record.
    for kind, (call, reference) in plan.sites:
        callee = _call_receiver(reference, plan)
        _copy_location(callee, call.func)
        call.args = [call.func, *call.args]
        call.func = callee


def lower_returned_closures(tree: ast.Module) -> list[dict[str, Any]]:
    """Convert returned closures in ``tree`` in place; return receipts."""

    if not isinstance(tree, ast.Module):
        return []
    taken = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    receipts: list[dict[str, Any]] = []
    for definition, owner in list(_definitions(tree)):
        plan = _closure_plan(definition, owner, taken)
        if plan is None:
            continue
        parents = _parents(tree)
        if not _plan_sites(tree, plan, parents):
            continue
        _apply(plan)
        taken.add(plan.call_name)
        receipts.append({
            "definition": definition.name,
            "owner": getattr(owner, "name", None),
            "variants": len(plan.variants),
            "captures": plan.slots,
            "call_sites": len(plan.sites),
        })
    return receipts
