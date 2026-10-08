"""Source-AST control normalisation passes (relocated from fortran_c_shell)."""

from __future__ import annotations

import ast
import copy
from typing import Any, Callable, Iterable


def _single_exit_tuple_arity(returns: Iterable[ast.Return]) -> int:
    """Arity shared by every published return when all are tuple literals.

    Mirrors the tail-recursion exit rewrite: a tuple literal of one common
    arity at every site is published one private name per lane, so each lane
    keeps its own SSA producer through the control merge.  Any other shape
    (mixed arities, a non-tuple return, a starred element) reports 0 and the
    whole value is published through one name as before.
    """

    statements = tuple(returns)
    if not statements:
        return 0
    arities = set()
    for statement in statements:
        value = statement.value
        if not isinstance(value, ast.Tuple) or any(
            isinstance(element, ast.Starred) for element in value.elts
        ):
            return 0
        arities.add(len(value.elts))
    if len(arities) != 1:
        return 0
    return int(next(iter(arities)))


def _normalize_top_level_guard_returns(
    tree: ast.Module,
    target_names: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    """Give selected functions one exit without inventing control semantics.

    Repository control regions are owned by branch bodies.  A top-level guard
    whose body contains only an early ``return`` therefore has no numerical
    region to own and historically disappeared before SSA control lowering.
    For compiler-bootstrap targets, rewrite only that canonical guard form to
    assignments to a private result followed by one final return.  ProcessGraph
    can then retain both arms as ordinary nested control.  The authored source
    and its hash remain the public compilation receipt; this is a deterministic
    compiler-complementary AST normalization recorded separately below.
    """

    requested = {str(name) for name in target_names if str(name)}
    if not requested:
        return ()
    receipts: list[dict[str, Any]] = []

    def selected(qualified_name: str, simple_name: str) -> bool:
        return qualified_name in requested or simple_name in requested

    def normalize_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        qualified_name: str,
    ) -> None:
        if not selected(qualified_name, node.name) or not node.body:
            return
        if not isinstance(node.body[-1], ast.Return):
            return
        terminal = node.body[-1]
        if terminal.value is None:
            return

        occupied = {
            candidate.id
            for candidate in ast.walk(node)
            if isinstance(candidate, ast.Name)
        }

        guard_lines: list[int] = []
        rewritten_returns: list[ast.Return] = []

        def nest(
            statements: list[ast.stmt],
            emit: Callable[[ast.Return], list[ast.stmt]],
        ) -> list[ast.stmt] | None:
            for index, statement in enumerate(statements[:-1]):
                if not (
                    isinstance(statement, ast.If)
                    and not statement.orelse
                    and statement.body
                    and isinstance(statement.body[-1], ast.Return)
                    and statement.body[-1].value is not None
                ):
                    continue
                tail = nest(statements[index + 1 :], emit)
                if tail is None:
                    tail = [
                        *statements[index + 1 : -1],
                        *emit(statements[-1]),
                    ]
                guarded_return = statement.body[-1]
                rewritten = ast.If(
                    test=statement.test,
                    body=[
                        *statement.body[:-1],
                        *emit(guarded_return),
                    ],
                    orelse=tail,
                )
                ast.copy_location(rewritten, statement)
                guard_lines.append(int(getattr(statement, "lineno", 0)))
                return [*statements[:index], rewritten]
            return None

        def collect(statement: ast.Return) -> list[ast.stmt]:
            rewritten_returns.append(statement)
            return []

        # First pass: decide whether the canonical guard form is present and
        # gather every return the rewrite will publish, so the exit shape is
        # chosen from all sites at once (as the tail-recursion rewrite does).
        if nest(list(node.body), collect) is None:
            return
        guard_lines.clear()
        tuple_result_arity = _single_exit_tuple_arity(rewritten_returns)
        if tuple_result_arity:
            result_names = []
            for lane in range(tuple_result_arity):
                result_index = lane
                while (
                    f"__turing_single_exit_result_{result_index}" in occupied
                ):
                    result_index += tuple_result_arity
                result_name = f"__turing_single_exit_result_{result_index}"
                result_names.append(result_name)
                occupied.add(result_name)
        else:
            result_name = "__turing_single_exit_result"
            suffix = 0
            while result_name in occupied:
                suffix += 1
                result_name = f"__turing_single_exit_result_{suffix}"
            result_names = [result_name]

        def result_assignment(statement: ast.Return) -> list[ast.stmt]:
            value = statement.value
            if tuple_result_arity:
                assert isinstance(value, ast.Tuple)
                assignments: list[ast.stmt] = []
                for name, expression in zip(
                    result_names, value.elts, strict=True,
                ):
                    assignment = ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=expression,
                    )
                    assignments.append(ast.copy_location(
                        assignment, statement,
                    ))
                return assignments
            assignment = ast.Assign(
                targets=[ast.Name(id=result_names[0], ctx=ast.Store())],
                value=value,
            )
            return [ast.copy_location(assignment, statement)]

        rewritten_body = nest(list(node.body), result_assignment)
        assert rewritten_body is not None
        final_return = ast.copy_location(
            ast.Return(value=(
                ast.Tuple(
                    elts=[
                        ast.Name(id=name, ctx=ast.Load())
                        for name in result_names
                    ],
                    ctx=ast.Load(),
                )
                if tuple_result_arity else
                ast.Name(id=result_names[0], ctx=ast.Load())
            )),
            terminal,
        )
        node.body = [*rewritten_body, final_return]
        receipts.append({
            "function": qualified_name,
            "result_name": result_names[0],
            "result_names": tuple(result_names),
            "tuple_result_arity": int(tuple_result_arity),
            "guard_count": len(guard_lines),
            "source_lines": tuple(sorted(guard_lines)),
        })

    def walk_scope(statements: Iterable[ast.stmt], prefix: str = "") -> None:
        for statement in statements:
            if isinstance(statement, ast.ClassDef):
                qualified = (
                    f"{prefix}.{statement.name}" if prefix else statement.name
                )
                walk_scope(statement.body, qualified)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = (
                    f"{prefix}.{statement.name}" if prefix else statement.name
                )
                normalize_function(statement, qualified)
                # Nested definitions retain their authored lexical identity.
                walk_scope(statement.body, f"{qualified}.<locals>")

    walk_scope(tree.body)
    if receipts:
        ast.fix_missing_locations(tree)
    return tuple(receipts)


def _normalize_direct_tail_recursion(
    tree: ast.Module,
) -> tuple[dict[str, Any], ...]:
    """Turn direct tail calls into an ordinary authored-control loop.

    A direct ``return f(...)`` in ``f`` does not require a native language
    stack frame: Python evaluates every argument before rebinding the next
    frame, exactly as a simultaneous assignment followed by ``continue``.
    Expressing that identity in the shared AST normalization keeps recursive
    call graphs finite for the deployment planner and gives every backend the
    same loop. Calls which are not direct tail returns, use ``*args`` or
    ``**kwargs``, or omit a required argument remain untouched.
    """

    receipts: list[dict[str, Any]] = []

    def normalize_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        qualified_name: str,
    ) -> None:
        arguments = node.args
        if arguments.vararg is not None or arguments.kwarg is not None:
            return
        positional = [*arguments.posonlyargs, *arguments.args]
        positional_names = [argument.arg for argument in positional]
        keyword_only_names = [argument.arg for argument in arguments.kwonlyargs]
        parameter_names = [*positional_names, *keyword_only_names]
        positional_defaults = {
            positional_names[
                len(positional_names) - len(arguments.defaults) + index
            ]: default
            for index, default in enumerate(arguments.defaults)
        }
        keyword_defaults = {
            name: default
            for name, default in zip(
                keyword_only_names, arguments.kw_defaults, strict=True,
            )
            if default is not None
        }

        class TailReturnRewriter(ast.NodeTransformer):
            def __init__(self) -> None:
                self.count = 0

            def visit_FunctionDef(self, nested):  # noqa: N802
                return nested

            def visit_AsyncFunctionDef(self, nested):  # noqa: N802
                return nested

            def visit_Lambda(self, nested):  # noqa: N802
                return nested

            def visit_ClassDef(self, nested):  # noqa: N802
                return nested

            def visit_Return(self, statement):  # noqa: N802
                value = statement.value
                if not (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == node.name
                    and not any(
                        keyword.arg is None for keyword in value.keywords
                    )
                    and len(value.args) <= len(positional_names)
                ):
                    return self.generic_visit(statement)
                supplied = {
                    name: expression
                    for name, expression in zip(positional_names, value.args)
                }
                for keyword in value.keywords:
                    assert keyword.arg is not None
                    if (
                        keyword.arg not in parameter_names
                        or keyword.arg in supplied
                    ):
                        return self.generic_visit(statement)
                    supplied[keyword.arg] = keyword.value
                rebound = []
                for name in positional_names:
                    expression = supplied.get(
                        name, positional_defaults.get(name)
                    )
                    if expression is None:
                        return self.generic_visit(statement)
                    rebound.append(copy.deepcopy(expression))
                for name in keyword_only_names:
                    expression = supplied.get(
                        name, keyword_defaults.get(name)
                    )
                    if expression is None:
                        return self.generic_visit(statement)
                    rebound.append(copy.deepcopy(expression))
                assignment = ast.Assign(
                    targets=[ast.Tuple(
                        elts=[
                            ast.Name(id=name, ctx=ast.Store())
                            for name in parameter_names
                        ],
                        ctx=ast.Store(),
                    )],
                    value=ast.Tuple(elts=rebound, ctx=ast.Load()),
                )
                ast.copy_location(assignment, statement)
                continuation = ast.copy_location(ast.Continue(), statement)
                self.count += 1
                return [assignment, continuation]

        rewriter = TailReturnRewriter()
        rewritten = []
        for statement in node.body:
            transformed = rewriter.visit(statement)
            if isinstance(transformed, list):
                rewritten.extend(transformed)
            elif transformed is not None:
                rewritten.append(transformed)
        if not rewriter.count:
            return
        occupied_names = {
            member.id for member in ast.walk(node)
            if isinstance(member, ast.Name)
        }
        remaining_returns = tuple(
            member for statement in rewritten for member in ast.walk(statement)
            if isinstance(member, ast.Return)
        )
        tuple_arities = {
            len(statement.value.elts)
            for statement in remaining_returns
            if isinstance(statement.value, ast.Tuple)
        }
        tuple_result_arity = (
            next(iter(tuple_arities))
            if len(tuple_arities) == 1
            and all(
                isinstance(statement.value, ast.Tuple)
                for statement in remaining_returns
            )
            else 0
        )
        result_names = []
        for lane in range(max(1, tuple_result_arity)):
            result_index = lane
            while f"__turing_tail_result_{result_index}" in occupied_names:
                result_index += max(1, tuple_result_arity)
            result_name = f"__turing_tail_result_{result_index}"
            result_names.append(result_name)
            occupied_names.add(result_name)

        class ExitReturnRewriter(ast.NodeTransformer):
            """Publish every non-recursive exit after the retry loop."""

            def visit_FunctionDef(self, nested):  # noqa: N802
                return nested

            def visit_AsyncFunctionDef(self, nested):  # noqa: N802
                return nested

            def visit_Lambda(self, nested):  # noqa: N802
                return nested

            def visit_ClassDef(self, nested):  # noqa: N802
                return nested

            def visit_Return(self, statement):  # noqa: N802
                value = (
                    statement.value
                    if statement.value is not None
                    else ast.Constant(value=None)
                )
                if tuple_result_arity:
                    assert isinstance(value, ast.Tuple)
                    assignments = []
                    for name, expression in zip(
                        result_names, value.elts, strict=True,
                    ):
                        assignment = ast.Assign(
                            targets=[ast.Name(id=name, ctx=ast.Store())],
                            value=expression,
                        )
                        assignments.append(ast.copy_location(
                            assignment, statement,
                        ))
                else:
                    assignment = ast.Assign(
                        targets=[ast.Name(
                            id=result_names[0], ctx=ast.Store(),
                        )],
                        value=value,
                    )
                    assignments = [ast.copy_location(assignment, statement)]
                return [
                    *assignments,
                    ast.copy_location(ast.Break(), statement),
                ]

        exit_rewriter = ExitReturnRewriter()
        exited = []
        for statement in rewritten:
            transformed = exit_rewriter.visit(statement)
            if isinstance(transformed, list):
                exited.extend(transformed)
            elif transformed is not None:
                exited.append(transformed)
        rewritten = exited
        leading = []
        if (
            rewritten
            and isinstance(rewritten[0], ast.Expr)
            and isinstance(rewritten[0].value, ast.Constant)
            and isinstance(rewritten[0].value.value, str)
        ):
            leading.append(rewritten.pop(0))
        loop = ast.While(
            test=ast.Constant(value=True), body=rewritten, orelse=[]
        )
        ast.copy_location(loop, node)
        final_return = ast.copy_location(
            ast.Return(value=(
                ast.Tuple(
                    elts=[
                        ast.Name(id=name, ctx=ast.Load())
                        for name in result_names
                    ],
                    ctx=ast.Load(),
                )
                if tuple_result_arity else
                ast.Name(id=result_names[0], ctx=ast.Load())
            )), node
        )
        node.body = [*leading, loop, final_return]
        receipts.append({
            "function": qualified_name,
            "tail_call_count": int(rewriter.count),
        })

    def walk_scope(statements: Iterable[ast.stmt], prefix: str = "") -> None:
        for statement in statements:
            if isinstance(statement, ast.ClassDef):
                qualified = (
                    f"{prefix}.{statement.name}" if prefix else statement.name
                )
                walk_scope(statement.body, qualified)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = (
                    f"{prefix}.{statement.name}" if prefix else statement.name
                )
                normalize_function(statement, qualified)
                walk_scope(statement.body, f"{qualified}.<locals>")

    walk_scope(tree.body)
    if receipts:
        ast.fix_missing_locations(tree)
    return tuple(receipts)


def _normalize_none_default_assignments(
    tree: ast.Module,
) -> tuple[dict[str, Any], ...]:
    """Turn scalar None-default guards into explicit SSA value choices."""

    receipts: list[dict[str, Any]] = []

    class Rewriter(ast.NodeTransformer):
        def visit_If(self, statement):  # noqa: N802
            statement = self.generic_visit(statement)
            test = statement.test
            if not (
                not statement.orelse
                and len(statement.body) == 1
                and isinstance(statement.body[0], ast.Assign)
                and len(statement.body[0].targets) == 1
                and isinstance(statement.body[0].targets[0], ast.Name)
                and isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Is)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None
                and isinstance(test.left, ast.Name)
                and test.left.id == statement.body[0].targets[0].id
                and not isinstance(
                    statement.body[0].value,
                    (ast.List, ast.Dict, ast.Set, ast.Tuple),
                )
            ):
                return statement
            assignment = statement.body[0]
            name = assignment.targets[0].id
            replacement = ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.IfExp(
                    test=test,
                    body=assignment.value,
                    orelse=ast.Name(id=name, ctx=ast.Load()),
                ),
            )
            receipts.append({"binding": name})
            return ast.copy_location(replacement, statement)

    Rewriter().visit(tree)
    if receipts:
        ast.fix_missing_locations(tree)
    return tuple(receipts)
