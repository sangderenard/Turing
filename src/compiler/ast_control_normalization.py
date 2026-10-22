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


def _pure_operand(node: ast.AST) -> bool:
    """Whether evaluating ``node`` twice is the same as evaluating it once:
    reads of names, constants, attributes and subscripts, and arithmetic or
    comparison over such reads.  A call (or anything else) is not."""

    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.Attribute):
        return _pure_operand(node.value)
    if isinstance(node, ast.Subscript):
        return _pure_operand(node.value) and _pure_operand(node.slice)
    if isinstance(node, ast.UnaryOp):
        return _pure_operand(node.operand)
    if isinstance(node, ast.BinOp):
        return _pure_operand(node.left) and _pure_operand(node.right)
    if isinstance(node, ast.Tuple):
        return all(_pure_operand(element) for element in node.elts)
    return False


def _normalize_chained_comparisons(
    tree: ast.Module,
) -> tuple[dict[str, Any], ...]:
    """Spell ``a < b < c`` as ``a < b and b < c``.

    Program: ``if 0 < n < 10: r += 1`` (scratch c208).  Wrong record: the
    planner classifies a chained Compare as dispatch metadata and nothing
    lowered it, so the predicate became a minted root input (always 0).
    Rule: Python defines the chain as the conjunction of its adjacent
    comparisons, each middle operand evaluated once.  A pure middle operand
    reads the same value twice; a chain with an impure middle operand (a
    call) is refused by name rather than evaluated twice.
    """

    receipts: list[dict[str, Any]] = []

    class Rewriter(ast.NodeTransformer):
        def visit_Compare(self, node):  # noqa: N802
            node = self.generic_visit(node)
            if len(node.ops) <= 1:
                return node
            middles = node.comparators[:-1]
            impure = next((
                middle for middle in middles if not _pure_operand(middle)
            ), None)
            if impure is not None:
                raise ValueError(
                    "chained comparison at line "
                    f"{getattr(node, 'lineno', '?')} has a middle operand "
                    f"with effects ({ast.unparse(impure)!r}); the chain "
                    "evaluates it once, which this lowering cannot spell"
                )
            operands = [node.left, *node.comparators]
            comparisons = []
            for index, operator in enumerate(node.ops):
                # A middle operand is the right side of one comparison and
                # the left side of the next: two occurrences, two AST
                # objects (graph nodes are keyed by occurrence).
                left = (
                    copy.deepcopy(operands[index]) if index > 0
                    else operands[index]
                )
                comparison = ast.Compare(
                    left=left, ops=[operator],
                    comparators=[operands[index + 1]],
                )
                ast.copy_location(comparison, operands[index])
                comparison.end_lineno = operands[index + 1].end_lineno
                comparison.end_col_offset = operands[index + 1].end_col_offset
                comparisons.append(comparison)
            receipts.append({
                "line": int(getattr(node, "lineno", -1)),
                "column": int(getattr(node, "col_offset", -1)),
                "comparisons": len(comparisons),
            })
            conjunction = ast.BoolOp(op=ast.And(), values=comparisons)
            return ast.copy_location(conjunction, node)

    Rewriter().visit(tree)
    return tuple(receipts)


def _normalize_constant_membership(
    tree: ast.Module,
) -> tuple[dict[str, Any], ...]:
    """Spell ``x in [1, 2, 3]`` as ``x == 1 or x == 2 or x == 3``.

    Program: ``if n in [6, 3, 4]: r = 1`` (fuzzer p3_25).  Wrong record:
    the membership against an inline literal reached the graph as a
    coordinator Call over a folded constant; no region and no resident
    query produced it, so its predicate had no value (refused at C
    emission).  Rule: Python defines membership in a literal collection as
    the disjunction of equalities with its members, in order (``not in``:
    the conjunction of inequalities).  Only numeric literal members and a
    pure left operand are spelled this way; any other membership keeps its
    existing lowering.
    """

    receipts: list[dict[str, Any]] = []

    def numeric(element: ast.AST) -> bool:
        return isinstance(element, ast.Constant) and isinstance(
            element.value, (bool, int, float)
        )

    class Rewriter(ast.NodeTransformer):
        def visit_Compare(self, node):  # noqa: N802
            node = self.generic_visit(node)
            if (
                len(node.ops) != 1
                or not isinstance(node.ops[0], (ast.In, ast.NotIn))
                or not isinstance(
                    node.comparators[0], (ast.List, ast.Tuple, ast.Set),
                )
                or not node.comparators[0].elts
                or not all(numeric(e) for e in node.comparators[0].elts)
                or not _pure_operand(node.left)
            ):
                return node
            negate = isinstance(node.ops[0], ast.NotIn)
            comparisons = []
            for index, member in enumerate(node.comparators[0].elts):
                comparison = ast.Compare(
                    left=(
                        node.left if index == 0 else copy.deepcopy(node.left)
                    ),
                    ops=[ast.NotEq() if negate else ast.Eq()],
                    comparators=[member],
                )
                ast.copy_location(comparison, node)
                comparisons.append(comparison)
            receipts.append({
                "line": int(getattr(node, "lineno", -1)),
                "column": int(getattr(node, "col_offset", -1)),
                "members": len(comparisons),
            })
            if len(comparisons) == 1:
                return comparisons[0]
            return ast.copy_location(ast.BoolOp(
                op=ast.And() if negate else ast.Or(), values=comparisons,
            ), node)

    Rewriter().visit(tree)
    if receipts:
        ast.fix_missing_locations(tree)
    return tuple(receipts)


def _normalize_binary_short_circuit(
    tree: ast.Module,
) -> tuple[dict[str, Any], ...]:
    """Right-nest every n-ary ``and``/``or`` into binary operators.

    Python evaluates ``a and b and c`` exactly as ``a and (b and c)``: the
    same operands, in the same order, each at most once, with the same
    result value.  A binary operator's second operand is then the whole
    guarded remainder, which is the arm the short-circuit conditional owns
    (``_short_circuit_control_program``).  The nested operator spans its
    operands' authored source, so lexical ownership reads real extents.
    """

    receipts: list[dict[str, Any]] = []

    class Rewriter(ast.NodeTransformer):
        def visit_BoolOp(self, node):  # noqa: N802
            node = self.generic_visit(node)
            if len(node.values) <= 2:
                return node
            receipts.append({
                "line": int(getattr(node, "lineno", -1)),
                "column": int(getattr(node, "col_offset", -1)),
                "operands": len(node.values),
                "operator": type(node.op).__name__,
            })
            remainder = node.values[-1]
            for operand in reversed(node.values[1:-1]):
                nested = ast.BoolOp(op=type(node.op)(), values=[operand, remainder])
                nested.lineno = operand.lineno
                nested.col_offset = operand.col_offset
                nested.end_lineno = remainder.end_lineno
                nested.end_col_offset = remainder.end_col_offset
                remainder = nested
            node.values = [node.values[0], remainder]
            return node

    Rewriter().visit(tree)
    return tuple(receipts)


def _normalize_loop_else(
    tree: ast.Module,
) -> tuple[dict[str, Any], ...]:
    """Lower a loop's ``else`` arm to the conditional Python defines it as.

    Program: ``for i in range(10): if i * i > n: break`` with ``else: return
    -1`` then ``return i`` (scratch c106).  Wrong record: Control IR has no
    loop ``else``; the arm was laid after the loop unconditionally, so every
    exit -- the ``break`` included -- returned -1 (also at the session base,
    with a clean audit).  Rule: ``for/while ... else E`` runs E exactly when
    no ``break`` of THAT loop ran.  Each such ``break`` first sets a private
    flag, and E follows the loop under ``if not flag``.  Breaks of nested
    loops and nested definitions are their own.  A loop with no break runs
    E unconditionally.  The synthesized statements carry the source
    positions of the ``break`` and of E, so lexical ownership reads real
    extents.
    """

    receipts: list[dict[str, Any]] = []

    def own_breaks(statements: list[ast.stmt]) -> list[ast.Break]:
        found: list[ast.Break] = []

        def walk(node: ast.AST) -> None:
            if isinstance(node, ast.Break):
                found.append(node)
                return
            if isinstance(node, (
                ast.For, ast.AsyncFor, ast.While,
                ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                ast.ClassDef,
            )):
                return
            for child in ast.iter_child_nodes(node):
                walk(child)

        for statement in statements:
            walk(statement)
        return found

    def located(node: ast.AST, source: ast.AST, end: ast.AST | None = None):
        end = end or source
        node.lineno = source.lineno
        node.col_offset = source.col_offset
        node.end_lineno = getattr(end, "end_lineno", source.lineno)
        node.end_col_offset = getattr(end, "end_col_offset", source.col_offset)
        return ast.fix_missing_locations(node)

    class Rewriter(ast.NodeTransformer):
        def rewrite_block(self, statements: list[ast.stmt]) -> list[ast.stmt]:
            rewritten: list[ast.stmt] = []
            for statement in statements:
                statement = self.visit(statement)
                if (
                    isinstance(statement, (ast.For, ast.AsyncFor, ast.While))
                    and statement.orelse
                ):
                    rewritten.extend(self.lower(statement))
                else:
                    rewritten.append(statement)
            return rewritten

        def lower(self, loop: ast.stmt) -> list[ast.stmt]:
            orelse = list(loop.orelse)
            loop.orelse = []
            # The loop's extent no longer includes the arm: the arm runs
            # after the loop, and readers of the loop's exit values there
            # are post-loop reads.
            loop.end_lineno = getattr(loop.body[-1], "end_lineno", loop.end_lineno)
            loop.end_col_offset = getattr(
                loop.body[-1], "end_col_offset", loop.end_col_offset,
            )
            breaks = own_breaks(loop.body)
            receipts.append({
                "line": int(loop.lineno),
                "column": int(loop.col_offset),
                "breaks": len(breaks),
            })
            if not breaks:
                return [loop, *orelse]
            flag = f"__turing_loop_broke_{loop.lineno}_{loop.col_offset}"
            first, last = orelse[0], orelse[-1]

            class BreakFlag(ast.NodeTransformer):
                def generic_visit(self, node):  # noqa: D401
                    if isinstance(node, (
                        ast.For, ast.AsyncFor, ast.While,
                        ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                        ast.ClassDef,
                    )) and node is not loop:
                        return node
                    for field, value in ast.iter_fields(node):
                        if isinstance(value, list) and value and all(
                            isinstance(item, ast.stmt) for item in value
                        ):
                            expanded: list[ast.stmt] = []
                            for item in value:
                                if isinstance(item, ast.Break) and any(
                                    item is own for own in breaks
                                ):
                                    expanded.append(located(ast.Assign(
                                        targets=[ast.Name(id=flag, ctx=ast.Store())],
                                        value=ast.Constant(value=True),
                                    ), item))
                                    expanded.append(item)
                                else:
                                    expanded.append(self.visit(item))
                            setattr(node, field, expanded)
                        elif isinstance(value, ast.AST):
                            self.visit(value)
                    return node

            BreakFlag().generic_visit(loop)
            initial = located(ast.Assign(
                targets=[ast.Name(id=flag, ctx=ast.Store())],
                value=ast.Constant(value=False),
            ), breaks[0])
            guard = located(ast.If(
                test=ast.UnaryOp(
                    op=ast.Not(), operand=ast.Name(id=flag, ctx=ast.Load()),
                ),
                body=orelse,
                orelse=[],
            ), first, last)
            return [initial, loop, guard]

        def generic_visit(self, node):  # noqa: D401
            for field, value in ast.iter_fields(node):
                if isinstance(value, list) and value and all(
                    isinstance(item, ast.stmt) for item in value
                ):
                    setattr(node, field, self.rewrite_block(value))
                elif isinstance(value, list):
                    setattr(node, field, [
                        self.visit(item) if isinstance(item, ast.AST) else item
                        for item in value
                    ])
                elif isinstance(value, ast.AST):
                    setattr(node, field, self.visit(value))
            return node

    Rewriter().visit(tree)
    return tuple(receipts)
