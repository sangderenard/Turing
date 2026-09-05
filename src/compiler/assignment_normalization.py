"""Normalize destructuring assignments into evaluate-once scalar assignments.

The ProcessGraph assignment path is deliberately simple: one target receives
one value.  Python destructuring is lowered to that vocabulary before graph
ingestion so every authored target goes through the ordinary assignment path.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class AssignmentNormalizationReceipt:
    line: int
    target_count: int
    temporary_names: tuple[str, ...]


class _DestructuringAssignmentNormalizer(ast.NodeTransformer):
    def __init__(self, tree: ast.AST):
        self._used_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        self._used_names.update(
            node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
        )
        self._next_temporary = 0
        self.receipts: list[AssignmentNormalizationReceipt] = []

    def _fresh_name(self) -> str:
        while True:
            name = f"__turing_assignment_value_{self._next_temporary}"
            self._next_temporary += 1
            if name not in self._used_names:
                self._used_names.add(name)
                return name

    @staticmethod
    def _load(name: str, location: ast.AST) -> ast.Name:
        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), location)

    @staticmethod
    def _projection(
        aggregate_name: str, projection: int | slice, location: ast.AST,
    ) -> ast.Subscript:
        if isinstance(projection, slice):
            slice_expression: ast.expr = ast.Slice(
                lower=(
                    None if projection.start is None
                    else ast.Constant(projection.start)
                ),
                upper=(
                    None if projection.stop is None
                    else ast.Constant(projection.stop)
                ),
                step=(
                    None if projection.step is None
                    else ast.Constant(projection.step)
                ),
            )
        else:
            slice_expression = ast.Constant(projection)
        return ast.copy_location(
            ast.Subscript(
                value=_DestructuringAssignmentNormalizer._load(
                    aggregate_name, location
                ),
                slice=slice_expression,
                ctx=ast.Load(),
            ),
            location,
        )

    def _iterate_assignment(
        self,
        target: ast.expr,
        value: ast.expr,
        *,
        location: ast.AST,
        temporaries: list[str],
        aggregate_name: str | None = None,
    ) -> Iterator[ast.Assign]:
        """Yield ordinary assignments for one target/value pair."""

        if not isinstance(target, (ast.Tuple, ast.List)):
            yield ast.copy_location(
                ast.Assign(targets=[target], value=value), location
            )
            return

        starred = tuple(
            index
            for index, element in enumerate(target.elts)
            if isinstance(element, ast.Starred)
        )
        if len(starred) > 1:
            # Python's parser normally rejects this.  Keep the invariant
            # explicit for programmatically constructed ASTs.
            raise ValueError("destructuring permits at most one starred target")
        starred_index = starred[0] if starred else None
        trailing = (
            len(target.elts) - starred_index - 1
            if starred_index is not None else 0
        )
        if aggregate_name is None:
            aggregate_name = self._fresh_name()
            temporaries.append(aggregate_name)
            yield ast.copy_location(
                ast.Assign(
                    targets=[ast.copy_location(
                        ast.Name(id=aggregate_name, ctx=ast.Store()), target
                    )],
                    value=value,
                ),
                location,
            )
        for index, element in enumerate(target.elts):
            if isinstance(element, ast.Starred):
                projection: int | slice = slice(
                    index, -trailing if trailing else None
                )
                binding_target = element.value
            else:
                projection = (
                    index - len(target.elts)
                    if starred_index is not None and index > starred_index
                    else index
                )
                binding_target = element
            projected_value: ast.expr = self._projection(
                aggregate_name, projection, binding_target
            )
            if isinstance(element, ast.Starred):
                # Python always binds a starred assignment target to a new
                # list, regardless of the source aggregate's concrete type.
                projected_value = ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="list", ctx=ast.Load()),
                        args=[projected_value],
                        keywords=[],
                    ),
                    binding_target,
                )
            yield from self._iterate_assignment(
                binding_target,
                projected_value,
                location=location,
                temporaries=temporaries,
            )

    def visit_Assign(self, node: ast.Assign):  # noqa: N802 - ast visitor API
        node = self.generic_visit(node)
        if not any(isinstance(target, (ast.Tuple, ast.List)) for target in node.targets):
            return node

        # A literal tuple/list RHS already exposes the exact source expression
        # paired with every flat target. Evaluate all elements into temporaries
        # first (preserving Python's evaluate-entire-RHS-before-stores rule),
        # then assign those real values directly. Building an aggregate temp
        # and synthetic ``temp[i]`` projections here loses the producer/source
        # correlation when numerical regions are later partitioned.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and isinstance(node.value, (ast.Tuple, ast.List))
            and len(node.targets[0].elts) == len(node.value.elts)
            and all(
                not isinstance(element, (ast.Tuple, ast.List, ast.Starred))
                for element in node.targets[0].elts
            )
        ):
            temporaries = [self._fresh_name() for _ in node.value.elts]
            normalized = [
                ast.copy_location(ast.Assign(
                    targets=[ast.copy_location(
                        ast.Name(id=name, ctx=ast.Store()), expression
                    )],
                    value=expression,
                ), node)
                for name, expression in zip(temporaries, node.value.elts)
            ]
            normalized.extend(
                ast.copy_location(ast.Assign(
                    targets=[target],
                    value=self._load(name, target),
                ), node)
                for target, name in zip(node.targets[0].elts, temporaries)
            )
            self.receipts.append(AssignmentNormalizationReceipt(
                line=int(getattr(node, "lineno", -1)),
                target_count=len(node.targets[0].elts),
                temporary_names=tuple(temporaries),
            ))
            return normalized

        temporaries: list[str] = []
        root_name = self._fresh_name()
        temporaries.append(root_name)
        normalized: list[ast.Assign] = [ast.copy_location(
            ast.Assign(
                targets=[ast.copy_location(
                    ast.Name(id=root_name, ctx=ast.Store()), node
                )],
                value=node.value,
            ),
            node,
        )]
        for target in node.targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                # The root temporary already owns the evaluate-once RHS.  Its
                # elements are the values paired with this authored target.
                for assignment in self._iterate_assignment(
                    target,
                    self._load(root_name, target),
                    location=node,
                    temporaries=temporaries,
                    aggregate_name=root_name,
                ):
                    normalized.append(assignment)
            else:
                normalized.append(ast.copy_location(
                    ast.Assign(
                        targets=[target],
                        value=self._load(root_name, target),
                    ),
                    node,
                ))
        self.receipts.append(AssignmentNormalizationReceipt(
            line=int(getattr(node, "lineno", -1)),
            target_count=sum(
                1
                for target in node.targets
                for member in ast.walk(target)
                if isinstance(member, (ast.Name, ast.Attribute, ast.Subscript))
                and isinstance(getattr(member, "ctx", None), ast.Store)
            ),
            temporary_names=tuple(temporaries),
        ))
        return normalized


def normalize_destructuring_assignments(
    tree: ast.AST,
) -> tuple[AssignmentNormalizationReceipt, ...]:
    """Rewrite tuple/list assignment in place and return audit receipts."""

    normalizer = _DestructuringAssignmentNormalizer(tree)
    normalizer.visit(tree)
    ast.fix_missing_locations(tree)
    return tuple(normalizer.receipts)


__all__ = [
    "AssignmentNormalizationReceipt",
    "normalize_destructuring_assignments",
]
