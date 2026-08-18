"""Turn SSA definitions back into real Python ``ast`` nodes.

``oop_schema`` names the two halves of an object crossing: ``ClassSchema`` is
the transportable description, and a *destination* reads it to build its own
construct. This module is the Python destination. It takes the definition
objects the SSA layer holds -- ``SSAClassDefinition``/``SSAClassTable``, or
the richer ``ClassSchema`` they project onto -- and manifests them as genuine
``ast.ClassDef`` and ``ast.FunctionDef`` nodes: inspectable, ``ast.unparse``
-able, and executable once compiled.

This is the direction that makes a definition executable Python again after a
round trip, and it is why the definition operators exist at all. A method that
crosses a boundary as a name arrives without its body, so an authored constant
inside that body -- a custom trig epsilon is the motivating case -- does not
survive. Materializing the body from the SSA function is what brings the
constant back with it.

What this module refuses to do is as much the point as what it does.

* **The scalar vocabulary is borrowed, never invented.** The spellings below
  are audited against ``ssa_llvm_backend``'s likeness tables at import, the
  same way ``ssa_reference_evaluator`` audits its own semantics. An opcode
  spelled here but absent there is a vocabulary this file made up, and the
  import fails. Opcodes with no exact Python form stay *declared* unimplemented
  and raise when reached -- ``ULt``/``ULe``/``SExt``/``ZExt``/``FpToUi`` need a
  stated bit width that a Python ``int`` does not carry, and picking one
  silently is precisely the failure this tree keeps paying for.
* **Only straight-line bodies.** A multi-block function is a control-flow
  graph, and turning one back into ``if``/``while`` is structural
  reconstruction -- a separate piece of work with its own correctness
  argument. Reaching one raises and names the blocks rather than emitting
  something that reads like Python and means something else.
* **A missing body is visible.** A method whose SSA function is not supplied
  materializes to a ``raise NotImplementedError`` naming what is missing, not
  to ``pass``. A class that silently no-ops is worse than one that will not
  run.

The generated code depends on ``math`` and nothing else; ``materialize_module``
emits the import when any body needs it.
"""

from __future__ import annotations

import ast
from typing import Any, Iterable, Mapping, Sequence

from ..transmogrifier.ssa import (
    SSAClassDefinition,
    SSAClassTable,
)
from .oop_schema import (
    UNKNOWN_TYPE,
    ClassSchema,
    MethodSchema,
    schemas_from_ssa_class_table,
)
from .ssa_llvm_backend import _BINARY as _LIKENESS_BINARY
from .ssa_llvm_backend import _UNARY as _LIKENESS_UNARY

# ``_declared_formal_names`` reads all three spellings a function may use to
# state its own port names. Reproducing that here would be a second reading of
# the same record, free to drift from the one the reference evaluator uses.
from .ssa_reference_evaluator import _declared_formal_names


class MaterializationError(RuntimeError):
    """Raised instead of emitting Python whose meaning is a guess."""


# -- the scalar vocabulary, as Python source fragments ----------------------
#
# Each entry is a format string over its operands, already parenthesised so it
# composes without precedence surprises. The spellings match
# ``ssa_reference_evaluator``'s semantics, which is the reading the backends
# are checked against -- notably ``Mod`` and ``FloorDiv``, where Python's own
# floored ``%`` and ``//`` are the correct form rather than C's truncating one.

_BINARY_SPELLING: dict[str, str] = {
    "Add": "({0} + {1})",
    "Sub": "({0} - {1})",
    "Mul": "({0} * {1})",
    "Div": "({0} / {1})",
    "Max": "max({0}, {1})",
    "Min": "min({0}, {1})",
    "Lt": "({0} < {1})",
    "Le": "({0} <= {1})",
    "Gt": "({0} > {1})",
    "Ge": "({0} >= {1})",
    "Eq": "({0} == {1})",
    "Ne": "({0} != {1})",
    "And": "({0} and {1})",
    "Or": "({0} or {1})",
    "Xor": "(bool({0}) != bool({1}))",
    "LAnd": "({0} and {1})",
    "LOr": "({0} or {1})",
    "BitAnd": "(int({0}) & int({1}))",
    "BitOr": "(int({0}) | int({1}))",
    "BitXor": "(int({0}) ^ int({1}))",
    "Shl": "(int({0}) << int({1}))",
    "Shr": "(int({0}) >> int({1}))",
    "Mod": "({0} % {1})",
    "FloorDiv": "({0} // {1})",
    "Pow": "({0} ** {1})",
}

_UNARY_SPELLING: dict[str, str] = {
    "Neg": "(-{0})",
    "Abs": "abs({0})",
    "Sqrt": "math.sqrt({0})",
    "Exp": "math.exp({0})",
    "Log": "math.log({0})",
    "Sin": "math.sin({0})",
    "Cos": "math.cos({0})",
    "Floor": "math.floor({0})",
    "Ceil": "math.ceil({0})",
    "Trunc": "math.trunc({0})",
    "Round": "round({0})",
    "Not": "(not {0})",
    "LNot": "(not {0})",
    "Invert": "(~int({0}))",
    "SIToFP": "float({0})",
    "SiToFp": "float({0})",
    "UiToFp": "float({0})",
    "FPToSI": "int({0})",
    "FpToSi": "int({0})",
}

# Spellings that reach for ``math``; the module import is emitted only when one
# of these is actually used, so a body of plain arithmetic stays dependency-free.
_NEEDS_MATH = frozenset(
    name for name, form in _UNARY_SPELLING.items() if form.startswith("math.")
)


def _audit_vocabulary() -> tuple[frozenset[str], frozenset[str]]:
    """Compare these spellings against the compiler's own likeness table.

    Same contract as ``ssa_reference_evaluator._audit_vocabulary``:
    ``invented`` must always be empty, while ``unimplemented`` is the honest,
    declared edge of this materializer.
    """

    ours = set(_BINARY_SPELLING) | set(_UNARY_SPELLING)
    theirs = set(_LIKENESS_BINARY) | set(_LIKENESS_UNARY)
    return frozenset(ours - theirs), frozenset(theirs - ours)


INVENTED, UNIMPLEMENTED = _audit_vocabulary()
if INVENTED:
    raise ImportError(
        "ssa_python_materializer spells opcodes absent from the compiler's "
        f"own likeness table: {sorted(INVENTED)}. The table in "
        "ssa_llvm_backend owns the vocabulary; this file may only supply a "
        "Python form for what is already in it."
    )


def _expression(source: str) -> ast.expr:
    """One source fragment as a real expression node.

    Parsed rather than hand-built, for the reason ``fused_program_python_backend``
    gives: the result is still genuine ``ast``, and a fragment that does not
    parse fails here rather than at ``unparse`` time.
    """

    return ast.parse(source, mode="eval").body


def _local(value_id: int) -> str:
    return f"t{int(value_id)}"


def _constant(literal: Any) -> str:
    return repr(literal)


class _BodyMaterializer:
    """One SSA function's single block, as Python statements."""

    def __init__(self, function: Any, *, argument_names: Mapping[int, str]):
        self.function = function
        self.names: dict[int, str] = dict(argument_names)
        self.statements: list[ast.stmt] = []
        self.uses_math = False

    def operand(self, value: Any) -> str:
        value_id = int(value.id)
        name = self.names.get(value_id)
        if name is None:
            raise MaterializationError(
                f"{self.function.name}: value %t{value_id} is used before it "
                "is produced; the block is not in dependency order"
            )
        return name

    def assign(self, result: Any, source: str) -> None:
        target = _local(result.id)
        self.names[int(result.id)] = target
        statement = ast.Assign(
            targets=[ast.Name(id=target, ctx=ast.Store())],
            value=_expression(source),
        )
        self.statements.append(statement)

    def step(self, instruction: Any) -> None:
        operation = str(instruction.op)
        attributes = instruction.attributes or {}
        result = instruction.res

        if operation in {"Const", "const"}:
            if "value" not in attributes:
                raise MaterializationError(
                    f"{self.function.name}: Const %t{int(result.id)} carries "
                    "no 'value' attribute, so there is no literal to emit"
                )
            self.assign(result, _constant(attributes["value"]))
            return

        if operation in {"GetAttr", "getattr"}:
            field = attributes.get("attribute")
            if not field:
                raise MaterializationError(
                    f"{self.function.name}: GetAttr %t{int(result.id)} names "
                    "no attribute; a slot number alone cannot be spelled in "
                    "Python without inventing the field's name"
                )
            receiver = self.operand(instruction.args[0])
            self.assign(result, f"{receiver}.{field}")
            return

        if operation in {"SetAttr", "setattr"}:
            field = attributes.get("attribute")
            if not field:
                raise MaterializationError(
                    f"{self.function.name}: SetAttr names no attribute"
                )
            receiver = self.operand(instruction.args[0])
            payload = self.operand(instruction.args[1])
            target = _expression(f"{receiver}.{field}")
            # Parsed as an expression, so it arrives in Load context; an
            # assignment target has to say Store or ``compile`` refuses it.
            target.ctx = ast.Store()
            self.statements.append(
                ast.Assign(targets=[target], value=_expression(payload))
            )
            # A write publishes no SSA value; the catalog records setattr as
            # ``returns: "void"`` for exactly this reason.
            return

        if operation in {"Select", "select"} and len(instruction.args) == 3:
            mask, when_true, when_false = (
                self.operand(argument) for argument in instruction.args
            )
            self.assign(
                result, f"({when_true} if {mask} else {when_false})"
            )
            return

        if operation in {"Call", "call"}:
            callee = str(attributes.get("callee") or "")
            if not callee:
                raise MaterializationError(
                    f"{self.function.name}: Call %t{int(result.id)} names no "
                    "callee"
                )
            operands = ", ".join(
                self.operand(argument) for argument in instruction.args
            )
            self.assign(result, f"{callee}({operands})")
            return

        if operation in _BINARY_SPELLING:
            left = self.operand(instruction.args[0])
            right = self.operand(instruction.args[1])
            self.assign(result, _BINARY_SPELLING[operation].format(left, right))
            return

        if operation in _UNARY_SPELLING:
            if operation in _NEEDS_MATH:
                self.uses_math = True
            operand = self.operand(instruction.args[0])
            self.assign(result, _UNARY_SPELLING[operation].format(operand))
            return

        if operation in UNIMPLEMENTED:
            raise MaterializationError(
                f"{self.function.name}: {operation!r} has no exact Python "
                "spelling here. It needs a stated bit width that a Python int "
                "does not carry, and choosing one silently would change the "
                "program's meaning."
            )

        raise MaterializationError(
            f"{self.function.name}: no Python form for {operation!r}. Add it "
            "deliberately rather than letting this materializer guess."
        )

    def finish(self, returned: Sequence[Any]) -> None:
        if not returned:
            return
        if len(returned) == 1:
            value = _expression(self.operand(returned[0]))
        else:
            value = _expression(
                "(" + ", ".join(self.operand(each) for each in returned) + ")"
            )
        self.statements.append(ast.Return(value=value))


def _single_block(function: Any) -> Any:
    blocks = list(getattr(function, "blocks", {}).values())
    if len(blocks) != 1:
        names = sorted(getattr(function, "blocks", {}))
        raise MaterializationError(
            f"{getattr(function, 'name', '<anonymous>')} has {len(blocks)} "
            f"blocks ({names}); reconstructing structured control flow from a "
            "CFG is a separate piece of work and is not attempted here"
        )
    return blocks[0]


def materialize_function_body(
    function: Any, *, parameter_names: Sequence[str] | None = None,
) -> tuple[list[ast.stmt], bool]:
    """One SSA function's body as statements, plus whether it needs ``math``.

    ``parameter_names`` binds positionally to the function's formals when the
    caller has better names than the function states -- a ``MethodSchema``'s
    signature, typically. When it is omitted the function's own declared port
    names are used, and a formal that names itself nowhere becomes ``t<id>``.
    """

    formals = tuple(getattr(function, "args", ()))
    if parameter_names is not None:
        if len(parameter_names) != len(formals):
            raise MaterializationError(
                f"{function.name}: {len(parameter_names)} parameter names for "
                f"{len(formals)} formals"
            )
        names = {
            int(formal.id): str(name)
            for formal, name in zip(formals, parameter_names)
        }
    else:
        declared = _declared_formal_names(function)
        by_position = {position: name for name, position in declared.items()}
        names = {
            int(formal.id): by_position.get(position, _local(formal.id))
            for position, formal in enumerate(formals)
        }

    block = _single_block(function)
    materializer = _BodyMaterializer(function, argument_names=names)
    returned: tuple[Any, ...] = ()
    for instruction in block.instrs:
        operation = str(instruction.op)
        if operation in {"Ret", "ret", "Return", "return"}:
            returned = tuple(instruction.args)
            break
        if operation in {"Br", "br", "CondBr", "condbr"}:
            raise MaterializationError(
                f"{function.name}: a branch reached a single-block body; the "
                "block list and the terminator disagree"
            )
        materializer.step(instruction)
    materializer.finish(returned)

    statements = materializer.statements or [ast.Pass()]
    return statements, materializer.uses_math


def _annotation(type_name: str) -> ast.expr | None:
    """The declared type as an annotation, or nothing when it is unknown.

    ``UNKNOWN_TYPE`` is deliberately not rendered: an unmapped type must stay
    visible as absent rather than be annotated with the word "unknown", which
    would read as a real type to anyone downstream.
    """

    name = str(type_name or "").strip()
    if not name or name == UNKNOWN_TYPE:
        return None
    try:
        return _expression(name)
    except SyntaxError:
        return None


def _missing_body(identity: str, method: MethodSchema) -> list[ast.stmt]:
    reference = method.function_name or method.body_reference
    detail = (
        f"SSA function {reference!r} was not supplied"
        if reference
        else "the schema carries no body reference"
    )
    return [
        ast.Raise(
            exc=_expression(
                "NotImplementedError("
                + repr(f"{identity}.{method.name}: {detail}")
                + ")"
            ),
            cause=None,
        )
    ]


def _method_ast(
    identity: str,
    method: MethodSchema,
    functions: Mapping[str, Any],
) -> tuple[ast.FunctionDef, bool]:
    receiver = () if method.is_static else ("self",)
    parameters = receiver + tuple(p.name for p in method.parameters)

    body: list[ast.stmt]
    uses_math = False
    source = functions.get(method.function_name or "")
    if source is None:
        body = _missing_body(identity, method)
    else:
        body, uses_math = materialize_function_body(
            source, parameter_names=parameters,
        )

    arguments = ast.arguments(
        posonlyargs=[],
        args=[
            ast.arg(
                arg=name,
                annotation=(
                    None
                    if index < len(receiver)
                    else _annotation(method.parameters[index - len(receiver)].type_name)
                ),
            )
            for index, name in enumerate(parameters)
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[
            _expression(repr(parameter.default))
            for parameter in method.parameters
            if parameter.has_default
        ],
    )

    definition = ast.FunctionDef(
        name="__init__" if method.is_constructor else method.name,
        args=arguments,
        body=body,
        decorator_list=(
            [ast.Name(id="staticmethod", ctx=ast.Load())]
            if method.is_static
            else []
        ),
        returns=_annotation(method.returns),
        type_comment=None,
        type_params=[],
    )
    return definition, uses_math


def _field_statements(schema: ClassSchema) -> list[ast.stmt]:
    """The instance layout as annotated class-body statements.

    Slot order is source order in Python, so the layout is carried by the
    order these are emitted in rather than by a number written beside them.
    That only holds if the slots are a gapless 0..N-1 block; a gap means the
    layout cannot be stated this way and is reported instead of being closed
    up silently, which would move every field after it.
    """

    ordered = sorted(schema.fields, key=lambda f: (f.slot is None, f.slot or 0))
    slots = [member.slot for member in ordered if member.slot is not None]
    if slots and slots != list(range(len(slots))):
        raise MaterializationError(
            f"{schema.identity}: field slots {slots} are not a gapless "
            "0..N-1 block, so source order cannot carry the layout"
        )

    statements: list[ast.stmt] = []
    for member in ordered:
        annotation = _annotation(member.type_name)
        if member.initial is not None:
            value = _expression(repr(member.initial))
            if annotation is None:
                statements.append(
                    ast.Assign(
                        targets=[ast.Name(id=member.name, ctx=ast.Store())],
                        value=value,
                    )
                )
            else:
                statements.append(
                    ast.AnnAssign(
                        target=ast.Name(id=member.name, ctx=ast.Store()),
                        annotation=annotation,
                        value=value,
                        simple=1,
                    )
                )
            continue
        statements.append(
            ast.AnnAssign(
                target=ast.Name(id=member.name, ctx=ast.Store()),
                annotation=annotation or _expression("object"),
                value=None,
                simple=1,
            )
        )
    return statements


def materialize_class(
    schema: ClassSchema | SSAClassDefinition,
    *,
    functions: Mapping[str, Any] | None = None,
) -> tuple[ast.ClassDef, bool]:
    """One class definition as an ``ast.ClassDef``, plus whether it needs ``math``.

    ``functions`` maps an SSA function name to the ``Function`` implementing
    it -- ``IRModule.functions`` is exactly that shape. A method whose function
    is absent still materializes, with a body that says what is missing.

    An ``SSAClassDefinition`` is accepted directly and goes through
    ``ClassSchema.from_ssa_class_definition``, so the lossiness of the SSA form
    is stated in one place rather than re-derived here.
    """

    if isinstance(schema, SSAClassDefinition):
        schema = ClassSchema.from_ssa_class_definition(schema)
    supplied = dict(functions or {})

    # A dotted identity names the class within its module; Python's class
    # statement can only carry the last segment, so the whole identity is kept
    # in the docstring rather than dropped.
    name = schema.identity.rsplit(".", 1)[-1]

    body: list[ast.stmt] = []
    if name != schema.identity:
        body.append(
            ast.Expr(value=ast.Constant(value=f"SSA identity: {schema.identity}"))
        )
    body.extend(_field_statements(schema))

    uses_math = False
    for method in schema.methods:
        definition, needed = _method_ast(schema.identity, method, supplied)
        uses_math = uses_math or needed
        body.append(definition)

    if not body:
        body.append(ast.Pass())

    return (
        ast.ClassDef(
            name=name,
            bases=[ast.Name(id=base, ctx=ast.Load()) for base in schema.bases],
            keywords=[],
            body=body,
            decorator_list=[],
            type_params=[],
        ),
        uses_math,
    )


def materialize_module(
    schemas: Iterable[ClassSchema | SSAClassDefinition] | SSAClassTable,
    *,
    functions: Mapping[str, Any] | None = None,
) -> ast.Module:
    """Every class in one importable ``ast.Module``, ``import math`` included
    when a materialized body needs it."""

    if isinstance(schemas, SSAClassTable):
        schemas = schemas_from_ssa_class_table(schemas)

    classes: list[ast.stmt] = []
    uses_math = False
    for schema in schemas:
        definition, needed = materialize_class(schema, functions=functions)
        uses_math = uses_math or needed
        classes.append(definition)

    body: list[ast.stmt] = []
    if uses_math:
        body.append(ast.Import(names=[ast.alias(name="math", asname=None)]))
    body.extend(classes)

    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))


def to_source(node: ast.AST) -> str:
    """The materialized definition as Python source."""

    return ast.unparse(ast.fix_missing_locations(node))


__all__ = [
    "INVENTED",
    "UNIMPLEMENTED",
    "MaterializationError",
    "materialize_class",
    "materialize_function_body",
    "materialize_module",
    "to_source",
]
