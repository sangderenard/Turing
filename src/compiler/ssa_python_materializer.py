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
* **The region-call ABI is spelled, not encoded.** A region call publishes
  its outputs as one aggregate that the caller projects with
  ``GetElementPtr``/``Load``. Python's tuple *is* that aggregate, so the
  projection is ordinary indexing and the ``Load`` is the identity. Emitting
  both rather than folding them keeps the Python step-for-step with the SSA,
  which is what makes a round trip usable as a diagnostic.
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


_TENSOR_RECEIVER = "AbstractTensor"
_TENSOR_IMPORT_MODULE = "src.common.tensors"


def _tensor_call_forms() -> dict[str, bool]:
    """Every catalogued tensor operation, and how Python spells a call to it.

    ``True`` means the operation is reached through the class
    (``AbstractTensor.arange(...)``), ``False`` through a receiver
    (``x.clip(...)``).

    That split is *read from* ``AbstractTensor`` with
    ``inspect.getattr_static`` rather than listed here, for the same reason
    the scalar table is audited against ``ssa_llvm_backend``: a hand-kept list
    of which operations are static is a vocabulary this file would be making
    up, and it would drift silently the first time the class changed.  The
    catalogue owns *which* operations exist; the class owns *how they are
    called*; this module supplies only the Python text.
    """

    import inspect

    from ..common.tensors.abstraction import AbstractTensor
    from ..common.tensors.operator_catalog import (
        CANONICAL_ABSTRACT_TENSOR_OPERATORS,
    )

    forms: dict[str, bool] = {}
    for name in CANONICAL_ABSTRACT_TENSOR_OPERATORS:
        try:
            static = inspect.getattr_static(AbstractTensor, name)
        except AttributeError:
            # Catalogued but not present on the class: an operation the SSA
            # vocabulary knows and this destination cannot call. Left out, so
            # reaching it raises the ordinary "no Python form" refusal rather
            # than emitting a call to something that does not exist.
            continue
        forms[name] = isinstance(static, staticmethod)
    return forms


TENSOR_CALL_FORMS = _tensor_call_forms()


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

    def __init__(
        self,
        function: Any,
        *,
        argument_names: Mapping[int, str],
        tensor_vocabulary: bool = False,
    ):
        self.function = function
        self.names: dict[int, str] = dict(argument_names)
        self.statements: list[ast.stmt] = []
        self.uses_math = False
        # Elementwise unary opcodes are shared between scalar and tensor
        # programs -- ``Log`` is what both ``math.log(x)`` and ``x.log()``
        # lower to, and the SSA carries nothing that separates them (a
        # genuine array arrives with ``shape=()`` like a scalar). So the
        # choice is declared by the caller rather than guessed from the IR:
        # a caller translating tensor material asks for the tensor
        # vocabulary, exactly as this module asks callers to add spellings
        # deliberately rather than let it invent one.
        self.tensor_vocabulary = bool(tensor_vocabulary)

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
        if result is None:
            raise MaterializationError(
                f"{self.function.name}: an instruction publishing no result "
                f"value reached assignment of {source!r}; a void instruction "
                "has no Python spelling here"
            )
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

        if operation in {"Cast", "CastLike", "cast_like"} and instruction.args:
            # Both backends render a cast as "produce the operand in the
            # RESULT's type" (see ssa_reference_evaluator, which follows
            # them). Under the double working type that is the identity, and
            # the loop-carried slot seed is emitted as exactly this shape.
            # Spelled as an assignment rather than elided so the emitted
            # Python stays step-for-step with the SSA.
            self.assign(result, self.operand(instruction.args[0]))
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
            if result is None:
                # A statement-form call publishes no SSA value (res=None).
                # ``assign`` cannot spell it, and reaching for ``result.id``
                # crashed here instead of refusing; say what it is.
                raise MaterializationError(
                    f"{self.function.name}: statement-form call to "
                    f"{callee!r} publishes no result value; only "
                    "value-producing calls are spelled here"
                )
            # A region call publishes its outputs as one aggregate, which the
            # caller then projects with GetElementPtr/Load. Python's tuple is
            # that aggregate exactly, so the convention needs no encoding --
            # the projection below is ordinary indexing.
            self.assign(result, f"{callee}({operands})")
            return

        if operation in {"GetElementPtr", "getelementptr"}:
            index = attributes.get("aggregate_index")
            if index is not None:
                base = self.operand(instruction.args[0])
                self.assign(result, f"{base}[{int(index)}]")
                return
            if len(instruction.args) == 2:
                # Ordinary element indexing, as opposed to the region-call
                # aggregate projection above: the address is (base, offset)
                # with no aggregate index, which is what an authored ``a[i]``
                # lowers to. Python spells that with the same subscript the
                # source used, so this reproduces the read rather than
                # guessing a layout -- the Load over it stays the identity,
                # exactly as it is for the projection case.
                #
                # This covers an authored SLICE as well as an element read. A
                # slice is not folded away at lowering: its bound arrives as a
                # value the block does not compute and is promoted to a formal,
                # so the emitted function takes the slice as a parameter. That
                # is the program parameterised, not the program damaged --
                # verified by executing it, where passing ``slice(2, None)``
                # reproduces the numpy original exactly on irregularly-spaced
                # input, and passing a different slice provably changes the
                # result. The subscript is spelled the same way either way;
                # what varies is only whether the offset is a constant the
                # block builds or an argument it receives.
                base = self.operand(instruction.args[0])
                offset = self.operand(instruction.args[1])
                self.assign(result, f"{base}[{offset}]")
                return
            raise MaterializationError(
                f"{self.function.name}: GetElementPtr %t{int(result.id)} "
                f"addresses by computed path over {len(instruction.args)} "
                "operands; only the region-call projection and single-offset "
                "element indexing are spelled here"
            )

        if operation in {"Load", "load"}:
            # The repository IR does not insert a Load before every use of an
            # address -- backends dereference implicitly -- so a Load over an
            # already-projected aggregate field is the identity here. Spelling
            # it as an assignment rather than eliding it keeps the emitted
            # Python step-for-step with the SSA it came from, which is the
            # point of a round trip used as a diagnostic.
            self.assign(result, self.operand(instruction.args[0]))
            return

        if operation in _BINARY_SPELLING:
            left = self.operand(instruction.args[0])
            right = self.operand(instruction.args[1])
            self.assign(result, _BINARY_SPELLING[operation].format(left, right))
            return

        if operation in _UNARY_SPELLING:
            operand = self.operand(instruction.args[0])
            tensor_name = operation.lower()
            if (
                self.tensor_vocabulary
                and operation in _NEEDS_MATH
                and TENSOR_CALL_FORMS.get(tensor_name) is False
            ):
                # Only the ``math.*`` spellings need redirecting: they are the
                # ones that genuinely fail on a tensor. The operator and
                # builtin forms (``-x``, ``abs(x)``) already mean the right
                # thing for both, so they are left exactly as they are rather
                # than rewritten for the sake of uniformity.
                self.assign(result, f"{operand}.{tensor_name}()")
                return
            if operation in _NEEDS_MATH:
                self.uses_math = True
            self.assign(result, _UNARY_SPELLING[operation].format(operand))
            return

        tensor_operation = str(
            attributes.get("tensor_operation")
            or attributes.get("tensor_candidate")
            or operation
        )
        if tensor_operation in TENSOR_CALL_FORMS:
            # A catalogued tensor operation. Every operand is positional in
            # the SSA -- verified on real lowered programs: ``clip`` arrives
            # as (x, lo, hi), ``index_select`` as (x, dim, indices) -- so the
            # Python call is the same sequence, split only by whether the
            # class or the first operand is the receiver.
            operands = [self.operand(argument) for argument in instruction.args]
            if TENSOR_CALL_FORMS[tensor_operation]:
                call = (
                    f"{_TENSOR_RECEIVER}.{tensor_operation}"
                    f"({', '.join(operands)})"
                )
            else:
                if not operands:
                    raise MaterializationError(
                        f"{self.function.name}: {tensor_operation!r} is a "
                        "method on a tensor but the SSA gives it no operand "
                        "to call it on"
                    )
                call = (
                    f"{operands[0]}.{tensor_operation}"
                    f"({', '.join(operands[1:])})"
                )
            self.assign(result, call)
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


_TERMINATORS = {
    "Br", "br", "CondBr", "condbr", "Ret", "ret", "Return", "return",
}


def _straight_line(materializer: "_BodyMaterializer", block: Any) -> None:
    """Lower one block's non-terminator, non-phi instructions in order."""

    for instruction in block.instrs:
        operation = str(instruction.op)
        if operation in _TERMINATORS or operation in {"Phi", "phi"}:
            continue
        materializer.step(instruction)


def _terminator(block: Any) -> Any | None:
    for instruction in reversed(block.instrs):
        if str(instruction.op) in _TERMINATORS:
            return instruction
    return None


def _counted_loop_shape(function: Any) -> dict[str, Any]:
    """Recognise the loop shape this compiler emits, or say what did not match.

    Deliberately narrow. This is not general control-flow reconstruction --
    that remains unattempted -- it recognises the five-block counted loop the
    lowering actually produces: a preheader branching to a header of phis and a
    condition, a body, a latch carrying the induction forward, and an exit.
    Anything else raises rather than being approximated.
    """

    blocks = dict(getattr(function, "blocks", {}) or {})
    if len(blocks) != 5:
        raise MaterializationError(
            f"{getattr(function, 'name', '<anonymous>')} has {len(blocks)} "
            f"blocks ({sorted(blocks)}); only the five-block counted loop is "
            "reconstructed here"
        )
    # Every block any terminator can reach -- a CondBr names two of them, and
    # missing those makes the body and the exit look like entry blocks.
    targeted: set[str] = set()
    for block in blocks.values():
        terminator = _terminator(block)
        if terminator is None:
            continue
        attributes = terminator.attributes or {}
        for key in ("target", "true_target", "false_target"):
            if attributes.get(key) is not None:
                targeted.add(str(attributes[key]))
    entries = [name for name in blocks if name not in targeted]
    conditional = [
        (name, _terminator(block))
        for name, block in blocks.items()
        if _terminator(block) is not None
        and str(_terminator(block).op) in {"CondBr", "condbr"}
    ]
    if len(entries) != 1 or len(conditional) != 1:
        raise MaterializationError(
            f"{getattr(function, 'name', '<anonymous>')}: expected one entry "
            f"and one conditional branch, found {len(entries)} and "
            f"{len(conditional)}"
        )
    entry_name = entries[0]
    header_name, branch = conditional[0]
    attributes = branch.attributes or {}
    body_name = str(attributes.get("true_target"))
    exit_name = str(attributes.get("false_target"))
    body = blocks.get(body_name)
    if body is None:
        raise MaterializationError(
            f"{getattr(function, 'name', '<anonymous>')}: the conditional "
            f"branch names {body_name!r}, which is not a block here"
        )
    latch_name = str((_terminator(body).attributes or {}).get("target"))
    if latch_name not in blocks:
        raise MaterializationError(
            f"{getattr(function, 'name', '<anonymous>')}: the body branches to "
            f"{latch_name!r}, which is not a block here"
        )
    return {
        "entry": blocks[entry_name],
        "header": blocks[header_name],
        "body": body,
        "latch": blocks[latch_name],
        "exit": blocks[exit_name],
        "branch": branch,
        "header_name": header_name,
        "entry_name": entry_name,
    }


def _conditional_diamond_shape(function: Any) -> dict[str, Any]:
    """Recognise the four-block if/else diamond ``lower_conditional`` emits.

    Deliberately narrow, like ``_counted_loop_shape``: an entry block ending
    in a CondBr, two arm blocks that each branch to one shared merge block,
    and the merge carrying the Phis and the return. Anything else raises
    rather than being approximated.
    """

    blocks = dict(getattr(function, "blocks", {}) or {})
    if len(blocks) != 4:
        raise MaterializationError(
            f"{getattr(function, 'name', '<anonymous>')} has {len(blocks)} "
            f"blocks ({sorted(blocks)}); only the four-block conditional "
            "diamond is reconstructed here"
        )
    targeted: set[str] = set()
    for block in blocks.values():
        terminator = _terminator(block)
        if terminator is None:
            continue
        attributes = terminator.attributes or {}
        for key in ("target", "true_target", "false_target"):
            if attributes.get(key) is not None:
                targeted.add(str(attributes[key]))
    entries = [name for name in blocks if name not in targeted]
    conditional = [
        (name, _terminator(block))
        for name, block in blocks.items()
        if _terminator(block) is not None
        and str(_terminator(block).op) in {"CondBr", "condbr"}
    ]
    if len(entries) != 1 or len(conditional) != 1:
        raise MaterializationError(
            f"{getattr(function, 'name', '<anonymous>')}: expected one entry "
            f"and one conditional branch, found {len(entries)} and "
            f"{len(conditional)}"
        )
    entry_name = entries[0]
    branch_name, branch = conditional[0]
    if branch_name != entry_name:
        raise MaterializationError(
            f"{getattr(function, 'name', '<anonymous>')}: the conditional "
            f"branch lives in {branch_name!r}, not the entry {entry_name!r}; "
            "that is not the single-diamond shape"
        )
    attributes = branch.attributes or {}
    true_name = str(attributes.get("true_target"))
    false_name = str(attributes.get("false_target"))
    arm_exits = {}
    for arm_name in (true_name, false_name):
        arm = blocks.get(arm_name)
        if arm is None:
            raise MaterializationError(
                f"{getattr(function, 'name', '<anonymous>')}: the branch "
                f"names {arm_name!r}, which is not a block here"
            )
        terminator = _terminator(arm)
        if terminator is None or str(terminator.op) not in {"Br", "br"}:
            raise MaterializationError(
                f"{getattr(function, 'name', '<anonymous>')}: arm "
                f"{arm_name!r} does not end in an unconditional branch"
            )
        arm_exits[arm_name] = str((terminator.attributes or {}).get("target"))
    merge_names = set(arm_exits.values())
    if len(merge_names) != 1:
        raise MaterializationError(
            f"{getattr(function, 'name', '<anonymous>')}: the arms branch to "
            f"{sorted(merge_names)}; a diamond has one merge block"
        )
    merge_name = merge_names.pop()
    if merge_name not in blocks or merge_name in {
        entry_name, true_name, false_name,
    }:
        raise MaterializationError(
            f"{getattr(function, 'name', '<anonymous>')}: the merge "
            f"{merge_name!r} is not a distinct block here"
        )
    return {
        "entry": blocks[entry_name],
        "true": blocks[true_name],
        "false": blocks[false_name],
        "merge": blocks[merge_name],
        "branch": branch,
        "true_name": true_name,
        "false_name": false_name,
    }


def _materialize_conditional_diamond(
    function: Any, materializer: "_BodyMaterializer",
) -> list[ast.stmt]:
    """The four-block diamond as a Python ``if``/``else``.

    Each merge Phi becomes an ordinary name assigned at the end of the arm
    its operand came from -- the same reading ``_materialize_counted_loop``
    gives loop phis, once you are allowed to mutate. The Phi's own
    ``incoming_blocks`` says which operand belongs to which arm; positional
    guessing would silently swap the arms whenever ``expect_true`` flipped
    the branch targets.
    """

    shape = _conditional_diamond_shape(function)
    phis = [
        instruction
        for instruction in shape["merge"].instrs
        if str(instruction.op) in {"Phi", "phi"}
    ]

    _straight_line(materializer, shape["entry"])
    condition = materializer.operand(shape["branch"].args[0])

    arm_statements: dict[str, list[ast.stmt]] = {}
    for arm_key in ("true", "false"):
        before = len(materializer.statements)
        _straight_line(materializer, shape[arm_key])
        arm_statements[arm_key] = materializer.statements[before:]
        del materializer.statements[before:]

    for instruction in phis:
        incoming = tuple(
            str(name)
            for name in (instruction.attributes or {}).get(
                "incoming_blocks"
            ) or ()
        )
        if len(incoming) != 2 or len(instruction.args) != 2 or set(
            incoming
        ) != {shape["true_name"], shape["false_name"]}:
            raise MaterializationError(
                f"{function.name}: merge phi %t{int(instruction.res.id)} "
                f"carries incoming blocks {incoming!r}; expected exactly "
                f"the two arms {shape['true_name']!r} and "
                f"{shape['false_name']!r}"
            )
        target = _local(instruction.res.id)
        for block_name, operand in zip(incoming, instruction.args):
            arm_key = "true" if block_name == shape["true_name"] else "false"
            arm_statements[arm_key].append(
                ast.Assign(
                    targets=[ast.Name(id=target, ctx=ast.Store())],
                    value=_expression(materializer.operand(operand)),
                )
            )
        materializer.names[int(instruction.res.id)] = target

    materializer.statements.append(
        ast.If(
            test=_expression(condition),
            body=arm_statements["true"] or [ast.Pass()],
            orelse=arm_statements["false"],
        )
    )

    _straight_line(materializer, shape["merge"])
    returned = _terminator(shape["merge"])
    if returned is not None and str(returned.op) in {
        "Ret", "ret", "Return", "return",
    }:
        materializer.finish(tuple(returned.args))
    return materializer.statements


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
    function: Any,
    *,
    parameter_names: Sequence[str] | None = None,
    tensor_vocabulary: bool = False,
) -> tuple[list[ast.stmt], bool]:
    """One SSA function's body as statements, plus whether it needs ``math``.

    ``tensor_vocabulary`` emits the AbstractTensor form of the elementwise
    unary operations (``x.log()``) instead of the scalar one
    (``math.log(x)``). See :class:`_BodyMaterializer` for why that is a
    caller's declaration rather than something inferred from the SSA.

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

    materializer = _BodyMaterializer(
        function, argument_names=names, tensor_vocabulary=tensor_vocabulary,
    )
    block_count = len(getattr(function, "blocks", {}) or {})
    if block_count == 4:
        statements = _materialize_conditional_diamond(function, materializer)
        return (statements or [ast.Pass()]), materializer.uses_math
    if block_count != 1:
        statements = _materialize_counted_loop(function, materializer)
        return (statements or [ast.Pass()]), materializer.uses_math

    block = _single_block(function)
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


def _materialize_counted_loop(
    function: Any, materializer: "_BodyMaterializer",
) -> list[ast.stmt]:
    """The five-block counted loop as a ``while`` with explicit carried names.

    Emitted as ``while True:`` with the header's condition tested inside and a
    ``break``, rather than hoisting the condition into the ``while`` clause.
    That is a mechanical, always-correct translation of the CFG: the header may
    compute anything before it compares, and a hoisted condition would have to
    duplicate that work or reorder it. Readability here is worth less than the
    emitted Python meaning exactly what the SSA meant.

    Each phi becomes an ordinary name: assigned before the loop from its
    preheader operand, and reassigned at the end of each iteration from its
    latch operand. That is what a phi *is*, once you are allowed to mutate.
    """

    shape = _counted_loop_shape(function)
    phis = [
        instruction
        for instruction in shape["header"].instrs
        if str(instruction.op) in {"Phi", "phi"}
    ]

    _straight_line(materializer, shape["entry"])

    # Seed every carried name from the preheader edge.
    incoming = [
        tuple(instruction.attributes.get("incoming_blocks") or ())
        for instruction in phis
    ]
    for instruction, blocks in zip(phis, incoming):
        if len(blocks) != 2 or blocks[0] != shape["entry_name"]:
            raise MaterializationError(
                f"{function.name}: phi carries incoming blocks {blocks!r}; the "
                f"first must be the preheader {shape['entry_name']!r}"
            )
        materializer.assign(instruction.res, materializer.operand(instruction.args[0]))

    before = len(materializer.statements)
    _straight_line(materializer, shape["header"])
    condition = materializer.operand(shape["branch"].args[0])
    header_statements = materializer.statements[before:]
    del materializer.statements[before:]

    body_start = len(materializer.statements)
    _straight_line(materializer, shape["body"])
    _straight_line(materializer, shape["latch"])
    # The latch operand of each phi is only in scope once body AND latch have
    # run, so the carried updates land after both.
    for instruction in phis:
        target = materializer.names[int(instruction.res.id)]
        materializer.statements.append(
            ast.Assign(
                targets=[ast.Name(id=target, ctx=ast.Store())],
                value=_expression(materializer.operand(instruction.args[1])),
            )
        )
    loop_statements = materializer.statements[body_start:]
    del materializer.statements[body_start:]

    materializer.statements.append(
        ast.While(
            test=ast.Constant(value=True),
            body=[
                *header_statements,
                ast.If(
                    test=_expression(f"not {condition}"),
                    body=[ast.Break()],
                    orelse=[],
                ),
                *loop_statements,
            ],
            orelse=[],
        )
    )

    # After the loop, a carried name means the loop's RESULT, and the exit
    # block reads it under the *updated* value's id -- the one produced inside
    # the body. That name only exists if the body ran, so a zero-iteration loop
    # would reference an unbound local. Binding the updated id to the phi's
    # name here is the same rebinding precompile_to_ssa performs with its
    # carried ports, and it makes the zero-trip case correct rather than
    # accidentally working.
    for instruction in phis:
        updated_id = (instruction.attributes or {}).get("updated_value_id")
        if updated_id is None:
            continue
        carried_name = materializer.names[int(instruction.res.id)]
        port_name = _local(int(updated_id))
        materializer.statements.append(
            ast.Assign(
                targets=[ast.Name(id=port_name, ctx=ast.Store())],
                value=_expression(carried_name),
            )
        )
        materializer.names[int(updated_id)] = port_name

    _straight_line(materializer, shape["exit"])
    returned = _terminator(shape["exit"])
    if returned is not None and str(returned.op) in {"Ret", "ret", "Return", "return"}:
        materializer.finish(tuple(returned.args))
    return materializer.statements


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


def _parameter_names(function: Any) -> list[str]:
    """Names for a function's formals, in the SSA's own argument order.

    ``parameter_names`` maps a source name to the value that name denotes, and
    for a parameter carried through a loop that value is the PHI, not the
    argument. So ``w`` points at the loop's carried value while the formal is
    the entry value it was seeded from. The link back is on the carried port's
    accounting (``source_value_id``), so it is followed rather than guessed.

    The resulting order is the SSA's, not the authored signature's -- that is
    the compiled ABI, and renaming would hide it rather than fix it.
    """

    formals = tuple(getattr(function, "args", ()))
    metadata = getattr(function, "metadata", None) or {}
    by_value: dict[int, str] = {}
    for label, value_id in (metadata.get("parameter_names") or ()):
        by_value.setdefault(int(value_id), str(label))
    for carried in (metadata.get("carried_port_values") or {}).values():
        accounting = getattr(carried, "accounting", None) or {}
        source = accounting.get("source_value_id")
        named = by_value.get(int(getattr(carried, "id", -1)))
        if source is not None and named is not None:
            by_value.setdefault(int(source), named)

    declared = _declared_formal_names(function)
    by_position = {position: label for label, position in declared.items()}
    return [
        by_value.get(int(formal.id))
        or by_position.get(position)
        or _local(formal.id)
        for position, formal in enumerate(formals)
    ]


def _region_output_contracts(module: Any) -> dict[str, Any]:
    """Recover each region's published outputs from the calls that project them.

    Returns ``name -> tuple_of_value_ids``, or ``name -> reason`` when two call
    sites disagree. A disagreement is not resolvable by picking one: the region
    would return a different aggregate depending on who asked, which is a real
    defect in the convention rather than something to paper over.
    """

    contracts: dict[str, Any] = {}
    for function in (getattr(module, "functions", {}) or {}).values():
        for block in (getattr(function, "blocks", {}) or {}).values():
            for instruction in block.instrs:
                if str(instruction.op) not in {"Call", "call"}:
                    continue
                attributes = instruction.attributes or {}
                callee = str(attributes.get("callee") or "")
                declared = attributes.get("output_ids")
                if not callee or declared is None:
                    continue
                outputs = tuple(int(each) for each in declared)
                existing = contracts.get(callee)
                if existing is None:
                    contracts[callee] = outputs
                elif isinstance(existing, tuple) and existing != outputs:
                    contracts[callee] = (
                        f"call sites disagree about what {callee!r} publishes: "
                        f"{existing} and {outputs}"
                    )
    return contracts


def materialize_ir_module(
    module: Any, *, tensor_vocabulary: bool = False,
) -> tuple[ast.Module, dict[str, str]]:
    """Every single-block function of an ``IRModule`` as one runnable module.

    ``tensor_vocabulary`` translates the program into AbstractTensor Python
    rather than scalar Python -- see :func:`materialize_function_body`.

    Returns the module and a mapping of the functions that could NOT be
    materialized to the reason why. Skipping is reported rather than silent:
    a Python module that quietly omits half a program is the same plausible
    wrong answer this materializer exists to avoid, and the omissions are
    exactly what a diagnostic round trip wants to look at.

    Region calls resolve by name because the region's own function is emitted
    into the same module, so the result executes rather than merely unparsing.
    """

    # A region function does not state what it returns. Its outputs are
    # declared at the CALL SITE, as the ``output_ids`` tuple on the Call, and
    # the callee's own metadata carries nothing about them. That is worth
    # noticing rather than working around quietly: the region calling
    # convention is only half-located, so the contract has to be recovered
    # from every caller and checked for agreement.
    published = _region_output_contracts(module)

    emitted: list[ast.stmt] = []
    skipped: dict[str, str] = {}
    uses_math = False

    for name, function in (getattr(module, "functions", {}) or {}).items():
        parameters = _parameter_names(function)
        # Declared workspace formals (the LAPACK-style storage ABI stamped in
        # Function.metadata) are not parameters an author wrote and no caller
        # could fill them -- they are the caller-provides-workspace chain
        # reaching this function. In the Python realization the lease model
        # applies directly: drop them from the signature and allocate them at
        # entry. A ``dynamic`` entry is a heap participant; the Python target
        # has no heap wiring yet, so it refuses by name rather than guessing
        # an extent.
        metadata = getattr(function, "metadata", None) or {}
        storage_entries = tuple(metadata.get("storage_formals") or ())
        storage_by_id = {int(entry["value_id"]): entry for entry in storage_entries}
        formal_ids = [int(a.id) for a in getattr(function, "args", ())]
        try:
            body, needed = materialize_function_body(
                function,
                parameter_names=parameters,
                tensor_vocabulary=tensor_vocabulary,
            )
        except MaterializationError as error:
            skipped[str(name)] = str(error)
            continue
        if storage_by_id:
            dynamic = [e for e in storage_entries if e.get("dynamic")]
            if dynamic:
                skipped[str(name)] = (
                    "storage formals with dynamic extent need heap "
                    f"participation, which the Python target does not wire "
                    f"yet: {[e['callee'] for e in dynamic]}"
                )
                continue
            allocations: list[ast.stmt] = []
            kept: list[str] = []
            for position, label in enumerate(parameters):
                value_id = formal_ids[position]
                entry = storage_by_id.get(value_id)
                if entry is None:
                    kept.append(label)
                    continue
                count = 1
                for extent in entry["shape"]:
                    count *= int(extent)
                source = (
                    "0.0" if not entry["shape"] else f"[0.0] * {count}"
                )
                allocations.append(ast.Assign(
                    targets=[ast.Name(id=label, ctx=ast.Store())],
                    value=_expression(source),
                ))
            parameters = kept
            body = allocations + body
        contract = published.get(str(name))
        if isinstance(contract, str):
            skipped[str(name)] = contract
            continue
        if contract is not None and not any(
            isinstance(statement, ast.Return) for statement in body
        ):
            # Publish the aggregate the callers project. Order is the call
            # site's, not this function's instruction order.
            body = body + [
                ast.Return(
                    value=_expression(
                        "(" + "".join(f"{_local(each)}, " for each in contract) + ")"
                    )
                )
            ]
        uses_math = uses_math or needed
        emitted.append(
            ast.FunctionDef(
                name=str(name),
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=label, annotation=None) for label in parameters],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=body,
                decorator_list=[],
                returns=None,
                type_comment=None,
                type_params=[],
            )
        )

    prologue: list[ast.stmt] = []
    if uses_math:
        prologue.append(ast.Import(names=[ast.alias(name="math", asname=None)]))
    return (
        ast.fix_missing_locations(
            ast.Module(body=prologue + emitted, type_ignores=[])
        ),
        skipped,
    )


def to_source(node: ast.AST) -> str:
    """The materialized definition as Python source."""

    return ast.unparse(ast.fix_missing_locations(node))


__all__ = [
    "INVENTED",
    "UNIMPLEMENTED",
    "MaterializationError",
    "materialize_class",
    "materialize_function_body",
    "materialize_ir_module",
    "materialize_module",
    "to_source",
]
