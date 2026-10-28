"""Import authored LLVM tensor algorithms into Turing's repository SSA.

LLVM is the already-implemented tensor algorithm language in the accelerated
backend.  This module does not create tensor opcodes.  It translates LLVM
instructions into the existing fundamental :class:`Handler` vocabulary and
expands LLVM ``switch`` terminators into ``Eq``/``CondBr`` chains.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from ....transmogrifier.dtype_layout import by_llvm_type
from ....transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSAValue,
)
from ....transmogrifier.ssa_registry import Handler


_DIRECT_HANDLERS = {
    "add": Handler.Add,
    "fadd": Handler.Add,
    "sub": Handler.Sub,
    "fsub": Handler.Sub,
    "mul": Handler.Mul,
    "fmul": Handler.Mul,
    "sdiv": Handler.Div,
    "udiv": Handler.Div,
    "fdiv": Handler.Div,
    "srem": Handler.Mod,
    "urem": Handler.Mod,
    "frem": Handler.Mod,
    "fneg": Handler.Neg,
    "and": Handler.And,
    "or": Handler.Or,
    "xor": Handler.Xor,
    "shl": Handler.Shl,
    "lshr": Handler.Shr,
    "ashr": Handler.AShr,
    "load": Handler.Load,
    "store": Handler.Store,
    "alloca": Handler.Alloca,
    "getelementptr": Handler.GetElementPtr,
    "trunc": Handler.Trunc,
    # ``fptrunc`` narrows a FLOAT's precision; ``Trunc`` is INTEGER
    # truncation.  Conflating them turned the float-narrowing kernels into
    # integer truncation wherever an imported body was re-spelled (Fortran
    # emitted ``int(x)`` for ``float(x)`` with no shortfall).  ``Cast`` with
    # the result's own declared type carries the correct meaning, exactly as
    # ``fpext`` below already does.
    "fptrunc": Handler.Cast,
    "zext": Handler.ZExt,
    "sext": Handler.SExt,
    "fptosi": Handler.FpToSi,
    "fptoui": Handler.FpToUi,
    "sitofp": Handler.SiToFp,
    "uitofp": Handler.UiToFp,
    "fpext": Handler.Cast,
    "bitcast": Handler.Cast,
    "ptrtoint": Handler.Cast,
    "inttoptr": Handler.Cast,
    "phi": Handler.Phi,
    "select": Handler.Select,
    "ret": Handler.Ret,
}

_INTEGER_PREDICATES = {
    "eq": Handler.Eq,
    "ne": Handler.Ne,
    "slt": Handler.Lt,
    "ult": Handler.ULt,
    "sle": Handler.Le,
    "ule": Handler.ULe,
    "sgt": Handler.Gt,
    "ugt": Handler.UGt,
    "sge": Handler.Ge,
    "uge": Handler.UGe,
}

_FLOAT_PREDICATES = {
    "oeq": Handler.Eq,
    "ueq": Handler.Eq,
    "one": Handler.Ne,
    "une": Handler.Ne,
    "olt": Handler.Lt,
    "ult": Handler.Lt,
    "ole": Handler.Le,
    "ule": Handler.Le,
    "ogt": Handler.Gt,
    "ugt": Handler.Gt,
    "oge": Handler.Ge,
    "uge": Handler.Ge,
}


def _layout_dtype(llvm_type: Any) -> str:
    """The layout name of an LLVM scalar spelling.

    Program: any imported repository kernel (``binary_double``).  Wrong
    record: its values carried LLVM's own spelling (``double``, ``i64``) as
    ``SSAValue.dtype`` while every lowered value of the same width carries
    the layout name (``float64``, ``int64``), so the concordance saw one
    value under two dtype spellings.  Rule: the importer states the
    declared layout name; a type with no scalar layout (vector, struct,
    ``void``) keeps its LLVM spelling.
    """

    layout = by_llvm_type(llvm_type)
    return str(llvm_type) if layout is None else layout.name


@dataclass(frozen=True, order=True)
class LLVMRepositorySSAShortfall:
    function: str
    block: str
    opcode: str
    instruction: str
    reason: str


@dataclass(frozen=True)
class LLVMRepositorySSAResult:
    module: IRModule
    shortfalls: tuple[LLVMRepositorySSAShortfall, ...]

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def shortfall_report(self) -> str:
        if not self.shortfalls:
            return "LLVM-to-repository-SSA lowering: complete"
        return "\n".join((
            "LLVM-to-repository-SSA shortfalls:",
            *(
                f"- {item.function}:{item.block} {item.opcode}; "
                f"{item.reason}: {item.instruction}"
                for item in self.shortfalls
            ),
        ))


class _FunctionImporter:
    def __init__(self, llvm_function: Any, first_value_id: int):
        self.llvm_function = llvm_function
        self.next_value_id = int(first_value_id)
        self.values: dict[tuple[str, str], SSAValue] = {}
        self.constants: dict[tuple[str, str], SSAValue] = {}
        self.constant_attributes: dict[int, dict[str, Any]] = {}
        #: Every value this import numbers, by id: (position in the kernel,
        #: kind, LLVM spelling).  The kernel's identity rows are posted from
        #: it per compile book (``post_repository_kernel_identity``).
        self.value_origins: dict[int, tuple[int, str, str]] = {}
        self.shortfalls: list[LLVMRepositorySSAShortfall] = []
        self.blocks = {
            str(block.name): BasicBlock(str(block.name))
            for block in llvm_function.blocks
        }

    def fresh(
        self, dtype: str | None, kind: str = "synthetic", text: str = "",
    ) -> SSAValue:
        value = SSAValue(self.next_value_id, dtype=dtype)
        self.value_origins[int(value.id)] = (
            len(self.value_origins), str(kind), str(text),
        )
        self.next_value_id += 1
        return value

    @staticmethod
    def _value_key(value: Any) -> tuple[str, str]:
        kind = value.value_kind.name
        if kind in {"argument", "instruction"}:
            return kind, str(value.name)
        return kind, str(value).strip()

    def define_values(self) -> list[SSAValue]:
        arguments = []
        for argument in self.llvm_function.arguments:
            value = self.fresh(
                _layout_dtype(argument.type), "argument", str(argument.name),
            )
            self.values[self._value_key(argument)] = value
            arguments.append(value)
        for block in self.llvm_function.blocks:
            for instruction in block.instructions:
                # Terminators have LLVM type ``void`` but still carry scalar
                # operands: ``ret`` values are the important example.
                # Register their constants before deciding whether the
                # instruction itself defines a value.
                for operand in instruction.operands:
                    self._register_constant(operand)
                if str(instruction.opcode) == "switch":
                    # A switch's case literals are not in ``.operands``
                    # (see ``_switch_cases``); register them from the
                    # instruction's own text so each ``Eq`` the switch
                    # expands to compares against a ``Const`` this
                    # function produces.
                    for case_type, literal, _target in self._switch_cases(
                        instruction,
                    )[1]:
                        self._register_constant_literal(case_type, literal)
                if str(instruction.type) == "void":
                    continue
                self.values[self._value_key(instruction)] = self.fresh(
                    _layout_dtype(instruction.type), "instruction",
                    str(instruction.name),
                )
        return arguments

    def _register_constant(self, operand: Any) -> None:
        if not operand.value_kind.name.startswith("constant"):
            return
        key = self._value_key(operand)
        if key in self.constants:
            return
        value = self.fresh(
            _layout_dtype(operand.type), "constant", str(operand).strip(),
        )
        self.constants[key] = value
        self.constant_attributes[value.id] = {
            "llvm_literal": str(operand).strip(),
        }

    def _register_constant_literal(self, llvm_type: str, literal: str) -> None:
        """Register an integer constant spelled ``<llvm_type> <literal>`` in
        the instruction text, under the same key, dtype, origin and
        ``llvm_literal`` attribute ``_register_constant`` gives the same
        constant when the binding hands it over as an operand, so a switch
        case ``i32 0`` and an ``add ..., 0`` elsewhere in the function share
        one ``Const``."""
        spelled = f"{llvm_type} {literal}"
        key = ("constant_int", spelled)
        if key in self.constants:
            return
        value = self.fresh(_layout_dtype(llvm_type), "constant", spelled)
        self.constants[key] = value
        self.constant_attributes[value.id] = {"llvm_literal": spelled}

    _SWITCH_HEAD = re.compile(
        r"switch\s+(?P<type>\S+)\s+(?P<selector>[^,]+),\s*label\s+%"
        r"(?P<default>\"[^\"]*\"|[^\s,\[\]]+)\s*\["
    )
    _SWITCH_CASE = re.compile(
        r"(?P<type>i\d+)\s+(?P<literal>-?\d+)\s*,\s*label\s+%"
        r"(?P<target>\"[^\"]*\"|[^\s,\[\]]+)"
    )

    @classmethod
    def _switch_cases(
        cls, instruction: Any,
    ) -> tuple[str, list[tuple[str, str, str]]]:
        """``(default_label, [(case_type, literal, target_label), ...])`` of
        a ``switch``, read from the instruction's own text.

        The binding's ``.operands`` of a switch is version-dependent: LLVM
        lists ``(selector, default, value0, block0, value1, block1, ...)``,
        but llvmlite 0.50 yields only ``(selector, default, block0,
        block1, ...)``, the case VALUES absent.  Reading the pairs by
        position from that list paired blocks with blocks and minted every
        destination block as a dangling "global" value (the ``binary_value``
        kernel carried eleven operands no instruction produced).  The text
        is the one spelling every version agrees on; the block list that
        IS exposed is checked against it in ``_switch``."""
        text = str(instruction)
        head = cls._SWITCH_HEAD.search(text)
        if head is None:
            raise ValueError(
                f"switch instruction text is not readable: {text.strip()}"
            )
        body = text[head.end():]
        cases = [
            (match.group("type"), match.group("literal"),
             match.group("target").strip('"'))
            for match in cls._SWITCH_CASE.finditer(body)
        ]
        return head.group("default").strip('"'), cases

    def operand_value(self, operand: Any) -> SSAValue:
        key = self._value_key(operand)
        if key in self.values:
            return self.values[key]
        if key in self.constants:
            return self.constants[key]
        # Global storage and declarations are ABI roots, not instructions in
        # the current function.  Preserve them as explicit Load arguments.
        value = self.fresh(_layout_dtype(operand.type), "global", str(operand.name))
        self.values[key] = value
        return value

    @staticmethod
    def _comparison_handler(instruction: Any) -> tuple[Handler, str]:
        text = str(instruction)
        match = re.search(r"\b(?:icmp|fcmp)\s+([a-z]+)\b", text)
        if match is None:
            raise KeyError("comparison predicate is absent")
        predicate = match.group(1)
        table = (
            _INTEGER_PREDICATES
            if instruction.opcode == "icmp"
            else _FLOAT_PREDICATES
        )
        return table.get(predicate, Handler.Call), predicate

    def _ordinary_instruction(
        self,
        instruction: Any,
        block: BasicBlock,
    ) -> None:
        opcode = str(instruction.opcode)
        operands = list(instruction.operands)
        attributes: dict[str, Any] = {"llvm_opcode": opcode}
        if opcode in {"icmp", "fcmp"}:
            handler, predicate = self._comparison_handler(instruction)
            attributes["predicate"] = predicate
            if handler is Handler.Call:
                attributes["callee"] = f"llvm.{opcode}.{predicate}"
        elif opcode == "call":
            handler = Handler.Call
            callee = operands.pop()
            attributes["callee"] = str(callee.name)
        else:
            handler = _DIRECT_HANDLERS.get(opcode)
        if handler is None:
            self.shortfalls.append(
                LLVMRepositorySSAShortfall(
                    str(self.llvm_function.name),
                    block.name,
                    opcode,
                    str(instruction).strip(),
                    "opcode has no existing repository SSA Handler",
                )
            )
            return
        result = (
            None
            if str(instruction.type) == "void"
            else self.values[self._value_key(instruction)]
        )
        if opcode == "phi":
            try:
                attributes["incoming_blocks"] = tuple(
                    str(incoming.name)
                    for incoming in instruction.incoming_blocks
                )
            except AttributeError:
                pass
        block.instrs.append(
            Instr(
                handler.value,
                [
                    self.operand_value(operand)
                    for operand in operands
                    if operand.value_kind.name
                    not in {"basic_block", "function"}
                ],
                result,
                attributes=attributes,
            )
        )

    def _branch(self, instruction: Any, block: BasicBlock) -> None:
        operands = list(instruction.operands)
        if len(operands) == 1:
            target = str(operands[0].name)
            block.instrs.append(
                Instr(
                    Handler.Br.value,
                    [],
                    None,
                    attributes={"target": target, "llvm_opcode": "br"},
                )
            )
            block.successors.append(target)
            return
        condition = self.operand_value(operands[0])
        # LLVM's binding exposes conditional labels as false, true.
        false_target = str(operands[1].name)
        true_target = str(operands[2].name)
        block.instrs.append(
            Instr(
                Handler.CondBr.value,
                [condition],
                None,
                attributes={
                    "true_target": true_target,
                    "false_target": false_target,
                    "llvm_opcode": "br",
                },
            )
        )
        block.successors.extend((true_target, false_target))

    def _switch(self, instruction: Any, block: BasicBlock) -> None:
        operands = list(instruction.operands)
        selector = self.operand_value(operands[0])
        default_target, spelled_cases = self._switch_cases(instruction)
        # The destination blocks the binding does expose must be the ones
        # the text names, in order; a disagreement is a reader fault, not
        # something to lower around.
        exposed_blocks = [
            str(operand.name)
            for operand in operands[1:]
            if operand.value_kind.name == "basic_block"
        ]
        named_blocks = [default_target, *(target for _, _, target in spelled_cases)]
        if exposed_blocks != named_blocks:
            raise ValueError(
                f"switch in {self.llvm_function.name}:{block.name} names "
                f"blocks {named_blocks} but the binding exposes "
                f"{exposed_blocks}: {str(instruction).strip()}"
            )
        cases = [
            (self.constants[("constant_int", f"{case_type} {literal}")], target)
            for case_type, literal, target in spelled_cases
        ]
        current = block
        for index, (literal, target) in enumerate(cases):
            is_last = index == len(cases) - 1
            next_name = (
                default_target
                if is_last
                else f"{block.name}.switch.{index}"
            )
            if not is_last:
                self.blocks[next_name] = BasicBlock(next_name)
            condition = self.fresh("bool", "switch_condition", next_name)
            current.instrs.extend((
                Instr(
                    Handler.Eq.value,
                    [selector, literal],
                    condition,
                    attributes={"llvm_opcode": "switch"},
                ),
                Instr(
                    Handler.CondBr.value,
                    [condition],
                    None,
                    attributes={
                        "true_target": target,
                        "false_target": next_name,
                        "llvm_opcode": "switch",
                    },
                ),
            ))
            current.successors.extend((target, next_name))
            if not is_last:
                current = self.blocks[next_name]

    def import_function(self) -> tuple[
        Function,
        tuple[LLVMRepositorySSAShortfall, ...],
        int,
    ]:
        arguments = self.define_values()
        entry_name = next(iter(self.blocks), "entry")
        entry = self.blocks.setdefault(entry_name, BasicBlock(entry_name))
        entry.instrs.extend(
            Instr(
                Handler.Const.value,
                [],
                value,
                attributes=self.constant_attributes[value.id],
            )
            for value in self.constants.values()
        )
        for llvm_block in self.llvm_function.blocks:
            block = self.blocks[str(llvm_block.name)]
            for instruction in llvm_block.instructions:
                if instruction.opcode == "br":
                    self._branch(instruction, block)
                elif instruction.opcode == "switch":
                    self._switch(instruction, block)
                else:
                    self._ordinary_instruction(instruction, block)
        return_type = str(self.llvm_function.global_value_type).split(
            "(", 1
        )[0].strip()
        import hashlib

        metadata: dict[str, Any] = {
            "llvm_argument_names": tuple(
                str(argument.name) for argument in self.llvm_function.arguments
            ),
            "llvm_return_dtype": _layout_dtype(return_type),
        }
        if return_type != "void":
            metadata["return_value"] = self.fresh(
                _layout_dtype(return_type), "return", "")
        # The kernel's source record, read by ``post_repository_kernel_identity``
        # in each compile book: the digest of its LLVM text and where every
        # value it numbered sits in it.  Provenance only; no decision reads it.
        metadata["llvm_repository_kernel"] = (
            hashlib.sha256(str(self.llvm_function).encode("utf-8")).hexdigest(),
            dict(self.value_origins),
        )
        return (
            Function(
                str(self.llvm_function.name),
                arguments,
                self.blocks,
                metadata=metadata,
            ),
            tuple(self.shortfalls),
            self.next_value_id,
        )


def import_llvm_to_repository_ssa(
    llvm_ir: str,
    *,
    include_declarations: bool = True,
) -> LLVMRepositorySSAResult:
    """Translate LLVM IR into the existing repository SSA vocabulary."""

    from llvmlite import binding as llvm

    llvm_module = llvm.parse_assembly(str(llvm_ir))
    llvm_module.verify()
    functions: dict[str, Function] = {}
    shortfalls: list[LLVMRepositorySSAShortfall] = []
    next_value_id = 0
    for llvm_function in llvm_module.functions:
        name = str(llvm_function.name)
        if llvm_function.is_declaration:
            if include_declarations:
                # Program: an imported intrinsic declaration (for example
                # ``llvm.sqrt.f64``), part of the imported repository like a
                # defined kernel.  Wrong record: its argument values were
                # minted with no row, so ``function_value_structure`` met a
                # value with no SSA identity.  Rule: the pass that creates a
                # value posts its row; the declaration carries the same
                # ``llvm_repository_kernel`` source record (digest of its LLVM
                # text, origin of each argument) so
                # ``post_repository_kernel_identity`` posts its rows.
                import hashlib

                arguments = []
                origins: dict[int, tuple[int, str, str]] = {}
                for argument in llvm_function.arguments:
                    origins[next_value_id] = (
                        len(origins), "argument", str(argument.name),
                    )
                    arguments.append(
                        SSAValue(next_value_id, dtype=_layout_dtype(argument.type))
                    )
                    next_value_id += 1
                functions[name] = Function(
                    name, arguments, {},
                    metadata={
                        "llvm_repository_kernel": (
                            hashlib.sha256(
                                str(llvm_function).encode("utf-8")
                            ).hexdigest(),
                            origins,
                        ),
                    },
                )
            continue
        importer = _FunctionImporter(llvm_function, next_value_id)
        function, function_shortfalls, next_value_id = (
            importer.import_function()
        )
        functions[name] = function
        shortfalls.extend(function_shortfalls)
    return LLVMRepositorySSAResult(
        IRModule(functions),
        tuple(shortfalls),
    )


def post_repository_kernel_identity(function: Any, book: Any = None) -> bool:
    """Post an imported kernel's identity rows on ``book`` (default: the
    active compile's), once per book.

    The import is cached process-wide (``c_backend_repository_ssa_reference``
    is an ``lru_cache``), so its Function objects reach many books; each
    book gets one ``repository_kernel_definition`` root per kernel,
    NOVEL(INGEST_SOURCE) keyed by the kernel name with the digest of its LLVM
    text, one ``repository_kernel_value`` row per imported value at its
    position in the kernel DERIVED from the root, and the value's
    ``ssa_value`` row DERIVED from that.  False when ``function`` is not an
    imported kernel."""

    from ....compiler.concordance_declarations import (
        INGESTION, INGEST_SOURCE, REPOSITORY_KERNEL_DEFINITION,
        REPOSITORY_KERNEL_VALUE, SSA_VALUE, RepositoryKernelDefinitionFact,
        RepositoryKernelValueFact, SSAValueFact, SSAValueOrigin,
    )
    from ....compiler.identity_concordance import (
        Derived, Mode, Novel, current_identity_book,
    )
    from ....compiler.ssa_record_return_state import function_scope_of

    record = (getattr(function, "metadata", None) or {}).get(
        "llvm_repository_kernel"
    )
    if record is None:
        return False
    if book is None:
        book = current_identity_book()
    digest, origins = record
    kernel = str(function.name)
    root_row = ("llvm_repository", kernel)
    # The root is posted once per book; the value rows are posted for every
    # numbering of the kernel that reaches this book (the import is cached
    # by text, and a second import of the same kernel numbers its values
    # from a different first id), so an existing root is read, not a
    # reason to return.
    root = book.latest_ref(REPOSITORY_KERNEL_DEFINITION, root_row)
    if root is None:
        root = book.post(
            REPOSITORY_KERNEL_DEFINITION, root_row,
            RepositoryKernelDefinitionFact(str(digest)),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
    values: dict[int, SSAValue] = {int(value.id): value for value in function.args}
    for block in function.blocks.values():
        for instruction in block.instrs:
            for value in (*instruction.args, instruction.res):
                if value is not None:
                    values.setdefault(int(value.id), value)
    returned = function.metadata.get("return_value")
    if returned is not None:
        values.setdefault(int(returned.id), returned)
    scope = function_scope_of(function)
    formal_ids = {int(value.id) for value in function.args}
    for value_id, (position, kind, text) in sorted(
        origins.items(), key=lambda item: item[1][0],
    ):
        value = values.get(int(value_id))
        if value is None:
            continue
        cell = book.latest_ref(
            REPOSITORY_KERNEL_VALUE, (root_row, int(position)),
        ) or book.post(
            REPOSITORY_KERNEL_VALUE, (root_row, int(position)),
            RepositoryKernelValueFact(str(kind), str(text)),
            stage=INGESTION, provenance=Derived((root,)), mode=Mode.CONCORD,
        )
        if book.latest_ref(SSA_VALUE, (scope, int(value_id))) is None:
            # Program: any imported repository kernel (binary_double, where_
            # double, ...).  Wrong record: its formals' rows were posted with
            # storage/rank at their defaults although ingestion is where the
            # kernel's declared ABI enters.  Rule: a formal posts the ABI the
            # LLVM signature itself declares, read from the imported value
            # the row derives from: a non-pointer formal is a ``scalar`` of
            # rank 0.  A pointer formal declares NO storage kind or rank --
            # those are the caller's contract, delivered on the call edge
            # (``span`` of the actual's rank) and posted where it is
            # delivered.  Posting ``reference`` here was measured: it made
            # every kernel pointer formal disagree with the span contract the
            # shell later delivers (4 type-disagreements per probe), so the
            # pointer's row stays unposted until the edge posts it.
            physical: dict[str, Any] = {}
            if (
                int(value_id) in formal_ids
                and value.dtype not in (None, "", "unknown", "ptr")
            ):
                physical = {"storage": "scalar", "rank": 0}
            book.post(
                SSA_VALUE, (scope, int(value_id)),
                SSAValueFact(
                    value.dtype, tuple(value.shape or ()),
                    SSAValueOrigin.MINTED, **physical,
                ),
                stage=INGESTION, provenance=Derived((cell,)),
                mode=Mode.REVISE,
            )
    return True


__all__ = [
    "LLVMRepositorySSAResult",
    "LLVMRepositorySSAShortfall",
    "import_llvm_to_repository_ssa",
    "post_repository_kernel_identity",
]
