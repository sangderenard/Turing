"""Lower repository SSA into Nodus KernelIR through the canonical-op membrane.

This is the emission-side mirror of ``machine_dialect_ssa``'s ingestion
boundary. Machine SSA stages foreign machine semantics on their way *into*
repository SSA, guarded by ``repository_ssa_legalized``; this module stages
repository semantics on their way *out* to a device kernel, guarded by the
closed canonical catalog: an instruction crosses only when its ``op`` spelling
resolves through ``canonical_ops.json`` (by canonical ``name`` or by turing
``handler`` member) to an append-only canonical ID, which becomes KernelIR's
``sub_op``. Anything else is a named shortfall — never a guess.

KernelIR (nodus ``kernel_isa.h``) is itself SSA — one value per output, a flat
instruction list — so this lowering is SSA-to-SSA and structure-preserving.
One repository ``Function`` becomes one named kernel; a multi-function program
stays multiple kernels. There is no FusedProgram anywhere on this path.

The produced kernel is the element-wise map of the scalar SSA chain: each
function argument becomes a read-only storage buffer, the output becomes a
writable one, and the body is ADDR/LOAD → compute → ADDR/STORE indexed by the
global-invocation sentinel the nodus assembler publishes
(``kGlobalIndexValue``, ``kernel_spirv_assembler.h``).

Scope mirrors ``ssa_spirv_backend.py``'s proven first slice: one basic block,
one output, scalar 32-bit values. Wider scope (control flow through KernelIR
``IF``, f64 once KernelIR grows it, reductions) is follow-up work the
shortfall structure already accommodates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa import Function, Instr, SSAValue

# The nodus assembler publishes gl_GlobalInvocationID.x under this sentinel
# value id (kernel_spirv_assembler.h::kGlobalIndexValue). ADDR reaches it with
# an ordinary value-reference operand.
GLOBAL_INDEX_VALUE = 0xFFFFFFFF

# KernelIR ScalarType (kernel_isa.h) — I32/U32/F32 only; F64 is an upstream
# TODO there, so float64 is a shortfall here, not a silent narrowing.
_SCALAR_TYPES: dict[str, str] = {
    "int32": "I32",
    "uint32": "U32",
    "float32": "F32",
}

# KernelIR OpCode names this lowering emits. The catalog routes bitand/bitor/
# bitxor/invert to KernelIR's dedicated logical opcodes (the same shape
# nodus's bitops_lowering.h emits); everything else selects by sub_op. VAR is
# deliberately absent: buffer declarations travel in
# ``buffer_value_ids``/``values``, not as instructions.
_EMITTED_OPCODES = (
    "ADDR", "LOAD", "STORE", "BINARY", "UNARY", "CMP", "CAST",
    "AND", "OR", "XOR", "NOT",
)

# Composite operations (catalog class "opaque") are NOT Tier-0 KernelIR
# instructions -- a reduction cannot be one instruction, which is why the
# catalog records kernel_op=null and names a tier1_class instead. They are
# DEFINED on the nodus side by Tier-1 recipes over Tier-0 primitives
# (include/tier1_lowering.h), so this lowering reports them as named
# shortfalls rather than inventing an opcode for them.


class KernelIRLoweringError(ValueError):
    """Raised when the canonical catalog itself is missing or malformed."""


def default_catalog_path() -> Path:
    """Sibling-repo discovery, same convention as backend_capability_audit."""

    return Path(__file__).resolve().parents[2] / ".." / "nodus" / "ops" / "canonical_ops.json"


def load_canonical_catalog(path: Path | None = None) -> tuple[dict[str, Any], ...]:
    """The append-only canonical operation list, ids = list positions."""

    catalog_path = Path(path) if path is not None else default_catalog_path()
    try:
        document = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise KernelIRLoweringError(
            f"canonical catalog not found at {catalog_path}; pass an explicit "
            "path if the nodus repository is not a sibling checkout"
        ) from error
    operations = document.get("ops")
    if not isinstance(operations, list) or not operations:
        raise KernelIRLoweringError(
            f"canonical catalog at {catalog_path} has no 'ops' list"
        )
    return tuple(operations)


@dataclass(frozen=True)
class KernelIRShortfall:
    """One SSA construct this lowering cannot express yet."""

    op: str
    block: str
    reason: str

    def format(self) -> str:
        return f"{self.op} [{self.block}]: {self.reason}"


@dataclass(frozen=True)
class KernelValue:
    """Mirror of kernel_isa.h ValueDef; id is the position in ``values``."""

    scalar: str  # "I32" | "U32" | "F32"
    is_buffer: bool = False
    is_readonly: bool = False
    debug_name: str = ""


@dataclass(frozen=True)
class KernelOperand:
    """Mirror of kernel_isa.h Operand: a value reference or one immediate."""

    kind: str  # "ref" | "i" | "u" | "f"
    value: int | float

    @staticmethod
    def ref(value_id: int) -> "KernelOperand":
        return KernelOperand("ref", int(value_id))


@dataclass(frozen=True)
class KernelInstruction:
    """Mirror of kernel_isa.h Instruction."""

    op: str  # OpCode name
    sub_op: int = -1
    inputs: tuple[KernelOperand, ...] = ()
    outputs: tuple[int, ...] = ()


@dataclass
class KernelIRProgram:
    """One lowered kernel region plus its honesty ledger."""

    name: str
    values: tuple[KernelValue, ...] = ()
    buffer_value_ids: tuple[int, ...] = ()
    instrs: tuple[KernelInstruction, ...] = ()
    local_size: tuple[int, int, int] = (64, 1, 1)
    element_count: int = 0
    shortfalls: tuple[KernelIRShortfall, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.shortfalls


def serialize_kernel_ir(program: KernelIRProgram) -> str:
    """KIRTEXT v1: one record per line, whitespace-separated, no escaping.

    The C++ loader (nodus ``kernel_ir_text.cpp``) parses exactly this. Debug
    names are restricted to identifier characters; anything else is dropped
    rather than escaped, because the name is diagnostic only.
    """

    lines = [
        "kirtext 1",
        f"kernel {program.name}",
        "local_size "
        f"{program.local_size[0]} {program.local_size[1]} {program.local_size[2]}",
        f"element_count {program.element_count}",
    ]
    for value in program.values:
        name = value.debug_name
        if name and not name.replace("_", "").isalnum():
            name = ""
        lines.append(
            "value "
            f"{value.scalar} "
            f"{'buffer' if value.is_buffer else 'plain'} "
            f"{'readonly' if value.is_readonly else 'writable'}"
            + (f" {name}" if name else "")
        )
    if program.buffer_value_ids:
        lines.append(
            "buffers " + " ".join(str(v) for v in program.buffer_value_ids)
        )
    for instr in program.instrs:
        parts = [f"instr {instr.op} {instr.sub_op}", "in"]
        for operand in instr.inputs:
            if operand.kind == "ref":
                parts.append(f"ref:{operand.value}")
            elif operand.kind == "f":
                parts.append(f"f:{operand.value!r}")
            else:
                parts.append(f"{operand.kind}:{operand.value}")
        parts.append("out")
        parts.extend(str(v) for v in instr.outputs)
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


class _Lowering:
    def __init__(
        self,
        function: Function,
        outputs: Sequence[SSAValue],
        catalog: Sequence[Mapping[str, Any]],
        element_count: int,
        local_size: tuple[int, int, int],
    ) -> None:
        self.function = function
        self.outputs = list(outputs)
        self.element_count = int(element_count)
        self.local_size = tuple(int(v) for v in local_size)
        self.shortfalls: list[KernelIRShortfall] = []
        self.values: list[KernelValue] = []
        self.instrs: list[KernelInstruction] = []
        self.buffer_ids: list[int] = []
        self._ssa_to_kernel: dict[int, int] = {}
        # Two spellings resolve into the one catalog: canonical tensor names
        # ("add", "tanh") and repository Handler members ("Add", "Shl").
        self._by_name: dict[str, tuple[int, Mapping[str, Any]]] = {}
        self._by_handler: dict[str, tuple[int, Mapping[str, Any]]] = {}
        for canonical_id, entry in enumerate(catalog):
            name = entry.get("name")
            handler = entry.get("handler")
            if isinstance(name, str) and name not in self._by_name:
                self._by_name[name] = (canonical_id, entry)
            if isinstance(handler, str) and handler not in self._by_handler:
                self._by_handler[handler] = (canonical_id, entry)

    # -- bookkeeping ---------------------------------------------------------

    def _shortfall(self, op: str, block: str, reason: str) -> None:
        self.shortfalls.append(KernelIRShortfall(op, block, reason))

    def _bail(self, op: str, block: str, reason: str) -> KernelIRProgram:
        self._shortfall(op, block, reason)
        return self._finish()

    def _finish(self) -> KernelIRProgram:
        return KernelIRProgram(
            name=self.function.name,
            values=tuple(self.values),
            buffer_value_ids=tuple(self.buffer_ids),
            instrs=tuple(self.instrs),
            local_size=self.local_size,  # type: ignore[arg-type]
            element_count=self.element_count,
            shortfalls=tuple(self.shortfalls),
        )

    def _new_value(
        self,
        scalar: str,
        *,
        is_buffer: bool = False,
        is_readonly: bool = False,
        debug_name: str = "",
    ) -> int:
        self.values.append(
            KernelValue(scalar, is_buffer, is_readonly, debug_name)
        )
        return len(self.values) - 1

    def _scalar_of(self, value: SSAValue, block: str) -> str | None:
        dtype = value.dtype or "float32"
        scalar = _SCALAR_TYPES.get(dtype)
        if scalar is None:
            self._shortfall(
                f"%t{value.id}", block,
                f"dtype {dtype} has no KernelIR ScalarType yet "
                "(kernel_isa.h holds I32/U32/F32; F64 is an upstream TODO)",
            )
        return scalar

    # -- lowering ------------------------------------------------------------

    def run(self) -> KernelIRProgram:
        function = self.function
        if len(function.blocks) != 1:
            return self._bail(
                "<function>", "*",
                "multi-block control flow (Br/CondBr/Phi) is not yet "
                "supported by the KernelIR lowering (KernelIR IF is the "
                "designated target, not yet emitted)",
            )
        block = next(iter(function.blocks.values()))
        if len(self.outputs) != 1:
            return self._bail(
                "<function>", block.name,
                "exactly one output is supported for the element-wise kernel "
                "frame (multi-output regions are future work)",
            )
        if self.element_count <= 0:
            return self._bail(
                "<function>", block.name,
                "element_count must be positive: the runnable kernel form is "
                "the element-wise map of the scalar chain over storage "
                "buffers",
            )
        # Report a composite plainly before the element-wise frame's own
        # premise (one element per argument) trips on its shaped input -- the
        # composite is the real reason, and the more specific message is the
        # useful one.
        for instr in block.instrs:
            if instr.op == "Ret":
                continue
            entry = self._by_name.get(instr.op) or self._by_handler.get(instr.op)
            if entry is not None and not entry[1].get("lowerable", False):
                self._lower_instr(instr, block.name)
                return self._finish()

        for value in function.args:
            if value.shape:
                return self._bail(
                    f"%t{value.id}", block.name,
                    "array-shaped SSA values are not yet lowered; the "
                    "element-wise frame maps a scalar chain over buffers",
                )

        gid = KernelOperand.ref(GLOBAL_INDEX_VALUE)

        # Arguments: one read-only buffer each, loaded at the global index.
        for argument in function.args:
            scalar = self._scalar_of(argument, block.name)
            if scalar is None:
                return self._finish()
            buffer_id = self._new_value(
                scalar, is_buffer=True, is_readonly=True,
                debug_name=f"t{argument.id}_in",
            )
            self.buffer_ids.append(buffer_id)
            pointer = self._new_value(scalar, debug_name=f"t{argument.id}_ptr")
            loaded = self._new_value(scalar, debug_name=f"t{argument.id}")
            self.instrs.append(KernelInstruction(
                "ADDR", -1, (KernelOperand.ref(buffer_id), gid), (pointer,),
            ))
            self.instrs.append(KernelInstruction(
                "LOAD", -1, (KernelOperand.ref(pointer),), (loaded,),
            ))
            self._ssa_to_kernel[argument.id] = loaded

        for instr in block.instrs:
            if instr.op == "Ret":
                continue
            self._lower_instr(instr, block.name)

        return self._emit_output_store(block.name)

    def _emit_output_store(self, block: str) -> KernelIRProgram:
        """Declare the output buffer and store the result at the global index.

        Shared by both frames: the element-wise kernel stores one element per
        invocation, and a whole-buffer reduction stores its single total from
        the one invocation its element_count=1 dispatch runs.
        """

        output = self.outputs[0]
        result = self._ssa_to_kernel.get(output.id)
        if result is None:
            self._shortfall(
                f"%t{output.id}", block,
                "declared output was never defined by a lowered instruction",
            )
            return self._finish()
        scalar = self._scalar_of(output, block)
        if scalar is None:
            return self._finish()
        out_buffer = self._new_value(
            scalar, is_buffer=True, debug_name=f"t{output.id}_out",
        )
        self.buffer_ids.append(out_buffer)
        out_pointer = self._new_value(scalar, debug_name=f"t{output.id}_optr")
        gid = KernelOperand.ref(GLOBAL_INDEX_VALUE)
        self.instrs.append(KernelInstruction(
            "ADDR", -1, (KernelOperand.ref(out_buffer), gid), (out_pointer,),
        ))
        self.instrs.append(KernelInstruction(
            "STORE", -1,
            (KernelOperand.ref(out_pointer), KernelOperand.ref(result)), (),
        ))
        return self._finish()

    def _immediate(self, scalar: str, raw: Any) -> KernelOperand:
        if scalar == "F32":
            return KernelOperand("f", float(raw))
        if scalar == "I32":
            return KernelOperand("i", int(raw))
        return KernelOperand("u", int(raw))

    def _lower_instr(self, instr: Instr, block: str) -> None:
        entry = self._by_name.get(instr.op) or self._by_handler.get(instr.op)
        if entry is None:
            self._shortfall(
                instr.op, block,
                "no canonical catalog entry under either the canonical name "
                "or the Handler spelling; the membrane admits catalog "
                "operations only",
            )
            return
        canonical_id, operation = entry
        if not operation.get("lowerable", False):
            family = operation.get("tier1_class")
            detail = (
                f"it is a Tier-1 '{family}' composite, defined by a recipe over "
                "Tier-0 primitives on the nodus side"
                if family
                else "it has no Tier-0 expression"
            )
            self._shortfall(
                instr.op, block,
                f"canonical operation '{operation.get('name')}' is not one "
                f"Tier-0 KernelIR instruction: {detail}",
            )
            return
        kernel_op = operation.get("kernel_op")
        if kernel_op not in _EMITTED_OPCODES:
            self._shortfall(
                instr.op, block,
                f"canonical operation '{operation.get('name')}' maps to "
                f"KernelIR opcode {kernel_op!r}, which this lowering does "
                "not emit yet",
            )
            return
        if instr.res is None:
            self._shortfall(
                instr.op, block, "instruction produces no result value",
            )
            return
        result_scalar = self._scalar_of(instr.res, block)
        if result_scalar is None:
            return

        operands: list[KernelOperand] = []
        for argument in instr.args:
            kernel_id = self._ssa_to_kernel.get(argument.id)
            if kernel_id is None:
                self._shortfall(
                    instr.op, block,
                    f"operand %t{argument.id} has no lowered definition",
                )
                return
            operands.append(KernelOperand.ref(kernel_id))

        # right_scalar/left_scalar attributes are folded exactly as the
        # SPIR-V backend folds them: one explicit operand plus one immediate.
        right = (instr.attributes or {}).get("right_scalar")
        left = (instr.attributes or {}).get("left_scalar")
        if right is not None and len(operands) == 1:
            operands.append(self._immediate(result_scalar, right))
        elif left is not None and len(operands) == 1:
            operands.insert(0, self._immediate(result_scalar, left))

        arity = operation.get("arity")
        if isinstance(arity, int) and arity != len(operands):
            self._shortfall(
                instr.op, block,
                f"canonical operation '{operation.get('name')}' expects "
                f"{arity} operand(s), instruction has {len(operands)}",
            )
            return

        result = self._new_value(result_scalar, debug_name=f"t{instr.res.id}")
        self.instrs.append(KernelInstruction(
            str(kernel_op), canonical_id, tuple(operands), (result,),
        ))
        self._ssa_to_kernel[instr.res.id] = result


def lower_function_to_kernel_ir(
    function: Function,
    outputs: Sequence[SSAValue],
    *,
    element_count: int,
    local_size: tuple[int, int, int] = (64, 1, 1),
    catalog: Sequence[Mapping[str, Any]] | None = None,
) -> KernelIRProgram:
    """Lower one repository-SSA function into one KernelIR kernel region.

    ``element_count`` is the buffer length of the element-wise map. The
    canonical catalog defaults to the sibling nodus checkout's
    ``ops/canonical_ops.json``; pass ``catalog`` explicitly to pin one.
    """

    resolved = tuple(catalog) if catalog is not None else load_canonical_catalog()
    return _Lowering(
        function, outputs, resolved, element_count, local_size,
    ).run()
