"""Validate Turing precompile artifacts against the existing SSA vocabulary.

This pass does not lower, optimize, rewrite, or canonicalize the program.  It
answers two separate questions:

1. Is this a structurally valid Turing ``FusedProgram`` precompile artifact?
2. Which operation names in that artifact do not yet have an operation in the
   existing :class:`transmogrifier.ssa_registry.Handler` set?
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from ..common.tensors.fused_ir import FusedProgram, Meta, OpStep
from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    translations_for_operation,
)
from ..transmogrifier.ssa_registry import Handler


@dataclass(frozen=True, order=True)
class PrecompileFormatIssue:
    code: str
    program: str
    step_index: int | None
    value_id: int | None
    message: str

    def format(self) -> str:
        location = self.program
        if self.step_index is not None:
            location += f":step[{self.step_index}]"
        value = "" if self.value_id is None else f" value={self.value_id}"
        return f"{self.code} [{location}]{value}: {self.message}"


@dataclass(frozen=True, order=True)
class SSAOperationCompatibility:
    precompile_name: str
    ssa_name: str
    count: int
    locations: tuple[str, ...]


@dataclass(frozen=True, order=True)
class SSACompatibilityShortfall:
    operation_name: str
    count: int
    locations: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class PrecompileSSAValidationResult:
    format_issues: tuple[PrecompileFormatIssue, ...]
    compatible_operations: tuple[SSAOperationCompatibility, ...]
    compatibility_shortfalls: tuple[SSACompatibilityShortfall, ...]

    @property
    def valid_precompile(self) -> bool:
        return not self.format_issues

    @property
    def ssa_compatible(self) -> bool:
        return self.valid_precompile and not self.compatibility_shortfalls

    def compatibility_shortfall_report(self) -> str:
        if not self.compatibility_shortfalls:
            return "precompile SSA compatibility: complete"
        lines = ["precompile SSA compatibility shortfalls:"]
        for shortfall in self.compatibility_shortfalls:
            lines.append(
                f"- {shortfall.operation_name}: {shortfall.count} "
                f"occurrence(s) at {', '.join(shortfall.locations)}; "
                f"{shortfall.reason}"
            )
        return "\n".join(lines)


class PrecompileSSAValidationError(ValueError):
    def __init__(self, result: PrecompileSSAValidationResult):
        self.result = result
        sections = []
        if result.format_issues:
            sections.append(
                "invalid Turing precompile:\n"
                + "\n".join(
                    issue.format() for issue in result.format_issues
                )
            )
        if result.compatibility_shortfalls:
            sections.append(result.compatibility_shortfall_report())
        super().__init__("\n".join(sections))


# This is a spelling bridge only.  It names which existing SSA Handler an
# already-recorded precompile operation corresponds to.  Absence is reported;
# the validator never invents a Handler or decomposes an operation.
PRECOMPILE_TO_SSA: dict[str, Handler] = {
    "add": Handler.Add,
    "sub": Handler.Sub,
    "mul": Handler.Mul,
    "truediv": Handler.Div,
    "div": Handler.Div,
    "floordiv": Handler.FloorDiv,
    "mod": Handler.Mod,
    "pow": Handler.Pow,
    "matmul": Handler.MatMul,
    "neg": Handler.Neg,
    "abs": Handler.Abs,
    # Shape operations remain canonical tensor calls in SSA. Backends such as
    # Fortran spell them as native array expressions from the preserved attrs.
    "reshape": Handler.Call,
    "view": Handler.Call,
    "broadcast_to": Handler.Call,
    "repeat": Handler.Call,
    "mean": Handler.Call,
    "zeros_like": Handler.Fill,
    "bitand": Handler.And,
    "bitor": Handler.Or,
    "bitxor": Handler.Xor,
    "invert": Handler.Not,
    "shl": Handler.Shl,
    "shr": Handler.Shr,
    "logical_and": Handler.LAnd,
    "logical_or": Handler.LOr,
    "logical_not": Handler.LNot,
    "equal": Handler.Eq,
    "not_equal": Handler.Ne,
    "less": Handler.Lt,
    "less_equal": Handler.Le,
    "greater": Handler.Gt,
    "greater_equal": Handler.Ge,
    "int_trunc": Handler.Trunc,
    "trunc": Handler.Trunc,
    "zext": Handler.ZExt,
    "sext": Handler.SExt,
    "fptosi": Handler.FpToSi,
    "fptoui": Handler.FpToUi,
    "sitofp": Handler.SiToFp,
    "uitofp": Handler.UiToFp,
}


def ssa_handler_for_precompile(operation_name: str) -> Handler | None:
    if operation_name == "tensor_from_list":
        return Handler.Const
    if translations_for_operation(operation_name):
        return Handler.Call
    mapped = PRECOMPILE_TO_SSA.get(operation_name)
    if mapped is not None:
        return mapped
    try:
        return Handler(operation_name)
    except ValueError:
        try:
            return Handler[operation_name]
        except KeyError:
            return None


def ssa_compatibility_name_for_precompile(
    operation_name: str,
) -> str | None:
    if operation_name == "tensor_from_list":
        return Handler.Const.value
    translations = translations_for_operation(operation_name)
    if translations:
        callees = "|".join(
            dict.fromkeys(item.llvm_symbol for item in translations)
        )
        return f"{Handler.Call.value}[{callees}]"
    handler = ssa_handler_for_precompile(operation_name)
    return None if handler is None else handler.value


def _programs(artifact: Any) -> tuple[tuple[str, Any], ...]:
    if isinstance(artifact, FusedProgram):
        return (("program", artifact),)
    program = getattr(artifact, "program", None)
    stages = tuple(getattr(artifact, "stages", ()) or ())
    if program is None:
        return (("program", artifact),)
    return (
        ("program", program),
        *((f"stage[{index}]", stage) for index, stage in enumerate(stages)),
    )


def _validate_program_format(
    label: str,
    program: Any,
    *,
    require_typed_metadata: bool,
) -> list[PrecompileFormatIssue]:
    issues: list[PrecompileFormatIssue] = []

    def report(
        code: str,
        message: str,
        *,
        step_index: int | None = None,
        value_id: int | None = None,
    ) -> None:
        issues.append(
            PrecompileFormatIssue(
                code, label, step_index, value_id, message
            )
        )

    if not isinstance(program, FusedProgram):
        report(
            "PRECOMPILE_NOT_FUSED_PROGRAM",
            f"expected FusedProgram, got {type(program).__name__}",
        )
        return issues
    if program.version != 1:
        report(
            "PRECOMPILE_VERSION",
            f"unsupported FusedProgram version {program.version!r}",
        )
    if not isinstance(program.feeds, set):
        report(
            "PRECOMPILE_FEEDS_TYPE",
            "feeds must be a set of integer value IDs",
        )
    feed_ids: set[int] = set()
    for feed_id in program.feeds if isinstance(program.feeds, set) else ():
        if isinstance(feed_id, bool) or not isinstance(feed_id, int):
            report(
                "PRECOMPILE_VALUE_ID",
                "feed ID must be an integer",
            )
        else:
            feed_ids.add(feed_id)

    if not isinstance(program.steps, list):
        report(
            "PRECOMPILE_STEPS_TYPE",
            "steps must be a list of OpStep records",
        )
        steps: Iterable[Any] = ()
    else:
        steps = program.steps

    available = set(feed_ids)
    for index, step in enumerate(steps):
        if not isinstance(step, OpStep):
            report(
                "PRECOMPILE_STEP_TYPE",
                f"expected OpStep, got {type(step).__name__}",
                step_index=index,
            )
            continue
        if (
            isinstance(step.step_id, bool)
            or not isinstance(step.step_id, int)
            or step.step_id < 0
        ):
            report(
                "PRECOMPILE_STEP_ID",
                "step_id must be a non-negative integer",
                step_index=index,
            )
        if not isinstance(step.op_name, str) or not step.op_name:
            report(
                "PRECOMPILE_OPERATION_NAME",
                "op_name must be a non-empty string",
                step_index=index,
            )
        if not isinstance(step.input_ids, list):
            report(
                "PRECOMPILE_INPUTS_TYPE",
                "input_ids must be a list of integer value IDs",
                step_index=index,
            )
            input_ids = ()
        else:
            input_ids = step.input_ids
        for input_id in input_ids:
            if isinstance(input_id, bool) or not isinstance(input_id, int):
                report(
                    "PRECOMPILE_VALUE_ID",
                    "input ID must be an integer",
                    step_index=index,
                )
            elif input_id not in available:
                report(
                    "PRECOMPILE_UNPRODUCED_INPUT",
                    "input is neither a declared feed nor an earlier result",
                    step_index=index,
                    value_id=input_id,
                )
        result_id = step.result_id
        if (
            isinstance(result_id, bool)
            or not isinstance(result_id, int)
            or result_id < 0
        ):
            report(
                "PRECOMPILE_RESULT_ID",
                "result_id must be a non-negative integer",
                step_index=index,
            )
        elif result_id in available:
            report(
                "PRECOMPILE_DUPLICATE_PRODUCER",
                "result ID is already a feed or earlier produced value",
                step_index=index,
                value_id=result_id,
            )
        else:
            available.add(result_id)
        if not isinstance(step.attrs, dict):
            report(
                "PRECOMPILE_ATTRIBUTES_TYPE",
                "step attrs must be a dictionary",
                step_index=index,
            )

    structural = (
        isinstance(program.extras, dict)
        and program.extras.get("kernel_kind") == "structural"
    )
    if not isinstance(program.outputs, dict) or (
        not program.outputs and not structural
    ):
        report(
            "PRECOMPILE_OUTPUTS",
            "outputs must be a non-empty name-to-value-ID dictionary",
        )
    else:
        for name, value_id in program.outputs.items():
            if not isinstance(name, str) or not name:
                report(
                    "PRECOMPILE_OUTPUT_NAME",
                    "output name must be a non-empty string",
                )
            if isinstance(value_id, bool) or not isinstance(value_id, int):
                report(
                    "PRECOMPILE_VALUE_ID",
                    "output ID must be an integer",
                )
            elif value_id not in available:
                report(
                    "PRECOMPILE_UNPRODUCED_OUTPUT",
                    "output has no feed or instruction producer",
                    value_id=value_id,
                )

    if program.meta is not None and not isinstance(program.meta, dict):
        report(
            "PRECOMPILE_METADATA_TYPE",
            "meta must be a value-ID-to-Meta dictionary or None",
        )
    metadata = program.meta if isinstance(program.meta, dict) else {}
    if require_typed_metadata:
        for value_id in sorted(available):
            meta = metadata.get(value_id)
            if not isinstance(meta, Meta):
                report(
                    "PRECOMPILE_METADATA_MISSING",
                    "typed precompile value has no Meta record",
                    value_id=value_id,
                )
                continue
            if meta.shape is None or meta.dtype is None:
                report(
                    "PRECOMPILE_METADATA_INCOMPLETE",
                    "typed precompile Meta requires shape and dtype",
                    value_id=value_id,
                )
    for value_id in metadata:
        if isinstance(value_id, bool) or not isinstance(value_id, int):
            report(
                "PRECOMPILE_VALUE_ID",
                "metadata key must be an integer value ID",
            )
            continue
        if value_id not in available:
            report(
                "PRECOMPILE_ORPHAN_METADATA",
                "Meta record names no feed or produced value",
                value_id=value_id,
            )
    if program.state_in is not None and not isinstance(
        program.state_in, set
    ):
        report(
            "PRECOMPILE_STATE_TYPE",
            "state_in must be a set of value IDs or None",
        )
    if program.extras is not None and not isinstance(program.extras, dict):
        report(
            "PRECOMPILE_EXTRAS_TYPE",
            "extras must be a dictionary or None",
        )
    return issues


def validate_precompile_ssa_compatibility(
    artifact: Any,
    *,
    require_typed_metadata: bool = True,
) -> PrecompileSSAValidationResult:
    """Scan one Turing precompile artifact without changing it."""

    labelled_programs = _programs(artifact)
    format_issues = tuple(
        issue
        for label, program in labelled_programs
        for issue in _validate_program_format(
            label,
            program,
            require_typed_metadata=require_typed_metadata,
        )
    )

    # The complete program is the semantic manifest. Stages are alternate
    # executable partitions of those same steps and are format-validated above
    # without double-counting operation compatibility.
    semantic = labelled_programs[0][1]
    compatible_locations: dict[tuple[str, str], list[str]] = {}
    shortfall_locations: dict[str, list[str]] = {}
    if isinstance(semantic, FusedProgram) and isinstance(
        semantic.steps, list
    ):
        for index, step in enumerate(semantic.steps):
            if not isinstance(step, OpStep) or not isinstance(
                step.op_name, str
            ):
                continue
            location = f"program:step[{index}]"
            compatibility_name = ssa_compatibility_name_for_precompile(
                step.op_name
            )
            if compatibility_name is None:
                shortfall_locations.setdefault(step.op_name, []).append(
                    location
                )
            else:
                compatible_locations.setdefault(
                    (step.op_name, compatibility_name), []
                ).append(location)
        kernel_kind = (
            semantic.extras.get("kernel_kind")
            if isinstance(semantic.extras, dict)
            else None
        )
        if kernel_kind not in {
            None,
            "passthrough",
            "structural",
        }:
            # ``kernel_kind`` is part of the executable precompile contract.
            # Validate its name independently so a compatible placeholder
            # step (for example add-zero around a reshape copy) cannot hide a
            # lowering operation that SSA does not represent.
            # ``mixed`` describes the precompile container.  It is not an
            # executable operation and must evaporate at this boundary.
            if (
                str(kernel_kind) != "mixed"
                and ssa_handler_for_precompile(str(kernel_kind)) is None
            ):
                name = f"kernel_kind:{kernel_kind}"
                shortfall_locations.setdefault(name, []).append(
                    "program:extras.kernel_kind"
                )

    compatible = tuple(
        SSAOperationCompatibility(
            precompile_name,
            ssa_name,
            len(locations),
            tuple(locations),
        )
        for (precompile_name, ssa_name), locations
        in sorted(compatible_locations.items())
    )
    shortfalls = tuple(
        SSACompatibilityShortfall(
            operation_name,
            len(locations),
            tuple(locations),
            "no compatible operation name exists in "
            "transmogrifier.ssa_registry.Handler",
        )
        for operation_name, locations in sorted(shortfall_locations.items())
    )
    return PrecompileSSAValidationResult(
        format_issues,
        compatible,
        shortfalls,
    )


def require_precompile_ssa_compatible(
    artifact: Any,
    *,
    require_typed_metadata: bool = True,
) -> Any:
    result = validate_precompile_ssa_compatibility(
        artifact,
        require_typed_metadata=require_typed_metadata,
    )
    if not result.ssa_compatible:
        raise PrecompileSSAValidationError(result)
    return artifact


__all__ = [
    "PRECOMPILE_TO_SSA",
    "PrecompileFormatIssue",
    "PrecompileSSAValidationError",
    "PrecompileSSAValidationResult",
    "SSACompatibilityShortfall",
    "SSAOperationCompatibility",
    "require_precompile_ssa_compatible",
    "ssa_handler_for_precompile",
    "ssa_compatibility_name_for_precompile",
    "validate_precompile_ssa_compatibility",
]
