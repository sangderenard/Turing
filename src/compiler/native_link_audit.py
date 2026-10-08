"""Native link audits: undefined repository operands and unmaterialised record parameters (relocated from fortran_c_shell)."""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any, Iterable, Mapping


def _full_native_link_failures(
    extraction_boundaries: Iterable[Mapping[str, Any]],
    unmaterialized_boundaries: Iterable[Mapping[str, Any]],
    unresolved_call_records: Iterable[Mapping[str, Any]],
    undefined_operands: Iterable[Mapping[str, Any]] = (),
    *,
    module: Any = None,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Classify every post-link condition forbidden by native execution."""

    non_native_boundaries = []
    for boundary in extraction_boundaries:
        contract = dict(boundary.get("extraction_contract") or {})
        parameters = dict(contract.get("parameters") or {})
        shell_profiles = tuple(map(
            str, parameters.get("shell_profiles") or (),
        ))
        reasons = []
        if str(contract.get("action") or "") == "python_host_call":
            reasons.append("python-host-call")
        if str(parameters.get("native_abi") or "") == "cpython-c-api":
            reasons.append("cpython-c-api")
        if "python" in shell_profiles or "cpython-c" in shell_profiles:
            reasons.append("python-shell-profile")
        if str(parameters.get("callbacks") or "") not in {"", "reject"}:
            reasons.append("callbacks-not-rejected")
        if reasons:
            non_native_boundaries.append({
                "identity": str(contract.get("identity") or ""),
                "action": str(contract.get("action") or ""),
                "reasons": tuple(reasons),
            })
    # Defined-as-a-formal is not proof of a legitimate program input. A
    # vanished authored computation can otherwise satisfy the undefined-use
    # check simply by becoming an invented argument in every calling frame.
    from .ssa_self_check import (
        check_formal_parity, check_optional_merges, check_structural_outputs,
    )

    unaccounted_formals = tuple(
        {"function": finding.function, "detail": finding.detail}
        for finding in check_formal_parity(module)
    ) if module is not None else ()
    unresolved_record_rows = tuple(
        {
            "function": str(function_name),
            **dict(row),
        }
        for function_name, function in (
            () if module is None else module.functions.items()
        )
        for row in function.metadata.get(
            "unresolved_record_sequence_rows", ()
        )
    )
    return {
        "unmaterialized_boundaries": tuple(unmaterialized_boundaries),
        "unresolved_call_records": tuple(unresolved_call_records),
        "undefined_operands": tuple(undefined_operands),
        "unaccounted_formals": unaccounted_formals,
        "unresolved_record_sequence_rows": unresolved_record_rows,
        "structural_outputs": tuple(
            {"function": finding.function, "detail": finding.detail}
            for finding in check_structural_outputs(module)
        ) if module is not None else (),
        "unrepresented_optional_merges": tuple(
            {"function": finding.function, "detail": finding.detail}
            for finding in check_optional_merges(module)
        ) if module is not None else (),
        "non_native_boundaries": tuple(non_native_boundaries),
    }


def _undefined_repository_ssa_operands(
    module: Any,
) -> tuple[Mapping[str, Any], ...]:
    """Inventory operands outside each function's complete value namespace."""

    findings = []
    for function_name, function in module.functions.items():
        defined = {int(value.id) for value in function.args}
        defined.update(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        seen = set()
        for block_name, block in function.blocks.items():
            for instruction in block.instrs:
                for argument in instruction.args:
                    value_id = int(argument.id)
                    key = (str(block_name), str(instruction.op), value_id)
                    accounting = dict(argument.accounting or {})
                    compiler_frame_definition = (
                        str(accounting.get("compiler_frame_storage") or "")
                        == str(function_name)
                    )
                    if (
                        value_id not in defined
                        and not compiler_frame_definition
                        and key not in seen
                    ):
                        seen.add(key)
                        findings.append({
                            "function": str(function_name),
                            "block": str(block_name),
                            "operation": str(instruction.op),
                            "value_id": value_id,
                            "value_accounting": accounting,
                            "value_names": tuple(
                                str(name)
                                for name, named_id in function.metadata.get(
                                    "value_names", ()
                                )
                                if int(named_id) == value_id
                            ),
                            "operand_ids": tuple(
                                int(item.id) for item in instruction.args
                            ),
                            "callee": instruction.attributes.get("callee"),
                            "attributes": dict(instruction.attributes or {}),
                            "block_trace": tuple(
                                (
                                    str(candidate.op),
                                    None if candidate.res is None else int(candidate.res.id),
                                    candidate.attributes.get("callee"),
                                    tuple(int(item.id) for item in candidate.args),
                                )
                                for candidate in block.instrs
                            ),
                        })
    return tuple(findings)


def _report_unmaterialised_record_parameters(module, extraction_contract) -> None:
    """Complain when a bound record never became a parameter.

    A LOUD COMPLAINT THAT NAMES THE INSTRUCTION IS WORTH MORE THAN A
    SILENT ADAPTATION. This one exists because the silent version cost a
    day: with a record declared in `program_abi.records` and bound in
    `program_abi.bindings`, the contract resolves the parameter's identity
    and the lowering drops the parameter anyway. Everything reached
    through it is then genuinely dead, the body empties, and the emission
    reports ZERO SHORTFALLS on a function whose whole content was

        define void @f(ptr %buffers, ptr %extents) { entry: ret void }

    Undeclared, the same program refused loudly and correctly --
    `opaque-state-effect`, naming the call. Declaring the record removed
    that guard without materialising anything, so a useful refusal became
    a no-op reported as success. This restores a complaint to that gap.

    It reports rather than raises: the gap is a compiler defect, not a
    fault in the program being compiled, and a build that is already
    working around it should not start failing. The receipt is on the
    function, and it is the thing to grep for when an emission looks
    suspiciously empty.
    """

    import warnings

    program_abi = getattr(extraction_contract, "program_abi", None)
    if program_abi is None or not getattr(program_abi, "records", None):
        return
    for qualified, function in module.functions.items():
        metadata = getattr(function, "metadata", None)
        if metadata is None:
            continue
        # ONLY BINDINGS AIMED AT THIS FUNCTION. The repository sheet binds
        # `state`, `targets`, `metrics`, `ctrl` and `controller` with a bare
        # `"*"`, so they match every function compiled and their absence
        # from any particular one means nothing. A binding written for a
        # named function -- `"*frame"`, `"_dependency_order"` -- is a claim
        # about THAT function, and its absence is the thing worth saying.
        targeted = {
            binding.parameter: program_abi.records[binding.record]
            for binding in program_abi.bindings
            if binding.function != "*"
            and fnmatchcase(str(qualified), binding.function)
            and binding.record in program_abi.records
        }
        if not targeted:
            continue
        bound = targeted
        present = set(dict(metadata.get("parameter_names", {}) or {}))
        missing = tuple(sorted(set(bound) - present))
        if not missing:
            continue
        metadata["unmaterialised_record_parameters"] = missing
        warnings.warn(
            f"{qualified}: {len(missing)} parameter(s) have a declared "
            f"program_abi record that never reached the ABI: "
            f"{', '.join(missing)}. Everything reached through them is dead, "
            f"so the emitted body may be empty with no shortfall. Declared "
            f"records: "
            + ", ".join(f"{name}={bound[name].identity}" for name in missing),
            RuntimeWarning,
            stacklevel=2,
        )
