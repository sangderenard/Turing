"""The calling contract for an auto-compiled program, written beside it.

A compiled artifact is useless without knowing how to call it, and until now
every caller rediscovered that by reading emitted source and guessing: which
subroutine is the entry point, how many arguments it takes, which are passed
by value and which by reference, what each one's element type is, and which
of them are results rather than inputs. Guessing wrong does not fail
cleanly -- passing an ``int`` by value where a Fortran dummy expects a
reference is an access violation, and passing the numeric subroutine instead
of the control one silently runs the loop body once.

So the contract is emitted with the artifact, as YAML, from the same
``Function`` objects the code generator used. It is a description of what was
generated, not a second source of truth that could disagree.

Read one back with ``load_api`` (or plain ``yaml.safe_load``; the file is
ordinary YAML with no custom tags).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..transmogrifier.ssa import Function, SSAValue

SCHEMA = "turing-compiled-program-api-v1"


# How an SSA dtype is spelled for a C-ABI caller, alongside the ctypes name a
# Python caller needs. Both are recorded because a caller in another language
# needs the first and cannot use the second.
_C_TYPES: dict[str, tuple[str, str]] = {
    "uint8": ("uint8_t", "c_uint8"),
    "u8": ("uint8_t", "c_uint8"),
    "float": ("float", "c_float"),
    "float32": ("float", "c_float"),
    "f32": ("float", "c_float"),
    "double": ("double", "c_double"),
    "float64": ("double", "c_double"),
    "f64": ("double", "c_double"),
    "int": ("int32_t", "c_int32"),
    "int32": ("int32_t", "c_int32"),
    "i32": ("int32_t", "c_int32"),
    "int64": ("int64_t", "c_int64"),
    "i64": ("int64_t", "c_int64"),
    "bool": ("bool", "c_bool"),
    "logical": ("bool", "c_bool"),
}


@dataclass(frozen=True)
class Parameter:
    """One dummy argument, in declaration order."""

    name: str
    role: str  # "extent" | "input" | "inout" | "workspace" | "output"
    dtype: str
    c_type: str
    ctypes_name: str
    # Fortran passes scalars declared `value` by value and everything else by
    # reference. A caller that gets this backwards gets an access violation,
    # not a wrong number, so it is stated rather than inferred from role.
    passing: str  # "value" | "reference"
    shape: tuple[int, ...] = ()
    extent: str | None = None
    # Stable source/IR name retained beside the ABI-local ``t<ID>`` spelling.
    # Keep this before newer optional fields because some established callers
    # construct Parameters positionally through this slot.
    source_name: str | None = None
    # Exact runtime dimensions for a shape-dynamic array. ``extent`` remains
    # as the rank-one/backward-compatible spelling.
    extents: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "c_type": self.c_type,
            "ctypes": self.ctypes_name,
            "passing": self.passing,
        }
        if self.shape:
            mapping["shape"] = list(self.shape)
        if self.extent is not None:
            mapping["extent"] = self.extent
        if self.extents:
            mapping["extents"] = list(self.extents)
        if self.source_name is not None:
            mapping["source_name"] = self.source_name
        return mapping


@dataclass(frozen=True)
class EntryPoint:
    name: str
    symbol: str
    kind: str  # "control" | "numerical" | "region"
    parameters: tuple[Parameter, ...] = ()
    # Which entry point a caller should actually invoke, and why. A program
    # with a loop has its iteration in the control subroutine; calling the
    # numerical one directly runs the body once.
    note: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "name": self.name,
            "symbol": self.symbol,
            "kind": self.kind,
            "parameters": [p.to_mapping() for p in self.parameters],
        }
        if self.note:
            mapping["note"] = self.note
        return mapping


@dataclass(frozen=True)
class CompiledProgramAPI:
    module: str
    language: str
    entry: str | None
    entry_points: tuple[EntryPoint, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "module": self.module,
            "language": self.language,
            "entry": self.entry,
            "metadata": dict(self.metadata),
            "entry_points": [e.to_mapping() for e in self.entry_points],
        }

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_mapping(), sort_keys=False)

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_yaml(), encoding="utf-8")
        return destination

    def entry_point(self, name: str) -> EntryPoint:
        for candidate in self.entry_points:
            if candidate.name == name:
                return candidate
        raise KeyError(f"{name!r} is not an entry point of {self.module}")


def _c_type_for(dtype: str | None) -> tuple[str, str]:
    return _C_TYPES.get(str(dtype or "float64"), ("double", "c_double"))


def describe_fortran_function(
    name: str,
    function: Function,
    *,
    extent_names: Sequence[str] = (),
    outputs: Iterable[SSAValue] = (),
    kind: str = "numerical",
    note: str | None = None,
    source_names: Mapping[int, str] | None = None,
    dynamic_array_extents: Mapping[int, str] | None = None,
    dynamic_array_dimensions: Mapping[int, Sequence[str]] | None = None,
    array_argument_ids: Iterable[int] = (),
    reference_argument_ids: Iterable[int] = (),
) -> EntryPoint:
    """Describe one emitted Fortran subroutine's calling contract.

    Argument order mirrors ``emit_function`` exactly: the extents it declares
    first (scalars, passed by value), then its SSA arguments, then the values
    named as outputs.
    """

    parameters: list[Parameter] = []
    source_names = dict(source_names or {})
    dynamic_array_extents = {
        int(value_id): str(extent)
        for value_id, extent in dict(dynamic_array_extents or {}).items()
    }
    dynamic_array_dimensions = {
        int(value_id): tuple(map(str, dimensions))
        for value_id, dimensions in dict(dynamic_array_dimensions or {}).items()
    }
    array_argument_ids = {int(value_id) for value_id in array_argument_ids}
    reference_argument_ids = {
        int(value_id) for value_id in reference_argument_ids
    }
    for extent in extent_names:
        parameters.append(
            Parameter(
                name=str(extent),
                role="extent",
                dtype="int32",
                c_type="int32_t",
                ctypes_name="c_int32",
                passing="value",
            )
        )
    output_ids = {value.id for value in outputs}
    for value in function.args:
        c_type, ctypes_name = _c_type_for(value.dtype)
        dynamic_extent = dynamic_array_extents.get(int(value.id))
        array = (
            bool(value.shape)
            or dynamic_extent is not None
            or int(value.id) in array_argument_ids
        )
        accounting = dict(value.accounting or {})
        workspace = (
            not accounting.get("program_abi_parameter")
            and (
                accounting.get("linked_call_frame_storage") is not None
                or accounting.get("returned_record_storage") is not None
            )
        )
        dimensions = dynamic_array_dimensions.get(int(value.id), ())
        authored_source_name = source_names.get(int(value.id))
        if authored_source_name is None and accounting.get("program_abi_parameter"):
            authored_source_name = str(accounting["program_abi_parameter"])
            if accounting.get("program_abi_field"):
                authored_source_name += "." + str(accounting["program_abi_field"])
        parameters.append(
            Parameter(
                name=f"t{value.id}",
                # An SSA value that is both an argument and a result is one
                # preallocated Fortran intent(inout) arena.  Keep it an input
                # in the shell contract so its initial contents are loaded;
                # feedback/output aliases refer to the same resident slot.
                role=(
                    "workspace"
                    if workspace
                    else "inout" if int(value.id) in output_ids else "input"
                ),
                dtype=str(value.dtype or "float64"),
                c_type=c_type,
                ctypes_name=ctypes_name,
                # A scalar dummy is declared `value` by the emitter; an array
                # never is.
                passing=(
                    "reference"
                    if (
                        array
                        or int(value.id) in output_ids
                        or int(value.id) in reference_argument_ids
                    )
                    else "value"
                ),
                shape=tuple(value.shape or ()),
                extent=(
                    dynamic_extent
                    or (dimensions[0] if len(dimensions) == 1 else None)
                    or (
                        str(extent_names[-1])
                        if array and extent_names and not dimensions
                        else None
                    )
                ),
                extents=dimensions,
                source_name=authored_source_name,
            )
        )
    for value in outputs:
        if value.id in {argument.id for argument in function.args}:
            continue
        c_type, ctypes_name = _c_type_for(value.dtype)
        dimensions = dynamic_array_dimensions.get(int(value.id), ())
        parameters.append(
            Parameter(
                name=f"t{value.id}",
                role="output",
                dtype=str(value.dtype or "float64"),
                c_type=c_type,
                ctypes_name=ctypes_name,
                passing="reference",
                shape=tuple(value.shape or ()),
                extent=(
                    dimensions[0]
                    if len(dimensions) == 1
                    else (
                        str(extent_names[-1])
                        if extent_names and not dimensions
                        else None
                    )
                ),
                extents=dimensions,
                source_name=source_names.get(int(value.id)),
            )
        )
    return EntryPoint(
        name=name, symbol=name, kind=kind, parameters=tuple(parameters), note=note
    )


def load_api(path: str | Path) -> dict[str, Any]:
    """Read a descriptor back."""

    import yaml

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping) or loaded.get("schema") != SCHEMA:
        raise ValueError(
            f"{path} is not a {SCHEMA} descriptor"
        )
    return dict(loaded)


__all__ = [
    "SCHEMA",
    "CompiledProgramAPI",
    "EntryPoint",
    "Parameter",
    "describe_fortran_function",
    "load_api",
]
