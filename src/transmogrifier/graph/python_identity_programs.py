"""Declarative Python identities expressed in ProcessGraph vocabulary.

An extraction contract answers whether implementation source may be entered.
These records answer the next question: what graph-native meaning replaces a
known Python callable?  A replacement is either one canonical operator, a
small operator program, a compiler-owned object/effect, or compile-time-only
metadata.  The records are inert descriptions; they never execute Python or
expand callable source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from types import MappingProxyType
from typing import Any, Mapping

from ...common.tensors.operator_catalog import (
    CANONICAL_ABSTRACT_TENSOR_OPERATORS,
)


_PROCESS_GRAPH_OPERATORS = frozenset({
    "Call", "Constant", "Deploy", "Deepcopy", "GetAttr", "Indexed",
    "IndexedStore", "Join", "SetAttr", "stream_publish",
})
_KNOWN_OPERATORS = CANONICAL_ABSTRACT_TENSOR_OPERATORS | _PROCESS_GRAPH_OPERATORS


@dataclass(frozen=True, slots=True)
class PythonProgramStep:
    name: str
    operator: str
    inputs: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a Python identity program step requires a name")
        if self.operator not in _KNOWN_OPERATORS:
            raise ValueError(f"unknown ProcessGraph operator {self.operator!r}")
        object.__setattr__(
            self, "attributes", MappingProxyType(dict(self.attributes))
        )

    def mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operator": self.operator,
            "inputs": self.inputs,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class PythonIdentityProgram:
    identities: tuple[str, ...]
    kind: str
    steps: tuple[PythonProgramStep, ...] = ()
    output: str | None = None
    object_type: str | None = None
    effects: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.identities:
            raise ValueError("a Python identity program requires an identity")
        if self.kind not in {"operator", "program", "object", "compile_time"}:
            raise ValueError(f"unknown Python identity replacement kind {self.kind!r}")
        names = {step.name for step in self.steps}
        if len(names) != len(self.steps):
            raise ValueError("Python identity program step names must be unique")
        if self.output is not None and self.output not in names:
            raise ValueError(f"unknown Python identity program output {self.output!r}")
        if self.kind == "operator" and len(self.steps) != 1:
            raise ValueError("an operator identity requires exactly one step")
        if self.kind == "object" and not self.object_type:
            raise ValueError("an object identity requires object_type")

    @property
    def direct_operator(self) -> str | None:
        return self.steps[0].operator if self.kind == "operator" else None

    @property
    def direct_attributes(self) -> dict[str, Any]:
        return dict(self.steps[0].attributes) if self.kind == "operator" else {}

    def mapping(self) -> dict[str, Any]:
        return {
            "identities": self.identities,
            "kind": self.kind,
            "steps": tuple(step.mapping() for step in self.steps),
            "output": self.output,
            "object_type": self.object_type,
            "effects": self.effects,
            "description": self.description,
        }


def _operator(
    *identities: str, operator: str, inputs: tuple[str, ...] = ("$arg0",),
    attributes: Mapping[str, Any] | None = None, effects: tuple[str, ...] = (),
    description: str = "",
) -> PythonIdentityProgram:
    return PythonIdentityProgram(
        identities=tuple(identities),
        kind="operator",
        steps=(PythonProgramStep("result", operator, inputs, attributes or {}),),
        output="result",
        effects=effects,
        description=description,
    )


PYTHON_IDENTITY_PROGRAMS: tuple[PythonIdentityProgram, ...] = (
    _operator(
        "src.common.tensors.abstraction.AbstractTensor.matmul",
        "src.common.tensors.abstraction.AbstractTensor.__matmul__",
        "src.common.tensors.abstraction.AbstractTensor.__rmatmul__",
        "src.common.tensors.abstraction.AbstractTensor.__imatmul__",
        "src.common.tensors.abstraction_methods.blas.matmul",
        "src.common.tensors.abstraction_methods.blas.rmatmul",
        "src.common.tensors.abstraction_methods.blas.imatmul",
        operator="matmul", inputs=("$receiver", "$arg0"),
        attributes={
            "semantic_library": "src.common.tensors.blas",
            "semantic_kernel": "gemm",
            "semantic_source_symbol": "GEMM_SOURCE",
            "semantic_parameters": {"alpha": 1.0, "beta": 0.0},
            "backend_intrinsic_family": "blas.gemm",
        },
        description=(
            "backend-neutral matrix product; matching backends may replace "
            "it through their receipted BLAS intrinsic identity library"
        ),
    ),
    _operator(
        "src.common.dt_system.dt_controller._restore_type",
        operator="cast_like", inputs=("$arg0", "$arg1"),
        attributes={"target_from_operand": 1},
        description=(
            "construct the reference operand's schema type from the value"
        ),
    ),
    _operator(
        "src.common.tensors.abstraction.AbstractTensor.tensor",
        operator="tensor",
        attributes={
            "ensures_schema_type": (
                "src.common.tensors.abstraction.AbstractTensor"
            ),
        },
        description=(
            "idempotent AbstractTensor normalization from any accepted input"
        ),
    ),
    _operator(
        "src.common.tensors.abstraction_methods.elementwise.maximum",
        operator="maximum", inputs=("$receiver", "$arg0"),
        description="AbstractTensor elementwise maximum",
    ),
    _operator(
        "src.common.tensors.abstraction_methods.elementwise.minimum",
        operator="minimum", inputs=("$receiver", "$arg0"),
        description="AbstractTensor elementwise minimum",
    ),
    _operator("builtins.float", operator="float", description="scalar extraction/cast"),
    _operator("builtins.int", operator="int", description="integer cast"),
    _operator("builtins.bool", operator="bool", description="truth-value cast"),
    _operator("builtins.abs", operator="abs"),
    _operator("builtins.min", operator="min", inputs=("$args",)),
    _operator("builtins.max", operator="max", inputs=("$args",)),
    _operator("builtins.sum", operator="sum", inputs=("$arg0", "$arg1?")),
    _operator("builtins.all", operator="all"),
    _operator("builtins.any", operator="any"),
    _operator("math.sqrt", operator="sqrt"),
    _operator("math.exp", operator="exp"),
    _operator("math.tanh", operator="tanh"),
    _operator(
        "random.Random.gauss", "random.gauss",
        operator="random_source", inputs=("$arg0", "$arg1", "$state:python_prng"),
        attributes={"distribution": "normal", "result_kind": "scalar"},
        effects=("read_write:python_prng",),
        description="stateful normal random scalar",
    ),
    _operator(
        "builtins.print", operator="stream_publish", inputs=("$args",),
        attributes={"stream": "text"}, effects=("publish:stdout",),
    ),
    _operator(
        "copy.deepcopy", operator="Deepcopy",
        description=(
            "Handler.Deepcopy, the SSA vocabulary's own deep copy -- CPython's "
            "copy implementation is never ingested"
        ),
    ),
    PythonIdentityProgram(
        identities=("builtins.len",), kind="program",
        steps=(
            PythonProgramStep("shape", "shape", ("$arg0",)),
            PythonProgramStep("axis", "Constant", attributes={"value": 0}),
            PythonProgramStep("result", "Indexed", ("$shape", "$axis")),
        ),
        output="result",
        description="Python len is the leading resident dimension, not numel",
    ),
    PythonIdentityProgram(
        identities=("builtins.range",), kind="object",
        object_type="arithmetic_sequence",
        description="planner-owned start/stop/step loop domain",
    ),
    PythonIdentityProgram(
        identities=("builtins.enumerate",), kind="object",
        object_type="enumerated_sequence",
        description="resident sequence plus planner induction projection",
    ),
    PythonIdentityProgram(
        identities=("builtins.zip",), kind="object",
        object_type="zipped_sequence",
        description="lockstep resident tuple projection",
    ),
    PythonIdentityProgram(
        identities=("builtins.id",), kind="object",
        object_type="stable_object_identity",
        description="compiler identity-table handle; never a CPython address",
    ),
    PythonIdentityProgram(
        identities=("builtins.getattr",), kind="program",
        steps=(PythonProgramStep(
            "result", "GetAttr", ("$arg0", "$arg1", "$arg2?"),
        ),), output="result",
        description="schema attribute projection with an optional default",
    ),
    PythonIdentityProgram(
        identities=("builtins.setattr",), kind="program",
        steps=(PythonProgramStep(
            "result", "SetAttr", ("$arg0", "$arg1", "$arg2"),
        ),), output="result", effects=("write:receiver",),
        description="schema attribute update",
    ),
    PythonIdentityProgram(
        identities=("builtins.hasattr",), kind="object",
        object_type="attribute_presence_guard",
        description="compile-time schema capability test or guarded projection",
    ),
    PythonIdentityProgram(
        identities=("builtins.isinstance",), kind="object",
        object_type="schema_type_guard",
        description="type-set test against compiler schema identities",
    ),
    PythonIdentityProgram(
        identities=("builtins.type",), kind="object",
        object_type="schema_type_identity",
        description="compiler schema identity; never a CPython type object",
    ),
    PythonIdentityProgram(
        identities=("builtins.callable",), kind="object",
        object_type="callable_capability_guard",
        description="test for a compiler-known call interface",
    ),
    PythonIdentityProgram(
        identities=("builtins.super",), kind="object",
        object_type="oop_super_dispatch",
        description="next implementation in the compiled class dispatch order",
    ),
    PythonIdentityProgram(
        identities=("builtins.list",), kind="object",
        object_type="resident_sequence",
        description="materialized sequence with length-cell storage ABI",
    ),
    PythonIdentityProgram(
        identities=("builtins.tuple",), kind="object",
        object_type="resident_tuple",
        description="fixed resident product sequence",
    ),
    PythonIdentityProgram(
        identities=(
            "src.common.tensors.abstract_nn.utils.set_seed",
            "random.seed", "numpy.random.seed", "torch.manual_seed",
            "torch.cuda.manual_seed_all",
        ),
        kind="object", object_type="prng_state",
        effects=("write:python_prng",),
        description="initialize compiler-owned deterministic PRNG state",
    ),
    PythonIdentityProgram(
        identities=("builtins.list.append", "list.append"), kind="program",
        steps=(
            PythonProgramStep(
                "result", "IndexedStore", ("$receiver", "$length", "$arg0"),
                {"index": "append"},
            ),
        ), output="result", effects=("write:receiver",),
        description="append at the resident sequence length cell",
    ),
    PythonIdentityProgram(
        identities=("builtins.list.extend", "list.extend"), kind="program",
        steps=(
            PythonProgramStep("lanes", "Deploy", ("$arg0",)),
            PythonProgramStep(
                "result", "IndexedStore", ("$receiver", "$lanes"),
                {"index": "append_each"},
            ),
        ), output="result", effects=("write:receiver",),
        description="append each source element to resident storage",
    ),
    PythonIdentityProgram(
        identities=(
            "logging.Logger.debug", "logging.Logger.info",
            "src.common.dt_system.debug.dbg",
            "src.common.tensors.autoautograd.*.annotate",
            "src.common.tensors.autoautograd.*.create_tensor_node",
            "src.common.tensors.autoautograd.*.record",
        ),
        kind="compile_time", effects=("metadata",),
        description="retain provenance when useful; emit no runtime operation",
    ),
)


def resolve_python_identity(identity: str | None) -> PythonIdentityProgram | None:
    """Return the first exact or glob identity replacement."""

    if not identity:
        return None
    text = str(identity)
    for program in PYTHON_IDENTITY_PROGRAMS:
        for pattern in program.identities:
            if text == pattern or ("*" in pattern and fnmatchcase(text, pattern)):
                return program
    return None


__all__ = [
    "PYTHON_IDENTITY_PROGRAMS",
    "PythonIdentityProgram",
    "PythonProgramStep",
    "resolve_python_identity",
]
