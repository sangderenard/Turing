"""Retain an AOT program as coordinated control and numerical cards.

``FusedProgram`` is deliberately only one numerical region.  This module keeps
the larger compiler product intact: one control card, every captured numeric
region, exact resident-value connections, authored class/function cards from
Map IR, and the outer-arena feedback policy.  Backends may compile cards lazily
or together, but cannot silently replace the control program with its observed
discovery trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..common.tensors.fused_ir import ordered_feed_ids
from .card_graph import build_card_graph
from .control_source import ControlProgram, ControlTarget, render_control_program


@dataclass(frozen=True)
class CardPort:
    name: str
    value_id: int
    role: str
    dtype: str = "float64"
    shape: tuple[int, ...] = ()
    storage_value_id: int | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_id": int(self.value_id),
            "storage_value_id": int(
                self.value_id
                if self.storage_value_id is None
                else self.storage_value_id
            ),
            "role": self.role,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "passing": "arena-address",
        }


@dataclass(frozen=True)
class ProgramCard:
    card_id: str
    kind: str
    inputs: tuple[CardPort, ...] = ()
    outputs: tuple[CardPort, ...] = ()
    region_index: int | None = None
    program: Any = field(default=None, repr=False, compare=False)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.card_id,
            "kind": self.kind,
            **(
                {"region_index": int(self.region_index)}
                if self.region_index is not None else {}
            ),
            "ports": [
                *(port.to_mapping() for port in self.inputs),
                *(port.to_mapping() for port in self.outputs),
            ],
            "cache_key": self.card_id,
        }


@dataclass(frozen=True)
class CardAliasEdge:
    source_card: str
    source_port: str
    target_card: str
    target_port: str
    storage_value_id: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": "resident-memory",
            "from": self.source_card,
            "to": self.target_card,
            "bindings": [{
                "from": self.source_port,
                "to": self.target_port,
                "storage_value_id": int(self.storage_value_id),
                "rewrite": "alias",
            }],
        }


@dataclass(frozen=True)
class ParametricCardProgram:
    name: str
    control: ControlProgram
    cards: tuple[ProgramCard, ...]
    connections: tuple[CardAliasEdge, ...]
    public_inputs: Mapping[str, int]
    public_outputs: Mapping[str, int]
    feedback: Mapping[str, str] = field(default_factory=dict)
    map_card_graph: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        regions = {
            int(card.region_index)
            for card in self.cards
            if card.region_index is not None
        }
        missing_regions = set(map(int, self.control.region_indices)) - regions
        if missing_regions:
            raise ValueError(
                "control card schedules absent numerical cards: "
                + ", ".join(map(str, sorted(missing_regions)))
            )
        producers: dict[int, set[str]] = {}
        for card in self.cards:
            for port in card.outputs:
                storage = int(
                    port.value_id
                    if port.storage_value_id is None
                    else port.storage_value_id
                )
                producers.setdefault(storage, set()).add(card.card_id)
        external = {
            int(port.storage_value_id or port.value_id)
            for card in self.cards if card.kind == "control"
            for port in card.inputs
        }
        unsatisfied = []
        for card in self.cards:
            if card.kind == "control":
                continue
            for port in card.inputs:
                storage = int(port.storage_value_id or port.value_id)
                if storage not in external and storage not in producers:
                    unsatisfied.append(f"{card.card_id}.{port.name}")
        if unsatisfied:
            raise ValueError(
                "card inputs lack an external or produced resident arena: "
                + ", ".join(sorted(unsatisfied))
            )
        unknown_feedback = (
            set(self.feedback) - set(self.public_inputs)
        ) | (set(self.feedback.values()) - set(self.public_outputs))
        if unknown_feedback:
            raise ValueError(
                "feedback names unknown public card ports: "
                + ", ".join(sorted(unknown_feedback))
            )

    def to_mapping(self) -> dict[str, Any]:
        self.validate()
        map_cards = list(self.map_card_graph.get("cards") or ())
        map_connections = list(
            self.map_card_graph.get("connections") or ()
        )
        return {
            "abi": "turing.parametric-card-program.v1",
            "name": self.name,
            "address_policy": {
                "arena": "outer-coordinator",
                "cache": "compiled-card",
                "execution": "control-read-head",
                "rebind": "every-dispatch",
                "inputs": "alias",
                "outputs": "alias",
                "feedback": "rotate-addresses",
            },
            "public_inputs": {
                str(name): int(value_id)
                for name, value_id in self.public_inputs.items()
            },
            "public_outputs": {
                str(name): int(value_id)
                for name, value_id in self.public_outputs.items()
            },
            "feedback": dict(self.feedback),
            "control": {
                "card": f"{self.name}::control",
                "region_indices": list(map(int, self.control.region_indices)),
                "python": render_control_program(
                    self.control, ControlTarget.PYTHON
                ),
                "recursion_regions": [
                    {
                        "region_id": int(region.region_id),
                        "kind": region.kind,
                        "lower_as": region.lower_as,
                        "feedback": [list(item) for item in region.feedback],
                    }
                    for region in self.control.recursion_regions
                ],
            },
            "cards": [
                *map_cards,
                *(card.to_mapping() for card in self.cards),
            ],
            "connections": [
                *map_connections,
                *(edge.to_mapping() for edge in self.connections),
            ],
        }


def _canonicalizer(aliases: Mapping[int, int]):
    redirects = {int(source): int(target) for source, target in aliases.items()}

    def canonical(value_id: int) -> int:
        current = int(value_id)
        seen = set()
        while current in redirects:
            if current in seen:
                raise ValueError("hierarchical value aliases contain a cycle")
            seen.add(current)
            current = redirects[current]
        return current

    return canonical


def build_parametric_card_program(
    compilation: Any,
    *,
    feedback: Mapping[str, str] | None = None,
    public_inputs: Mapping[str, int] | None = None,
    public_outputs: Mapping[str, int] | None = None,
) -> ParametricCardProgram:
    """Project one complete :class:`AOTCompilation` without flattening it."""

    control = compilation.shell_control_program
    if not isinstance(control, ControlProgram):
        raise ValueError("AOT compilation has no retained ControlProgram")
    region_programs = dict(compilation.region_programs or {})
    canonical = _canonicalizer(
        compilation.hierarchical_value_aliases or {}
    )
    names_by_id: dict[int, str] = {}
    for name, history in dict(compilation.identity_table or {}).items():
        for value_id in history:
            names_by_id.setdefault(int(value_id), str(name))
    names_by_id.update({
        int(value_id): str(name)
        for name, value_id in dict(
            compilation.public_input_value_ids or {}
        ).items()
    })
    names_by_id.update({
        int(value_id): str(name)
        for name, value_id in dict(
            compilation.public_output_value_ids or {}
        ).items()
    })

    def port(program: Any, value_id: int, role: str, name: str | None = None):
        metadata = (program.meta or {}).get(int(value_id))
        return CardPort(
            name=str(name or names_by_id.get(int(value_id), f"value_{value_id}")),
            value_id=int(value_id),
            role=role,
            dtype=str(getattr(metadata, "dtype", None) or "float64"),
            shape=tuple(map(int, getattr(metadata, "shape", ()) or ())),
            storage_value_id=canonical(int(value_id)),
        )

    public_inputs = dict(
        compilation.public_input_value_ids or {}
        if public_inputs is None else public_inputs
    )
    public_outputs = dict(
        compilation.public_output_value_ids or {}
        if public_outputs is None else public_outputs
    )
    numerical_cards = []
    for region_index in sorted(region_programs):
        program = region_programs[region_index]
        numerical_cards.append(ProgramCard(
            card_id=f"{compilation.entrypoint}::region::{int(region_index)}",
            kind="numerical-region",
            region_index=int(region_index),
            inputs=tuple(
                port(program, value_id, "input")
                for value_id in ordered_feed_ids(program)
            ),
            outputs=tuple(
                port(program, value_id, "output", output_name)
                for output_name, value_id in program.outputs.items()
            ),
            program=program,
        ))
    produced_storage_ids = {
        canonical(int(value_id))
        for program in region_programs.values()
        for value_id in program.outputs.values()
    }
    public_storage_ids = {
        canonical(int(value_id)) for value_id in public_inputs.values()
    }
    for program in region_programs.values():
        for value_id in ordered_feed_ids(program):
            storage = canonical(int(value_id))
            if storage in produced_storage_ids or storage in public_storage_ids:
                continue
            authored_name = names_by_id.get(int(value_id))
            if authored_name is None:
                authored_name = f"captured_value_{int(value_id)}"
            candidate = authored_name
            suffix = 1
            while candidate in public_inputs and int(public_inputs[candidate]) != int(value_id):
                suffix += 1
                candidate = f"{authored_name}_{suffix}"
            public_inputs[candidate] = int(value_id)
            public_storage_ids.add(storage)

    value_meta_program = next(iter(region_programs.values()), None)
    control_inputs = tuple(
        port(value_meta_program, value_id, "input", name)
        if value_meta_program is not None else CardPort(
            str(name), int(value_id), "input",
            storage_value_id=canonical(int(value_id)),
        )
        for name, value_id in public_inputs.items()
    )
    control_outputs = tuple(
        port(value_meta_program, value_id, "output", name)
        if value_meta_program is not None else CardPort(
            str(name), int(value_id), "output",
            storage_value_id=canonical(int(value_id)),
        )
        for name, value_id in public_outputs.items()
    )
    control_card = ProgramCard(
        card_id=f"{compilation.entrypoint}::control",
        kind="control",
        inputs=control_inputs,
        outputs=control_outputs,
        program=control,
    )
    cards = (control_card, *numerical_cards)

    producers: dict[int, list[tuple[str, str]]] = {}
    consumers: dict[int, list[tuple[str, str]]] = {}
    for card in cards:
        for card_port in card.outputs:
            producers.setdefault(
                int(card_port.storage_value_id or card_port.value_id), []
            ).append((card.card_id, card_port.name))
        for card_port in card.inputs:
            consumers.setdefault(
                int(card_port.storage_value_id or card_port.value_id), []
            ).append((card.card_id, card_port.name))
    edges = []
    for storage_value_id in sorted(set(producers) & set(consumers)):
        for source_card, source_port in producers[storage_value_id]:
            for target_card, target_port in consumers[storage_value_id]:
                if source_card == target_card:
                    continue
                edges.append(CardAliasEdge(
                    source_card,
                    source_port,
                    target_card,
                    target_port,
                    storage_value_id,
                ))

    result = ParametricCardProgram(
        name=str(compilation.entrypoint),
        control=control,
        cards=tuple(cards),
        connections=tuple(edges),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
        feedback=dict(feedback or {}),
        map_card_graph=build_card_graph(compilation.map_ir or {}),
    )
    result.validate()
    return result


def capture_ast_card_program(
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    *,
    python_bindings: Mapping[str, Any] | None = None,
    mutable_parameters: Sequence[str] = (),
    feedback: Mapping[str, str] | None = None,
    checkpoint: bool | str = False,
) -> ParametricCardProgram:
    """Capture AST/class/control/numerical regions as a parametric card set."""

    from ..common.tensors.accelerator_backends.aot_compile import compile_ast_aot

    compilation = compile_ast_aot(
        source,
        entrypoint,
        dict(feeds),
        backend="c",
        precompile_only=True,
        bake_mode="whole_program",
        python_bindings=dict(python_bindings or {}),
        mutable_parameters=tuple(mutable_parameters),
        checkpoint=checkpoint,
    )
    return build_parametric_card_program(compilation, feedback=feedback)


__all__ = [
    "CardAliasEdge",
    "CardPort",
    "ParametricCardProgram",
    "ProgramCard",
    "build_parametric_card_program",
    "capture_ast_card_program",
]
