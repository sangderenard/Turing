"""Lift one validated machine-tape lineage into effect-aware trace SSA.

This representation specializes the observed run. It preserves the original
binary instruction identity, state versions, external effects, and tape
dependencies, making it suitable for slicing and replay compilation. It does
not claim to be a whole-program symbolic decompilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .amd64_machine_semantics import PagedByteMemory
from .machine_execution import MachineExecutionState
from .machine_system_tape import MachineSystemTape


@dataclass(frozen=True, slots=True)
class MachineTraceSSAValue:
    resource: str
    version: int
    observed: str

    @property
    def identity(self) -> str:
        return f"{self.resource}@{self.version}"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.identity,
            "resource": self.resource,
            "version": self.version,
            "observed": self.observed,
        }


@dataclass(frozen=True, slots=True)
class MachineTraceSSAOperation:
    operation_id: str
    sequence: int
    event: str
    address: int | None
    instruction_token: str | None
    semantic_token: str | None
    encoded: str
    inputs: tuple[str, ...]
    output_inputs: tuple[tuple[str, tuple[str, ...]], ...]
    outputs: tuple[MachineTraceSSAValue, ...]
    effect_domains: tuple[str, ...]
    tape_dependencies: tuple[tuple[str, int], ...]
    pure: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.operation_id,
            "sequence": self.sequence,
            "event": self.event,
            "address": self.address,
            "instruction_token": self.instruction_token,
            "semantic_token": self.semantic_token,
            "encoded": self.encoded,
            "inputs": list(self.inputs),
            "output_inputs": {
                output: list(inputs) for output, inputs in self.output_inputs
            },
            "outputs": [item.to_mapping() for item in self.outputs],
            "effect_domains": list(self.effect_domains),
            "tape_dependencies": [
                {"kind": kind, "sequence": sequence}
                for kind, sequence in self.tape_dependencies
            ],
            "pure": self.pure,
        }


@dataclass(frozen=True, slots=True)
class MachineTraceSSAProgram:
    core: int
    source_sequences: tuple[int, ...]
    operations: tuple[MachineTraceSSAOperation, ...]
    final_values: Mapping[str, str]
    specialization: str = "observed-tape-lineage"
    reduction_witness: Mapping[str, Any] | None = None

    def to_mapping(self) -> dict[str, Any]:
        result = {
            "schema": "turing.machine-trace-ssa.v1",
            "specialization": self.specialization,
            "core": self.core,
            "source_sequences": list(self.source_sequences),
            "operations": [item.to_mapping() for item in self.operations],
            "final_values": dict(self.final_values),
        }
        if self.reduction_witness is not None:
            result["reduction_witness"] = dict(self.reduction_witness)
        return result

    def backward_slice(
        self,
        resources: Sequence[str],
        *,
        include_control: bool = True,
    ) -> "MachineTraceSSAProgram":
        """Retain operations needed to produce selected final resources."""

        producers = {
            output.identity: operation
            for operation in self.operations
            for output in operation.outputs
        }
        dependencies = {
            output: inputs
            for operation in self.operations
            for output, inputs in operation.output_inputs
        }
        pending = [
            self.final_values[resource]
            for resource in resources
            if resource in self.final_values
        ]
        selected: set[str] = set()
        visited: set[str] = set()
        while pending:
            value = pending.pop()
            if value in visited:
                continue
            visited.add(value)
            operation = producers.get(value)
            if operation is None:
                continue
            selected.add(operation.operation_id)
            pending.extend(
                item for item in dependencies.get(value, operation.inputs)
                if include_control or not item.startswith("control@")
            )
        operations = tuple(
            item for item in self.operations if item.operation_id in selected
        )
        produced = {
            output.identity
            for operation in operations
            for output in operation.outputs
        }
        finals = {
            resource: value for resource, value in self.final_values.items()
            if value in produced or resource in resources
        }
        removed = tuple(
            item.sequence for item in self.operations
            if item.operation_id not in selected
        )
        witness = MappingProxyType({
            "schema": "turing.machine-trace-reduction-witness.v1",
            "rewrite": "backward-slice",
            "seed_resources": tuple(str(item) for item in resources),
            "include_control": bool(include_control),
            "source_operation_count": len(self.operations),
            "retained_source_sequences": tuple(item.sequence for item in operations),
            "removed_source_sequences": removed,
        })
        return MachineTraceSSAProgram(
            self.core,
            tuple(item.sequence for item in operations),
            operations,
            MappingProxyType(finals),
            self.specialization,
            witness,
        )

    def reduction_summary(self, sliced: "MachineTraceSSAProgram") -> Mapping[str, int]:
        return MappingProxyType({
            "source_operations": len(self.operations),
            "retained_operations": len(sliced.operations),
            "removed_operations": len(self.operations) - len(sliced.operations),
            "retained_pure_operations": sum(item.pure for item in sliced.operations),
            "retained_effect_operations": sum(not item.pure for item in sliced.operations),
        })


InstructionLookup = Callable[[int], Any | None]


def _observed(value: Any) -> str:
    if isinstance(value, bytes):
        if len(value) <= 64:
            return "hex:" + value.hex()
        return f"sha256:{sha256(value).hexdigest()}:bytes={len(value)}"
    return str(value)


def _page_map(state: MachineExecutionState) -> Mapping[int, bytes]:
    memory = state.memory
    if isinstance(memory, PagedByteMemory):
        return memory.pages
    return {}


def _changed_resources(
    before: MachineExecutionState,
    after: MachineExecutionState,
) -> tuple[dict[str, Any], set[str]]:
    changed: dict[str, Any] = {}
    domains: set[str] = set()
    before_registers = before.register_contents()
    after_registers = after.register_contents()
    for name, value in after_registers.items():
        if before_registers.get(name) != value:
            changed[f"register.{name}"] = value
            domains.add("register")
    before_pages, after_pages = _page_map(before), _page_map(after)
    for page in sorted(set(before_pages) | set(after_pages)):
        if before_pages.get(page) != after_pages.get(page):
            changed[f"memory.page.{page:#x}"] = after_pages.get(page, b"")
            domains.add("memory")
    mappings = (
        ("system", before.system_state, after.system_state),
        ("environment", before.environment_state, after.environment_state),
        ("text", before.text_state, after.text_state),
        ("device", before.device_state, after.device_state),
    )
    for domain, left, right in mappings:
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                changed[f"{domain}.{key}"] = right.get(key, "<deleted>")
                domains.add(domain)
                if (
                    domain == "device" and str(key).startswith("pipe.")
                    or domain == "system" and str(key).startswith("windows.pipe.")
                ):
                    domains.add("pipe")
    if before.virtual_filesystem != after.virtual_filesystem:
        generation = -1 if after.virtual_filesystem is None else after.virtual_filesystem.generation
        changed["filesystem.state"] = generation
        domains.add("filesystem")
    if before.virtual_registry != after.virtual_registry:
        generation = -1 if after.virtual_registry is None else after.virtual_registry.generation
        changed["registry.state"] = generation
        domains.add("registry")
    if before.virtual_memory != after.virtual_memory:
        generation = -1 if after.virtual_memory is None else after.virtual_memory.generation
        changed["virtual_memory.state"] = generation
        domains.add("virtual_memory")
    if before.external_requests != after.external_requests:
        changed["external.pending"] = tuple(
            f"{item.reference.library}!{item.reference.symbol}"
            for item in after.external_requests
        )
        domains.add("external")
    for name in ("termination_requested", "halted", "exit_code"):
        if getattr(before, name) != getattr(after, name):
            changed[f"process.{name}"] = getattr(after, name)
            domains.add("process")
    changed["control"] = after.pc
    domains.add("control")
    return changed, domains


def lift_tape_lineage_to_trace_ssa(
    tape: MachineSystemTape,
    *,
    core: int = 0,
    sequence: int | None = None,
    instruction_lookup: InstructionLookup | None = None,
) -> MachineTraceSSAProgram:
    """Version every changed machine resource along one validated tape path."""

    lineage = tape.lineage_states(core, sequence=sequence)
    versions: dict[str, int] = {}
    current: dict[str, str] = {}
    operations: list[MachineTraceSSAOperation] = []
    for (parent_node, before), (node, after) in zip(lineage, lineage[1:]):
        changed, domains = _changed_resources(before, after)
        prior_current = dict(current)
        inputs = []
        outputs = []
        for resource, value in changed.items():
            prior = current.get(resource)
            if prior is not None:
                inputs.append(prior)
            version = versions.get(resource, 0) + 1
            versions[resource] = version
            output = MachineTraceSSAValue(resource, version, _observed(value))
            outputs.append(output)
            current[resource] = output.identity
        output_inputs = []
        for output in outputs:
            dependencies = []
            prior = prior_current.get(output.resource)
            if prior is not None:
                dependencies.append(prior)
            if output.resource != "control":
                control = prior_current.get("control")
                if control is not None:
                    dependencies.append(control)
            if output.resource.startswith("device.") and node.event == "external_completion":
                dependencies.append(prior_current.get("external.pending", "external.pending@0"))
                dependencies.append("memory.state@0")
            output_inputs.append((output.identity, tuple(dict.fromkeys(dependencies))))
        inputs = [
            dependency
            for _output, dependencies in output_inputs
            for dependency in dependencies
        ]
        instruction = None
        address = before.pc if node.event == "forward" else None
        if address is not None and instruction_lookup is not None:
            instruction = instruction_lookup(address)
        operation = MachineTraceSSAOperation(
            f"trace.{core}.{node.sequence}",
            node.sequence,
            node.event,
            address,
            None if instruction is None else instruction.token.name,
            None if instruction is None else instruction.semantic.name,
            "" if instruction is None else instruction.encoded.hex(),
            tuple(dict.fromkeys(inputs)),
            tuple(output_inputs),
            tuple(outputs),
            tuple(sorted(domains)),
            node.dependencies,
            node.event == "forward" and domains <= {"register", "control"},
        )
        operations.append(operation)
    return MachineTraceSSAProgram(
        core,
        tuple(node.sequence for node, _state in lineage),
        tuple(operations),
        MappingProxyType(dict(current)),
    )


def iter_path_state_head_trace_ssa_operations(
    store,
    head_id: str | int,
    *,
    core: int = 0,
    final_values: dict[str, str] | None = None,
    source_positions: list[int] | None = None,
):
    """Stream SSA operations from an exact path while retaining bounded chunks."""

    lineage = iter(store.iter_states(head_id, include_metadata=True))
    try:
        previous = next(lineage)
    except StopIteration:
        if final_values is not None:
            final_values.clear()
        return
    if source_positions is not None:
        source_positions.append(int(previous[0]))
    versions: dict[str, int] = {}
    current: dict[str, str] = {}
    for active in lineage:
        before_position, before, _before_record = previous
        position, after, record = active
        if source_positions is not None:
            source_positions.append(int(position))
        changed, domains = _changed_resources(before, after)
        metadata = dict(record.get("metadata", {}))
        event = str(record.get("event", "forward"))
        prior_current = dict(current)
        outputs = []
        output_inputs = []
        for resource, value in changed.items():
            version = versions.get(resource, 0) + 1
            versions[resource] = version
            output = MachineTraceSSAValue(resource, version, _observed(value))
            outputs.append(output)
            dependencies = []
            prior = prior_current.get(resource)
            if prior is not None:
                dependencies.append(prior)
            if resource != "control":
                control = prior_current.get("control")
                if control is not None:
                    dependencies.append(control)
            output_inputs.append((output.identity, tuple(dict.fromkeys(dependencies))))
            current[resource] = output.identity
        inputs = tuple(dict.fromkeys(
            dependency
            for _output, dependencies in output_inputs
            for dependency in dependencies
        ))
        address_value = metadata.get("instruction_address")
        address = None if address_value is None else int(address_value)
        yield MachineTraceSSAOperation(
            f"path.{head_id}.{position}",
            int(position), event, address,
            metadata.get("instruction_token"), metadata.get("semantic"),
            str(metadata.get("instruction_bytes", "")),
            inputs, tuple(output_inputs), tuple(outputs),
            tuple(sorted(domains)),
            (("path_parent", int(before_position)),),
            event == "forward" and domains <= {"register", "control"},
        )
        previous = active
    if final_values is not None:
        final_values.clear()
        final_values.update(current)


def lift_path_state_head_to_trace_ssa(
    store,
    head_id: str | int,
    *,
    core: int = 0,
) -> MachineTraceSSAProgram:
    """Materialize one exact path trace when an in-memory SSA view is wanted."""

    final_values: dict[str, str] = {}
    source_positions: list[int] = []
    operations = tuple(iter_path_state_head_trace_ssa_operations(
        store, head_id, core=core,
        final_values=final_values, source_positions=source_positions,
    ))
    return MachineTraceSSAProgram(
        int(core),
        tuple(source_positions), operations, MappingProxyType(dict(final_values)),
        "observed-possible-world-path",
    )


def segment_path_state_head_to_trace_ssa(
    path_store,
    head_id: str | int,
    trace_store,
    *,
    core: int = 0,
    parent_head_id: str | int | None = None,
    fork_position: int | None = None,
    operations_per_segment: int = 256,
):
    """Stream an exact path directly into its content-addressed SSA suffix."""

    final_values: dict[str, str] = {}
    operations = iter_path_state_head_trace_ssa_operations(
        path_store, head_id, core=core, final_values=final_values,
    )
    head = path_store.heads[str(head_id)]
    return trace_store.add_operation_stream(
        head_id,
        (operation.to_mapping() for operation in operations),
        core=core,
        specialization="observed-possible-world-path",
        final_values=final_values,
        parent_head_id=parent_head_id,
        fork_sequence=fork_position,
        constraints=head.constraints,
        operations_per_segment=operations_per_segment,
    )


__all__ = [
    "MachineTraceSSAOperation", "MachineTraceSSAProgram", "MachineTraceSSAValue",
    "iter_path_state_head_trace_ssa_operations",
    "lift_path_state_head_to_trace_ssa", "lift_tape_lineage_to_trace_ssa",
    "segment_path_state_head_to_trace_ssa",
]
