"""Recursive reduction contracts from ProcessGraph toward physical tape.

The reduction bridge does not own source or target semantics.  It catalogs
well-founded graph rewrites, records their parent/child morphisms in the
append-only :mod:`evolution_metagraph`, and projects terminal Turing graphs onto
a vector of physical tape costs.  Text operation names are boundary spellings;
stable integer tokens are the identities carried by rules and graph records.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
import math
from typing import Any, Callable, Hashable, Iterable, Mapping

import networkx as nx

from .bitops_process_graph import expand_bitops_process_graph
from .evolution_metagraph import (
    EvolutionComponentRef,
    EvolutionGraphRef,
    EvolutionMetaGraph,
    TokenPathAtlas,
)
from .machine_turing_graph import MachineTuringGraph, TuringOperatorToken
from .machine_reference_vocabulary import X86Register
from .object_process_bridge import (
    RaisedObjectMethod,
    raise_object_method_to_process_graph,
)
from ..hardware.analog_spec import BiosHeader, InstructionWord, Opcode
from ..hardware.constants import (
    BIT_FRAME_MS,
    LANES,
    MOTOR_RAMP_MS,
    NOISE_SOURCES,
    REGISTERS,
)
from ..turing_machine.tape_map import TapeMap
from ..transmogrifier.graph.graph_express2 import ProcessGraph


class ReductionLayer(IntEnum):
    """Ordinal decreases as a description approaches physical execution."""

    PHYSICAL = 0
    TAPE = 1
    TURING = 2
    BITOPS = 3
    PROCESS = 4
    OBJECT = 5


class ReductionNamespace(IntEnum):
    BITOPS = 0
    TURING = 1
    TAPE = 2
    RULE = 3
    PHYSICAL = 4
    OBJECT = 5
    PROCESS = 6


class BitOpsReductionToken(IntEnum):
    BIT_AND = 0
    BIT_OR = 1
    BIT_XOR = 2
    INVERT = 3
    ADD = 4
    SUBTRACT = 5
    MULTIPLY = 6


_BITOPS_SPELLINGS = {
    "bitand": BitOpsReductionToken.BIT_AND,
    "bitor": BitOpsReductionToken.BIT_OR,
    "bitxor": BitOpsReductionToken.BIT_XOR,
    "invert": BitOpsReductionToken.INVERT,
    "add": BitOpsReductionToken.ADD,
    "sub": BitOpsReductionToken.SUBTRACT,
    "mul": BitOpsReductionToken.MULTIPLY,
}

_TURING_SPELLINGS = {
    "nand": TuringOperatorToken.NAND,
    "sigma_L": TuringOperatorToken.MOTION_LEFT,
    "sigma_R": TuringOperatorToken.MOTION_RIGHT,
    "concat": TuringOperatorToken.CONCAT,
    "slice": TuringOperatorToken.SLICE,
    "mu": TuringOperatorToken.SELECT,
    "length": TuringOperatorToken.LENGTH,
    "zeros": TuringOperatorToken.ZEROS,
}

_TAPE_BINDINGS = {
    "nand": Opcode.NAND,
    "sigma_L": Opcode.SIGL,
    "sigma_R": Opcode.SIGR,
    "concat": Opcode.CONCAT,
    "slice": Opcode.SLICE,
    "mu": Opcode.MU,
    "length": Opcode.LENGTH,
    "zeros": Opcode.ZEROS,
}


@dataclass(frozen=True, slots=True, order=True)
class ReductionRank:
    """Lexicographic termination measure for one reduction level."""

    layer: ReductionLayer
    structural_depth: int = 0

    def __post_init__(self) -> None:
        if self.structural_depth < 0:
            raise ValueError("structural_depth cannot be negative")


@dataclass(frozen=True, slots=True)
class ReductionRule:
    """Self-description of one graph rewrite family."""

    token_id: int
    source_token_id: int
    source_spelling: str
    source_rank: ReductionRank
    target_rank: ReductionRank
    input_roles: tuple[str, ...]
    output_roles: tuple[str, ...] = ("result",)
    target_token_ids: tuple[int, ...] = ()
    reducer: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.source_spelling:
            raise ValueError("reduction rule requires a diagnostic spelling")
        if not self.target_rank < self.source_rank:
            raise ValueError("reduction rule must strictly decrease its rank")
        if len(set(self.input_roles)) != len(self.input_roles):
            raise ValueError("input roles must be unique")


class ReductionCatalog:
    """Numeric-token registry for recursive reduction rules."""

    def __init__(self, *, atlas: TokenPathAtlas | None = None) -> None:
        self.atlas = atlas or TokenPathAtlas()
        self._rules: dict[tuple[ReductionLayer, int], ReductionRule] = {}
        self._spellings: dict[tuple[ReductionLayer, str], int] = {}

    def register(self, rule: ReductionRule) -> None:
        key = (rule.source_rank.layer, int(rule.source_token_id))
        spelling_key = (rule.source_rank.layer, rule.source_spelling)
        if key in self._rules or spelling_key in self._spellings:
            raise ValueError(f"duplicate reduction rule for {rule.source_spelling!r}")
        self._rules[key] = rule
        self._spellings[spelling_key] = int(rule.source_token_id)

    def resolve_token(
        self, layer: ReductionLayer, token_id: int,
    ) -> ReductionRule | None:
        return self._rules.get((layer, int(token_id)))

    def resolve_spelling(
        self, layer: ReductionLayer, spelling: str,
    ) -> ReductionRule | None:
        token = self._spellings.get((layer, str(spelling)))
        return None if token is None else self.resolve_token(layer, token)

    def rules(self) -> tuple[ReductionRule, ...]:
        return tuple(sorted(self._rules.values(), key=lambda rule: rule.token_id))


def bitops_turing_reduction_catalog(
    *, atlas: TokenPathAtlas | None = None,
) -> ReductionCatalog:
    """Describe the live BitOps recipes without reimplementing their algebra."""

    catalog = ReductionCatalog(atlas=atlas)
    terminal_tokens = tuple(
        catalog.atlas.consume((
            int(ReductionNamespace.TURING), int(token),
        ))
        for token in _TURING_SPELLINGS.values()
    )
    arities = {
        "invert": ("value",),
        "bitand": ("left", "right"),
        "bitor": ("left", "right"),
        "bitxor": ("left", "right"),
        "add": ("left", "right"),
        "sub": ("left", "right"),
        "mul": ("left", "right"),
    }
    for spelling, token in _BITOPS_SPELLINGS.items():
        source_token = catalog.atlas.consume((
            int(ReductionNamespace.BITOPS), int(token),
        ))
        rule_token = catalog.atlas.consume((
            int(ReductionNamespace.RULE),
            int(ReductionNamespace.BITOPS),
            int(token),
            int(ReductionNamespace.TURING),
        ))
        catalog.register(ReductionRule(
            token_id=rule_token,
            source_token_id=source_token,
            source_spelling=spelling,
            source_rank=ReductionRank(ReductionLayer.BITOPS, 0),
            target_rank=ReductionRank(ReductionLayer.TURING, 0),
            input_roles=arities[spelling],
            target_token_ids=terminal_tokens,
            reducer="BitOpsTranslator.apply_bits",
            description=(
                f"Expand {spelling} through the instrumented Turing carrier"
            ),
        ))
    return catalog


@dataclass(frozen=True, slots=True)
class TapeCostVector:
    """Composable physical counts and conservative, uncalibrated energy units."""

    tape_distance_frames: int = 0
    seeks: int = 0
    read_frames: int = 0
    write_frames: int = 0
    operator_events: int = 0
    bit_transition_upper_bound: int = 0
    storage_frames: int = 0
    latency_seconds: float = 0.0
    mechanical_work_units: float = 0.0
    signal_energy_frame_units: float = 0.0
    noise_exposure_frame_sources: int = 0
    peak_parallel_lanes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "tape_distance_frames", "seeks", "read_frames", "write_frames",
            "operator_events", "bit_transition_upper_bound", "storage_frames",
            "noise_exposure_frame_sources", "peak_parallel_lanes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "latency_seconds", "mechanical_work_units",
            "signal_energy_frame_units",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def serial(self, other: "TapeCostVector") -> "TapeCostVector":
        return TapeCostVector(
            tape_distance_frames=(
                self.tape_distance_frames + other.tape_distance_frames
            ),
            seeks=self.seeks + other.seeks,
            read_frames=self.read_frames + other.read_frames,
            write_frames=self.write_frames + other.write_frames,
            operator_events=self.operator_events + other.operator_events,
            bit_transition_upper_bound=(
                self.bit_transition_upper_bound
                + other.bit_transition_upper_bound
            ),
            storage_frames=max(self.storage_frames, other.storage_frames),
            latency_seconds=self.latency_seconds + other.latency_seconds,
            mechanical_work_units=(
                self.mechanical_work_units + other.mechanical_work_units
            ),
            signal_energy_frame_units=(
                self.signal_energy_frame_units
                + other.signal_energy_frame_units
            ),
            noise_exposure_frame_sources=(
                self.noise_exposure_frame_sources
                + other.noise_exposure_frame_sources
            ),
            peak_parallel_lanes=max(
                self.peak_parallel_lanes, other.peak_parallel_lanes,
            ),
        )

    def parallel(self, other: "TapeCostVector") -> "TapeCostVector":
        combined = self.serial(other)
        return TapeCostVector(
            **{
                name: getattr(combined, name)
                for name in combined.__dataclass_fields__
                if name != "latency_seconds"
            },
            latency_seconds=max(self.latency_seconds, other.latency_seconds),
        )


@dataclass(frozen=True, slots=True)
class TapeReliabilityEstimate:
    """Distribution-free union bound over exposed physical frame sources."""

    exposed_frame_sources: int
    per_source_error_probability_upper_bound: float
    failure_probability_upper_bound: float
    success_probability_lower_bound: float


@dataclass(frozen=True, slots=True)
class GraphConcurrencyProfile:
    """Dependency parallelism retained before serial tape scheduling."""

    logical_nodes: int
    operator_nodes: int
    critical_path_operator_events: int
    maximum_parallel_operator_events: int
    operator_width_by_level: tuple[int, ...]
    operator_levels: Mapping[Hashable, int]
    operator_frontiers: tuple[tuple[Hashable, ...], ...]
    average_available_parallelism: float
    physical_parallel_lanes: int
    serial_to_critical_path_ratio: float

    def __post_init__(self) -> None:
        for name in (
            "logical_nodes",
            "operator_nodes",
            "critical_path_operator_events",
            "maximum_parallel_operator_events",
            "physical_parallel_lanes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.physical_parallel_lanes == 0:
            raise ValueError("physical_parallel_lanes must be positive")
        if any(width < 0 for width in self.operator_width_by_level):
            raise ValueError("operator level widths cannot be negative")
        if tuple(len(frontier) for frontier in self.operator_frontiers) != (
            self.operator_width_by_level
        ):
            raise ValueError("operator frontiers do not match level widths")
        if len(self.operator_levels) != self.operator_nodes:
            raise ValueError("operator_levels does not cover every operator")
        for name in (
            "average_available_parallelism",
            "serial_to_critical_path_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def analyze_graph_concurrency(
    graph: nx.MultiDiGraph | nx.DiGraph,
    *,
    output_nodes: Iterable[Hashable] | None = None,
    physical_parallel_lanes: int = 1,
) -> GraphConcurrencyProfile:
    """Measure available DAG work independently of the chosen scheduler."""

    if physical_parallel_lanes <= 0:
        raise ValueError("physical_parallel_lanes must be positive")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("concurrency analysis requires an acyclic graph")
    if output_nodes is None:
        live = set(graph)
    else:
        outputs = tuple(output_nodes)
        if any(node not in graph for node in outputs):
            raise ValueError("concurrency output is absent from the graph")
        live = set(outputs)
        for output in outputs:
            live.update(nx.ancestors(graph, output))
    transparent = {"input", "const", "constant", "return", "parameter"}
    depth: dict[Hashable, int] = {}
    widths: dict[int, int] = {}
    operator_nodes = 0
    for node in nx.topological_sort(graph):
        if node not in live:
            continue
        parents = tuple(
            parent for parent in _ordered_parents(graph, node) if parent in live
        )
        parent_depth = max((depth[parent] for parent in parents), default=0)
        payload = graph.nodes[node]
        op = str(payload.get("op") or payload.get("label") or "")
        is_operator = op not in transparent
        node_depth = parent_depth + (1 if is_operator else 0)
        depth[node] = node_depth
        if is_operator:
            operator_nodes += 1
            widths[node_depth] = widths.get(node_depth, 0) + 1
    critical = max(widths, default=0)
    width_by_level = tuple(widths.get(level, 0) for level in range(1, critical + 1))
    operator_levels = {
        node: node_depth
        for node, node_depth in depth.items()
        if str(
            graph.nodes[node].get("op")
            or graph.nodes[node].get("label")
            or ""
        ) not in transparent
    }
    frontiers = tuple(
        tuple(sorted(
            (
                node for node, node_depth in operator_levels.items()
                if node_depth == level
            ),
            key=repr,
        ))
        for level in range(1, critical + 1)
    )
    maximum = max(width_by_level, default=0)
    average = operator_nodes / critical if critical else 0.0
    return GraphConcurrencyProfile(
        logical_nodes=len(live),
        operator_nodes=operator_nodes,
        critical_path_operator_events=critical,
        maximum_parallel_operator_events=maximum,
        operator_width_by_level=width_by_level,
        operator_levels=operator_levels,
        operator_frontiers=frontiers,
        average_available_parallelism=average,
        physical_parallel_lanes=int(physical_parallel_lanes),
        serial_to_critical_path_ratio=(
            operator_nodes / critical if critical else 0.0
        ),
    )


def estimate_tape_reliability(
    cost: TapeCostVector,
    *,
    per_source_error_probability_upper_bound: float,
) -> TapeReliabilityEstimate:
    """Conservatively bound failure without assuming independent noise."""

    probability = float(per_source_error_probability_upper_bound)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("per-source error probability must be between zero and one")
    exposed = int(cost.noise_exposure_frame_sources)
    failure = min(1.0, exposed * probability)
    return TapeReliabilityEstimate(
        exposed,
        probability,
        failure,
        max(0.0, 1.0 - failure),
    )


@dataclass(frozen=True, slots=True)
class TapePlacement:
    offsets: Mapping[Hashable, int]
    extents: Mapping[Hashable, int]
    total_frames: int


@dataclass(frozen=True, slots=True)
class TapeFeasibilityReport:
    placement: TapePlacement
    cost: TapeCostVector
    opcodes: Mapping[Hashable, int]
    instruction_nodes: tuple[Hashable, ...]
    initialized_data_nodes: tuple[Hashable, ...]
    structural_nodes: tuple[Hashable, ...]
    shortfalls: tuple[tuple[Hashable, str], ...]

    @property
    def complete(self) -> bool:
        return not self.shortfalls


@dataclass(frozen=True, slots=True)
class TerminalTapeProgram:
    """Executable NAND-terminal tape image with graph/value correspondence."""

    tape_map: TapeMap
    instructions: tuple[InstructionWord, ...]
    instruction_frames: tuple[tuple[int, ...], ...]
    instruction_sources: tuple[Hashable | None, ...]
    initial_register_values: Mapping[int, int]
    initial_spill_values: Mapping[int, int]
    spill_slots: Mapping[Hashable, int]
    node_registers: Mapping[Hashable, int]
    output_registers: Mapping[Hashable, int]
    output_spill_slots: Mapping[Hashable, int]
    bit_width: int
    storage_mode: str


@dataclass(frozen=True, slots=True)
class ScalarizedTuringGraph:
    """Structural Turing vectors reduced to scalar NAND/data topology."""

    graph: nx.MultiDiGraph
    output_bits: Mapping[Hashable, tuple[Hashable, ...]]
    input_bits: Mapping[Hashable, tuple[Hashable, int]]


@dataclass(frozen=True, slots=True)
class TapeExecutionEvent:
    instruction_index: int
    source_node: Hashable | None
    opcode: int
    dest: int
    reg_a: int
    reg_b: int
    param: int
    audio_start: int
    audio_end: int
    head_start_inches: float
    head_end_inches: float
    cost: TapeCostVector


@dataclass(frozen=True, slots=True)
class TapeExecutionWitness:
    outputs: Mapping[Hashable, int]
    events: tuple[TapeExecutionEvent, ...]
    audio_samples: int
    halted: bool
    setup_cost: TapeCostVector
    execution_cost: TapeCostVector
    observation_cost: TapeCostVector
    total_cost: TapeCostVector
    reliability: TapeReliabilityEstimate


@dataclass(frozen=True, slots=True)
class MachineTapeExecution:
    """Machine-derived Turing graph executed with ownership provenance."""

    raised: MachineTuringGraph
    program: TerminalTapeProgram
    witness: TapeExecutionWitness
    provenance: nx.MultiDiGraph

    def physical_descendants(self, instruction_address: int) -> tuple[int, ...]:
        source = ("machine", int(instruction_address))
        if source not in self.provenance:
            raise KeyError(f"unknown machine instruction {instruction_address:#x}")
        frontier = [source]
        visited = {source}
        physical: list[int] = []
        while frontier:
            parent = frontier.pop()
            for _left, child, edge in self.provenance.out_edges(parent, data=True):
                if edge.get("relation") != "ownership" or child in visited:
                    continue
                visited.add(child)
                frontier.append(child)
                if child[0] == "physical":
                    physical.append(int(child[1]))
        return tuple(sorted(physical))

    def cost_for_instruction(self, instruction_address: int) -> TapeCostVector:
        total = TapeCostVector()
        by_index = {
            event.instruction_index: event.cost for event in self.witness.events
        }
        for index in self.physical_descendants(instruction_address):
            total = total.serial(by_index[index])
        return total

    def reliability_for_instruction(
        self,
        instruction_address: int,
        *,
        per_source_error_probability_upper_bound: float,
    ) -> TapeReliabilityEstimate:
        return estimate_tape_reliability(
            self.cost_for_instruction(instruction_address),
            per_source_error_probability_upper_bound=(
                per_source_error_probability_upper_bound
            ),
        )


@dataclass(frozen=True, slots=True)
class ScalarMachineTapeAssembly:
    """General structural machine lowering ready for scalar tape execution."""

    raised: MachineTuringGraph
    scalarized: ScalarizedTuringGraph
    program: TerminalTapeProgram
    output_register: X86Register
    output_node: Hashable
    output_bits: tuple[Hashable, ...]
    input_bit_values: Mapping[Hashable, int]
    provenance: nx.MultiDiGraph

    def pack_output(self, bit_values: Mapping[Hashable, int]) -> int:
        value = 0
        for bit in self.output_bits:
            try:
                scalar = int(bit_values[bit])
            except KeyError as error:
                raise ValueError(f"missing scalar output bit {bit!r}") from error
            if scalar not in (0, 1):
                raise ValueError(f"scalar output bit {bit!r} is not Boolean")
            value = (value << 1) | scalar
        return value

    def tape_descendants(self, instruction_address: int) -> tuple[int, ...]:
        source = ("machine", int(instruction_address))
        if source not in self.provenance:
            raise KeyError(f"unknown machine instruction {instruction_address:#x}")
        descendants = nx.descendants(self.provenance, source)
        return tuple(sorted(
            int(node[1]) for node in descendants if node[0] == "tape"
        ))

    @property
    def spill_slot_count(self) -> int:
        return max(self.program.spill_slots.values(), default=-1) + 1

    @property
    def opcode_counts(self) -> Mapping[int, int]:
        counts: dict[int, int] = {}
        for instruction in self.program.instructions:
            token = int(instruction.opcode.value)
            counts[token] = counts.get(token, 0) + 1
        return counts

    @property
    def execution_cost_estimate(self) -> TapeCostVector:
        return estimate_terminal_tape_execution_cost(self.program)

    @property
    def execution_event_cost_estimates(self) -> tuple[TapeCostVector, ...]:
        return estimate_terminal_tape_execution_event_costs(self.program)

    def cost_for_instruction(self, instruction_address: int) -> TapeCostVector:
        total = TapeCostVector()
        events = self.execution_event_cost_estimates
        for index in self.tape_descendants(instruction_address):
            total = total.serial(events[index])
        return total

    def reliability_for_instruction(
        self,
        instruction_address: int,
        *,
        per_source_error_probability_upper_bound: float,
    ) -> TapeReliabilityEstimate:
        return estimate_tape_reliability(
            self.cost_for_instruction(instruction_address),
            per_source_error_probability_upper_bound=(
                per_source_error_probability_upper_bound
            ),
        )

    def estimate_reliability(
        self,
        per_source_error_probability_upper_bound: float,
    ) -> TapeReliabilityEstimate:
        return estimate_tape_reliability(
            self.execution_cost_estimate,
            per_source_error_probability_upper_bound=(
                per_source_error_probability_upper_bound
            ),
        )

    @property
    def concurrency_profile(self) -> GraphConcurrencyProfile:
        return analyze_graph_concurrency(
            self.scalarized.graph,
            output_nodes=self.output_bits,
            physical_parallel_lanes=max(
                self.execution_cost_estimate.peak_parallel_lanes, 1,
            ),
        )


def _node_extent(payload: Mapping[str, Any], default_width: int) -> int:
    metadata = payload.get("metadata") or {}
    tensor = payload.get("tensor") or {}
    result_length = metadata.get("result_length")
    if isinstance(result_length, int) and result_length > 0:
        return result_length
    shape = tensor.get("shape") or ()
    if shape and isinstance(shape[0], int) and shape[0] > 0:
        return int(shape[0])
    return max(int(default_width), 1)


def estimate_turing_tape_feasibility(
    graph: nx.MultiDiGraph | nx.DiGraph,
    *,
    bit_width: int | None = None,
    placement: TapePlacement | None = None,
    bits_per_inch: int = 800,
    seek_speed_ips: float = 30.0,
) -> TapeFeasibilityReport:
    """Place a terminal graph linearly and accumulate physical cost counts."""

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("tape feasibility requires an acyclic terminal graph")
    if bits_per_inch <= 0 or seek_speed_ips <= 0:
        raise ValueError("physical tape geometry must be positive")
    width = int(bit_width or graph.graph.get("bit_width", 1))
    if placement is None:
        offsets: dict[Hashable, int] = {}
        extents: dict[Hashable, int] = {}
        cursor = 0
        for node in nx.topological_sort(graph):
            extent = _node_extent(graph.nodes[node], width)
            offsets[node] = cursor
            extents[node] = extent
            cursor += extent
        active_placement = TapePlacement(offsets, extents, cursor)
    else:
        if set(placement.offsets) != set(graph):
            raise ValueError("placement offsets must cover every graph node")
        if set(placement.extents) != set(graph):
            raise ValueError("placement extents must cover every graph node")
        if any(int(value) < 0 for value in placement.offsets.values()):
            raise ValueError("placement offsets cannot be negative")
        if any(int(value) <= 0 for value in placement.extents.values()):
            raise ValueError("placement extents must be positive")
        required = max(
            int(placement.offsets[node]) + int(placement.extents[node])
            for node in graph
        ) if graph else 0
        if placement.total_frames < required:
            raise ValueError("placement total_frames does not cover its extents")
        active_placement = placement
    offsets = active_placement.offsets
    extents = active_placement.extents

    head = 0
    distance = seeks = reads = writes = operator_events = 0
    initialized: list[Hashable] = []
    structural: list[Hashable] = []
    instructions: list[Hashable] = []
    opcodes: dict[Hashable, int] = {}
    shortfalls: list[tuple[Hashable, str]] = []
    for node in nx.topological_sort(graph):
        payload = graph.nodes[node]
        op = str(payload.get("op") or payload.get("label") or "")
        metadata = payload.get("metadata") or {}
        if op in {"input", "const", "constant"}:
            initialized.append(node)
            continue
        if op in {"return"}:
            structural.append(node)
            continue
        if op not in _TAPE_BINDINGS:
            shortfalls.append((node, op or "<missing-op>"))
            continue
        instructions.append(node)
        opcodes[node] = int(_TAPE_BINDINGS[op].value)
        operator_events += 1
        incoming = sorted(
            graph.in_edges(node, data=True),
            key=lambda edge: int(edge[2].get("arg_pos", 0)),
        )
        for source, _target, _edge in incoming:
            target = offsets[source]
            delta = abs(target - head)
            if delta:
                seeks += 1
                distance += delta
            head = target + extents[source]
            reads += extents[source]
        target = offsets[node]
        delta = abs(target - head)
        if delta:
            seeks += 1
            distance += delta
        head = target + extents[node]
        writes += extents[node]

    traversed = distance + reads + writes
    frame_seconds = BIT_FRAME_MS / 1000.0
    seek_seconds = distance / bits_per_inch / seek_speed_ips
    ramp_frame_units = seeks * (MOTOR_RAMP_MS / BIT_FRAME_MS)
    cost = TapeCostVector(
        tape_distance_frames=distance,
        seeks=seeks,
        read_frames=reads,
        write_frames=writes,
        operator_events=operator_events,
        bit_transition_upper_bound=writes * max(width, 1),
        storage_frames=active_placement.total_frames,
        latency_seconds=seek_seconds + (reads + writes) * frame_seconds,
        mechanical_work_units=distance + ramp_frame_units,
        signal_energy_frame_units=reads + writes + operator_events,
        noise_exposure_frame_sources=traversed * NOISE_SOURCES,
        peak_parallel_lanes=min(max(width, 1), LANES),
    )
    return TapeFeasibilityReport(
        active_placement,
        cost,
        opcodes,
        tuple(instructions),
        tuple(initialized),
        tuple(structural),
        tuple(shortfalls),
    )


def _ordered_parents(
    graph: nx.MultiDiGraph | nx.DiGraph,
    node: Hashable,
) -> tuple[Hashable, ...]:
    payload = graph.nodes[node]
    declared = payload.get("parents")
    if declared:
        return tuple(parent for parent, _role in declared)
    if graph.is_multigraph():
        incoming = graph.in_edges(node, keys=True, data=True)
        return tuple(
            source for source, _target, _key, _edge in sorted(
                incoming,
                key=lambda item: int(item[3].get("arg_pos", 0)),
            )
        )
    incoming = graph.in_edges(node, data=True)
    return tuple(
        source for source, _target, edge in sorted(
            incoming,
            key=lambda item: int(item[2].get("arg_pos", 0)),
        )
    )


def _terminal_output_nodes(
    graph: nx.MultiDiGraph | nx.DiGraph,
    output_nodes: Iterable[Hashable] | None,
) -> tuple[Hashable, ...]:
    if output_nodes is not None:
        return tuple(output_nodes)
    return tuple(
        parent
        for node, payload in graph.nodes(data=True)
        if str(payload.get("op") or payload.get("label") or "") == "return"
        for parent in _ordered_parents(graph, node)
    )


def _requires_scalarization(
    graph: nx.MultiDiGraph | nx.DiGraph,
    output_nodes: Iterable[Hashable] | None = None,
) -> bool:
    outputs = _terminal_output_nodes(graph, output_nodes)
    live = set(outputs)
    for output in outputs:
        if output in graph:
            live.update(nx.ancestors(graph, output))
    terminal = {"input", "const", "constant", "nand"}
    return any(
        str(
            graph.nodes[node].get("op")
            or graph.nodes[node].get("label")
            or ""
        ) not in terminal
        for node in live
    )


def _initialized_terminal_value(
    graph: nx.MultiDiGraph | nx.DiGraph,
    node: Hashable,
    input_values: Mapping[Hashable, int],
) -> int:
    payload = graph.nodes[node]
    metadata = payload.get("metadata") or {}
    if node in input_values:
        return int(input_values[node])
    if metadata.get("kind") == "constant" and "value" in metadata:
        return int(metadata["value"])
    if payload.get("constant") is not None:
        return int(payload["constant"])
    raise ValueError(f"missing initialized value for terminal input {node!r}")


def scalarize_turing_operator_graph(
    graph: nx.MultiDiGraph | nx.DiGraph,
    *,
    output_nodes: Iterable[Hashable],
) -> ScalarizedTuringGraph:
    """Eliminate vector structure and SELECT into scalar NAND topology.

    This is a vocabulary lowering rather than an arithmetic recipe. CONCAT,
    SLICE, motions, LENGTH, and ZEROS alter compile-time carrier topology; MU
    becomes its general four-NAND Boolean selector. Input words become ordered
    scalar leaves so the tape layer remains free to choose a physical packing.
    """

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Turing scalarization requires an acyclic graph")
    requested = tuple(output_nodes)
    if not requested or any(node not in graph for node in requested):
        raise ValueError("scalarization requires present output nodes")
    live = set(requested)
    for output in requested:
        live.update(nx.ancestors(graph, output))

    scalar = nx.MultiDiGraph(
        kind="scalar-turing-nand",
        bit_width=1,
        source_word_width=int(graph.graph.get("bit_width", 0)),
    )
    vectors: dict[Hashable, tuple[Hashable, ...]] = {}
    scalar_values: dict[Hashable, int] = {}
    input_bits: dict[Hashable, tuple[Hashable, int]] = {}
    scalar_cache: dict[tuple[Any, ...], Hashable] = {}
    constant_values: dict[Hashable, int] = {}
    next_node = 0

    def new_node(
        op: str,
        source: Hashable,
        *,
        bit_index: int | None = None,
        value: int | None = None,
        parents: tuple[Hashable, ...] = (),
    ) -> Hashable:
        nonlocal next_node
        if op == "nand":
            left, right = parents
            left_constant = constant_values.get(left)
            right_constant = constant_values.get(right)
            if left_constant == 0 or right_constant == 0:
                return new_node("constant", source, value=1)
            if left_constant == 1 and right_constant == 1:
                return new_node("constant", source, value=0)
            if left_constant == 1:
                parents = (right, right)
            elif right_constant == 1:
                parents = (left, left)
        cache_key: tuple[Any, ...] | None = None
        if op == "constant" and value is not None:
            cache_key = ("constant", int(value) & 1)
        elif op == "nand":
            cache_key = ("nand", *sorted(parents))
        if cache_key is not None and cache_key in scalar_cache:
            existing = scalar_cache[cache_key]
            existing_metadata = scalar.nodes[existing]["metadata"]
            sources = set(existing_metadata.get("source_turing_nodes", ()))
            sources.add(source)
            existing_metadata["source_turing_nodes"] = tuple(
                sorted(sources, key=repr)
            )
            return existing
        node = next_node
        next_node += 1
        source_metadata = dict(graph.nodes[source].get("metadata") or {})
        metadata = {
            "result_length": 1,
            "source_turing_node": source,
            "source_turing_nodes": (source,),
            **{
                key: source_metadata[key]
                for key in (
                    "machine_instruction_address",
                    "machine_semantic_token",
                    "machine_mnemonic",
                )
                if key in source_metadata
            },
        }
        if bit_index is not None:
            metadata["bit_index"] = int(bit_index)
        if value is not None:
            metadata.update(kind="constant", value=int(value) & 1)
        scalar.add_node(
            node,
            op=op,
            token_id=(
                int(TuringOperatorToken.NAND)
                if op == "nand"
                else int(TuringOperatorToken.INPUT)
            ),
            metadata=metadata,
        )
        if cache_key is not None:
            scalar_cache[cache_key] = node
        if op == "constant" and value is not None:
            constant_values[node] = int(value) & 1
        for position, parent in enumerate(parents):
            scalar.add_edge(
                parent, node, arg_pos=position, role=f"arg:{position}",
            )
        return node

    def literal(payload: Mapping[str, Any], position: int) -> int:
        literals = (payload.get("metadata") or {}).get("literal_args", {})
        try:
            return int(literals[position])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{payload.get('op')!r} lacks scalar argument {position}"
            ) from error

    for source in nx.topological_sort(graph):
        if source not in live:
            continue
        payload = graph.nodes[source]
        op = str(payload.get("op") or payload.get("label") or "")
        parents = _ordered_parents(graph, source)
        metadata = payload.get("metadata") or {}
        if op in {"input", "const", "constant"}:
            width = int(
                metadata.get("width")
                or metadata.get("result_length")
                or graph.graph.get("bit_width", 0)
            )
            if width <= 0:
                raise ValueError(f"input {source!r} has no positive bit width")
            constant = (
                int(metadata["value"])
                if metadata.get("kind") == "constant" and "value" in metadata
                else None
            )
            bits: list[Hashable] = []
            for index in range(width):
                value = None if constant is None else (
                    constant >> (width - 1 - index)
                ) & 1
                bit = new_node(
                    "constant" if value is not None else "input",
                    source,
                    bit_index=index,
                    value=value,
                )
                bits.append(bit)
                if value is None:
                    input_bits[bit] = (source, index)
            vectors[source] = tuple(bits)
        elif op == "zeros":
            width = literal(payload, 0)
            vectors[source] = tuple(
                new_node("constant", source, bit_index=index, value=0)
                for index in range(width)
            )
        elif op == "length":
            if len(parents) != 1:
                raise ValueError("LENGTH requires one carrier")
            scalar_values[source] = len(vectors[parents[0]])
        elif op in {"sigma_L", "sigma_R"}:
            if len(parents) != 1:
                raise ValueError(f"{op} requires one carrier")
            amount = literal(payload, 1)
            source_bits = vectors[parents[0]]
            if op == "sigma_L":
                padding = tuple(
                    new_node("constant", source, bit_index=index, value=0)
                    for index in range(amount)
                )
                vectors[source] = source_bits + padding
            else:
                vectors[source] = (
                    source_bits[:-amount] if amount else source_bits
                )
        elif op == "concat":
            if len(parents) != 2:
                raise ValueError("CONCAT requires two carriers")
            vectors[source] = vectors[parents[0]] + vectors[parents[1]]
        elif op == "slice":
            if len(parents) != 1:
                raise ValueError("SLICE requires one carrier")
            start, stop = literal(payload, 1), literal(payload, 2)
            vectors[source] = vectors[parents[0]][start:stop]
        elif op == "nand":
            if len(parents) != 2:
                raise ValueError("NAND requires two carriers")
            left, right = vectors[parents[0]], vectors[parents[1]]
            if len(left) != len(right):
                raise ValueError("NAND carrier lengths differ")
            vectors[source] = tuple(
                new_node("nand", source, bit_index=index, parents=(a, b))
                for index, (a, b) in enumerate(zip(left, right))
            )
        elif op == "mu":
            if len(parents) != 3:
                raise ValueError("MU requires two carriers and a selector")
            left, right, selector = (vectors[parent] for parent in parents)
            if len(left) != len(right) or len(left) != len(selector):
                raise ValueError("MU carrier lengths differ")
            results: list[Hashable] = []
            for index, (a, b, select) in enumerate(zip(left, right, selector)):
                not_select = new_node(
                    "nand", source, bit_index=index, parents=(select, select),
                )
                masked_left = new_node(
                    "nand", source, bit_index=index, parents=(a, not_select),
                )
                masked_right = new_node(
                    "nand", source, bit_index=index, parents=(b, select),
                )
                results.append(new_node(
                    "nand", source, bit_index=index,
                    parents=(masked_left, masked_right),
                ))
            vectors[source] = tuple(results)
        else:
            raise ValueError(f"cannot scalarize Turing operator {op!r}")

    outputs = {node: vectors[node] for node in requested}
    scalar.graph["output_bits"] = outputs
    return ScalarizedTuringGraph(scalar, outputs, input_bits)


def _assemble_spilled_nand_terminal_tape_program(
    graph: nx.MultiDiGraph | nx.DiGraph,
    *,
    bit_width: int,
    input_values: Mapping[Hashable, int],
    outputs: tuple[Hashable, ...],
    order: list[Hashable],
) -> TerminalTapeProgram:
    """Materialize live values in liveness-reused numbered tape slots."""

    live = set(order)
    output_set = set(outputs)
    depth: dict[Hashable, int] = {}
    for node in reversed(order):
        depth[node] = 1 + max(
            (depth[target] for _source, target in graph.out_edges(node)
             if target in live),
            default=0,
        )
    remaining_predecessors = {
        node: sum(parent in live for parent in _ordered_parents(graph, node))
        for node in order
    }
    ready = {
        node for node in order if remaining_predecessors[node] == 0
    }
    uses = {node: (1 if node in output_set else 0) for node in live}
    for consumer in order:
        for parent in _ordered_parents(graph, consumer):
            if parent in uses:
                uses[parent] += 1
    scheduled: list[Hashable] = []
    while ready:
        def scheduling_key(candidate: Hashable) -> tuple[int, int, int, str]:
            parents = _ordered_parents(graph, candidate)
            occurrences = {
                parent: parents.count(parent) for parent in set(parents)
            }
            releases = sum(
                uses[parent] == count and parent not in output_set
                for parent, count in occurrences.items()
            )
            return (
                releases,
                depth[candidate],
                graph.out_degree(candidate),
                repr(candidate),
            )

        node = max(ready, key=scheduling_key)
        ready.remove(node)
        scheduled.append(node)
        for parent in _ordered_parents(graph, node):
            if parent in uses:
                uses[parent] -= 1
        for _source, target in graph.out_edges(node):
            if target not in live:
                continue
            remaining_predecessors[target] -= 1
            if remaining_predecessors[target] == 0:
                ready.add(target)
    if len(scheduled) != len(order):
        raise ValueError("terminal spill scheduler did not cover the live DAG")

    # Recompute uses for physical slot lifetime accounting. A dead operand slot
    # may be reused for its consumer because LOADs occur before STORE.
    uses = {node: (1 if node in output_set else 0) for node in live}
    for consumer in scheduled:
        for parent in _ordered_parents(graph, consumer):
            if parent in uses:
                uses[parent] += 1
    active_slots: dict[Hashable, int] = {}
    assigned_slots: dict[Hashable, int] = {}
    free_slots: list[int] = []
    next_slot = 0

    def allocate(node: Hashable) -> int:
        nonlocal next_slot
        slot = free_slots.pop() if free_slots else next_slot
        if not free_slots and slot == next_slot:
            next_slot += 1
        if slot >= 64:
            raise ValueError(
                "terminal spill liveness exceeds the 64-slot instruction envelope"
            )
        active_slots[node] = slot
        assigned_slots[node] = slot
        return slot

    mask = (1 << bit_width) - 1
    initial_spill_values: dict[int, int] = {}
    instructions: list[InstructionWord] = []
    instruction_sources: list[Hashable | None] = []
    supported_data = {"input", "const", "constant"}
    # Priming materializes every initialized leaf before instruction execution.
    # They therefore need distinct resident slots regardless of when the DAG
    # scheduler first makes their consumers ready.
    for node in scheduled:
        payload = graph.nodes[node]
        op = str(payload.get("op") or payload.get("label") or "")
        if op not in supported_data:
            continue
        slot = allocate(node)
        initial_spill_values[slot] = (
            _initialized_terminal_value(graph, node, input_values) & mask
        )
    for node in scheduled:
        payload = graph.nodes[node]
        op = str(payload.get("op") or payload.get("label") or "")
        if op in supported_data:
            continue
        parents = _ordered_parents(graph, node)
        if len(parents) != 2:
            raise ValueError("NAND terminal instruction requires two operands")
        instructions.append(InstructionWord(
            Opcode.LOAD,
            reg_a=0,
            reg_b=0,
            dest=0,
            param=active_slots[parents[0]],
        ))
        instruction_sources.append(parents[0])
        if parents[1] == parents[0]:
            right_register = 0
        else:
            instructions.append(InstructionWord(
                Opcode.LOAD,
                reg_a=0,
                reg_b=0,
                dest=1,
                param=active_slots[parents[1]],
            ))
            instruction_sources.append(parents[1])
            right_register = 1
        instructions.append(InstructionWord(
            Opcode.NAND,
            reg_a=0,
            reg_b=right_register,
            dest=2,
            param=0,
        ))
        instruction_sources.append(node)
        for parent in parents:
            uses[parent] -= 1
            if uses[parent] == 0 and parent not in output_set:
                free_slots.append(active_slots.pop(parent))
        result_slot = allocate(node)
        instructions.append(InstructionWord(
            Opcode.STORE,
            reg_a=2,
            reg_b=0,
            dest=0,
            param=result_slot,
        ))
        instruction_sources.append(node)

    output_registers: dict[Hashable, int] = {}
    output_spill_slots: dict[Hashable, int] = {}
    for register, output in enumerate(outputs[:REGISTERS]):
        instructions.append(InstructionWord(
            Opcode.LOAD,
            reg_a=0,
            reg_b=0,
            dest=register,
            param=active_slots[output],
        ))
        instruction_sources.append(output)
        output_registers[output] = register
    for output in outputs[REGISTERS:]:
        output_spill_slots[output] = active_slots[output]
    instructions.append(InstructionWord(
        Opcode.HALT,
        reg_a=0,
        reg_b=0,
        dest=0,
        param=0,
    ))
    instruction_sources.append(None)
    bios = BiosHeader(
        calib_fast_ms=10.0,
        calib_read_ms=50.0,
        drift_ms=1.0,
        inputs=[],
        outputs=sorted(output_registers.values()),
        instr_start_addr=0,
    )
    tape_map = TapeMap(bios, instruction_frames=len(instructions))
    tape_map.bios.instr_start_addr = tape_map.instr_start
    from .tape_compiler import TapeCompiler

    frames = TapeCompiler.binarize_instructions(instructions)
    return TerminalTapeProgram(
        tape_map=tape_map,
        instructions=tuple(instructions),
        instruction_frames=tuple(tuple(frame) for frame in frames),
        instruction_sources=tuple(instruction_sources),
        initial_register_values={},
        initial_spill_values=initial_spill_values,
        spill_slots=assigned_slots,
        node_registers=dict(output_registers),
        output_registers=output_registers,
        output_spill_slots=output_spill_slots,
        bit_width=bit_width,
        storage_mode="spilled",
    )


def assemble_nand_terminal_tape_program(
    graph: nx.MultiDiGraph | nx.DiGraph,
    *,
    bit_width: int,
    input_values: Mapping[Hashable, int],
    output_nodes: Iterable[Hashable] | None = None,
) -> TerminalTapeProgram:
    """Allocate a live NAND graph into the physical three-register machine."""

    if bit_width <= 0:
        raise ValueError("bit_width must be positive")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("terminal tape assembly requires an acyclic graph")
    if output_nodes is None:
        outputs = tuple(
            parent
            for node, payload in graph.nodes(data=True)
            if str(payload.get("op") or payload.get("label") or "") == "return"
            for parent in _ordered_parents(graph, node)
        )
    else:
        outputs = tuple(output_nodes)
    if not outputs:
        raise ValueError("terminal tape assembly requires at least one output")
    if any(node not in graph for node in outputs):
        raise ValueError("terminal output is not present in the graph")
    live = set(outputs)
    for output in outputs:
        live.update(nx.ancestors(graph, output))
    order = [node for node in nx.topological_sort(graph) if node in live]
    supported_data = {"input", "const", "constant"}
    for node in order:
        op = str(graph.nodes[node].get("op") or graph.nodes[node].get("label") or "")
        if op not in supported_data | {"nand"}:
            raise ValueError(f"NAND terminal assembler cannot encode {op!r}")
    if sum(
        str(graph.nodes[node].get("op") or graph.nodes[node].get("label") or "")
        in supported_data
        for node in order
    ) > REGISTERS:
        return _assemble_spilled_nand_terminal_tape_program(
            graph,
            bit_width=bit_width,
            input_values=input_values,
            outputs=outputs,
            order=order,
        )

    # Count operand occurrences, not merely successor nodes.  NAND(x, x) is a
    # two-use instruction even though a simple graph reports one successor.
    uses = {node: (1 if node in outputs else 0) for node in live}
    for consumer in order:
        for parent in _ordered_parents(graph, consumer):
            if parent in uses:
                uses[parent] += 1
    node_registers: dict[Hashable, int] = {}
    register_owner: dict[int, Hashable] = {}
    initial_register_values: dict[int, int] = {}
    instructions: list[InstructionWord] = []
    instruction_sources: list[Hashable | None] = []
    mask = (1 << bit_width) - 1

    def allocate_initial(node: Hashable) -> None:
        free = next(
            (register for register in range(REGISTERS) if register not in register_owner),
            None,
        )
        if free is None:
            raise ValueError("terminal inputs exceed the physical register file")
        value = _initialized_terminal_value(graph, node, input_values)
        node_registers[node] = free
        register_owner[free] = node
        initial_register_values[free] = value & mask

    for node in order:
        payload = graph.nodes[node]
        op = str(payload.get("op") or payload.get("label") or "")
        if op in supported_data:
            allocate_initial(node)

    pending = [
        node for node in order
        if str(graph.nodes[node].get("op") or graph.nodes[node].get("label") or "")
        == "nand"
    ]
    completed = set(node_registers)
    while pending:
        ready = [
            node for node in pending
            if all(parent in completed for parent in _ordered_parents(graph, node))
        ]
        if not ready:
            raise ValueError("terminal NAND scheduling reached an invalid dependency state")

        def scheduling_key(candidate: Hashable) -> tuple[int, int, int]:
            candidate_parents = _ordered_parents(graph, candidate)
            occurrences = {
                parent: candidate_parents.count(parent)
                for parent in set(candidate_parents)
            }
            released = sum(
                uses[parent] == count and parent not in outputs
                for parent, count in occurrences.items()
            )
            # Prefer an instruction that releases storage, then one combining
            # distinct values.  The latter schedules shared NAND terms before
            # destructive self-NANDs (the critical three-register XOR case).
            return released, len(occurrences), -order.index(candidate)

        node = max(ready, key=scheduling_key)
        pending.remove(node)
        parents = _ordered_parents(graph, node)
        if len(parents) != 2:
            raise ValueError("NAND terminal instruction requires two operands")
        if any(parent not in node_registers for parent in parents):
            raise ValueError("terminal graph operand was not allocated")
        source_registers = tuple(node_registers[parent] for parent in parents)
        for parent in parents:
            uses[parent] -= 1
        reusable = next((
            register
            for parent, register in zip(parents, source_registers)
            if uses[parent] == 0 and parent not in outputs
        ), None)
        destination = reusable
        if destination is None:
            destination = next((
                register
                for register in range(REGISTERS)
                if register not in register_owner
            ), None)
        if destination is None:
            return _assemble_spilled_nand_terminal_tape_program(
                graph,
                bit_width=bit_width,
                input_values=input_values,
                outputs=outputs,
                order=order,
            )
        displaced = register_owner.get(destination)
        if displaced is not None:
            register_owner.pop(destination)
        for parent, register in zip(parents, source_registers):
            if uses[parent] == 0 and parent not in outputs and register != destination:
                register_owner.pop(register, None)
        node_registers[node] = destination
        register_owner[destination] = node
        instructions.append(InstructionWord(
            Opcode.NAND,
            reg_a=source_registers[0],
            reg_b=source_registers[1],
            dest=destination,
            param=0,
        ))
        instruction_sources.append(node)
        completed.add(node)

    output_registers = {node: node_registers[node] for node in outputs}
    instructions.append(InstructionWord(
        Opcode.HALT, reg_a=0, reg_b=0, dest=0, param=0,
    ))
    instruction_sources.append(None)
    bios = BiosHeader(
        calib_fast_ms=10.0,
        calib_read_ms=50.0,
        drift_ms=1.0,
        inputs=sorted(initial_register_values),
        outputs=sorted(set(output_registers.values())),
        instr_start_addr=0,
    )
    tape_map = TapeMap(bios, instruction_frames=len(instructions))
    tape_map.bios.instr_start_addr = tape_map.instr_start
    from .tape_compiler import TapeCompiler

    frames = TapeCompiler.binarize_instructions(instructions)
    return TerminalTapeProgram(
        tape_map=tape_map,
        instructions=tuple(instructions),
        instruction_frames=tuple(tuple(frame) for frame in frames),
        instruction_sources=tuple(instruction_sources),
        initial_register_values=initial_register_values,
        initial_spill_values={},
        spill_slots={},
        node_registers=node_registers,
        output_registers=output_registers,
        output_spill_slots={},
        bit_width=bit_width,
        storage_mode="registers",
    )


@dataclass(frozen=True, slots=True)
class _CassetteCounters:
    seeks: int
    seek_distance_inches: float
    reads: int
    writes: int
    audio_samples: int


def _cassette_counters(cassette: Any) -> _CassetteCounters:
    return _CassetteCounters(
        int(cassette._seek_operations),
        float(cassette._seek_distance_inches),
        int(cassette._read_frames),
        int(cassette._write_frames),
        int(cassette._audio_cursor),
    )


def _observed_tape_cost(
    cassette: Any,
    before: _CassetteCounters,
    after: _CassetteCounters,
    *,
    operator_events: int = 0,
) -> TapeCostVector:
    seeks = after.seeks - before.seeks
    distance = int(round(
        (after.seek_distance_inches - before.seek_distance_inches)
        * cassette.bits_per_inch
    ))
    reads = after.reads - before.reads
    writes = after.writes - before.writes
    audio_samples = after.audio_samples - before.audio_samples
    traversed = distance + reads + writes
    return TapeCostVector(
        tape_distance_frames=distance,
        seeks=seeks,
        read_frames=reads,
        write_frames=writes,
        operator_events=operator_events,
        bit_transition_upper_bound=writes,
        storage_frames=cassette.total_bits,
        latency_seconds=audio_samples / cassette.sample_rate_hz,
        mechanical_work_units=(
            distance + seeks * (MOTOR_RAMP_MS / BIT_FRAME_MS)
        ),
        signal_energy_frame_units=reads + writes + operator_events,
        noise_exposure_frame_sources=traversed * NOISE_SOURCES,
        peak_parallel_lanes=1 if traversed else 0,
    )


def estimate_terminal_tape_execution_event_costs(
    program: TerminalTapeProgram,
    *,
    read_write_speed_ips: float = 1.875,
    seek_speed_ips: float = 30.0,
) -> tuple[TapeCostVector, ...]:
    """Predict transport cost for each encoded terminal instruction.

    The estimator follows the actual head order, including the deliberately
    literal sixteen-lane instruction fetch. Each lane read advances one frame,
    so fetching the next lane at the same instruction frame incurs a one-frame
    rewind. This exposes present hardware-model costs instead of idealizing
    them away.
    """

    if read_write_speed_ips <= 0 or seek_speed_ips <= 0:
        raise ValueError("tape speeds must be positive")
    # Boot reads the BIOS directly through the cassette and leaves the physical
    # head immediately after its final frame. TapeTransport is constructed
    # earlier and retains an independent logical cursor at zero.
    head = int(program.tape_map.instr_start)
    transport_cursor = 0
    seeks = 0
    distance = 0
    reads = 0
    writes = 0
    data_start = int(program.tape_map.data_start)
    width = int(program.bit_width)
    spill_extent = max(program.spill_slots.values(), default=-1) + 1
    storage = data_start + (REGISTERS + spill_extent) * width + 1
    fast_frame_seconds = (
        BIT_FRAME_MS / 1000.0
        * float(read_write_speed_ips) / float(seek_speed_ips)
    )
    event_costs: list[TapeCostVector] = []

    def access(target: int, *, writing: bool) -> None:
        nonlocal head, seeks, distance, reads, writes
        delta = abs(int(target) - head)
        if delta:
            seeks += 1
            distance += delta
        head = int(target) + 1
        if writing:
            writes += 1
        else:
            reads += 1

    def seek_to(target: int) -> None:
        nonlocal head, seeks, distance
        delta = abs(int(target) - head)
        if delta:
            seeks += 1
            distance += delta
        head = int(target)

    def advance_transport(target: int) -> None:
        nonlocal transport_cursor
        target = int(target)
        if target < transport_cursor:
            seek_to(target)
            transport_cursor = target
        while transport_cursor < target:
            access(transport_cursor, writing=False)
            transport_cursor += 1

    def read_slice(start: int) -> None:
        nonlocal transport_cursor
        advance_transport(start)
        for offset in range(width):
            access(start + offset, writing=False)
            transport_cursor = start + offset + 1

    def write_slice(start: int) -> None:
        nonlocal transport_cursor
        advance_transport(start)
        for offset in range(width):
            access(start + offset, writing=True)
            transport_cursor = start + offset + 1

    for index, instruction in enumerate(program.instructions):
        before = seeks, distance, reads, writes
        fetch_address = int(program.tape_map.instr_start) + index
        for _lane in range(16):
            access(fetch_address, writing=False)
        opcode = instruction.opcode
        register_a = data_start + int(instruction.reg_a) * width
        register_b = data_start + int(instruction.reg_b) * width
        destination = data_start + int(instruction.dest) * width
        if opcode == Opcode.LOAD:
            source = data_start + (
                REGISTERS + int(instruction.param)
            ) * width
            read_slice(source)
            write_slice(destination)
        elif opcode == Opcode.STORE:
            target = data_start + (
                REGISTERS + int(instruction.param)
            ) * width
            read_slice(register_a)
            write_slice(target)
        elif opcode == Opcode.NAND:
            read_slice(register_a)
            read_slice(register_b)
            write_slice(destination)
        elif opcode != Opcode.HALT:
            raise ValueError(
                f"terminal cost estimator cannot model {opcode.name}"
            )
        before_seeks, before_distance, before_reads, before_writes = before
        event_seeks = seeks - before_seeks
        event_distance = distance - before_distance
        event_reads = reads - before_reads
        event_writes = writes - before_writes
        traversed = event_distance + event_reads + event_writes
        event_costs.append(TapeCostVector(
            tape_distance_frames=event_distance,
            seeks=event_seeks,
            read_frames=event_reads,
            write_frames=event_writes,
            operator_events=1,
            bit_transition_upper_bound=event_writes,
            storage_frames=storage,
            latency_seconds=(
                event_distance * fast_frame_seconds
                + (event_reads + event_writes) * (BIT_FRAME_MS / 1000.0)
            ),
            mechanical_work_units=(
                event_distance
                + event_seeks * (MOTOR_RAMP_MS / BIT_FRAME_MS)
            ),
            signal_energy_frame_units=event_reads + event_writes + 1,
            noise_exposure_frame_sources=traversed * NOISE_SOURCES,
            peak_parallel_lanes=1 if traversed else 0,
        ))
    return tuple(event_costs)


def estimate_terminal_tape_execution_cost(
    program: TerminalTapeProgram,
    *,
    read_write_speed_ips: float = 1.875,
    seek_speed_ips: float = 30.0,
) -> TapeCostVector:
    """Aggregate the instruction-level static cassette cost trace."""

    total = TapeCostVector()
    for event in estimate_terminal_tape_execution_event_costs(
        program,
        read_write_speed_ips=read_write_speed_ips,
        seek_speed_ips=seek_speed_ips,
    ):
        total = total.serial(event)
    return total


def execute_terminal_tape_program(
    program: TerminalTapeProgram,
    *,
    tape_length: int | None = None,
    per_source_error_probability_upper_bound: float = 0.0,
    activity_callback: Callable[..., None] | None = None,
    time_scale_factor: float = 0.0,
    play_audio: bool = False,
) -> TapeExecutionWitness:
    """Prime, boot, and physically execute one terminal program with audio."""

    if not math.isfinite(float(time_scale_factor)) or time_scale_factor < 0:
        raise ValueError("time_scale_factor must be finite and nonnegative")

    from ..hardware.cassette_tape import CassetteTapeBackend
    from ..turing_machine.survival_computer import prime_tape_with_program
    from ..turing_machine.tape_machine import TapeMachine

    spill_extent = (
        max(program.spill_slots.values(), default=-1) + 1
    )
    required = (
        program.tape_map.data_start
        + (REGISTERS + spill_extent) * program.bit_width
        + 1
    )
    cassette = CassetteTapeBackend(
        tape_length=max(int(tape_length or 0), required),
        time_scale_factor=float(time_scale_factor),
        activity_callback=activity_callback,
        play_audio=bool(play_audio),
    )
    try:
        setup_before = _cassette_counters(cassette)
        prime_tape_with_program(
            cassette,
            program.tape_map,
            program.instruction_frames,
        )
        for register, value in program.initial_register_values.items():
            address = program.tape_map.data_start + register * program.bit_width
            for index in range(program.bit_width):
                bit = (int(value) >> (program.bit_width - 1 - index)) & 1
                cassette.write_bit(0, 0, address + index, bit)
        for slot, value in program.initial_spill_values.items():
            address = (
                program.tape_map.data_start
                + (REGISTERS + int(slot)) * program.bit_width
            )
            for index in range(program.bit_width):
                bit = (int(value) >> (program.bit_width - 1 - index)) & 1
                cassette.write_bit(0, 0, address + index, bit)
        machine = TapeMachine(cassette, program.bit_width)
        machine._boot(len(program.instructions))
        setup_after = _cassette_counters(cassette)
        setup_cost = _observed_tape_cost(
            cassette, setup_before, setup_after,
        )
        execution_before = setup_after
        events: list[TapeExecutionEvent] = []
        for index, source_node in enumerate(program.instruction_sources):
            event_before = _cassette_counters(cassette)
            audio_start = cassette._audio_cursor
            head_start = cassette._head_pos_inches
            opcode, dest, reg_a, reg_b, param = machine._fetch_decode()
            machine._execute(opcode, dest, reg_a, reg_b, param)
            event_after = _cassette_counters(cassette)
            events.append(TapeExecutionEvent(
                index,
                source_node,
                int(opcode.value),
                dest,
                reg_a,
                reg_b,
                param,
                audio_start,
                cassette._audio_cursor,
                head_start,
                cassette._head_pos_inches,
                _observed_tape_cost(
                    cassette,
                    event_before,
                    event_after,
                    operator_events=1,
                ),
            ))
            if machine.halted:
                break
        execution_after = _cassette_counters(cassette)
        execution_cost = _observed_tape_cost(
            cassette,
            execution_before,
            execution_after,
            operator_events=len(events),
        )
        observation_before = execution_after
        outputs: dict[Hashable, int] = {}
        for node, register in program.output_registers.items():
            address = machine.data_registers[register]
            value = 0
            for index in range(program.bit_width):
                value = (value << 1) | cassette.read_bit(
                    0, 0, address + index,
                )
            outputs[node] = value
        for node, slot in program.output_spill_slots.items():
            address = (
                program.tape_map.data_start
                + (REGISTERS + int(slot)) * program.bit_width
            )
            value = 0
            for index in range(program.bit_width):
                value = (value << 1) | cassette.read_bit(
                    0, 0, address + index,
                )
            outputs[node] = value
        observation_after = _cassette_counters(cassette)
        observation_cost = _observed_tape_cost(
            cassette,
            observation_before,
            observation_after,
        )
        total_cost = setup_cost.serial(execution_cost).serial(observation_cost)
        return TapeExecutionWitness(
            outputs,
            tuple(events),
            cassette._audio_cursor,
            machine.halted,
            setup_cost,
            execution_cost,
            observation_cost,
            total_cost,
            estimate_tape_reliability(
                total_cost,
                per_source_error_probability_upper_bound=(
                    per_source_error_probability_upper_bound
                ),
            ),
        )
    finally:
        cassette.close()


def assemble_scalar_machine_tape_program(
    raised: MachineTuringGraph,
    *,
    output_register: X86Register,
    input_register_values: Mapping[X86Register, int] | None = None,
) -> ScalarMachineTapeAssembly:
    """Lower the full structural Turing vocabulary to scalar NAND tape.

    Unlike the compact word-NAND execution path, this accepts arithmetic cones
    containing CONCAT, SLICE, MU, LENGTH, ZEROS, and tape motions. Structural
    operators disappear into carrier topology before physical slot allocation.
    """

    if not raised.complete:
        raise ValueError("machine Turing graph is incomplete")
    output_node = raised.register_outputs.get(output_register)
    if output_node is None:
        raise ValueError(f"machine graph has no output for {output_register.name}")
    graph = raised.operator_graph
    word_width = int(graph.graph.get("bit_width", 0))
    if word_width <= 0:
        raise ValueError("machine graph does not declare a positive bit width")
    scalarized = scalarize_turing_operator_graph(
        graph,
        output_nodes=(output_node,),
    )
    supplied = dict(input_register_values or {})
    input_bit_values: dict[Hashable, int] = {}
    for bit, (source, bit_index) in scalarized.input_bits.items():
        metadata = graph.nodes[source].get("metadata") or {}
        if metadata.get("kind") != "register":
            raise ValueError(f"scalar input source {source!r} is not a register")
        try:
            register = X86Register[str(metadata["register"])]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"invalid lifted register metadata on node {source!r}"
            ) from error
        if register not in supplied:
            raise ValueError(f"missing initial value for {register.name}")
        input_bit_values[bit] = (
            int(supplied[register]) >> (word_width - 1 - int(bit_index))
        ) & 1
    output_bits = scalarized.output_bits[output_node]
    program = assemble_nand_terminal_tape_program(
        scalarized.graph,
        bit_width=1,
        input_values=input_bit_values,
        output_nodes=output_bits,
    )

    provenance = nx.MultiDiGraph(kind="machine-structural-scalar-tape")
    for instruction in raised.report.instructions:
        provenance.add_node(
            ("machine", instruction.address),
            layer="machine",
            token_id=int(instruction.semantic),
            diagnostic=instruction.token.name.lower(),
        )
    turing_sources: set[Hashable] = set()
    for _scalar_node, payload in scalarized.graph.nodes(data=True):
        turing_sources.update((payload.get("metadata") or {}).get(
            "source_turing_nodes", (),
        ))
    for node in turing_sources:
        payload = graph.nodes[node]
        metadata = payload.get("metadata") or {}
        provenance.add_node(
            ("turing", node),
            layer="turing",
            token_id=payload.get("token_id"),
            diagnostic=payload.get("op"),
        )
        owner = metadata.get("machine_instruction_address")
        if owner is not None and ("machine", int(owner)) in provenance:
            provenance.add_edge(
                ("machine", int(owner)),
                ("turing", node),
                relation="ownership",
            )
    for node, payload in scalarized.graph.nodes(data=True):
        provenance.add_node(
            ("scalar", node),
            layer="scalar-turing",
            token_id=payload.get("token_id"),
            diagnostic=payload.get("op"),
        )
        for source in (payload.get("metadata") or {}).get(
            "source_turing_nodes", (),
        ):
            if ("turing", source) in provenance:
                provenance.add_edge(
                    ("turing", source),
                    ("scalar", node),
                    relation="ownership",
                )
    for index, (instruction, source) in enumerate(zip(
        program.instructions,
        program.instruction_sources,
    )):
        provenance.add_node(
            ("tape", index),
            layer="tape",
            token_id=int(instruction.opcode.value),
            diagnostic=instruction.opcode.name.lower(),
        )
        if source is not None and ("scalar", source) in provenance:
            provenance.add_edge(
                ("scalar", source),
                ("tape", index),
                relation="ownership",
            )
    return ScalarMachineTapeAssembly(
        raised=raised,
        scalarized=scalarized,
        program=program,
        output_register=output_register,
        output_node=output_node,
        output_bits=output_bits,
        input_bit_values=input_bit_values,
        provenance=provenance,
    )


def execute_machine_turing_graph(
    raised: MachineTuringGraph,
    *,
    output_register: X86Register,
    input_register_values: Mapping[X86Register, int] | None = None,
    per_source_error_probability_upper_bound: float = 0.0,
) -> MachineTapeExecution:
    """Execute one complete machine lift and retain instruction ownership."""

    if not raised.complete:
        raise ValueError("machine Turing graph is incomplete")
    output_node = raised.register_outputs.get(output_register)
    if output_node is None:
        raise ValueError(f"machine graph has no output for {output_register.name}")
    graph = raised.operator_graph
    bit_width = int(graph.graph.get("bit_width", 0))
    if bit_width <= 0:
        raise ValueError("machine graph does not declare a positive bit width")
    supplied = dict(input_register_values or {})
    live = {output_node, *nx.ancestors(graph, output_node)}
    node_values: dict[Hashable, int] = {}
    for node in live:
        payload = graph.nodes[node]
        metadata = payload.get("metadata") or {}
        if payload.get("op") != "input" or metadata.get("kind") != "register":
            continue
        try:
            register = X86Register[metadata["register"]]
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid lifted register metadata on node {node!r}") from error
        if register not in supplied:
            raise ValueError(f"missing initial value for {register.name}")
        node_values[node] = int(supplied[register])
    program = assemble_nand_terminal_tape_program(
        graph,
        bit_width=bit_width,
        input_values=node_values,
        output_nodes=(output_node,),
    )
    witness = execute_terminal_tape_program(
        program,
        per_source_error_probability_upper_bound=(
            per_source_error_probability_upper_bound
        ),
    )
    provenance = nx.MultiDiGraph(kind="machine-to-physical-ownership")
    for instruction in raised.report.instructions:
        provenance.add_node(
            ("machine", instruction.address),
            layer="machine",
            token_id=int(instruction.semantic),
            diagnostic=instruction.token.name.lower(),
        )
    for node, payload in graph.nodes(data=True):
        metadata = payload.get("metadata") or {}
        provenance.add_node(
            ("turing", node),
            layer="turing",
            token_id=payload.get("token_id"),
            diagnostic=payload.get("op"),
        )
        owner = metadata.get("machine_instruction_address")
        if owner is not None and ("machine", int(owner)) in provenance:
            provenance.add_edge(
                ("machine", int(owner)),
                ("turing", node),
                relation="ownership",
            )
    for index, (instruction, source_node) in enumerate(zip(
        program.instructions,
        program.instruction_sources,
    )):
        provenance.add_node(
            ("tape", index),
            layer="tape",
            token_id=int(instruction.opcode.value),
            diagnostic=instruction.opcode.name.lower(),
            spill_slot=(
                instruction.param
                if instruction.opcode in {Opcode.LOAD, Opcode.STORE}
                else None
            ),
        )
        if source_node is not None:
            provenance.add_edge(
                ("turing", source_node),
                ("tape", index),
                relation="ownership",
            )
    for event in witness.events:
        physical = ("physical", event.instruction_index)
        provenance.add_node(
            physical,
            layer="physical",
            token_id=event.opcode,
            diagnostic=Opcode(event.opcode).name.lower(),
            audio_start=event.audio_start,
            audio_end=event.audio_end,
            tape_distance_frames=event.cost.tape_distance_frames,
            latency_seconds=event.cost.latency_seconds,
            mechanical_work_units=event.cost.mechanical_work_units,
            noise_exposure_frame_sources=(
                event.cost.noise_exposure_frame_sources
            ),
        )
        provenance.add_edge(
            ("tape", event.instruction_index),
            physical,
            relation="ownership",
        )
    return MachineTapeExecution(raised, program, witness, provenance)


@dataclass(frozen=True, slots=True)
class ReductionArtifact:
    source: ProcessGraph
    target: ProcessGraph
    catalog: ReductionCatalog
    metagraph: EvolutionMetaGraph
    process_graph_ref: EvolutionGraphRef
    source_graph_ref: EvolutionGraphRef
    target_graph_ref: EvolutionGraphRef
    process_lineage: Mapping[Hashable, tuple[Hashable, ...]]
    lineage: Mapping[Hashable, tuple[Hashable, ...]]
    tape: TapeFeasibilityReport

    def verify_lineage(self) -> bool:
        source_nodes = set(self.source.G)
        target_nodes = set(self.target.G)
        if set(self.process_lineage) != source_nodes:
            return False
        if any(
            child not in source_nodes
            for children in self.process_lineage.values()
            for child in children
        ):
            return False
        if not set(self.lineage) <= source_nodes:
            return False
        return all(
            child in target_nodes
            for children in self.lineage.values()
            for child in children
        )

    def journey(self) -> "ReductionJourney":
        return ReductionJourney(
            stages=(
                ReductionStage(
                    ReductionRank(ReductionLayer.PROCESS),
                    self.process_graph_ref,
                    tuple(self.source.G.nodes),
                ),
                ReductionStage(
                    ReductionRank(ReductionLayer.BITOPS),
                    self.source_graph_ref,
                    tuple(self.source.G.nodes),
                ),
                ReductionStage(
                    ReductionRank(
                        ReductionLayer.TURING,
                        1 if _requires_scalarization(self.target.G) else 0,
                    ),
                    self.target_graph_ref,
                    tuple(self.target.G.nodes),
                ),
            ),
            morphisms=(
                ReductionMorphism(
                    self.process_graph_ref,
                    self.source_graph_ref,
                    dict(self.process_lineage),
                ),
                ReductionMorphism(
                    self.source_graph_ref,
                    self.target_graph_ref,
                    dict(self.lineage),
                ),
            ),
            metagraph=self.metagraph,
        )


@dataclass(frozen=True, slots=True)
class ReductionStage:
    rank: ReductionRank
    graph: EvolutionGraphRef
    nodes: tuple[Hashable, ...]


@dataclass(frozen=True, slots=True)
class ReductionMorphism:
    source_graph: EvolutionGraphRef
    target_graph: EvolutionGraphRef
    parent_to_children: Mapping[Hashable, tuple[Hashable, ...]]


@dataclass(frozen=True, slots=True)
class ReductionJourney:
    """Composable ancestry across any number of strictly descending stages."""

    stages: tuple[ReductionStage, ...]
    morphisms: tuple[ReductionMorphism, ...]
    metagraph: EvolutionMetaGraph

    def __post_init__(self) -> None:
        if len(self.morphisms) != max(len(self.stages) - 1, 0):
            raise ValueError("one morphism is required between adjacent stages")
        for index, morphism in enumerate(self.morphisms):
            source = self.stages[index]
            target = self.stages[index + 1]
            if (
                morphism.source_graph != source.graph
                or morphism.target_graph != target.graph
            ):
                raise ValueError("morphism endpoints must match adjacent stages")
            if not target.rank < source.rank:
                raise ValueError("reduction journey ranks must strictly descend")

    def descendants(
        self,
        stage_index: int,
        node: Hashable,
        *,
        target_stage: int | None = None,
    ) -> tuple[Hashable, ...]:
        """Follow one node through every requested parent/child morphism."""

        count = len(self.stages)
        start = stage_index if stage_index >= 0 else count + stage_index
        stop = (
            count - 1
            if target_stage is None
            else (target_stage if target_stage >= 0 else count + target_stage)
        )
        if not 0 <= start < count or not start <= stop < count:
            raise IndexError("invalid reduction journey stage range")
        current = (node,)
        for morphism in self.morphisms[start:stop]:
            current = tuple(
                child
                for parent in current
                for child in morphism.parent_to_children.get(parent, ())
            )
        return current


@dataclass(frozen=True, slots=True)
class ExecutedReductionArtifact:
    """One reduction plus its encoded and physically observed lower stages."""

    reduction: ReductionArtifact
    program: TerminalTapeProgram
    witness: TapeExecutionWitness
    journey: ReductionJourney
    tape_graph_ref: EvolutionGraphRef
    physical_graph_ref: EvolutionGraphRef
    scalarized: ScalarizedTuringGraph | None = None
    scalar_graph_ref: EvolutionGraphRef | None = None

    def physical_events_for_ancestor(
        self,
        stage_index: int,
        node: Hashable,
    ) -> tuple[int, ...]:
        stage_count = len(self.journey.stages)
        resolved_stage = (
            stage_index if stage_index >= 0 else stage_count + stage_index
        )
        if not 0 <= resolved_stage < stage_count:
            raise IndexError("invalid reduction journey stage")
        if node not in self.journey.stages[resolved_stage].nodes:
            raise KeyError(
                f"node {node!r} is absent from journey stage {resolved_stage}"
            )
        return tuple(sorted(set(self.journey.descendants(
            resolved_stage,
            node,
            target_stage=stage_count - 1,
        ))))

    def cost_for_ancestor(
        self,
        stage_index: int,
        node: Hashable,
    ) -> TapeCostVector:
        costs = {
            event.instruction_index: event.cost for event in self.witness.events
        }
        total = TapeCostVector()
        for index in self.physical_events_for_ancestor(stage_index, node):
            total = total.serial(costs[index])
        return total

    def reliability_for_ancestor(
        self,
        stage_index: int,
        node: Hashable,
        *,
        per_source_error_probability_upper_bound: float,
    ) -> TapeReliabilityEstimate:
        return estimate_tape_reliability(
            self.cost_for_ancestor(stage_index, node),
            per_source_error_probability_upper_bound=(
                per_source_error_probability_upper_bound
            ),
        )

    @property
    def concurrency_profile(self) -> GraphConcurrencyProfile:
        physical_lanes = max(
            self.witness.execution_cost.peak_parallel_lanes, 1,
        )
        if self.scalarized is not None:
            outputs = tuple(
                bit
                for bits in self.scalarized.output_bits.values()
                for bit in bits
            )
            return analyze_graph_concurrency(
                self.scalarized.graph,
                output_nodes=outputs,
                physical_parallel_lanes=physical_lanes,
            )
        outputs = tuple(self.program.output_registers) + tuple(
            self.program.output_spill_slots
        )
        return analyze_graph_concurrency(
            self.reduction.target.G,
            output_nodes=outputs,
            physical_parallel_lanes=physical_lanes,
        )


@dataclass(frozen=True, slots=True)
class ObjectReductionArtifact:
    """Selected object method joined to its Process/BitOps reduction."""

    raised: RaisedObjectMethod
    reduction: ReductionArtifact
    object_graph_ref: EvolutionGraphRef
    journey: ReductionJourney


@dataclass(frozen=True, slots=True)
class ExecutedObjectMethod:
    object_reduction: ObjectReductionArtifact
    execution: ExecutedReductionArtifact


def execute_reduction_artifact(
    artifact: ReductionArtifact,
    *,
    bit_width: int,
    input_values: Mapping[Hashable, int],
    output_nodes: Iterable[Hashable] | None = None,
    tape_length: int | None = None,
    per_source_error_probability_upper_bound: float = 0.0,
    base_journey: ReductionJourney | None = None,
) -> ExecutedReductionArtifact:
    """Assemble and execute an artifact, extending ancestry to physics."""

    selected_outputs = _terminal_output_nodes(
        artifact.target.G, output_nodes,
    )
    scalarized: ScalarizedTuringGraph | None = None
    if _requires_scalarization(artifact.target.G, selected_outputs):
        scalarized = scalarize_turing_operator_graph(
            artifact.target.G,
            output_nodes=selected_outputs,
        )
        scalar_inputs: dict[Hashable, int] = {}
        for bit, (source, bit_index) in scalarized.input_bits.items():
            if source not in input_values:
                raise ValueError(
                    f"missing initialized value for Turing input {source!r}"
                )
            scalar_inputs[bit] = (
                int(input_values[source])
                >> (bit_width - 1 - int(bit_index))
            ) & 1
        scalar_outputs = tuple(
            bit
            for output in selected_outputs
            for bit in scalarized.output_bits[output]
        )
        program = assemble_nand_terminal_tape_program(
            scalarized.graph,
            bit_width=1,
            input_values=scalar_inputs,
            output_nodes=scalar_outputs,
        )
    else:
        program = assemble_nand_terminal_tape_program(
            artifact.target.G,
            bit_width=bit_width,
            input_values=input_values,
            output_nodes=selected_outputs,
        )
    raw_witness = execute_terminal_tape_program(
        program,
        tape_length=tape_length,
        per_source_error_probability_upper_bound=(
            per_source_error_probability_upper_bound
        ),
    )
    if scalarized is None:
        witness = raw_witness
    else:
        packed_outputs: dict[Hashable, int] = {}
        for output in selected_outputs:
            value = 0
            for bit in scalarized.output_bits[output]:
                value = (value << 1) | int(raw_witness.outputs[bit])
            packed_outputs[output] = value
        witness = replace(raw_witness, outputs=packed_outputs)
    evolution = artifact.metagraph
    scalar_ref: EvolutionGraphRef | None = None
    turing_to_scalar: dict[Hashable, list[Hashable]] = {}
    active_graph_ref = artifact.target_graph_ref
    if scalarized is not None:
        scalar_ref = evolution.open_graph(
            "scalar-turing", "structural Turing scalarization",
        )
        scalar_components: dict[Hashable, EvolutionComponentRef] = {}
        for node, payload in scalarized.graph.nodes(data=True):
            metadata = payload.get("metadata") or {}
            source_nodes = tuple(metadata.get("source_turing_nodes", ()))
            sources = tuple(
                source_ref
                for source in source_nodes
                if evolution.has_component(source_ref := EvolutionComponentRef(
                    artifact.target_graph_ref.id, str(source),
                ))
            )
            component = evolution.component(
                scalar_ref,
                node,
                label=str(payload.get("op") or "scalar"),
                kind="scalar-turing-primitive",
                attributes={
                    "bit_index": metadata.get("bit_index"),
                    "source_turing_nodes": source_nodes,
                },
                consumes=sources,
                token_id=payload.get("token_id"),
            )
            scalar_components[node] = component
            for source in source_nodes:
                turing_to_scalar.setdefault(source, []).append(node)
            if sources:
                evolution.handoff(
                    component,
                    sources,
                    transformation="structural-turing-to-scalar-turing",
                )
        for source, target, edge in scalarized.graph.edges(data=True):
            evolution.relationship(
                scalar_ref,
                scalar_components[source],
                scalar_components[target],
                role=str(edge.get("role") or f"arg:{edge.get('arg_pos', 0)}"),
            )
        evolution.close_graph(scalar_ref)
        active_graph_ref = scalar_ref
    tape_ref = evolution.open_graph("tape", "encoded cassette instructions")
    tape_components: dict[int, EvolutionComponentRef] = {}
    active_to_tape: dict[Hashable, list[int]] = {}
    previous_tape: EvolutionComponentRef | None = None
    spill_nodes = {
        slot: node for node, slot in program.spill_slots.items()
    }
    for index, (instruction, source_node) in enumerate(zip(
        program.instructions,
        program.instruction_sources,
    )):
        sources: tuple[EvolutionComponentRef, ...] = ()
        if source_node is not None:
            source_ref = EvolutionComponentRef(
                active_graph_ref.id,
                str(source_node),
            )
            if evolution.has_component(source_ref):
                sources = (source_ref,)
                active_to_tape.setdefault(source_node, []).append(index)
        token_id = artifact.catalog.atlas.consume((
            int(ReductionNamespace.TAPE),
            int(instruction.opcode.value),
        ))
        instruction_attributes: dict[str, Any] = {
            "opcode": int(instruction.opcode.value),
            "reg_a": instruction.reg_a,
            "reg_b": instruction.reg_b,
            "dest": instruction.dest,
            "param": instruction.param,
            "frame": tuple(program.instruction_frames[index]),
            "storage_mode": program.storage_mode,
        }
        if instruction.opcode in {Opcode.LOAD, Opcode.STORE}:
            instruction_attributes.update({
                "spill_slot": instruction.param,
                "spill_node": spill_nodes.get(instruction.param),
            })
        component = evolution.component(
            tape_ref,
            index,
            label=instruction.opcode.name.lower(),
            kind="encoded-instruction",
            attributes=instruction_attributes,
            consumes=sources,
            token_id=token_id,
        )
        tape_components[index] = component
        if sources:
            evolution.handoff(
                component,
                sources,
                transformation="turing-to-tape-instruction",
            )
        if previous_tape is not None:
            evolution.relationship(
                tape_ref, previous_tape, component, role="next-instruction",
            )
        previous_tape = component
    evolution.close_graph(tape_ref)

    physical_ref = evolution.open_graph(
        "physical", "observed cassette execution",
    )
    physical_components: dict[int, EvolutionComponentRef] = {}
    tape_to_physical: dict[int, tuple[int, ...]] = {}
    previous_physical: EvolutionComponentRef | None = None
    for event in witness.events:
        tape_source = tape_components[event.instruction_index]
        cost_attributes = {
            name: getattr(event.cost, name)
            for name in event.cost.__dataclass_fields__
        }
        component = evolution.component(
            physical_ref,
            event.instruction_index,
            label=f"physical-{Opcode(event.opcode).name.lower()}",
            kind="cassette-execution-event",
            attributes={
                "audio_start": event.audio_start,
                "audio_end": event.audio_end,
                "head_start_inches": event.head_start_inches,
                "head_end_inches": event.head_end_inches,
                **cost_attributes,
            },
            consumes=(tape_source,),
            token_id=artifact.catalog.atlas.consume((
                int(ReductionNamespace.PHYSICAL),
                int(event.opcode),
            )),
        )
        physical_components[event.instruction_index] = component
        tape_to_physical[event.instruction_index] = (
            event.instruction_index,
        )
        evolution.handoff(
            component,
            (tape_source,),
            transformation="tape-instruction-to-physical-event",
        )
        if previous_physical is not None:
            evolution.relationship(
                physical_ref,
                previous_physical,
                component,
                role="next-event",
            )
        previous_physical = component
    evolution.close_graph(physical_ref)

    base = base_journey or artifact.journey()
    if (
        not base.stages
        or base.stages[-1].graph != artifact.target_graph_ref
        or base.stages[-1].rank.layer != ReductionLayer.TURING
    ):
        raise ValueError("execution journey must terminate at the artifact's Turing graph")
    scalar_stages: tuple[ReductionStage, ...] = ()
    scalar_morphisms: tuple[ReductionMorphism, ...] = ()
    if scalarized is not None:
        if scalar_ref is None:  # pragma: no cover - construction invariant
            raise RuntimeError("scalar graph reference was not created")
        scalar_stages = (ReductionStage(
            ReductionRank(ReductionLayer.TURING),
            scalar_ref,
            tuple(scalarized.graph.nodes),
        ),)
        scalar_morphisms = (ReductionMorphism(
            artifact.target_graph_ref,
            scalar_ref,
            {
                parent: tuple(children)
                for parent, children in turing_to_scalar.items()
            },
        ),)
    journey = ReductionJourney(
        stages=base.stages + scalar_stages + (
            ReductionStage(
                ReductionRank(ReductionLayer.TAPE),
                tape_ref,
                tuple(tape_components),
            ),
            ReductionStage(
                ReductionRank(ReductionLayer.PHYSICAL),
                physical_ref,
                tuple(physical_components),
            ),
        ),
        morphisms=base.morphisms + scalar_morphisms + (
            ReductionMorphism(
                active_graph_ref,
                tape_ref,
                {
                    parent: tuple(children)
                    for parent, children in active_to_tape.items()
                },
            ),
            ReductionMorphism(
                tape_ref,
                physical_ref,
                tape_to_physical,
            ),
        ),
        metagraph=evolution,
    )
    return ExecutedReductionArtifact(
        artifact,
        program,
        witness,
        journey,
        tape_ref,
        physical_ref,
        scalarized,
        scalar_ref,
    )


def reduce_object_method_source(
    source: str | Any,
    *,
    class_name: str,
    method_name: str,
    bit_width: int,
    source_filename: str = "<object-source>",
    catalog: ReductionCatalog | None = None,
    metagraph: EvolutionMetaGraph | None = None,
) -> ObjectReductionArtifact:
    """Raise a selected OOP method and join it to Process/BitOps ancestry."""

    evolution = metagraph or EvolutionMetaGraph()
    active_catalog = catalog or bitops_turing_reduction_catalog()
    raised = raise_object_method_to_process_graph(
        source,
        class_name=class_name,
        method_name=method_name,
        source_filename=source_filename,
        materialize_memory=False,
    )
    identity = raised.identity
    object_ref = evolution.open_graph("object", identity.graph_identity)
    encoded_identity = identity.graph_identity.encode("utf-8")
    method_component = evolution.component(
        object_ref,
        identity.graph_identity,
        label=identity.graph_identity,
        kind="object-method",
        attributes={
            "class_name": identity.class_name,
            "method_name": identity.method_name,
            "class_source_span": identity.class_source_span,
            "method_source_span": identity.method_source_span,
            "decorators": identity.decorators,
            "source_filename": raised.source_filename,
        },
        token_id=active_catalog.atlas.consume((
            int(ReductionNamespace.OBJECT),
            len(encoded_identity),
            *encoded_identity,
        )),
    )
    evolution.close_graph(object_ref)
    reduction = reduce_bitops_process_graph(
        raised.process_graph,
        bit_width=bit_width,
        catalog=active_catalog,
        metagraph=evolution,
    )
    process_children: list[Hashable] = []
    for node in reduction.source.G:
        process_component = EvolutionComponentRef(
            reduction.process_graph_ref.id,
            str(node),
        )
        if not evolution.has_component(process_component):
            continue
        process_children.append(node)
        evolution.handoff(
            process_component,
            (method_component,),
            transformation="object-method-to-process-graph",
            detail={"graph_identity": identity.graph_identity},
        )
    base = reduction.journey()
    journey = ReductionJourney(
        stages=(ReductionStage(
            ReductionRank(ReductionLayer.OBJECT),
            object_ref,
            (identity.graph_identity,),
        ),) + base.stages,
        morphisms=(ReductionMorphism(
            object_ref,
            reduction.process_graph_ref,
            {identity.graph_identity: tuple(process_children)},
        ),) + base.morphisms,
        metagraph=evolution,
    )
    return ObjectReductionArtifact(
        raised,
        reduction,
        object_ref,
        journey,
    )


def execute_object_method_source(
    source: str | Any,
    *,
    class_name: str,
    method_name: str,
    bit_width: int,
    input_values_by_name: Mapping[str, int],
    source_filename: str = "<object-source>",
    tape_length: int | None = None,
    per_source_error_probability_upper_bound: float = 0.0,
) -> ExecutedObjectMethod:
    """Run the selected object method through every currently live stage."""

    object_reduction = reduce_object_method_source(
        source,
        class_name=class_name,
        method_name=method_name,
        bit_width=bit_width,
        source_filename=source_filename,
    )
    target = object_reduction.reduction.target.G
    available_inputs = {
        str(payload.get("label")): node
        for node, payload in target.nodes(data=True)
        if str(payload.get("op")) == "input"
    }
    unknown = set(input_values_by_name) - set(available_inputs)
    if unknown:
        raise ValueError(f"unknown object-method inputs: {sorted(unknown)!r}")
    node_values = {
        available_inputs[name]: int(value)
        for name, value in input_values_by_name.items()
    }
    execution = execute_reduction_artifact(
        object_reduction.reduction,
        bit_width=bit_width,
        input_values=node_values,
        tape_length=tape_length,
        per_source_error_probability_upper_bound=(
            per_source_error_probability_upper_bound
        ),
        base_journey=object_reduction.journey,
    )
    return ExecutedObjectMethod(object_reduction, execution)


def reduce_bitops_process_graph(
    source: ProcessGraph,
    *,
    bit_width: int,
    catalog: ReductionCatalog | None = None,
    metagraph: EvolutionMetaGraph | None = None,
) -> ReductionArtifact:
    """Run the live BitOps expansion and record its cross-stage morphism."""

    active_catalog = catalog or bitops_turing_reduction_catalog()
    evolution = metagraph or EvolutionMetaGraph()
    process_ref = evolution.open_graph("process", "source ProcessGraph")
    process_components: dict[Hashable, EvolutionComponentRef] = {}
    for node, payload in source.G.nodes(data=True):
        op = str(payload.get("op") or payload.get("label") or "")
        process_components[node] = evolution.component(
            process_ref,
            node,
            label=op or "node",
            kind="process-element",
        )
    for source_node, target_node, edge in source.G.edges(data=True):
        evolution.relationship(
            process_ref,
            process_components[source_node],
            process_components[target_node],
            role=str(edge.get("role", "data")),
        )
    evolution.close_graph(process_ref)

    source_ref = evolution.open_graph("bitops", "self-describing BitOps")
    source_components: dict[Hashable, EvolutionComponentRef] = {}
    process_lineage: dict[Hashable, tuple[Hashable, ...]] = {}
    for node, payload in source.G.nodes(data=True):
        op = str(payload.get("op") or payload.get("label") or "")
        rule = active_catalog.resolve_spelling(ReductionLayer.BITOPS, op)
        process_component = process_components[node]
        source_components[node] = evolution.component(
            source_ref,
            node,
            label=op or "node",
            kind=("bitops-element" if rule is not None else "boundary"),
            consumes=(process_component,),
            token_id=None if rule is None else rule.source_token_id,
        )
        process_lineage[node] = (node,)
        evolution.handoff(
            source_components[node],
            (process_component,),
            transformation=(
                "process-operation-to-bitops"
                if rule is not None
                else "process-boundary-to-bitops-boundary"
            ),
        )
    for source_node, target_node, edge in source.G.edges(data=True):
        evolution.relationship(
            source_ref,
            source_components[source_node],
            source_components[target_node],
            role=str(edge.get("role", "data")),
        )

    target = expand_bitops_process_graph(source, bit_width=bit_width)
    target_ref = evolution.open_graph("turing", "Turing super-reduction")
    target_components: dict[Hashable, EvolutionComponentRef] = {}
    lineage_lists: dict[Hashable, list[Hashable]] = {
        node: [] for node in source.G
    }
    for node, payload in target.G.nodes(data=True):
        op = str(payload.get("op") or payload.get("label") or "")
        control = payload.get("control") or {}
        parent = control.get("source_node")
        consumes = (
            (source_components[parent],)
            if parent in source_components
            else (
                (source_components[node],)
                if node in source_components
                else ()
            )
        )
        primitive = _TURING_SPELLINGS.get(op)
        token_id = (
            None
            if primitive is None
            else active_catalog.atlas.consume((
                int(ReductionNamespace.TURING), int(primitive),
            ))
        )
        target_components[node] = evolution.component(
            target_ref,
            node,
            label=op or "node",
            kind=("turing-primitive" if primitive is not None else "boundary"),
            consumes=consumes,
            token_id=token_id,
        )
        if consumes:
            source_node = parent if parent in source_components else node
            lineage_lists[source_node].append(node)
            rule = active_catalog.resolve_spelling(
                ReductionLayer.BITOPS,
                str(control.get("source_operation") or op),
            )
            evolution.handoff(
                target_components[node],
                consumes,
                transformation=(
                    "identity"
                    if rule is None
                    else f"reduction-rule:{rule.token_id}"
                ),
            )
    for source_node, target_node, edge in target.G.edges(data=True):
        evolution.relationship(
            target_ref,
            target_components[source_node],
            target_components[target_node],
            role=str(edge.get("role", "data")),
        )
    evolution.close_graph(source_ref)
    evolution.close_graph(target_ref)
    lineage = {
        parent: tuple(children)
        for parent, children in lineage_lists.items()
        if children
    }
    tape = estimate_turing_tape_feasibility(
        target.G, bit_width=bit_width,
    )
    return ReductionArtifact(
        source=source,
        target=target,
        catalog=active_catalog,
        metagraph=evolution,
        process_graph_ref=process_ref,
        source_graph_ref=source_ref,
        target_graph_ref=target_ref,
        process_lineage=process_lineage,
        lineage=lineage,
        tape=tape,
    )


__all__ = [
    "BitOpsReductionToken",
    "ExecutedReductionArtifact",
    "ExecutedObjectMethod",
    "GraphConcurrencyProfile",
    "MachineTapeExecution",
    "ObjectReductionArtifact",
    "ReductionArtifact",
    "ReductionCatalog",
    "ReductionLayer",
    "ReductionJourney",
    "ReductionMorphism",
    "ReductionNamespace",
    "ReductionRank",
    "ReductionRule",
    "ReductionStage",
    "ScalarMachineTapeAssembly",
    "ScalarizedTuringGraph",
    "TapeExecutionEvent",
    "TapeExecutionWitness",
    "TapeCostVector",
    "TapeFeasibilityReport",
    "TapePlacement",
    "TapeReliabilityEstimate",
    "TerminalTapeProgram",
    "assemble_nand_terminal_tape_program",
    "assemble_scalar_machine_tape_program",
    "analyze_graph_concurrency",
    "bitops_turing_reduction_catalog",
    "estimate_turing_tape_feasibility",
    "estimate_terminal_tape_execution_cost",
    "estimate_terminal_tape_execution_event_costs",
    "estimate_tape_reliability",
    "execute_reduction_artifact",
    "execute_object_method_source",
    "execute_machine_turing_graph",
    "execute_terminal_tape_program",
    "reduce_bitops_process_graph",
    "reduce_object_method_source",
    "scalarize_turing_operator_graph",
]
