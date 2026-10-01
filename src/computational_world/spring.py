"""AbstractTensor port of the existing BoundSpring physics state.

Force equations, rest-length activation, growth, glow, damping, repulsion,
and spherical containment follow ``transmogrifier/bound_spring.py``. The one
deliberate ownership change is timestep selection: this module never
subdivides ``dt``. It reports an admissible ceiling and rejects an oversized
candidate so ``dt_system`` performs rollback and retry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from ..common.dt_system.dt_scaler import Metrics
from ..common.tensors.abstraction import AbstractTensor as AT
from .state import ComputationalWorldState


REPULSION_MODES = ("none", "full", "topk")


@dataclass(frozen=True, slots=True)
class BoundSpringParameters:
    k_stretch: float = 8.0
    c_repulse: float = 0.4
    damping: float = 0.902
    eps_rep: float = 1.0e-4
    level_target: float = 0.9
    type_target: float = 0.75
    role_target: float = 0.5
    relax_rate: float = 0.1
    growth_rate: float = 0.1
    max_force: float | None = None
    max_velocity: float | None = None
    max_displacement: float | None = None
    c_frac: float = 0.5
    boundary_radius: float | None = None
    cycle_period: float = 1.0 / 60.0
    nominal_dt: float = 1.0 / 60.0
    beta_level: float = 0.5
    beta_type: float = 0.7
    beta_role: float = 0.9
    glow_rise: float = 0.3
    glow_decay: float = 0.05
    glow_peak_alpha: float = 1.0
    glow_floor_alpha: float = 0.1
    glow_peak_radius: float = 0.2
    glow_floor_radius: float = 0.1
    #: Bytes one tile of the force assembly may hold.  The incidence matmul
    #: and the pairwise repulsion are dense (nodes x edges, nodes x nodes x
    #: 3); measured on a 13,024-node, 46,348-edge world, one untiled call
    #: peaked at 12.8 GB.  ``_forces`` splits edges and repulsion rows into
    #: tiles under this budget; a world that fits in one tile runs the
    #: untiled code unchanged.  Compute layout only, not physics.
    force_tile_bytes: int = 512 * 2 ** 20
    #: Which node pairs the repulsion -- the one force that is not carried by
    #: an edge -- sums over.  ``"full"``: every same-network pair (the legacy
    #: N x N sum).  ``"topk"``: each node's ``repulse_k`` nearest same-network
    #: nodes, chosen afresh from the positions at every force evaluation.
    #: ``"none"``: no repulsion, whatever ``c_repulse`` says.
    repulsion: str = "full"
    repulse_k: int = 8

    def __post_init__(self) -> None:
        if self.k_stretch < 0.0 or self.c_repulse < 0.0:
            raise ValueError("spring coefficients cannot be negative")
        if self.damping < 0.0 or self.eps_rep <= 0.0:
            raise ValueError("spring damping/epsilon are invalid")
        if self.cycle_period <= 0.0 or self.nominal_dt <= 0.0:
            raise ValueError("spring managed periods must be positive")
        if self.c_frac <= 0.0:
            raise ValueError("spring relativistic fraction must be positive")
        if int(self.force_tile_bytes) <= 0:
            raise ValueError("spring force tile budget must be positive")
        if self.repulsion not in REPULSION_MODES:
            raise ValueError(f"spring repulsion must be one of {REPULSION_MODES}")
        if int(self.repulse_k) <= 0:
            raise ValueError("spring top-k repulsion needs k >= 1")


def _tensor(value, dtype: str):
    return AT.tensor(value, dtype=dtype)


def install_bound_spring(
    state: ComputationalWorldState,
    positions: Sequence[Sequence[float]],
    edges: Sequence[tuple[int, int]],
    *,
    edge_level_mask: Sequence[Sequence[bool]] | None = None,
    edge_type_mask: Sequence[Sequence[bool]] | None = None,
    edge_role_mask: Sequence[Sequence[bool]] | None = None,
    node_level_mask: Sequence[Sequence[bool]] | None = None,
    node_type_mask: Sequence[Sequence[bool]] | None = None,
    node_role_mask: Sequence[Sequence[bool]] | None = None,
    parameters: BoundSpringParameters | None = None,
    surface: bool = False,
) -> None:
    """Install one legacy-compatible BoundSpring network into world state.

    ``surface=True`` puts the network ON its boundary sphere instead of inside
    it: every step projects each node back to the radius and strips the whole
    radial velocity, so the network lives on the surface (the shell of a
    globe) while another network, installed without it, lives in the volume.
    """

    cfg = parameters or BoundSpringParameters()
    position = _tensor(positions, "float32")
    if len(position.shape) != 2 or tuple(position.shape[1:]) != (3,):
        raise ValueError("BoundSpring positions must have shape (S, 3)")
    node_count = int(position.shape[0])
    edge_pairs = tuple((int(source), int(target)) for source, target in edges)
    if any(
        source < 0 or target < 0 or source >= node_count or target >= node_count
        for source, target in edge_pairs
    ):
        raise ValueError("BoundSpring edge endpoint is outside node storage")
    edge_index = _tensor(
        [
            [source for source, _target in edge_pairs],
            [target for _source, target in edge_pairs],
        ],
        "int64",
    )
    edge_count = len(edge_pairs)
    if edge_count:
        source = edge_index[0]
        target = edge_index[1]
        displacement = position.index_select(0, target) - position.index_select(0, source)
        lengths = (displacement * displacement).sum(dim=1).sqrt()
    else:
        lengths = _tensor([], "float32")

    def masks(edge_mask, node_mask):
        if edge_mask is None and node_mask is None:
            return [[True] * edge_count], [[True] * node_count]
        if edge_mask is None or node_mask is None:
            raise ValueError("BoundSpring edge and node masks must be supplied together")
        return edge_mask, node_mask

    lvl_e, lvl_n = masks(edge_level_mask, node_level_mask)
    typ_e, typ_n = masks(edge_type_mask, node_type_mask)
    rol_e, rol_n = masks(edge_role_mask, node_role_mask)
    group_count = len(lvl_e)
    if not (
        len(typ_e) == len(rol_e) == len(lvl_n) == len(typ_n) == len(rol_n)
        == group_count
    ):
        raise ValueError("BoundSpring activation group counts must match")

    state.spring_position = position
    state.spring_node_count = _tensor([node_count], "int64")
    state.spring_edge_count = _tensor([edge_count], "int64")
    state.spring_group_count = _tensor([group_count], "int64")
    state.spring_velocity = AT.zeros_like(position)
    state.spring_edge_index = edge_index
    state.spring_mass = _tensor([1.0] * node_count, "float32")
    state.spring_rest_length = lengths.clone()
    state.spring_base_length = lengths.clone()
    state.spring_natural_rest_length = _tensor([1.0] * edge_count, "float32")
    state.spring_done_growing = _tensor([False] * edge_count, "bool")
    state.spring_edge_level_mask = _tensor(lvl_e, "bool")
    state.spring_edge_type_mask = _tensor(typ_e, "bool")
    state.spring_edge_role_mask = _tensor(rol_e, "bool")
    state.spring_node_level_mask = _tensor(lvl_n, "bool")
    state.spring_node_type_mask = _tensor(typ_n, "bool")
    state.spring_node_role_mask = _tensor(rol_n, "bool")
    state.spring_glow_alpha = _tensor(
        [[cfg.glow_floor_alpha] for _ in range(node_count)], "float32"
    )
    state.spring_glow_radius = _tensor(
        [[cfg.glow_floor_radius] for _ in range(node_count)], "float32"
    )
    state.spring_group_index = _tensor([0], "int32")
    state.spring_cycle_time = _tensor([0.0], "float64")
    if node_count:
        center = position.mean(dim=0, keepdim=True)
        radius = float(
            ((position - center) * (position - center)).sum(dim=1).sqrt().max().item()
        )
    else:
        center = _tensor([], "float32").reshape((0, 3))
        radius = 0.0
    state.spring_boundary_center = center
    state.spring_boundary_radius = _tensor(
        (
            [cfg.boundary_radius if cfg.boundary_radius is not None else 2.0 * radius]
            if node_count else []
        ),
        "float32",
    )
    state.spring_node_network = _tensor([0] * node_count, "int32")
    state.spring_edge_network = _tensor([0] * edge_count, "int32")
    state.spring_network_surface = _tensor([bool(surface)] if node_count else [], "bool")
    state.spring_external_force = AT.zeros_like(position)
    state.validate_sparse_shapes()


def _pad_mask_rows(mask, rows: int):
    missing = int(rows) - int(mask.shape[0])
    if missing <= 0:
        return mask
    return AT.cat([
        mask,
        AT.zeros((missing, int(mask.shape[1])), dtype="bool"),
    ], dim=0)


def append_bound_spring(
    state: ComputationalWorldState,
    positions: Sequence[Sequence[float]],
    edges: Sequence[tuple[int, int]],
    **kwargs,
) -> int:
    """Append an independent network to the same lean tensor state machine."""

    if int(state.spring_position.shape[0]) == 0:
        install_bound_spring(state, positions, edges, **kwargs)
        return 0
    incoming = ComputationalWorldState.empty()
    install_bound_spring(incoming, positions, edges, **kwargs)
    network_id = max(
        (int(value) for value in state.spring_node_network.tolist()),
        default=-1,
    ) + 1
    node_offset = int(state.spring_position.shape[0])

    state.spring_position = AT.cat(
        [state.spring_position, incoming.spring_position], dim=0
    )
    state.spring_velocity = AT.cat(
        [state.spring_velocity, incoming.spring_velocity], dim=0
    )
    state.spring_edge_index = AT.cat([
        state.spring_edge_index,
        incoming.spring_edge_index + node_offset,
    ], dim=1)
    for name in (
        "spring_mass", "spring_rest_length", "spring_base_length",
        "spring_natural_rest_length", "spring_done_growing",
        "spring_glow_alpha", "spring_glow_radius", "spring_external_force",
    ):
        setattr(state, name, AT.cat([
            getattr(state, name), getattr(incoming, name)
        ], dim=0))
    state.spring_node_network = AT.cat([
        state.spring_node_network,
        _tensor([network_id] * int(incoming.spring_position.shape[0]), "int32"),
    ], dim=0)
    state.spring_edge_network = AT.cat([
        state.spring_edge_network,
        _tensor([network_id] * int(incoming.spring_edge_index.shape[1]), "int32"),
    ], dim=0)

    groups = max(
        int(state.spring_edge_level_mask.shape[0]),
        int(incoming.spring_edge_level_mask.shape[0]),
    )
    for edge_name, node_name in (
        ("spring_edge_level_mask", "spring_node_level_mask"),
        ("spring_edge_type_mask", "spring_node_type_mask"),
        ("spring_edge_role_mask", "spring_node_role_mask"),
    ):
        existing_edge = _pad_mask_rows(getattr(state, edge_name), groups)
        incoming_edge = _pad_mask_rows(getattr(incoming, edge_name), groups)
        setattr(state, edge_name, AT.cat([existing_edge, incoming_edge], dim=1))
        existing_node = _pad_mask_rows(getattr(state, node_name), groups)
        incoming_node = _pad_mask_rows(getattr(incoming, node_name), groups)
        setattr(state, node_name, AT.cat([existing_node, incoming_node], dim=1))
    state.spring_boundary_center = AT.cat([
        state.spring_boundary_center, incoming.spring_boundary_center
    ], dim=0)
    state.spring_boundary_radius = AT.cat([
        state.spring_boundary_radius, incoming.spring_boundary_radius
    ], dim=0)
    state.spring_network_surface = AT.cat([
        state.spring_network_surface, incoming.spring_network_surface
    ], dim=0)
    state.spring_node_count = _tensor(
        [int(state.spring_position.shape[0])], "int64"
    )
    state.spring_edge_count = _tensor(
        [int(state.spring_edge_index.shape[1])], "int64"
    )
    state.spring_group_count = _tensor([groups], "int64")
    state.validate_sparse_shapes()
    return network_id


def _scaled_fraction(per_nominal_step: float, dt: float, nominal_dt: float) -> float:
    """Preserve the legacy factor at nominal dt while making it dt coherent."""

    if per_nominal_step <= 0.0:
        return 0.0
    if per_nominal_step >= 1.0:
        return 1.0
    return 1.0 - (1.0 - per_nominal_step) ** (float(dt) / nominal_dt)


def _resolved_caps(
    state: ComputationalWorldState,
    cfg: BoundSpringParameters,
) -> tuple[float, float, float]:
    """Resolve the same scale-derived safety defaults as legacy BoundSpring."""

    edge_count = int(state.spring_edge_count.item())
    if edge_count:
        mean_length = float(state.spring_base_length[:edge_count].mean().item())
    else:
        mean_length = 1.0
    max_displacement = (
        float(cfg.max_displacement)
        if cfg.max_displacement is not None
        else 0.5 * mean_length
    )
    max_force = (
        float(cfg.max_force)
        if cfg.max_force is not None
        else cfg.k_stretch * mean_length
    )
    max_velocity = (
        float(cfg.max_velocity)
        if cfg.max_velocity is not None
        else max_displacement  # legacy default_dt is exactly 1
    )
    return max_force, max_velocity, max_displacement


def bound_spring_stretch_force(
    edge_displacement,
    source_incidence,
    target_incidence,
    rest_length,
    k_stretch,
):
    """Pure tensor kernel for the legacy Hooke/incidence force calculation.

    This deliberately ordinary Python function is the compiler-facing numeric
    surface as well as the implementation used by the live state machine.  It
    contains no private timestep logic and introduces no operator beyond the
    existing AbstractTensor/process-graph vocabulary.
    """

    length = (
        (edge_displacement * edge_displacement)
        .sum(dim=1, keepdim=True)
        .sqrt()
        + 1.0e-9
    )
    direction = edge_displacement / length
    delta = length.reshape((-1,)) - rest_length
    edge_force = k_stretch * delta.reshape((-1, 1)) * direction
    return source_incidence @ (-edge_force) + target_incidence @ edge_force


def _forces(
    state: ComputationalWorldState,
    cfg: BoundSpringParameters,
    *,
    rest_length=None,
):
    node_count = int(state.spring_node_count.item())
    edge_count = int(state.spring_edge_count.item())
    position = state.spring_position[:node_count].clone()
    force = AT.zeros_like(position)
    if edge_count:
        source = state.spring_edge_index[0, :edge_count].clone()
        target = state.spring_edge_index[1, :edge_count].clone()
        displacement = position.index_select(0, source) - position.index_select(0, target)
        # Incidence accumulation handles repeated endpoints without backend
        # indexed-assignment semantics. It is the same spring force sum as
        # the legacy pair of ``index_add_`` calls.
        node_ids = AT.arange(node_count, dtype="int64").reshape((-1, 1))
        active_rest_length = (
            state.spring_rest_length[:edge_count].clone()
            if rest_length is None else rest_length
        )
        # Autotile over edges: per edge the two incidence columns cost about
        # 10 bytes per node (float32 + the bool compare, twice).
        edge_tile = max(1, int(cfg.force_tile_bytes) // max(10 * node_count, 1))
        if edge_count == 1:
            source_incidence = (node_ids == source.reshape((1, -1))).astype("float32")
            target_incidence = (node_ids == target.reshape((1, -1))).astype("float32")
            length = (
                (displacement * displacement).sum(dim=1, keepdim=True).sqrt()
                + 1.0e-9
            )
            direction = displacement / length
            delta = length.reshape((-1,)) - active_rest_length
            edge_force = cfg.k_stretch * delta.reshape((-1, 1)) * direction
            force = (
                source_incidence[:, :, None] * (-edge_force)[None, :, :]
                + target_incidence[:, :, None] * edge_force[None, :, :]
            ).sum(dim=1)
        elif edge_tile >= edge_count:
            source_incidence = (node_ids == source.reshape((1, -1))).astype("float32")
            target_incidence = (node_ids == target.reshape((1, -1))).astype("float32")
            force = bound_spring_stretch_force(
                displacement,
                source_incidence,
                target_incidence,
                active_rest_length,
                cfg.k_stretch,
            )
        else:
            for e0 in range(0, edge_count, edge_tile):
                e1 = min(edge_count, e0 + edge_tile)
                force = force + bound_spring_stretch_force(
                    displacement[e0:e1],
                    (node_ids == source[e0:e1].reshape((1, -1))).astype("float32"),
                    (node_ids == target[e0:e1].reshape((1, -1))).astype("float32"),
                    active_rest_length[e0:e1],
                    cfg.k_stretch,
                )
    # Autotile over repulsion rows: per row about 52 bytes per node live at
    # once (displacement and its product, x3 float32, plus four row planes).
    row_tile = max(1, int(cfg.force_tile_bytes) // max(52 * node_count, 1))
    repulse = cfg.c_repulse and node_count and cfg.repulsion != "none"
    if repulse and cfg.repulsion == "topk":
        if node_count > 1:
            force = force + _topk_repulsion(state, cfg, position, node_count)
    elif repulse and row_tile >= node_count:
        displacement = position[:, None, :] - position[None, :, :]
        distance2 = (displacement * displacement).sum(dim=2) + cfg.eps_rep
        off_diagonal = 1.0 - AT.eye(node_count, dtype="float32")
        network = state.spring_node_network[:node_count].clone().reshape((-1, 1))
        network_rows = AT.broadcast_to(network, (node_count, node_count))
        network_columns = AT.broadcast_to(
            network.T(), (node_count, node_count)
        )
        same_network = (network_rows == network_columns).astype("float32")
        inverse = off_diagonal * same_network / distance2
        force = force + (
            cfg.c_repulse * inverse[:, :, None] * displacement
        ).sum(dim=1)
    elif repulse:
        # The same rows of the same pairwise sum, a tile of rows at a time:
        # the identity and the same-network mask are the full ones' rows.
        network = state.spring_node_network[:node_count].clone().reshape((-1, 1))
        column_ids = AT.arange(node_count, dtype="int64").reshape((1, -1))
        repulsion = []
        for r0 in range(0, node_count, row_tile):
            r1 = min(node_count, r0 + row_tile)
            rows = r1 - r0
            displacement = position[r0:r1][:, None, :] - position[None, :, :]
            distance2 = (displacement * displacement).sum(dim=2) + cfg.eps_rep
            row_ids = AT.arange(r0, r1, dtype="int64").reshape((-1, 1))
            off_diagonal = 1.0 - (row_ids == column_ids).astype("float32")
            network_rows = AT.broadcast_to(network[r0:r1], (rows, node_count))
            network_columns = AT.broadcast_to(network.T(), (rows, node_count))
            same_network = (network_rows == network_columns).astype("float32")
            inverse = off_diagonal * same_network / distance2
            repulsion.append((
                cfg.c_repulse * inverse[:, :, None] * displacement
            ).sum(dim=1))
        force = force + AT.cat(repulsion, dim=0)
    # The host's force layer, added before the integrator and before the
    # causal ceiling, so the dt system admits it like every other force.
    force = force + state.spring_external_force[:node_count]
    proposed_acceleration = force / state.spring_mass[:node_count].clone().reshape((-1, 1))
    _max_force, max_velocity, _max_displacement = _resolved_caps(state, cfg)
    c_abs = cfg.c_frac * max_velocity
    velocity_magnitude = (
        state.spring_velocity[:node_count] * state.spring_velocity[:node_count]
    ).sum(dim=1, keepdim=True).sqrt()
    gamma = 1.0 / (
        1.0 - (velocity_magnitude / max(c_abs, 1.0e-12)) ** 2
    ).clamp(min=1.0e-9).sqrt()
    acceleration = proposed_acceleration / (gamma ** 3)
    return force, acceleration


def _topk_repulsion(state, cfg: BoundSpringParameters, position, node_count: int):
    """Repulsion from each node's ``repulse_k`` nearest same-network nodes.

    The full sum's term, ``c_repulse * d / (|d|^2 + eps)``, over a chosen
    neighbour set instead of every pair.  Selection ranks one N x N plane per
    row tile: within a row ``|p - q|^2 = |p|^2 - 2 (p.q - |q|^2 / 2)`` and
    ``|p|^2`` is the row's constant, so the nearest q are the largest
    ``p.q - |q|^2 / 2`` -- one matmul and one broadcast subtract, no N x N x 3
    displacement.  Self and other-network columns are set to -1e30 by two
    ``where``.  The force is computed from the exact gathered displacements,
    and a pick that is self or another network (a network smaller than
    k + 1) carries zero weight.  Rows are tiled under ``force_tile_bytes`` at
    about 16 bytes per node per row (score, two masked scores, two bool planes).
    """
    k = min(int(cfg.repulse_k), node_count - 1)
    network = state.spring_node_network[:node_count].clone().reshape((-1, 1))
    network_flat = network.reshape((-1,))
    network_columns = network.T()
    column_ids = AT.arange(node_count, dtype="int64").reshape((1, -1))
    half_norm2 = (0.5 * (position * position).sum(dim=1)).reshape((1, -1))
    position_t = position.T()
    row_tile = max(1, int(cfg.force_tile_bytes) // max(16 * node_count, 1))
    repulsion = []
    for r0 in range(0, node_count, row_tile):
        r1 = min(node_count, r0 + row_tile)
        rows = r1 - r0
        row_position = position[r0:r1]
        row_ids = AT.arange(r0, r1, dtype="int64").reshape((-1, 1))
        row_network = network[r0:r1]
        closeness = row_position @ position_t - half_norm2
        # Nested ``where`` rather than ``&``: AbstractTensor's ``__and__``
        # runs value by value in Python (1.2M calls for one 484-node tile).
        candidate = AT.where(
            row_network == network_columns,
            AT.where(row_ids != column_ids, closeness, -1.0e30),
            -1.0e30,
        )
        nearest = AT.get_tensor(AT.topk(candidate, k, dim=1).indices)
        neighbour = nearest.reshape((-1,))
        displacement = (
            row_position[:, None, :]
            - position.index_select(0, neighbour).reshape((rows, k, 3))
        )
        neighbour_network = network_flat.index_select(0, neighbour).reshape((rows, k))
        valid = (
            (neighbour_network == row_network).astype("float32")
            * (nearest != row_ids).astype("float32")
        )
        inverse = valid / ((displacement * displacement).sum(dim=2) + cfg.eps_rep)
        repulsion.append((cfg.c_repulse * inverse[:, :, None] * displacement).sum(dim=1))
    return AT.cat(repulsion, dim=0) if len(repulsion) > 1 else repulsion[0]


def _causal_limit(
    state: ComputationalWorldState,
    cfg: BoundSpringParameters,
    force,
    acceleration,
) -> float:
    node_count = int(state.spring_node_count.item())
    if not node_count:
        return float("inf")
    force_mag = (force * force).sum(dim=1).sqrt()
    acceleration_mag = (acceleration * acceleration).sum(dim=1).sqrt()
    velocity_mag = (
        state.spring_velocity[:node_count] * state.spring_velocity[:node_count]
    ).sum(dim=1).sqrt()
    limits = []
    max_force, max_velocity, max_displacement = _resolved_caps(state, cfg)
    limits.append(float((max_force / (force_mag + 1.0e-9)).min().item()))
    limits.append(float((max_velocity / (acceleration_mag + 1.0e-9)).min().item()))
    limits.append(float((max_displacement / (velocity_mag + 1.0e-9)).min().item()))
    cycle_remaining = cfg.cycle_period - float(state.spring_cycle_time.item())
    if cycle_remaining <= 1.0e-12:
        # The boundary is here: a step of any size from this state crosses it
        # (``advance_bound_spring`` crosses at period - 1e-15), so the ceiling
        # is the boundary after it.  Measured with a 1/60 s frame dividing the
        # period exactly: the old floor of 1e-12 forced a 1e-12 s slice, and
        # the controller then climbed back over 377 slices of 4.4e-5 s.
        cycle_remaining += cfg.cycle_period
    limits.append(cycle_remaining)
    return min(limits)


def advance_bound_spring(
    state: ComputationalWorldState,
    dt: float,
    parameters: BoundSpringParameters,
) -> tuple[bool, Metrics]:
    """Attempt one spring step without internally changing the admitted dt."""

    node_count = int(state.spring_node_count.item())
    edge_count = int(state.spring_edge_count.item())
    if not node_count:
        return True, Metrics(0.0, 0.0, 0.0, 0.0, advanced_dt=float(dt))
    group_count = int(state.spring_group_count.item())
    group = int(state.spring_group_index.item()) % max(group_count, 1)
    cycle_time = float(state.spring_cycle_time.item()) + float(dt)
    crosses_cycle = cycle_time >= parameters.cycle_period - 1.0e-15
    active_group = (
        (group + 1) % group_count
        if crosses_cycle and group_count
        else group
    )
    proposed_rest_length = state.spring_rest_length[:edge_count]
    if group_count:
        level = state.spring_edge_level_mask[active_group, :edge_count].astype("float32")
        typ = state.spring_edge_type_mask[active_group, :edge_count].astype("float32")
        role = state.spring_edge_role_mask[active_group, :edge_count].astype("float32")
        scale = float(dt) / parameters.nominal_dt
        contraction = (
            state.spring_base_length[:edge_count] * (1.0 - parameters.level_target) * level
            + state.spring_base_length[:edge_count] * (1.0 - parameters.type_target) * typ
            + state.spring_base_length[:edge_count] * (1.0 - parameters.role_target) * role
        ) * scale
        relaxed = state.spring_rest_length[:edge_count] - contraction
        relax = _scaled_fraction(
            parameters.relax_rate, dt, parameters.nominal_dt
        )
        proposed_rest_length = relaxed + (
            state.spring_base_length[:edge_count] - relaxed
        ) * relax

    force, acceleration = _forces(
        state, parameters, rest_length=proposed_rest_length
    )
    ceiling = _causal_limit(state, parameters, force, acceleration)
    if float(dt) > ceiling + 1.0e-15:
        return False, Metrics(
            max_vel=float(
                (state.spring_velocity[:node_count] * state.spring_velocity[:node_count])
                .sum(dim=1).sqrt().max().item()
            ),
            max_flux=0.0,
            div_inf=0.0,
            mass_err=0.0,
            dt_limit=ceiling,
            error_channels=AT.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(dt) - ceiling, 0.0, 0.0, 0.0, 0.0]),
            error_present=AT.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            advanced_dt=0.0,
        )
    state.spring_rest_length[:edge_count] = proposed_rest_length

    growth = _scaled_fraction(
        parameters.growth_rate, dt, parameters.nominal_dt
    )
    delta_growth = (
        state.spring_base_length[:edge_count] - state.spring_natural_rest_length[:edge_count]
    )
    grow_mask = (~state.spring_done_growing[:edge_count]) & (delta_growth.abs() > 1.0e-2)
    state.spring_base_length[:edge_count] = state.spring_base_length[:edge_count] - (
        delta_growth * growth * grow_mask.astype("float32")
    )
    state.spring_done_growing[:edge_count] = state.spring_done_growing[:edge_count] | (
        delta_growth.abs() <= 1.0e-2
    )

    velocity = (
        state.spring_velocity[:node_count] + acceleration * float(dt)
    ) * math.exp(-parameters.damping * float(dt))
    position = state.spring_position[:node_count] + velocity * float(dt)

    network_index = state.spring_node_network[:node_count].astype("int64")
    center = state.spring_boundary_center.index_select(0, network_index)
    radial = position - center
    distance = (radial * radial).sum(dim=1, keepdim=True).sqrt()
    radius = state.spring_boundary_radius.index_select(
        0, network_index
    ).reshape((-1, 1))
    if int(radius.shape[0]):
        normal = radial / (distance + 1.0e-9)
        # A surface network is projected every step, both ways, and loses its
        # whole radial velocity; a contained one only when it escapes, and
        # only the outward part.
        on_surface = state.spring_network_surface.index_select(
            0, network_index
        ).astype("float32").reshape((-1, 1))
        escaped = ((distance > radius).astype("float32") + on_surface) > 0.5
        projected = center + normal * radius
        outward = (velocity * normal).sum(dim=1, keepdim=True)
        radial_loss = outward.clamp(min=0.0) * (1.0 - on_surface) + outward * on_surface
        slipped = velocity - radial_loss * normal
        position = AT.where(escaped, projected, position)
        velocity = AT.where(escaped, slipped, velocity)
    state.spring_position[:node_count] = position
    state.spring_velocity[:node_count] = velocity

    if group_count:
        level_n = state.spring_node_level_mask[active_group, :node_count].astype("float32").reshape((-1, 1))
        type_n = state.spring_node_type_mask[active_group, :node_count].astype("float32").reshape((-1, 1))
        role_n = state.spring_node_role_mask[active_group, :node_count].astype("float32").reshape((-1, 1))
        weight = (
            level_n * parameters.beta_level
            + type_n * parameters.beta_type
            + role_n * parameters.beta_role
        )
        alpha_target = parameters.glow_floor_alpha + (
            parameters.glow_peak_alpha - parameters.glow_floor_alpha
        ) * weight
        radius_target = parameters.glow_floor_radius + (
            parameters.glow_peak_radius - parameters.glow_floor_radius
        ) * weight
        rise_alpha = AT.where(
            alpha_target > state.spring_glow_alpha[:node_count],
            _scaled_fraction(parameters.glow_rise, dt, parameters.nominal_dt),
            _scaled_fraction(parameters.glow_decay, dt, parameters.nominal_dt),
        )
        rise_radius = AT.where(
            radius_target > state.spring_glow_radius[:node_count],
            _scaled_fraction(parameters.glow_rise, dt, parameters.nominal_dt),
            _scaled_fraction(parameters.glow_decay, dt, parameters.nominal_dt),
        )
        state.spring_glow_alpha[:node_count] = state.spring_glow_alpha[:node_count] + (
            alpha_target - state.spring_glow_alpha[:node_count]
        ) * rise_alpha
        state.spring_glow_radius[:node_count] = state.spring_glow_radius[:node_count] + (
            radius_target - state.spring_glow_radius[:node_count]
        ) * rise_radius

    if cycle_time >= parameters.cycle_period - 1.0e-15:
        cycle_time = max(0.0, cycle_time - parameters.cycle_period)
        if group_count:
            state.spring_group_index = _tensor([active_group], "int32")
    state.spring_cycle_time = _tensor([cycle_time], "float64")
    max_velocity = float(
        (velocity * velocity).sum(dim=1).sqrt().max().item()
    )
    return True, Metrics(
        max_vel=max_velocity,
        max_flux=0.0,
        div_inf=0.0,
        mass_err=0.0,
        dt_limit=_causal_limit(state, parameters, *_forces(state, parameters)),
        error_channels=AT.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        error_present=AT.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        advanced_dt=float(dt),
    )


__all__ = [
    "BoundSpringParameters",
    "advance_bound_spring",
    "append_bound_spring",
    "bound_spring_stretch_force",
    "install_bound_spring",
]
