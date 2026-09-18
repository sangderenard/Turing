"""Vectorized Python/AbstractTensor balloon-tire assembly authority.

Batch, wheel, vertex, face, face-corner, contact-surface, and XYZ are tensor
axes.  Fixed incidence and Laplacian matrices replace the handwritten scalar
C loops.  Native C and web shaders are downstream compiler products.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np

from src.common.tensors.riemann.mesh_laplace import build_cotangent_geometry

from .abstract_ui_vehicles import WHEEL_NAMES, load_default_car_configuration
from .vehicle_balloon_tire import balloon_tire_graph_abi


MAX_PLANES_PER_WHEEL = 2

#: Same four ring stations as vehicle_tire_ring_model.py's Pappus volume law
#: and vehicle_tire_reduced_contact_law.py's fallback-spring law -- one
#: shared geometry, not independently tuned per law.
_STATIONS = ("bead_inboard", "shoulder_inboard", "shoulder_outboard", "bead_outboard")


BALLOON_TIRE_VECTOR_SOURCE = '''
def vector_cross(left, right):
    return AbstractTensor.stack([
        left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
        left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],
        left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0],
    ], dim=-1)


def vector_norm(value, epsilon):
    return ((value * value).sum(dim=-1) + epsilon).sqrt()


def balloon_tire_vector_initialize(inputs, state, wheel_input_indices, rest):
    batch_count = inputs.shape[0]
    wheel_count = wheel_input_indices.shape[0]
    wheel_input = inputs.gather(wheel_input_indices.reshape((-1,)), dim=1).reshape((batch_count, wheel_count, 41))
    basis = wheel_input[:, :, 6:15].reshape((batch_count, wheel_count, 3, 3))
    hub = wheel_input[:, :, 0:3]
    hub_velocity = wheel_input[:, :, 3:6]
    angle = wheel_input[:, :, 18]
    cosine = angle.cos().reshape((batch_count, wheel_count, 1))
    sine = angle.sin().reshape((batch_count, wheel_count, 1))
    local = rest.reshape((1, 1, -1, 3))
    rotated_local = AbstractTensor.stack([
        cosine * local[:, :, :, 0] - sine * local[:, :, :, 1],
        sine * local[:, :, :, 0] + cosine * local[:, :, :, 1],
        local[:, :, :, 2] * AbstractTensor.ones_like(cosine),
    ], dim=-1)
    radius = AbstractTensor.matmul(rotated_local, basis)
    omega = wheel_input[:, :, 15:18] + wheel_input[:, :, 19].reshape((batch_count, wheel_count, 1)) * basis[:, :, 2, :]
    state[:, :, :, 0:3] = hub.reshape((batch_count, wheel_count, 1, 3)) + radius
    state[:, :, :, 3:6] = hub_velocity.reshape((batch_count, wheel_count, 1, 3)) + vector_cross(omega.reshape((batch_count, wheel_count, 1, 3)), radius)
    return state


def balloon_tire_reduced_vector_step(inputs, state, output, wheel_input_indices, rest,
                                     face_vertices, face_rest, face_scatter,
                                     bending_incidence, bending_scatter,
                                     bending_weight, vertex_area, bead_mask,
                                     face_material):
    """The reduced/wrench/wrench-per-vertex tire fidelity modes (1/2/3),
    called by the host INSTEAD OF balloon_tire_vector_step -- not a branch
    inside it.  fine (mode 0) is training-data-only; deployed/live rig work
    uses one of these.  Never runs the membrane, bending, or bead-implicit
    solve (the actually expensive part per step) -- the tire mesh is purely
    kinematic (rigidly follows the hub, exactly like balloon_tire_vector_
    initialize), and rim_force/rim_moment come directly from whichever
    reduced contact law tire_fidelity_mode selects.  Same signature as
    balloon_tire_vector_step (drop-in host dispatch) even though the mesh
    geometry/material arguments are unused here -- extend never shrink.
    A trimmed-signature variant was tried and reverted: it did not fix the
    (4,4,144,144)x(4,1,144,3) shape-mismatch this hits during standalone
    native-shim lowering (docs/PLAN_TIRE_FIDELITY_LADDER.md) -- the same
    error appeared one stage earlier instead, so unused arguments were not
    the actual cause. Root cause still open.
    """

    batch_count = inputs.shape[0]
    wheel_count = wheel_input_indices.shape[0]
    wheel_input = inputs.gather(wheel_input_indices.reshape((-1,)), dim=1).reshape((batch_count, wheel_count, 41))
    hub = wheel_input[:, :, 0:3]
    hub_velocity = wheel_input[:, :, 3:6]
    basis = wheel_input[:, :, 6:15].reshape((batch_count, wheel_count, 3, 3))
    angle = wheel_input[:, :, 18]
    cosine = angle.cos().reshape((batch_count, wheel_count, 1))
    sine = angle.sin().reshape((batch_count, wheel_count, 1))
    local = rest.reshape((1, 1, -1, 3))
    rotated_local = AbstractTensor.stack([
        cosine * local[:, :, :, 0] - sine * local[:, :, :, 1],
        sine * local[:, :, :, 0] + cosine * local[:, :, :, 1],
        local[:, :, :, 2] * AbstractTensor.ones_like(cosine),
    ], dim=-1)
    reference = hub.reshape((batch_count, wheel_count, 1, 3)) + AbstractTensor.matmul(rotated_local, basis)
    total_omega = wheel_input[:, :, 15:18] + wheel_input[:, :, 19].reshape((batch_count, wheel_count, 1)) * basis[:, :, 2, :]
    velocity = hub_velocity.reshape((batch_count, wheel_count, 1, 3)) + vector_cross(
        total_omega.reshape((batch_count, wheel_count, 1, 3)), reference - hub.reshape((batch_count, wheel_count, 1, 3)))
    position = reference
    state[:, :, :, 0:3] = position
    state[:, :, :, 3:6] = velocity

    batch_offset = AbstractTensor.arange(position.shape[0]).reshape((-1, 1, 1, 1)) * (wheel_count * position.shape[2])
    wheel_vertex_offset = AbstractTensor.arange(wheel_count).reshape((1, wheel_count, 1, 1)) * position.shape[2]
    face_index = batch_offset + wheel_vertex_offset + face_vertices.reshape((1, 1, -1, 3))
    face_position = position.reshape((-1, 3)).gather(face_index.reshape((-1,)), dim=0).reshape((batch_count, wheel_count, face_vertices.shape[0], 3, 3))
    volume = (face_position[:, :, :, 0, :] * vector_cross(
        face_position[:, :, :, 1, :], face_position[:, :, :, 2, :])).sum(dim=-1).sum(dim=2) / 6.0
    gas_result = balloon_tire_gas(
        volume, inputs[:, 7].reshape((-1, 1)), inputs[:, 12].reshape((-1, 1)),
        (inputs[:, 3] * inputs[:, 4].maximum(0.0)).reshape((-1, 1)),
        inputs[:, 6].reshape((-1, 1)), inputs[:, 5].reshape((-1, 1)))
    gas_pressure = gas_result[0]
    volume_ratio = gas_result[1]
    gas_temperature = gas_result[2]

    fidelity_mode = inputs[:, 24].reshape((-1, 1))
    station_r = [inputs[:, 25].reshape((-1, 1)), inputs[:, 26].reshape((-1, 1)),
                inputs[:, 27].reshape((-1, 1)), inputs[:, 28].reshape((-1, 1))]
    station_z = [inputs[:, 29].reshape((-1, 1)), inputs[:, 30].reshape((-1, 1)),
                inputs[:, 31].reshape((-1, 1)), inputs[:, 32].reshape((-1, 1))]
    ground_y = wheel_input[:, :, 24]
    roller_engaged = (wheel_input[:, :, 22] > 0.5).to_dtype("float64")
    depth = (hub[:, :, 1] - ground_y).reshape((batch_count, wheel_count))
    _gauss5_nodes = (0.0, 0.5384693101056831, -0.5384693101056831,
                     0.9061798459386640, -0.9061798459386640)
    _gauss5_weights = (0.5688888888888889, 0.4786286704993665, 0.4786286704993665,
                       0.2369268850561891, 0.2369268850561891)
    contact_patch_area = depth * 0.0
    for _segment in range(3):
        r0, r1 = station_r[_segment], station_r[_segment + 1]
        z0, z1 = station_z[_segment], station_z[_segment + 1]
        half_length = ((z1 - z0) / 2.0).reshape((batch_count, 1))
        segment_area = depth * 0.0
        for _node, _weight in zip(_gauss5_nodes, _gauss5_weights):
            _t = (_node + 1.0) / 2.0
            radius_here = (r0 + _t * (r1 - r0)).reshape((batch_count, 1))
            chord_width = 2.0 * (radius_here * radius_here - depth * depth).maximum(0.0).sqrt()
            segment_area = segment_area + _weight * chord_width
        contact_patch_area = contact_patch_area + segment_area * half_length.abs()
    bead_selector = bead_mask.reshape((1, 1, -1, 1)).to_dtype("float64")
    bead_vertex_count = bead_mask.reshape((1, -1)).to_dtype("float64").sum(dim=1).maximum(1.0)
    bead_mean_velocity_y = (velocity[:, :, :, 1] * bead_selector[:, :, :, 0]).sum(dim=2) / bead_vertex_count
    reduced_damping_force_y = inputs[:, 20].reshape((-1, 1)) * bead_mean_velocity_y
    reduced_force_y_raw = ((gas_pressure * contact_patch_area - reduced_damping_force_y)
                           * roller_engaged).maximum(0.0)
    mode1_gate = ((fidelity_mode - 1.0).abs() < 0.5).to_dtype("float64")
    mode2_gate = ((fidelity_mode - 2.0).abs() < 0.5).to_dtype("float64")
    mode3_gate = ((fidelity_mode - 3.0).abs() < 0.5).to_dtype("float64")
    reduced_force_y = reduced_force_y_raw * mode1_gate.reshape((batch_count, 1))
    rim_force = AbstractTensor.stack([
        reduced_force_y * 0.0, reduced_force_y, reduced_force_y * 0.0], dim=-1)
    rim_moment = rim_force * 0.0

    shoulder_r = (0.5 * (station_r[1] + station_r[2])).reshape((batch_count, 1, 1))
    surface_kind_q = wheel_input[:, :, 20].reshape((batch_count, wheel_count, 1))
    cylinder_radius_q = wheel_input[:, :, 21].reshape((batch_count, wheel_count, 1))
    plane_point_q = wheel_input[:, :, 23:26].reshape((batch_count, wheel_count, 1, 3))
    plane_normal_q = wheel_input[:, :, 26:29].reshape((batch_count, wheel_count, 1, 3))
    wrench_k = inputs[:, 33].reshape((batch_count, 1, 1))
    wrench_c = inputs[:, 34].reshape((batch_count, 1, 1))

    def _wrench_force(query_position, query_velocity):
        to_plane = ((query_position - plane_point_q) * plane_normal_q).sum(dim=-1)
        radial = query_position[:, :, :, 0:2] - plane_point_q[:, :, :, 0:2]
        radial_length = ((radial * radial).sum(dim=-1) + 1.0e-24).sqrt()
        to_cylinder = radial_length - cylinder_radius_q
        cylinder_direction = AbstractTensor.stack([
            radial[:, :, :, 0] / radial_length, radial[:, :, :, 1] / radial_length,
            radial_length * 0.0], dim=-1)
        is_cylinder = (surface_kind_q >= 0.5).reshape(surface_kind_q.shape[:3] + (1,))
        surface_distance = AbstractTensor.where(surface_kind_q >= 0.5, to_cylinder, to_plane)
        direction = AbstractTensor.where(is_cylinder, cylinder_direction,
                                         plane_normal_q * AbstractTensor.ones_like(cylinder_direction))
        penetration = (shoulder_r - surface_distance).maximum(0.0)
        closing_velocity = (query_velocity * direction).sum(dim=-1)
        magnitude = (wrench_k.reshape(wrench_k.shape[:2] + (1,)) * penetration
                    - wrench_c.reshape(wrench_c.shape[:2] + (1,)) * closing_velocity).maximum(0.0)
        engaged = roller_engaged.reshape(roller_engaged.shape + (1,)) * (penetration > 0.0).to_dtype("float64")
        return magnitude.reshape(magnitude.shape + (1,)) * engaged.reshape(engaged.shape + (1,)) * direction

    hub_query_position = hub.reshape((batch_count, wheel_count, 1, 3))
    hub_query_velocity = hub_velocity.reshape((batch_count, wheel_count, 1, 3))
    wrench_force_hub = _wrench_force(hub_query_position, hub_query_velocity)[:, :, 0, :]
    rim_force = rim_force + wrench_force_hub * mode2_gate.reshape((batch_count, 1, 1))

    wrench_force_per_vertex = _wrench_force(position, velocity) * bead_selector
    arm = position - hub.reshape((batch_count, wheel_count, 1, 3))
    moment_per_vertex = AbstractTensor.stack([
        arm[..., 1] * wrench_force_per_vertex[..., 2] - arm[..., 2] * wrench_force_per_vertex[..., 1],
        arm[..., 2] * wrench_force_per_vertex[..., 0] - arm[..., 0] * wrench_force_per_vertex[..., 2],
        arm[..., 0] * wrench_force_per_vertex[..., 1] - arm[..., 1] * wrench_force_per_vertex[..., 0],
    ], dim=-1)
    rim_force = rim_force + wrench_force_per_vertex.sum(dim=2) * mode3_gate.reshape((batch_count, 1, 1))
    rim_moment = rim_moment + moment_per_vertex.sum(dim=2) * mode3_gate.reshape((batch_count, 1, 1))

    output[:, :, 0:3], output[:, :, 3:6] = rim_force, rim_moment
    output[:, :, 6], output[:, :, 7], output[:, :, 8] = gas_pressure, volume_ratio, gas_temperature
    output[:, :, 9] = roller_engaged * (mode1_gate + mode2_gate + mode3_gate).reshape((batch_count, 1))
    output[:, :, 10] = position[:, :, :, 1].min(dim=2)
    output[:, :, 11], output[:, :, 12], output[:, :, 13] = (
        output[:, :, 11] * 0.0, output[:, :, 12] * 0.0, output[:, :, 13] * 0.0)
    return state, output


def balloon_tire_reduced_microstep_loop(inputs, state, output, wheel_input_indices, rest,
                                        face_vertices, face_rest, face_scatter,
                                        bending_incidence, bending_scatter,
                                        bending_weight, vertex_area, bead_mask,
                                        face_material, repeat_count):
    """The native-shim entry point: one Python/eager dispatch runs
    ``repeat_count`` reduced-mode microsteps internally, instead of the host
    paying full eager AbstractTensor call overhead once per microstep (see
    docs/PLAN_TIRE_FIDELITY_LADDER.md's native-shim-microstepping section).
    ``repeat_count`` is a genuine Python int (like ``vehicle_tire_recurrence``'s
    ``microstep_count``), not a tensor -- the loop trip count is fixed at
    lowering/specialization time, exactly the existing precedent.
    """

    for _ in range(repeat_count):
        state, output = balloon_tire_reduced_vector_step(
            inputs, state, output, wheel_input_indices, rest,
            face_vertices, face_rest, face_scatter,
            bending_incidence, bending_scatter,
            bending_weight, vertex_area, bead_mask, face_material)
    return state, output


def balloon_tire_vector_step(inputs, state, output, wheel_input_indices, rest,
                             face_vertices, face_rest, face_scatter,
                             bending_incidence, bending_scatter,
                             bending_weight, vertex_area, bead_mask,
                             face_material):
    batch_count = inputs.shape[0]
    wheel_count = wheel_input_indices.shape[0]
    wheel_input = inputs.gather(wheel_input_indices.reshape((-1,)), dim=1).reshape((batch_count, wheel_count, 41))
    dt_wheel = inputs[:, 0].reshape((-1, 1))
    dt_vertex = inputs[:, 0].reshape((-1, 1, 1))
    dt_vector = inputs[:, 0].reshape((-1, 1, 1, 1))
    dt_plane = inputs[:, 0].reshape((-1, 1, 1, 1, 1))
    gravity_y = inputs[:, 1].reshape((-1, 1, 1))
    mass = inputs[:, 2].reshape((-1, 1, 1))
    position = state[:, :, :, 0:3].reshape((batch_count, wheel_count, rest.shape[0], 3))
    velocity = state[:, :, :, 3:6].reshape((batch_count, wheel_count, rest.shape[0], 3))
    hub = wheel_input[:, :, 0:3]
    hub_velocity = wheel_input[:, :, 3:6]
    basis = wheel_input[:, :, 6:15].reshape((batch_count, wheel_count, 3, 3))
    angle = wheel_input[:, :, 18]
    cosine = angle.cos().reshape((batch_count, wheel_count, 1))
    sine = angle.sin().reshape((batch_count, wheel_count, 1))
    local = rest.reshape((1, 1, -1, 3))
    rotated_local = AbstractTensor.stack([
        cosine * local[:, :, :, 0] - sine * local[:, :, :, 1],
        sine * local[:, :, :, 0] + cosine * local[:, :, :, 1],
        local[:, :, :, 2] * AbstractTensor.ones_like(cosine),
    ], dim=-1)
    reference = hub.reshape((batch_count, wheel_count, 1, 3)) + AbstractTensor.matmul(rotated_local, basis)
    total_omega = wheel_input[:, :, 15:18] + wheel_input[:, :, 19].reshape((batch_count, wheel_count, 1)) * basis[:, :, 2, :]
    target_velocity = hub_velocity.reshape((batch_count, wheel_count, 1, 3)) + vector_cross(
        total_omega.reshape((batch_count, wheel_count, 1, 3)), reference - hub.reshape((batch_count, wheel_count, 1, 3)))

    batch_offset = AbstractTensor.arange(position.shape[0]).reshape((-1, 1, 1, 1)) * (wheel_count * position.shape[2])
    wheel_vertex_offset = AbstractTensor.arange(wheel_count).reshape((1, wheel_count, 1, 1)) * position.shape[2]
    face_index = batch_offset + wheel_vertex_offset + face_vertices.reshape((1, 1, -1, 3))
    face_position = position.reshape((-1, 3)).gather(face_index.reshape((-1,)), dim=0).reshape((batch_count, wheel_count, face_vertices.shape[0], 3, 3))
    face_velocity = velocity.reshape((-1, 3)).gather(face_index.reshape((-1,)), dim=0).reshape((batch_count, wheel_count, face_vertices.shape[0], 3, 3))
    face_reference = reference.reshape((-1, 3)).gather(face_index.reshape((-1,)), dim=0).reshape((batch_count, wheel_count, face_vertices.shape[0], 3, 3))
    volume = (face_position[:, :, :, 0, :] * vector_cross(
        face_position[:, :, :, 1, :], face_position[:, :, :, 2, :])).sum(dim=-1).sum(dim=2) / 6.0
    gas_result = balloon_tire_gas(
        volume, inputs[:, 7].reshape((-1, 1)), inputs[:, 12].reshape((-1, 1)),
        (inputs[:, 3] * inputs[:, 4].maximum(0.0)).reshape((-1, 1)),
        inputs[:, 6].reshape((-1, 1)), inputs[:, 5].reshape((-1, 1)))
    gas_pressure = gas_result[0]
    volume_ratio = gas_result[1]
    gas_temperature = gas_result[2]
    membrane = balloon_tire_membrane_face(
        gas_pressure.reshape((batch_count, wheel_count, 1)), face_material[:, 1].reshape((1, 1, -1)),
        face_material[:, 2].reshape((1, 1, -1)),
        face_material[:, 3].reshape((1, 1, -1)),
        face_material[:, 4].reshape((1, 1, -1)),
        face_rest[:, 5].reshape((1, 1, -1)),
        face_rest[:, 6].reshape((1, 1, -1)),
        face_rest[:, 7].reshape((1, 1, -1)),
        face_material[:, 5].reshape((1, 1, -1)),
        face_material[:, 6].reshape((1, 1, -1)),
        face_material[:, 7].reshape((1, 1, -1)),
        face_material[:, 8].reshape((1, 1, -1)),
        face_material[:, 9].reshape((1, 1, -1)),
        face_material[:, 10].reshape((1, 1, -1)),
        face_reference[:, :, :, 0, 0], face_reference[:, :, :, 0, 1], face_reference[:, :, :, 0, 2],
        face_reference[:, :, :, 1, 0], face_reference[:, :, :, 1, 1], face_reference[:, :, :, 1, 2],
        face_reference[:, :, :, 2, 0], face_reference[:, :, :, 2, 1], face_reference[:, :, :, 2, 2],
        # Construction prestress at the CURRENT gas charge, the same reference
        # the gas law is given.  The authored topology is the molded shape;
        # the carcass tension that holds it is whatever balances the gas that
        # is actually in it, so pressure and construction cancel face-for-face
        # at that shape at any charge.  With the prestress pinned to the rated
        # pressure, a tyre started at ambient charge (0.75 of rated) received
        # a 34 kPa net inward load on every face in one step at t=0: the whole
        # skin reached 7.8 m/s in the first 244 us, the sidewalls caved, the
        # volume collapsed and the gas law read 500 kPa.  Inflating now raises
        # pressure and stiffness without a shape jump.
        (inputs[:, 3] * inputs[:, 4].maximum(0.0)).reshape((-1, 1, 1)), face_rest[:, 4], face_rest[:, 0], face_rest[:, 1], face_rest[:, 2], face_rest[:, 3], face_material[:, 0].reshape((1, 1, -1)),
        face_velocity[:, :, :, 0, 0], face_velocity[:, :, :, 0, 1], face_velocity[:, :, :, 0, 2],
        face_velocity[:, :, :, 1, 0], face_velocity[:, :, :, 1, 1], face_velocity[:, :, :, 1, 2],
        face_velocity[:, :, :, 2, 0], face_velocity[:, :, :, 2, 1], face_velocity[:, :, :, 2, 2],
        face_position[:, :, :, 0, 0], face_position[:, :, :, 0, 1], face_position[:, :, :, 0, 2],
        face_position[:, :, :, 1, 0], face_position[:, :, :, 1, 1], face_position[:, :, :, 1, 2],
        face_position[:, :, :, 2, 0], face_position[:, :, :, 2, 1], face_position[:, :, :, 2, 2])
    total_face_force = AbstractTensor.stack([
        membrane[5], membrane[10], membrane[15], membrane[20], membrane[25],
        membrane[30], membrane[35], membrane[40], membrane[45]], dim=-1).reshape((batch_count, wheel_count, face_vertices.shape[0], 3, 3))
    damping_face_force = AbstractTensor.stack([
        membrane[7], membrane[12], membrane[17], membrane[22], membrane[27],
        membrane[32], membrane[37], membrane[42], membrane[47]], dim=-1).reshape((batch_count, wheel_count, face_vertices.shape[0], 3, 3))
    elastic_face_force = total_face_force - damping_face_force
    force = (AbstractTensor.matmul(face_scatter[0], elastic_face_force[:, :, :, 0, :])
             + AbstractTensor.matmul(face_scatter[1], elastic_face_force[:, :, :, 1, :])
             + AbstractTensor.matmul(face_scatter[2], elastic_face_force[:, :, :, 2, :]))
    damping_force = (AbstractTensor.matmul(face_scatter[0], damping_face_force[:, :, :, 0, :])
                     + AbstractTensor.matmul(face_scatter[1], damping_face_force[:, :, :, 1, :])
                     + AbstractTensor.matmul(face_scatter[2], damping_face_force[:, :, :, 2, :]))
    relative_velocity = velocity - hub_velocity.reshape((batch_count, wheel_count, 1, 3))
    damping_power = (relative_velocity * damping_force).sum(dim=3).sum(dim=2)
    damping_quadratic = (damping_force * damping_force).sum(dim=3).sum(dim=2) / mass.reshape((-1, 1))
    damping_scale = (-1.9 * damping_power / (dt_wheel * damping_quadratic + 1.0e-24)).maximum(0.0).minimum(1.0)
    force = force + damping_scale.reshape((batch_count, wheel_count, 1, 1)) * damping_force
    displacement = position - reference
    # Preserve the original sparse cotangent-shell evaluation exactly.  The
    # incidence product forms edge differences before any accumulation, so a
    # rigid translation cancels by construction rather than by a dense
    # row-sum-zero dot product.  The transpose/scatter then reconstructs the
    # symmetric cotangent numerator Q used by the retained native assembly.
    edge_displacement = AbstractTensor.matmul(
        bending_incidence, displacement)
    weighted_edge_displacement = (
        bending_weight.reshape((-1, 1)) * edge_displacement)
    laplace_numerator = -AbstractTensor.matmul(
        bending_scatter, weighted_edge_displacement)
    bending_dual = (
        inputs[:, 19].reshape((-1, 1, 1, 1))
        * laplace_numerator
        / vertex_area.reshape((1, 1, -1, 1))
    )
    edge_bending_dual = AbstractTensor.matmul(
        bending_incidence, bending_dual)
    force = force + AbstractTensor.matmul(
        bending_scatter,
        bending_weight.reshape((-1, 1)) * edge_bending_dual)
    bending_energy = 0.5 * (laplace_numerator * bending_dual).sum(dim=3).sum(dim=2)
    gravity = AbstractTensor.stack([position[:, :, :, 0] * 0.0,
        position[:, :, :, 1] * 0.0 + mass * gravity_y, position[:, :, :, 2] * 0.0], dim=-1)
    free_velocity = velocity + dt_vector * (force + gravity) / mass.reshape((-1, 1, 1, 1))

    bead = balloon_tire_bead_implicit_step(
        inputs[:, 20].reshape((-1, 1, 1)), inputs[:, 18].reshape((-1, 1, 1)), dt_vertex,
        free_velocity[:, :, :, 0], free_velocity[:, :, :, 1], free_velocity[:, :, :, 2],
        hub[:, :, 0].reshape((batch_count, wheel_count, 1)), hub[:, :, 1].reshape((batch_count, wheel_count, 1)), hub[:, :, 2].reshape((batch_count, wheel_count, 1)),
        target_velocity[:, :, :, 0], target_velocity[:, :, :, 1], target_velocity[:, :, :, 2],
        reference[:, :, :, 0], reference[:, :, :, 1], reference[:, :, :, 2], mass,
        position[:, :, :, 0], position[:, :, :, 1], position[:, :, :, 2])
    bead_next_position = AbstractTensor.stack([bead[0], bead[4], bead[8]], dim=-1)
    bead_next_velocity = AbstractTensor.stack([bead[1], bead[5], bead[9]], dim=-1)
    bead_selector = bead_mask.reshape((1, 1, -1, 1))
    predicted = AbstractTensor.where(bead_selector, bead_next_position, position + dt_vector * free_velocity)
    next_velocity = AbstractTensor.where(bead_selector, bead_next_velocity, free_velocity)
    rim_force = AbstractTensor.stack([bead[2], bead[6], bead[10]], dim=-1) * bead_selector
    rim_moment = AbstractTensor.stack([bead[3], bead[7], bead[11]], dim=-1) * bead_selector

    plane = wheel_input[:, :, 23:41].reshape((batch_count, wheel_count, 2, 3, 3))
    plane_point = plane[:, :, :, 0, :]
    plane_normal = plane[:, :, :, 1, :]
    plane_velocity = plane[:, :, :, 2, :]
    previous = position.reshape((batch_count, wheel_count, position.shape[2], 1, 3)) + dt_plane * plane_velocity.reshape((batch_count, wheel_count, 1, 2, 3))
    current = predicted.reshape((batch_count, wheel_count, predicted.shape[2], 1, 3))
    normal = plane_normal.reshape((batch_count, wheel_count, 1, 2, 3))
    point = plane_point.reshape((batch_count, wheel_count, 1, 2, 3))
    skin_offset = inputs[:, 21].reshape((-1, 1, 1, 1))
    previous_distance = ((previous - point) * normal).sum(dim=-1) - skin_offset
    current_distance = ((current - point) * normal).sum(dim=-1) - skin_offset
    toi = (previous_distance / (previous_distance - current_distance + 1.0e-18)).maximum(0.0).minimum(1.0)
    contact_position = previous + toi.reshape((batch_count, wheel_count, toi.shape[2], 2, 1)) * (current - previous) - skin_offset.reshape((-1, 1, 1, 1, 1)) * normal
    surface_kind = wheel_input[:, :, 20].reshape((batch_count, wheel_count, 1, 1)) >= 0.5
    cylinder_radius = wheel_input[:, :, 21].reshape((batch_count, wheel_count, 1, 1))
    delta = current - previous
    qx = previous[:, :, :, :, 0] - point[:, :, :, :, 0]
    qy = previous[:, :, :, :, 1] - point[:, :, :, :, 1]
    inflated_radius = cylinder_radius + skin_offset
    qa = delta[:, :, :, :, 0] ** 2 + delta[:, :, :, :, 1] ** 2 + 1.0e-30
    qb = 2.0 * (qx * delta[:, :, :, :, 0] + qy * delta[:, :, :, :, 1])
    qc = qx ** 2 + qy ** 2 - inflated_radius ** 2
    discriminant = (qb ** 2 - 4.0 * qa * qc).maximum(0.0)
    cylinder_toi = ((-qb - discriminant.sqrt()) / (2.0 * qa)).maximum(0.0).minimum(1.0)
    swept = previous + cylinder_toi.reshape(
        (batch_count, wheel_count, cylinder_toi.shape[2], 2, 1)) * delta
    radial = swept[:, :, :, :, 0:2] - point[:, :, :, :, 0:2]
    radial_length = vector_norm(radial, 1.0e-30)
    cylinder_normal = AbstractTensor.stack([
        radial[:, :, :, :, 0] / radial_length,
        radial[:, :, :, :, 1] / radial_length,
        radial_length * 0.0], dim=-1)
    cylinder_previous_distance = (qx * qx + qy * qy + 1.0e-30).sqrt() - inflated_radius
    cylinder_current_delta = current[:, :, :, :, 0:2] - point[:, :, :, :, 0:2]
    cylinder_current_distance = vector_norm(cylinder_current_delta, 1.0e-30) - inflated_radius
    cylinder_contact = AbstractTensor.stack([
        point[:, :, :, :, 0] + cylinder_radius * cylinder_normal[:, :, :, :, 0],
        point[:, :, :, :, 1] + cylinder_radius * cylinder_normal[:, :, :, :, 1],
        swept[:, :, :, :, 2]], dim=-1)
    toi = AbstractTensor.where(surface_kind, cylinder_toi, toi)
    previous_distance = AbstractTensor.where(surface_kind, cylinder_previous_distance, previous_distance)
    current_distance = AbstractTensor.where(surface_kind, cylinder_current_distance, current_distance)
    surface_selector = surface_kind.reshape((batch_count, wheel_count, 1, 1, 1))
    normal = AbstractTensor.where(surface_selector, cylinder_normal, normal)
    contact_position = AbstractTensor.where(
        surface_selector, cylinder_contact, contact_position)
    plane_enabled = AbstractTensor.arange(2).reshape((1, 1, 1, 2)) < wheel_input[:, :, 22].reshape((batch_count, wheel_count, 1, 1))
    contact_active = plane_enabled * (current_distance <= 0.0)
    relative_contact_velocity = next_velocity.reshape((batch_count, wheel_count, next_velocity.shape[2], 1, 3)) - plane_velocity.reshape((batch_count, wheel_count, 1, 2, 3))
    normal_velocity = (relative_contact_velocity * normal).sum(dim=-1)
    normal_impulse = contact_active * (-(1.0 + inputs[:, 22].reshape((-1, 1, 1, 1))) * normal_velocity.minimum(0.0)) * mass.reshape((-1, 1, 1, 1))
    tangent_velocity = relative_contact_velocity - normal_velocity.reshape((batch_count, wheel_count, normal_velocity.shape[2], 2, 1)) * normal
    tangent_speed = vector_norm(tangent_velocity, 1.0e-24)
    tangent_impulse = (tangent_speed * mass.reshape((-1, 1, 1, 1))).minimum(inputs[:, 23].reshape((-1, 1, 1, 1)) * normal_impulse)
    impulse = (
        normal_impulse.reshape(
            (batch_count, wheel_count, normal_impulse.shape[2], 2, 1)) * normal
        - tangent_impulse.reshape(
            (batch_count, wheel_count, tangent_impulse.shape[2], 2, 1))
        * tangent_velocity
        / tangent_speed.reshape(
            (batch_count, wheel_count, tangent_speed.shape[2], 2, 1)))
    next_velocity = next_velocity + impulse.sum(dim=3) / mass.reshape(
        (-1, 1, 1, 1))
    # A disabled surface is absent, matching the original finite plane loop;
    # it must not win the minimum merely because its zero-filled geometry has
    # a non-positive signed distance.  Choose exactly one enabled slot on a
    # tie as well, otherwise summing two equal contact positions doubles the
    # vertex coordinate.  This is assembly selection, not a contact law.
    eligible_distance = AbstractTensor.where(
        plane_enabled, current_distance, current_distance * 0.0 + 1.0e300)
    selection_distance = (
        eligible_distance
        + AbstractTensor.arange(2).reshape((1, 1, 1, 2)) * 1.0e-12
    )
    selection_minimum = selection_distance.min(dim=3)
    deepest = eligible_distance.min(dim=3)
    selected = (
        selection_distance
        == selection_minimum.reshape((
            selection_distance.shape[0],
            selection_distance.shape[1],
            selection_distance.shape[2],
            1,
        ))
    ).reshape((batch_count, wheel_count, current_distance.shape[2], 2, 1))
    selected_contact = (contact_position * selected).sum(dim=3)
    selected_normal = (normal * selected).sum(dim=3)
    selected_toi = (toi * selected[:, :, :, :, 0]).sum(dim=3)
    corrected = (
        selected_contact
        + inputs[:, 21].reshape((-1, 1, 1, 1)) * selected_normal
        + (1.0 - selected_toi.reshape(
            (batch_count, wheel_count, selected_toi.shape[2], 1)))
        * dt_vector * next_velocity)
    next_position = AbstractTensor.where(
        (deepest <= 0.0).reshape((batch_count, wheel_count, deepest.shape[2], 1)),
        corrected,
        predicted)
    state[:, :, :, 0:3], state[:, :, :, 3:6] = next_position, next_velocity
    output[:, :, 0:3], output[:, :, 3:6] = rim_force.sum(dim=2), rim_moment.sum(dim=2)
    output[:, :, 6], output[:, :, 7], output[:, :, 8] = gas_pressure, volume_ratio, gas_temperature
    output[:, :, 9] = contact_active.sum(dim=3).sum(dim=2)
    output[:, :, 10] = next_position[:, :, :, 1].min(dim=2)
    output[:, :, 11], output[:, :, 12], output[:, :, 13] = membrane[0].sum(dim=2), membrane[1].sum(dim=2), bending_energy
    return state, output
'''


@dataclass(frozen=True, slots=True)
class BalloonTirePythonProgram:
    source: str
    entrypoint: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    state_scalar_count: int
    vertex_count: int
    face_count: int
    constants: Mapping[str, np.ndarray]
    face_zones: tuple[str, ...]
    pneumatic_mode: str
    material_profile: str


def _face_material_basis(
    rest: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """Angle from each triangle's first tangent to tire circumference."""

    angles = np.zeros((len(faces),), dtype=np.float64)
    for index, (ia, ib, ic) in enumerate(faces):
        a, b, c = rest[ia], rest[ib], rest[ic]
        tangent_a = b - a
        tangent_a /= max(1.0e-15, float(np.linalg.norm(tangent_a)))
        normal = np.cross(b - a, c - a)
        normal /= max(1.0e-15, float(np.linalg.norm(normal)))
        tangent_b = np.cross(normal, tangent_a)
        centroid = (a + b + c) / 3.0
        circumferential = np.asarray((-centroid[1], centroid[0], 0.0))
        circumferential -= normal * float(np.dot(circumferential, normal))
        circumferential /= max(
            1.0e-15, float(np.linalg.norm(circumferential)))
        angles[index] = np.arctan2(
            float(np.dot(circumferential, tangent_b)),
            float(np.dot(circumferential, tangent_a)))
    return angles


def _oriented_face_stiffness(
    abi: Mapping[str, object], material_profile: str,
) -> np.ndarray:
    """Integrate oriented reinforcement into a per-face tangent stiffness.

    Columns follow compiled argument order Q11,Q12,Q16,Q22,Q26,Q66 in each triangle's reference
    tangent basis.  Rubber remains in the isotropic Lamé terms; these entries
    are the directional cord/steel contribution.
    """

    topology = abi["topology"]
    stacks = abi["layer_stacks"]
    result = np.zeros((len(topology.faces), 6), dtype=np.float64)
    directional_modulus = {"composite-cord": 115_000_000.0,
                           "steel": 1_250_000_000.0}
    for face, zone in enumerate(topology.face_zones):
        if zone == "rim-closure":
            continue
        layers = [layer for layer in stacks[zone]
                  if layer["material"] in directional_modulus]
        if material_profile == "cheap-commercial-retread":
            # Reused radial carcass plus economical replacement cap.  The
            # tread retains crossed steel belts; sidewall cords run bead to
            # bead; bead wire is circumferential.
            if zone == "tread":
                layers = [
                    {"material": "steel", "thickness_m": 0.00135,
                     "orientation_rad": np.deg2rad(22.0)},
                    {"material": "steel", "thickness_m": 0.00135,
                     "orientation_rad": np.deg2rad(-22.0)},
                    {"material": "composite-cord", "thickness_m": 0.0012,
                     "orientation_rad": np.pi / 2.0},
                ]
            elif zone == "sidewall":
                layers = [
                    {"material": "composite-cord", "thickness_m": 0.0018,
                     "orientation_rad": np.pi / 2.0},
                ]
            else:
                layers = [
                    {"material": "steel", "thickness_m": 0.0024,
                     "orientation_rad": 0.0},
                    {"material": "composite-cord", "thickness_m": 0.0014,
                     "orientation_rad": np.pi / 2.0},
                ]
        zone_thickness = max(
            1.0e-9, sum(float(layer["thickness_m"])
                        for layer in stacks[zone]))
        q11 = q22 = q12 = q66 = q16 = q26 = 0.0
        for layer in layers:
            modulus = directional_modulus[str(layer["material"])]
            weight = float(layer["thickness_m"]) / zone_thickness
            # Q is expressed directly in the persistent material chart:
            # 0 rad is circumferential u; pi/2 is bead-to-bead v.
            angle = float(layer["orientation_rad"])
            cosine, sine = np.cos(angle), np.sin(angle)
            stiffness = modulus * weight
            q11 += stiffness * cosine ** 4
            q22 += stiffness * sine ** 4
            q12 += stiffness * cosine ** 2 * sine ** 2
            q66 += stiffness * cosine ** 2 * sine ** 2
            q16 += stiffness * cosine ** 3 * sine
            q26 += stiffness * cosine * sine ** 3
        result[face] = (q11, q12, q16, q22, q26, q66)
    return result


@lru_cache(maxsize=32)
def balloon_tire_python_program(
    wheel_names: tuple[str, ...] = WHEEL_NAMES,
    *,
    tire_radius_m: float | None = None,
    tire_section_radius_m: float | None = None,
    tire_width_m: float | None = None,
    tire_mass_kg: float | None = None,
    reference_pressure_pa: float | None = None,
    rim_radius_m: float | None = None,
    pneumatic_mode: str | None = None,
    material_profile: str = "configured",
) -> BalloonTirePythonProgram:
    """Author one tire-axis specialization for the loaded wheel identities."""

    wheel_names = tuple(str(name) for name in wheel_names)
    if len(set(wheel_names)) != len(wheel_names):
        raise ValueError("wheel identities must be unique")
    config = copy.deepcopy(load_default_car_configuration().source)
    if tire_radius_m is not None:
        config["tires"]["radius"] = float(tire_radius_m)
    if tire_section_radius_m is not None:
        config["tires"]["toroid_section_radius_m"] = float(
            tire_section_radius_m)
    if tire_width_m is not None:
        config["tires"]["width"] = float(tire_width_m)
    if tire_mass_kg is not None:
        config["drivetrain"]["tire_mass_kg"] = float(tire_mass_kg)
    if reference_pressure_pa is not None:
        config["tires"]["pressure_pa"] = float(reference_pressure_pa)
    if rim_radius_m is not None:
        config["wheels"]["rim_radius"] = float(rim_radius_m)
    if pneumatic_mode is not None:
        if pneumatic_mode not in {"tubeless", "tube"}:
            raise ValueError("pneumatic mode must be tubeless or tube")
        config["tire_skin"]["pneumatic_mode"] = pneumatic_mode
    if material_profile == "cheap-commercial-retread":
        # A reused radial carcass with an economical replacement tread cap.
        # These are solver-active zone properties: they become one row per
        # triangle in ``face_material`` below and enter the membrane equation.
        skin = config["tire_skin"]
        skin["material_profile"] = material_profile
        skin.update({
            "circumferential_segments": 32,
            "section_segments": 24,
            "bending_stiffness_nm": 95.0,
            "skin_thickness_m": 0.0115,
            "lame_lambda_pa": 5_600_000.0,
            "lame_mu_pa": 3_650_000.0,
            "membrane_damping_lambda_pa_s": 6_100.0,
            "membrane_damping_mu_pa_s": 4_050.0,
            "tread_thickness_scale": 1.62,
            "tread_stiffness_scale": 1.35,
            "tread_damping_scale": 1.55,
            "sidewall_thickness_scale": 0.78,
            "sidewall_stiffness_scale": 0.58,
            "sidewall_damping_scale": 1.18,
            "bead_thickness_scale": 1.58,
            "bead_stiffness_scale": 2.35,
            "bead_damping_scale": 1.75,
        })
    elif material_profile != "configured":
        raise ValueError(f"unknown tire material profile {material_profile!r}")
    abi = balloon_tire_graph_abi(config)
    topology = abi["topology"]
    rest = np.asarray(topology.rest_positions, dtype=np.float64)
    faces = np.asarray(topology.faces, dtype=np.int64)
    face_material_basis = _face_material_basis(rest, faces)
    oriented_stiffness = _oriented_face_stiffness(
        abi, material_profile)
    geometry = build_cotangent_geometry(rest, faces)
    vertex_count, face_count = len(rest), len(faces)
    face_scatter = np.zeros((3, vertex_count, face_count), dtype=np.float64)
    for corner in range(3):
        face_scatter[corner, faces[:, corner], np.arange(face_count)] = 1.0
    laplacian = np.zeros((vertex_count, vertex_count), dtype=np.float64)
    for (left, right), weight in zip(geometry.edges, geometry.cotangent_weights):
        laplacian[left, right] += weight / geometry.lumped_vertex_areas[left]
        laplacian[right, left] += weight / geometry.lumped_vertex_areas[right]
        laplacian[left, left] -= weight / geometry.lumped_vertex_areas[left]
        laplacian[right, right] -= weight / geometry.lumped_vertex_areas[right]
    bending_incidence = np.zeros(
        (len(geometry.edges), vertex_count), dtype=np.float64)
    bending_incidence[np.arange(len(geometry.edges)), geometry.edges[:, 0]] = -1.0
    bending_incidence[np.arange(len(geometry.edges)), geometry.edges[:, 1]] = 1.0
    bead = {index for ring in topology.bead_rings for index in ring}
    parameter_names = (
        "vertex_mass_kg", "reference_pressure_pa", "gas_charge_fraction", "reference_volume_m3",
        "reference_temperature_k", "gas_polytropic_exponent", "gas_molar_mass_kg_per_mol",
        "gas_specific_heat_ratio", "membrane_gas_permeability_mol_m_per_m2_s_pa",
        "gas_permeability_activation_energy_j_per_mol", "minimum_volume_fraction", "skin_thickness_m",
        "lame_lambda_pa", "lame_mu_pa", "membrane_damping_lambda_pa_s",
        "membrane_damping_mu_pa_s", "bead_stiffness_n_per_m", "bending_stiffness_nm",
        "bead_damping_n_s_per_m", "contact_skin_offset_m", "contact_restitution", "friction_coefficient",
        "tire_fidelity_mode",
        *(f"ring_{name}_r_m" for name in _STATIONS), *(f"ring_{name}_z_m" for name in _STATIONS),
        "wrench_spring_n_per_m", "wrench_damping_n_s_per_m")
    #: tire_fidelity_mode selects which contact force law augments the
    #: mesh's own dynamics; inert (multiplies to zero) unless selected:
    #:   0.0 = fine (today's full deformable-mesh contact only; default,
    #:         unchanged behavior)
    #:   1.0 = reduced -- lumped fallback-spring force from the real
    #:         ring-geometry contact-patch integral (see
    #:         vehicle_tire_reduced_contact_law.py), applied at the bead ring
    #:   2.0 = wrench -- bare-bones spring-damper from the hub to the
    #:         nearest ground/roller surface, one-sided (engages only on
    #:         boundary penetration); no moment arm (single query point)
    #:   3.0 = wrench-per-vertex -- the same spring-damper evaluated at each
    #:         bead vertex independently, giving a real moment for off-center
    #:         contact, still far cheaper than the full membrane
    #: All kept selectable (not removed as others are added) so they can be
    #: compared/tuned against each other and against the fine mesh.  The
    #: ring_*_r_m/ring_*_z_m/wrench_*_n_per_m fields are ordinary scalar
    #: parameters (not new function arguments) so every existing call site
    #: keeps working unchanged.  See docs/PLAN_TIRE_FIDELITY_LADDER.md.
    wheel_fields = [
        *(f"hub_position_{axis}" for axis in "xyz"), *(f"hub_velocity_{axis}" for axis in "xyz"),
        *(f"hub_basis_{local}_{world}" for local in "xyz" for world in "xyz"),
        *(f"hub_angular_velocity_{axis}" for axis in "xyz"), "hub_angle_rad", "hub_angular_velocity_z",
        "surface_kind", "cylinder_radius_m", "plane_count",
        *(f"plane_{plane}_{quantity}_{axis}" for plane in range(2)
          for quantity in ("point", "normal", "velocity") for axis in "xyz")]
    input_names = ("dt", "gravity_y", *parameter_names, *(
        f"{wheel}.{field}" for wheel in wheel_names for field in wheel_fields))
    wheel_output_fields = (
        *(f"rim_force_{axis}_n" for axis in "xyz"), *(f"rim_moment_{axis}_nm" for axis in "xyz"),
        "gas_pressure_pa", "volume_ratio", "gas_temperature_k", "contact_count",
        "minimum_skin_y_m", "strain_energy_j", "dissipation_power_w", "bending_energy_j")
    output_names = tuple(f"{wheel}.{field}" for wheel in wheel_names for field in wheel_output_fields)
    default_input = np.zeros((len(input_names),), dtype=np.float64)
    input_index = {name: index for index, name in enumerate(input_names)}
    default_input[input_index["dt"]] = 1.0 / 4096.0
    default_input[input_index["gravity_y"]] = -9.81
    for name in parameter_names:
        if name == "tire_fidelity_mode" or name.startswith("ring_") or name.startswith("wrench_"):
            continue  # tire_fidelity_mode defaults to 0.0 (already zero-filled);
            # ring_* stations are set below from the real rest mesh;
            # wrench_* defaults are set below (not part of abi["parameters"]).
        default_input[input_index[name]] = (
            1.0 if name == "gas_charge_fraction" else float(abi["parameters"][name]))
    for wheel in wheel_names:
        for axis in "xyz":
            default_input[input_index[f"{wheel}.hub_basis_{axis}_{axis}"]] = 1.0

    # Real ring-station geometry, derived from the actual rest mesh (not
    # hand-picked): vertex layout is iu*section_rows+iv (confirmed against
    # topology.bead_rings' (0, section_segments)-style rows), bead stations
    # are rows 0/last, shoulder stations are the max-radius row in each half.
    circumferential_segments = topology.circumferential_segments
    section_rows = vertex_count // circumferential_segments
    _radius = np.linalg.norm(rest[:, :2], axis=-1)

    def _row_indices(iv: int) -> np.ndarray:
        return np.array([iu * section_rows + iv for iu in range(circumferential_segments)])

    def _station(iv: int) -> tuple[float, float]:
        row = _row_indices(iv)
        return float(np.mean(_radius[row])), float(np.mean(rest[row, 2]))

    _half = section_rows // 2
    _inboard_iv = int(np.argmax([np.mean(_radius[_row_indices(iv)]) for iv in range(0, _half)]))
    _outboard_iv = _half + int(np.argmax(
        [np.mean(_radius[_row_indices(iv)]) for iv in range(_half, section_rows)]))
    _station_rz = dict(zip(_STATIONS, (
        _station(0), _station(_inboard_iv), _station(_outboard_iv), _station(section_rows - 1))))
    for name, (station_r, station_z) in _station_rz.items():
        default_input[input_index[f"ring_{name}_r_m"]] = station_r
        default_input[input_index[f"ring_{name}_z_m"]] = station_z

    # Seeded from the already-tuned bead spring/damper, not invented values;
    # real tuning (by hand or, once wanted, ADAM against a fine-mesh
    # reference) is expected to move these -- they are deliberately
    # parameterized, not baked in.  See docs/PLAN_TIRE_FIDELITY_LADDER.md.
    default_input[input_index["wrench_spring_n_per_m"]] = (
        float(abi["parameters"]["bead_stiffness_n_per_m"]))
    default_input[input_index["wrench_damping_n_s_per_m"]] = (
        float(abi["parameters"]["bead_damping_n_s_per_m"]))

    face_material = np.concatenate((np.asarray([
        ([0.0, 0.0, 0.0, 0.0, 0.0]
         if zone == "rim-closure" else [
            float(config["tire_skin"]["skin_thickness_m"]) * float(config["tire_skin"][f"{zone}_thickness_scale"]),
            float(config["tire_skin"]["lame_lambda_pa"]) * float(config["tire_skin"][f"{zone}_stiffness_scale"]),
            float(config["tire_skin"]["lame_mu_pa"]) * float(config["tire_skin"][f"{zone}_stiffness_scale"]),
            float(config["tire_skin"]["membrane_damping_lambda_pa_s"]) * float(config["tire_skin"][f"{zone}_damping_scale"]),
            float(config["tire_skin"]["membrane_damping_mu_pa_s"]) * float(config["tire_skin"][f"{zone}_damping_scale"]),
        ])
        for zone in topology.face_zones
    ], dtype=np.float64), oriented_stiffness), axis=1)
    constants = {
        "wheel_input_indices": np.arange(
            2 + len(parameter_names), len(input_names), dtype=np.int64
        ).reshape((len(wheel_names), 41)),
        "rest": rest, "face_vertices": faces,
        "face_rest": np.asarray(topology.face_rest_data, dtype=np.float64),
        "face_scatter": face_scatter, "laplacian": laplacian,
        "bending_incidence": bending_incidence,
        "bending_scatter": bending_incidence.T.copy(),
        "bending_weight": np.asarray(
            geometry.cotangent_weights, dtype=np.float64),
        "vertex_area": np.asarray(
            geometry.lumped_vertex_areas, dtype=np.float64),
        "bead_mask": np.asarray([index in bead for index in range(vertex_count)], dtype=bool),
        "face_material": face_material,
        "face_material_basis_rad": face_material_basis,
        "material_coordinates_uv": np.asarray(
            topology.material_coordinates_uv, dtype=np.float64),
        "natural_position_uv": np.asarray(
            topology.natural_position_uv, dtype=np.float64),
        "face_material_uv": np.asarray(
            topology.face_material_uv, dtype=np.float64),
        "face_natural_jacobian_uv": np.asarray(
            topology.face_natural_jacobian_uv, dtype=np.float64),
        "face_natural_metric_uv": np.asarray(
            topology.face_rest_data, dtype=np.float64)[:, 5:8],
        "face_directional_coefficients_uv": face_material[:, 5:11],
        "flexible_face_mask": np.asarray(
            topology.flexible_face_mask, dtype=bool),
        "rim_closure_face_mask": np.asarray(
            topology.rim_closure_face_mask, dtype=bool),
        "default_input": default_input}
    return BalloonTirePythonProgram(
        BALLOON_TIRE_VECTOR_SOURCE, "balloon_tire_vector_step",
        tuple(input_names), output_names,
        len(wheel_names) * vertex_count * 6,
        vertex_count, face_count, constants, tuple(topology.face_zones),
        str(abi["pneumatic_mode"]), material_profile)


__all__ = ["BALLOON_TIRE_VECTOR_SOURCE", "BalloonTirePythonProgram",
           "MAX_PLANES_PER_WHEEL", "balloon_tire_python_program"]
