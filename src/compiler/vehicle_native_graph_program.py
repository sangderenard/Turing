"""Vectorized Python/AbstractTensor closed vehicle graph.

All independent native-rig work is expressed on tensor axes.  The only Python
loop left in the numerical source is the tire microstep recurrence; each
recurrence evaluates every batch lane, wheel, vertex, face, edge, and contact
surface together.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Mapping

import numpy as np

from .abstract_ui_vehicles import load_default_car_configuration, _vehicle_mechanical_graph


RIG_POINT_COUNT = 16
BATCH_CAPACITY = 8


VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE = '''
def graph_cross(left, right):
    return AbstractTensor.stack([
        left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
        left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],
        left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0],
    ], dim=-1)


def graph_norm(value):
    return ((value * value).sum(dim=-1) + 1.0e-24).sqrt()


def vehicle_rig_points_vector(body_position, body_velocity, attitude,
                              angular_velocity, rig_points):
    roll, pitch, yaw = attitude[:, 0], attitude[:, 1], attitude[:, 2]
    cr, sr, cp, sp = roll.cos(), roll.sin(), pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()
    local = rig_points[:, :, 2:5]
    x1 = cy.reshape((-1, 1)) * local[:, :, 0] - sy.reshape((-1, 1)) * local[:, :, 2]
    z1 = sy.reshape((-1, 1)) * local[:, :, 0] + cy.reshape((-1, 1)) * local[:, :, 2]
    y1 = cr.reshape((-1, 1)) * local[:, :, 1] - sr.reshape((-1, 1)) * z1
    radius = AbstractTensor.stack([
        cp.reshape((-1, 1)) * x1 - sp.reshape((-1, 1)) * y1,
        sp.reshape((-1, 1)) * x1 + cp.reshape((-1, 1)) * y1,
        sr.reshape((-1, 1)) * local[:, :, 1] + cr.reshape((-1, 1)) * z1,
    ], dim=-1)
    point_velocity = body_velocity.reshape((-1, 1, 3)) + graph_cross(
        angular_velocity.reshape((-1, 1, 3)), radius)
    target, target_velocity = rig_points[:, :, 5:8], rig_points[:, :, 8:11]
    command, stiffness, damping = rig_points[:, :, 11:14], rig_points[:, :, 14:17], rig_points[:, :, 17:20]
    position_force = stiffness * (target - body_position.reshape((-1, 1, 3)) - radius) + damping * (target_velocity - point_velocity)
    velocity_force = damping * (target_velocity - point_velocity)
    mode = rig_points[:, :, 1].reshape((-1, 16, 1))
    force = AbstractTensor.where(mode == 1, position_force,
        AbstractTensor.where(mode == 2, command,
        AbstractTensor.where(mode == 3, velocity_force, position_force * 0.0)))
    force = force * rig_points[:, :, 0].reshape((-1, 16, 1))
    magnitude = graph_norm(force)
    maximum = rig_points[:, :, 20]
    scale = AbstractTensor.where((maximum > 0.0) * (magnitude > maximum), maximum / magnitude, 1.0)
    force = force * scale.reshape((-1, 16, 1))
    moment = graph_cross(radius, force)
    moment[:, :, 1] = -moment[:, :, 1]
    reactions = AbstractTensor.concat([-force, -moment], dim=-1)
    return force.sum(dim=1), moment.sum(dim=1), reactions


def vehicle_material_nodes_vector(body_position, body_velocity, attitude,
                                  angular_velocity, compression,
                                  compression_velocity, node_reference,
                                  node_structural_support_binding):
    roll, pitch, yaw = attitude[:, 0], attitude[:, 1], attitude[:, 2]
    cr, sr, cp, sp = roll.cos(), roll.sin(), pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()
    reference = node_reference.reshape((1, -1, 3))
    x1 = cy.reshape((-1, 1)) * reference[:, :, 0] - sy.reshape((-1, 1)) * reference[:, :, 2]
    z1 = sy.reshape((-1, 1)) * reference[:, :, 0] + cy.reshape((-1, 1)) * reference[:, :, 2]
    y1 = cr.reshape((-1, 1)) * reference[:, :, 1] - sr.reshape((-1, 1)) * z1
    radius = AbstractTensor.stack([
        cp.reshape((-1, 1)) * x1 - sp.reshape((-1, 1)) * y1,
        sp.reshape((-1, 1)) * x1 + cp.reshape((-1, 1)) * y1,
        sr.reshape((-1, 1)) * reference[:, :, 1] + cr.reshape((-1, 1)) * z1,
    ], dim=-1)
    suspension_axis = AbstractTensor.stack([-sp * cr, cp * cr, sr], dim=-1)
    node_compression = AbstractTensor.matmul(
        compression, node_structural_support_binding.swapaxes(0, 1))
    node_compression_velocity = AbstractTensor.matmul(
        compression_velocity,
        node_structural_support_binding.swapaxes(0, 1))
    radius = radius + node_compression.reshape((compression.shape[0], -1, 1)) * suspension_axis.reshape((-1, 1, 3))
    position = body_position.reshape((-1, 1, 3)) + radius
    velocity = body_velocity.reshape((-1, 1, 3)) + graph_cross(
        angular_velocity.reshape((-1, 1, 3)), radius)
    velocity = velocity + node_compression_velocity.reshape((compression.shape[0], -1, 1)) * suspension_axis.reshape((-1, 1, 3))
    return position, velocity


def vehicle_member_material_vector(dt, axial_strain, axial_rate,
                                   material_state, edge_geometry):
    zero = axial_strain * 0.0
    return vehicle_member_material_step(
        material_state[:, :, 3], axial_strain, axial_rate,
        edge_geometry[:, 10], zero, zero, edge_geometry[:, 10],
        material_state[:, :, 4], dt.reshape((-1, 1)), material_state[:, :, 5],
        edge_geometry[:, 8], edge_geometry[:, 9], edge_geometry[:, 7],
        edge_geometry[:, 5], edge_geometry[:, 2], material_state[:, :, 0],
        material_state[:, :, 1], material_state[:, :, 2], edge_geometry[:, 4],
        zero, zero, edge_geometry[:, 10], edge_geometry[:, 6], edge_geometry[:, 3])


def vehicle_material_bank_vector(node_positions, node_velocities,
                                 material_state, edge_nodes,
                                 edge_geometry, structural_support_edge_mask,
                                 dt):
    batch_count = node_positions.shape[0]
    support_count = structural_support_edge_mask.shape[0]
    if edge_nodes.shape[0] == 0:
        support_zero = node_positions[:, 0, 0].reshape(
            (batch_count, 1)) * 0.0
        support_zero = support_zero * (
            structural_support_edge_mask.sum(dim=1).reshape(
                (1, support_count)) * 0.0 + 1.0)
        return (material_state, material_state[:, :, 0:8],
                support_zero, support_zero + 1.0)
    left = node_positions.gather(edge_nodes[:, 0], dim=1)
    right = node_positions.gather(edge_nodes[:, 1], dim=1)
    delta = right - left
    velocity_delta = node_velocities.gather(edge_nodes[:, 1], dim=1) - node_velocities.gather(edge_nodes[:, 0], dim=1)
    length = graph_norm(delta)
    axial_kinematic = edge_geometry[:, 12]
    axial_strain = axial_kinematic * (length - edge_geometry[:, 0]) / edge_geometry[:, 0]
    axial_rate = axial_kinematic * (delta * velocity_delta).sum(dim=-1) / (length * edge_geometry[:, 0])
    result = vehicle_member_material_vector(
        dt, axial_strain, axial_rate, material_state,
        edge_geometry)
    material_state[:, :, 0] = result[0]
    material_state[:, :, 1] = result[1]
    material_state[:, :, 2] = result[2]
    material_state[:, :, 3] = result[3]
    material_state[:, :, 4] = result[16]
    material_state[:, :, 5] = result[6]
    material_state[:, :, 6] = axial_strain
    material_state[:, :, 7:9] = material_state[:, :, 7:9] * 0.0
    diagnostics = AbstractTensor.stack([
        axial_strain, result[0], result[3], result[4], result[6],
        result[7], result[12], result[16]], dim=-1)
    plastic_length = result[0].abs() * edge_geometry[:, 0]
    support_plastic = AbstractTensor.where(
        structural_support_edge_mask.reshape((1, support_count, -1)), plastic_length.reshape((-1, 1, plastic_length.shape[1])), 0.0).max(dim=2)
    support_survival = AbstractTensor.where(
        structural_support_edge_mask.reshape((1, support_count, -1)), (1.0 - result[6]).reshape((-1, 1, result[6].shape[1])), 1.0).min(dim=2)
    return material_state, diagnostics, support_plastic, support_survival


def vehicle_periodic_terrain_vector(hub_xz, phase_xz, period_xz, base_y,
                                    terrain_triangles):
    batch_count, wheel_count = hub_xz.shape[0], hub_xz.shape[1]
    cell = period_xz / 8.0
    uv = (hub_xz + phase_xz.reshape((-1, 1, 2))) % period_xz.reshape((-1, 1, 2))
    cell_index = (uv // cell.reshape((-1, 1, 2))).to_dtype("int64")
    fraction = (uv - cell_index * cell.reshape((-1, 1, 2))) / cell.reshape((-1, 1, 2))
    upper = (fraction[:, :, 0] + fraction[:, :, 1] > 1.0).to_dtype("int64")
    triangle_index = ((cell_index[:, :, 1] * 8 + cell_index[:, :, 0]) * 2 + upper).reshape((-1,))
    unit_triangle = terrain_triangles.gather(triangle_index, dim=0).reshape((batch_count, wheel_count, 3, 3))
    triangle = AbstractTensor.stack([
        unit_triangle[:, :, :, 0] * period_xz[:, 0].reshape((-1, 1, 1)),
        unit_triangle[:, :, :, 1],
        unit_triangle[:, :, :, 2] * period_xz[:, 1].reshape((-1, 1, 1)),
    ], dim=-1)
    edge_1, edge_2 = triangle[:, :, 1, :] - triangle[:, :, 0, :], triangle[:, :, 2, :] - triangle[:, :, 0, :]
    normal = graph_cross(edge_1, edge_2)
    normal = normal / graph_norm(normal).reshape((batch_count, wheel_count, 1))
    normal = AbstractTensor.where((normal[:, :, 1] < 0.0).reshape((batch_count, wheel_count, 1)), -normal, normal)
    point = AbstractTensor.stack([hub_xz[:, :, 0], base_y + triangle[:, :, 0, 1], hub_xz[:, :, 1]], dim=-1)
    return point, normal


def vehicle_roller_fixture_vector(fixture_global, fixture_wheel,
                                  fixture_surface, roller_load):
    batch_count = fixture_global.shape[0]
    dt = fixture_global[:, 0].reshape((batch_count, 1))
    global_mode = fixture_global[:, 1].reshape((batch_count, 1))
    gravity = fixture_global[:, 2].reshape((batch_count, 1))
    floor_y = fixture_global[:, 3].reshape((batch_count, 1))
    carriage_mass = fixture_global[:, 4].reshape((batch_count, 1))
    neutral_buoyancy = fixture_global[:, 5].reshape((batch_count, 1))
    passive_damping = fixture_global[:, 6].reshape((batch_count, 1))
    lock_stiffness = fixture_global[:, 7].reshape((batch_count, 1))
    lock_damping = fixture_global[:, 8].reshape((batch_count, 1))
    maximum_force = fixture_global[:, 9].reshape((batch_count, 1))
    hub_velocity = fixture_wheel[:, :, 1]
    carriage_y = fixture_wheel[:, :, 2]
    carriage_velocity = fixture_wheel[:, :, 3]
    command_y = fixture_wheel[:, :, 4]
    command_velocity = fixture_wheel[:, :, 5]
    mode = global_mode.maximum(fixture_wheel[:, :, 7]).maximum(0.0).minimum(1.0)
    passive_force = passive_damping * (carriage_velocity - hub_velocity).maximum(0.0)
    lock_force = (lock_stiffness * (command_y - carriage_y)
                  + lock_damping * (command_velocity - carriage_velocity))
    lock_force = lock_force.maximum(-maximum_force).minimum(maximum_force)
    actuator_force = -(1.0 - mode) * passive_force + mode * lock_force
    gravity_force = carriage_mass * gravity
    compensation_force = -neutral_buoyancy * gravity_force
    acceleration = (gravity_force + compensation_force + actuator_force
                    - roller_load) / carriage_mass
    candidate_velocity = carriage_velocity + dt * acceleration
    candidate_y = carriage_y + dt * candidate_velocity
    next_y = candidate_y.maximum(floor_y)
    next_velocity = (next_y - carriage_y) / dt
    corner_output = AbstractTensor.stack([
        next_y, next_velocity, actuator_force,
        (1.0 - mode) * passive_force,
        compensation_force * AbstractTensor.ones_like(carriage_y),
    ], dim=2)
    surface_output = AbstractTensor.stack([
        fixture_surface[:, 1] + fixture_global[:, 0] * fixture_surface[:, 3],
        fixture_surface[:, 2] + fixture_global[:, 0] * fixture_surface[:, 4],
        fixture_surface[:, 0].maximum(0.0).minimum(1.0),
        fixture_surface[:, 5].maximum(0.01),
        fixture_surface[:, 6].maximum(0.01),
    ], dim=1)
    return corner_output, surface_output


def vehicle_prepare_wheel_boundary(body_position, body_velocity, attitude,
                                   angular_velocity, attachment, compression,
                                   compression_velocity, wheel_angle, wheel_omega,
                                   fixture_wheel, fixture_surface, roller_anchor,
                                   roller_anchor_valid, terrain_triangles):
    batch_count, wheel_count = attachment.shape[0], attachment.shape[1]
    roll, pitch, yaw = attitude[:, 0], attitude[:, 1], attitude[:, 2]
    cr, sr, cp, sp = roll.cos(), roll.sin(), pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()
    basis = AbstractTensor.stack([
        cp * cy + sp * sr * sy, sp * cy - cp * sr * sy, cr * sy,
        -sp * cr, cp * cr, sr,
        -cp * sy + sp * sr * cy, -sp * sy - cp * sr * cy, cr * cy,
    ], dim=-1).reshape((-1, 3, 3))
    rotated_attachment = AbstractTensor.matmul(attachment, basis)
    hub = body_position.reshape((-1, 1, 3)) + rotated_attachment
    hub[:, :, 1] = hub[:, :, 1] + compression
    hub_velocity = body_velocity.reshape((-1, 1, 3)) + graph_cross(
        angular_velocity.reshape((-1, 1, 3)), rotated_attachment)
    hub_velocity = hub_velocity + compression_velocity.reshape((batch_count, wheel_count, 1)) * basis[:, 1, :].reshape((batch_count, 1, 3))
    hub_xz = AbstractTensor.stack([hub[:, :, 0], hub[:, :, 2]], dim=-1)
    anchor = AbstractTensor.where(
        roller_anchor_valid.reshape((batch_count, wheel_count, 1)), roller_anchor, hub_xz)
    carriage_y = fixture_wheel[:, :, 2]
    carriage_velocity_y = fixture_wheel[:, :, 3]
    terrain_mode = fixture_surface[:, 0] >= 0.5
    terrain_point, terrain_normal = vehicle_periodic_terrain_vector(
        hub_xz, fixture_surface[:, 1:3], fixture_surface[:, 5:7],
        carriage_y, terrain_triangles)
    roller_x = AbstractTensor.stack([anchor[:, :, 0] - 0.18, anchor[:, :, 0] + 0.18], dim=2)
    roller_point = AbstractTensor.stack([
        roller_x,
        carriage_y.reshape((batch_count, wheel_count, 1)) * AbstractTensor.ones_like(roller_x),
        anchor[:, :, 1].reshape((batch_count, wheel_count, 1)) * AbstractTensor.ones_like(roller_x),
    ], dim=-1)
    roller_normal = roller_point * 0.0
    roller_normal[:, :, :, 1] = 1.0
    point = AbstractTensor.where(terrain_mode.reshape((batch_count, 1, 1, 1)), terrain_point.reshape((batch_count, wheel_count, 1, 3)), roller_point)
    normal = AbstractTensor.where(terrain_mode.reshape((batch_count, 1, 1, 1)), terrain_normal.reshape((batch_count, wheel_count, 1, 3)), roller_normal)
    surface_velocity = point * 0.0
    surface_velocity[:, :, :, 0] = fixture_surface[:, 3].reshape((-1, 1, 1))
    surface_velocity[:, :, :, 2] = fixture_surface[:, 4].reshape((-1, 1, 1))
    surface_velocity[:, :, :, 1] = carriage_velocity_y.reshape((batch_count, wheel_count, 1))
    wheel_basis = basis.reshape((batch_count, 1, 3, 3)) * AbstractTensor.ones_like(
        attachment[:, :, 0]).reshape((batch_count, wheel_count, 1, 1))
    return hub, hub_velocity, wheel_basis, wheel_angle, wheel_omega, point, normal, surface_velocity, anchor


def vehicle_tire_recurrence(tire_inputs, tire_state, tire_output, tire_history,
                            tire_history_valid, tire_constants, outer_dt,
                            microstep_count):
    wheel_input_indices = tire_constants[0]
    batch_count = tire_inputs.shape[0]
    wheel_count = wheel_input_indices.shape[0]
    wheel_input = tire_inputs.gather(
        wheel_input_indices.reshape((-1,)), dim=1).reshape((batch_count, wheel_count, 41))
    previous_hub, previous_basis, previous_angle, previous_plane = tire_history
    current_hub = wheel_input[:, :, 0:3]
    current_basis = wheel_input[:, :, 6:15].reshape((batch_count, wheel_count, 3, 3))
    current_angle = wheel_input[:, :, 18]
    current_plane = wheel_input[:, :, 23:41].reshape((batch_count, wheel_count, 2, 3, 3))[:, :, :, 0, :]
    surface_velocity = wheel_input[:, :, 23:41].reshape((batch_count, wheel_count, 2, 3, 3))[:, :, :, 2, :]
    previous_hub = AbstractTensor.where(
        tire_history_valid.reshape((-1, 1, 1)), previous_hub, current_hub)
    previous_basis = AbstractTensor.where(
        tire_history_valid.reshape((-1, 1, 1, 1)), previous_basis,
        current_basis)
    previous_angle = AbstractTensor.where(
        tire_history_valid.reshape((-1, 1)), previous_angle, current_angle)
    previous_plane = AbstractTensor.where(
        tire_history_valid.reshape((-1, 1, 1, 1)), previous_plane,
        current_plane)
    wrench_sum = tire_output[:, :, 0:6] * 0.0
    contact_peak = tire_output[:, :, 9] * 0.0
    minimum_skin = tire_output[:, :, 10] * 0.0 + 1.0e300
    # The tire kernel owns no clock: the enclosing vehicle recurrence must
    # publish the duration of each accepted microstep.  The retained native C
    # assembly performs the same write before its loop.  Without it, this port
    # silently retains the standalone fixture's default dt and advances a
    # different amount of physical time from the boundary interpolation.
    tire_inputs[:, 0] = outer_dt / microstep_count
    for microstep in range(microstep_count):
        alpha = (microstep + 1.0) / microstep_count
        wheel_input[:, :, 0:3] = previous_hub + alpha * (current_hub - previous_hub)
        wheel_input[:, :, 6:15] = (previous_basis + alpha * (current_basis - previous_basis)).reshape((batch_count, wheel_count, 9))
        wheel_input[:, :, 18] = previous_angle + alpha * (current_angle - previous_angle)
        plane = previous_plane + alpha * (current_plane - previous_plane)
        geometric_velocity = (current_plane - previous_plane) / outer_dt.reshape((-1, 1, 1, 1))
        packed_plane = wheel_input[:, :, 23:41].reshape((batch_count, wheel_count, 2, 3, 3))
        packed_plane[:, :, :, 0, :] = plane
        packed_plane[:, :, :, 2, :] = surface_velocity + geometric_velocity
        wheel_input[:, :, 23:41] = packed_plane.reshape((batch_count, wheel_count, 18))
        tire_inputs[:, wheel_input_indices.reshape((-1,))] = wheel_input.reshape((-1, 164))
        tire_state, tire_output = balloon_tire_vector_step(
            tire_inputs, tire_state, tire_output, *tire_constants)
        wrench_sum = wrench_sum + tire_output[:, :, 0:6]
        contact_peak = contact_peak.maximum(tire_output[:, :, 9])
        minimum_skin = minimum_skin.minimum(tire_output[:, :, 10])
    tire_output[:, :, 0:6] = wrench_sum / microstep_count
    tire_output[:, :, 9], tire_output[:, :, 10] = contact_peak, minimum_skin
    return (tire_state, tire_output,
            (current_hub, current_basis, current_angle, current_plane),
            tire_history_valid * 0.0 + 1.0)


def vehicle_close_contact_graph(vehicle_input, contact_input,
                                tire_output, pillar_force, wheel_assembly_alpha,
                                tire_assembly_alpha):
    batch_count, wheel_count = tire_output.shape[0], tire_output.shape[1]
    present = wheel_assembly_alpha * tire_assembly_alpha
    rim_force = tire_output[:, :, 0:3] * present.reshape((batch_count, wheel_count, 1))
    rim_force[:, :, 1] = rim_force[:, :, 1] + pillar_force
    roller_load = (tire_output[:, :, 1] * present).maximum(0.0)
    wheel_load = rim_force[:, :, 1].maximum(0.0)
    residual_force = rim_force
    residual_force[:, :, 1] = rim_force[:, :, 1] - wheel_load
    contact_input[:, :, 6] = tire_output[:, :, 9] > 0.0
    contact_input[:, :, 7:9] = contact_input[:, :, 7:9] * 0.0
    attachment = contact_input[:, :, 3:6]
    contact_moment = graph_cross(attachment, residual_force)
    contact_moment[:, :, 0:2] = contact_moment[:, :, 0:2] + tire_output[:, :, 3:5] * present.reshape((batch_count, wheel_count, 1))
    chassis_force = residual_force.sum(dim=1)
    chassis_torque = contact_moment.sum(dim=1)
    reaction_torque = -tire_output[:, :, 5] * present
    return vehicle_input, contact_input, roller_load, wheel_load, reaction_torque, chassis_force, chassis_torque


def vehicle_graph_tick_vector(vehicle_input, contact_input,
                              fixture_global, fixture_wheel, fixture_surface,
                              tire_input, tire_state, tire_output,
                              tire_previous_hub, tire_previous_basis,
                              tire_previous_angle, tire_previous_plane,
                              rig_points, material_state, node_reference,
                              node_structural_support_binding,
                              edge_nodes, edge_geometry,
                              structural_support_edge_mask,
                              wheel_to_structural_support,
                              structural_support_position,
                              tire_wheel_input_indices, tire_rest,
                              tire_face_vertices, tire_face_rest,
                              tire_face_scatter, tire_bending_incidence,
                              tire_bending_scatter, tire_bending_weight,
                              tire_vertex_area, tire_bead_mask,
                              tire_face_material,
                              wheel_assembly_alpha, tire_assembly_alpha,
                              compression, compression_velocity,
                              wheel_angle, wheel_omega,
                              roller_anchor, roller_anchor_valid,
                              terrain_triangles, pillar_alpha, pillar_pose,
                              lock_stiffness, lock_damping,
                              maximum_actuator_force, tire_initialized,
                              tire_history_valid,
                              outer_dt,
                              microstep_count):
    batch_count = vehicle_input.shape[0]
    wheel_count = tire_wheel_input_indices.shape[0]
    tire_history = (
        tire_previous_hub, tire_previous_basis,
        tire_previous_angle, tire_previous_plane)
    tire_constants = (
        tire_wheel_input_indices, tire_rest, tire_face_vertices,
        tire_face_rest, tire_face_scatter, tire_bending_incidence,
        tire_bending_scatter, tire_bending_weight,
        tire_vertex_area, tire_bead_mask, tire_face_material)
    body_position = vehicle_input[:, 0:3]
    body_velocity = vehicle_input[:, 3:6]
    attitude = vehicle_input[:, 6:9]
    angular_velocity = vehicle_input[:, 9:12]
    rig_force, rig_torque, rig_reactions = vehicle_rig_points_vector(
        body_position, body_velocity, attitude, angular_velocity, rig_points)
    node_positions, node_velocities = vehicle_material_nodes_vector(
        body_position, body_velocity, attitude, angular_velocity,
        compression, compression_velocity, node_reference,
        node_structural_support_binding)
    material_state, material_diagnostics, support_plastic, support_survival = vehicle_material_bank_vector(
        node_positions, node_velocities, material_state, edge_nodes,
        edge_geometry, structural_support_edge_mask, outer_dt)
    wheel_compression = AbstractTensor.matmul(
        compression, wheel_to_structural_support.swapaxes(0, 1))
    wheel_compression_velocity = AbstractTensor.matmul(
        compression_velocity, wheel_to_structural_support.swapaxes(0, 1))
    hub, hub_velocity, basis, wheel_angle, wheel_omega, plane_point, plane_normal, plane_velocity, roller_anchor = vehicle_prepare_wheel_boundary(
        body_position, body_velocity, attitude, angular_velocity,
        contact_input[:, :, 3:6], wheel_compression,
        wheel_compression_velocity,
        wheel_angle, wheel_omega, fixture_wheel, fixture_surface, roller_anchor,
        roller_anchor_valid, terrain_triangles)
    pillar_force = pillar_alpha * (
        lock_stiffness.reshape((-1, 1)) * (pillar_pose[:, :, 1] - hub[:, :, 1])
        - lock_damping.reshape((-1, 1)) * hub_velocity[:, :, 1])
    pillar_force = pillar_force.maximum(-maximum_actuator_force.reshape((-1, 1))).minimum(
        maximum_actuator_force.reshape((-1, 1)))
    wheel_input_indices = tire_constants[0]
    tire_wheel_input = tire_input.gather(
        wheel_input_indices.reshape((-1,)), dim=1).reshape((batch_count, wheel_count, 41))
    tire_wheel_input[:, :, 0:3], tire_wheel_input[:, :, 3:6] = hub, hub_velocity
    tire_wheel_input[:, :, 6:15] = basis.reshape((batch_count, wheel_count, 9))
    tire_wheel_input[:, :, 15:18] = angular_velocity.reshape((-1, 1, 3))
    tire_wheel_input[:, :, 18], tire_wheel_input[:, :, 19] = wheel_angle, wheel_omega
    packed_plane = tire_wheel_input[:, :, 23:41].reshape((batch_count, wheel_count, 2, 3, 3))
    packed_plane[:, :, :, 0, :], packed_plane[:, :, :, 1, :] = plane_point, plane_normal
    packed_plane[:, :, :, 2, :] = plane_velocity
    tire_wheel_input[:, :, 23:41] = packed_plane.reshape((batch_count, wheel_count, 18))
    tire_input[:, wheel_input_indices.reshape((-1,))] = tire_wheel_input.reshape((batch_count, wheel_count * 41))
    initialized_tire_state = balloon_tire_vector_initialize(
        tire_input, tire_state * 0.0, tire_constants[0], tire_constants[1])
    tire_enabled = tire_assembly_alpha.sum(dim=1) > 0.0
    # Tensor backends evaluate both sides of ``where``.  Always give the tire
    # recurrence a geometrically valid candidate even while the installation
    # gate keeps that candidate out of persistent state.  Advancing an all-zero
    # closed skin computes zero volume and can overflow before masking.
    recurrence_tire_state = AbstractTensor.where(
        tire_initialized.reshape((-1, 1, 1, 1)), tire_state,
        initialized_tire_state)
    tire_initialized = AbstractTensor.where(
        tire_enabled, tire_initialized * 0.0 + 1.0, tire_initialized)
    prior_tire_state, prior_tire_output = tire_state, tire_output
    next_tire_state, next_tire_output, next_tire_history, next_tire_history_valid = vehicle_tire_recurrence(
        tire_input, recurrence_tire_state, tire_output,
        tire_history, tire_history_valid,
        tire_constants, outer_dt, microstep_count)
    tire_state = AbstractTensor.where(
        tire_enabled.reshape((-1, 1, 1, 1)), next_tire_state,
        prior_tire_state)
    tire_output = AbstractTensor.where(
        tire_enabled.reshape((-1, 1, 1)), next_tire_output,
        prior_tire_output * 0.0)
    tire_history = (
        AbstractTensor.where(tire_enabled.reshape((-1, 1, 1)),
                             next_tire_history[0], tire_history[0]),
        AbstractTensor.where(tire_enabled.reshape((-1, 1, 1, 1)),
                             next_tire_history[1], tire_history[1]),
        AbstractTensor.where(tire_enabled.reshape((-1, 1)),
                             next_tire_history[2], tire_history[2]),
        AbstractTensor.where(tire_enabled.reshape((-1, 1, 1, 1)),
                             next_tire_history[3], tire_history[3]),
    )
    tire_history_valid = AbstractTensor.where(
        tire_enabled, next_tire_history_valid, tire_history_valid * 0.0)
    vehicle_input, contact_input, roller_load, wheel_load, reaction_torque, contact_force, contact_torque = vehicle_close_contact_graph(
        vehicle_input, contact_input, tire_output, pillar_force,
        wheel_assembly_alpha, tire_assembly_alpha)
    fixture_corner_output, fixture_surface_output = vehicle_roller_fixture_vector(
        fixture_global, fixture_wheel, fixture_surface, roller_load)
    total_force, total_torque = rig_force + contact_force, rig_torque + contact_torque
    structural_support_load = AbstractTensor.matmul(
        wheel_load + fixture_corner_output[:, :, 3],
        wheel_to_structural_support)
    structural_support_reaction_torque = AbstractTensor.matmul(
        reaction_torque, wheel_to_structural_support)
    vehicle_output = vehicle_physics_step_vector(
        vehicle_input, structural_support_load,
        structural_support_reaction_torque,
        total_force, total_torque, support_plastic, support_survival,
        structural_support_position)
    return (vehicle_output, contact_input, fixture_corner_output,
            fixture_surface_output, tire_input, tire_state,
            tire_output, tire_history, rig_reactions, material_state,
            material_diagnostics, roller_anchor, -pillar_force,
            roller_anchor_valid * 0.0 + 1.0, tire_initialized,
            tire_history_valid)


def vehicle_energy_diagnostics_vector(tire_input, tire_state, tire_output):
    velocity = tire_state[:, :, :, 3:6]
    # The velocity norm is summed over vertex, xyz and wheel first, leaving
    # one scalar per batch lane; the per-lane mass multiplies that (batch,)
    # vector directly.  A (batch, 1, 1) reshape here broadcast against the
    # summed vector into a rank-3 tensor, and the stack below rejected it.
    kinetic = 0.5 * tire_input[:, 2] * (velocity * velocity).sum(dim=3).sum(dim=2).sum(dim=1)
    potential = (tire_output[:, :, 11] + tire_output[:, :, 13]).sum(dim=1)
    dissipation = tire_output[:, :, 12].sum(dim=1)
    contacts = tire_output[:, :, 9].sum(dim=1)
    return AbstractTensor.stack([kinetic, potential, dissipation, contacts], dim=1)
'''


@dataclass(frozen=True, slots=True)
class VehicleNativeGraphPythonProgram:
    source: str = VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE
    entrypoint: str = "vehicle_graph_tick_vector"
    source_language: str = "python"
    vehicle_specific_c_lines: int = 0
    vector_axes: tuple[str, ...] = (
        "batch", "wheel", "vertex", "face", "edge", "rig-point",
        "contact-surface", "xyz")
    constants: "VehicleGraphConstants | None" = None


@dataclass(frozen=True, slots=True)
class VehicleGraphConstants:
    node_reference: np.ndarray
    node_structural_support_binding: np.ndarray
    edge_nodes: np.ndarray
    edge_geometry: np.ndarray
    structural_support_edge_mask: np.ndarray


@lru_cache(maxsize=1)
def vehicle_graph_constants() -> VehicleGraphConstants:
    """Build tensor constants that the old C material structs embedded."""

    graph = _vehicle_mechanical_graph(load_default_car_configuration())
    nodes = tuple(graph["nodes"])
    node_index = {node["identity"]: index for index, node in enumerate(nodes)}
    corners = ("front_left", "front_right", "rear_left", "rear_right")
    reference = np.asarray([node["reference_position"] for node in nodes], dtype=np.float64)
    binding = np.zeros((len(nodes), 4), dtype=np.float64)
    for index, node in enumerate(nodes):
        coordinate = str(node.get("generalized_coordinate") or "")
        moves_with = node.get("moves_with")
        chassis_bound = node.get("fixed_to") == "chassis" or moves_with == "chassis"
        for corner, name in enumerate(corners):
            if coordinate == f"compression_{name}" and not chassis_bound:
                binding[index, corner] = 1.0
    edge_nodes = []
    edge_geometry = []
    edge_corners = []
    for edge in graph["edges"]:
        damage = edge.get("damage", {})
        if damage.get("model") != "elastic-plastic-member-with-shear-fracture":
            continue
        left, right = node_index[edge["a"]], node_index[edge["b"]]
        authored_rest = float(damage.get("natural_rest_length", edge["rest_length"]))
        reference_rest = math.dist(reference[left], reference[right])
        binding_pair = set(np.flatnonzero(binding[left] + binding[right]).tolist())
        axial_kinematic = float(
            authored_rest > 1.0e-8 and reference_rest > 1.0e-8
            and (np.array_equal(binding[left], binding[right])))
        rest = authored_rest if axial_kinematic else 1.0
        radius = max(1.0e-4, float(edge.get("radius", 0.018)))
        yield_stress = float(damage.get("yield_strength_pa", 350_000_000.0))
        yield_force = max(1.0, float(damage.get("axial_yield_force_n", 1.0)))
        area = max(float(damage.get("section_area_m2", yield_force / yield_stress)),
                   math.pi * radius * radius * 0.04)
        youngs = float(damage.get("youngs_modulus_pa", 205_000_000_000.0))
        fracture = max(float(damage.get("fracture_strain", 0.075)), 1.0e-4)
        corner_rows = np.flatnonzero(binding[left] + binding[right])
        corner = int(corner_rows[0]) if len(corner_rows) == 1 else -1
        edge_nodes.append((left, right))
        edge_corners.append(corner)
        edge_geometry.append((
            rest, area, area * rest, youngs, youngs / 2.6, yield_stress,
            max(yield_stress * 1.35, yield_stress + 1.0), youngs * 0.01,
            fracture, 0.35, youngs * 5.0e-5, float(corner), axial_kinematic,
        ))
    corner_mask = np.asarray([
        [corner == wheel for corner in edge_corners] for wheel in range(4)
    ], dtype=bool)
    return VehicleGraphConstants(
        reference, binding, np.asarray(edge_nodes, dtype=np.int64),
        np.asarray(edge_geometry, dtype=np.float64), corner_mask)


def vehicle_graph_constants_from_model(
    model: Mapping[str, Any],
    support_count: int | None = None,
) -> VehicleGraphConstants:
    """Build structural graph constants from a loaded assembly model."""

    graph = model["mechanical_graph"]
    nodes = tuple(graph["nodes"])
    node_index = {str(node["identity"]): index
                  for index, node in enumerate(nodes)}
    reference = np.asarray([
        node.get("reference_position", (0.0, 0.0, 0.0)) for node in nodes
    ], dtype=np.float64).reshape((len(nodes), 3))
    if support_count is None:
        support_count = len(model.get("structure", {}).get(
            "support_corners", ()))
    binding = np.zeros((len(nodes), support_count), dtype=np.float64)
    edge_nodes: list[tuple[int, int]] = []
    edge_geometry: list[tuple[float, ...]] = []
    edge_supports: list[int] = []
    for edge in graph["edges"]:
        damage = edge.get("damage", {})
        if damage.get("model") != "elastic-plastic-member-with-shear-fracture":
            continue
        endpoints = edge.get("nodes")
        if endpoints is None:
            endpoints = (edge["a"], edge["b"])
        left, right = (node_index[str(endpoint)] for endpoint in endpoints)
        rest = max(1.0e-8, math.dist(reference[left], reference[right]))
        radius = max(1.0e-4, float(edge.get("radius", 0.018)))
        yield_stress = float(damage.get("yield_strength_pa", 350_000_000.0))
        area = max(math.pi * radius * radius * 0.04,
                   float(damage.get("section_area_m2", 1.0e-4)))
        youngs = float(damage.get("youngs_modulus_pa", 205_000_000_000.0))
        fracture = max(float(damage.get("fracture_strain", 0.075)), 1.0e-4)
        edge_nodes.append((left, right))
        edge_supports.append(-1)
        edge_geometry.append((
            rest, area, area * rest, youngs, youngs / 2.6, yield_stress,
            max(yield_stress * 1.35, yield_stress + 1.0), youngs * 0.01,
            fracture, 0.35, youngs * 5.0e-5, -1.0, 1.0,
        ))
    edge_count = len(edge_nodes)
    support_mask = np.zeros((support_count, edge_count), dtype=bool)
    return VehicleGraphConstants(
        reference,
        binding,
        np.asarray(edge_nodes, dtype=np.int64).reshape((edge_count, 2)),
        np.asarray(edge_geometry, dtype=np.float64).reshape((edge_count, 13)),
        support_mask,
    )


def vehicle_native_graph_python_program() -> VehicleNativeGraphPythonProgram:
    return VehicleNativeGraphPythonProgram(constants=vehicle_graph_constants())


__all__ = ["BATCH_CAPACITY", "RIG_POINT_COUNT", "VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE",
           "VehicleGraphConstants", "VehicleNativeGraphPythonProgram",
           "vehicle_graph_constants", "vehicle_graph_constants_from_model",
           "vehicle_native_graph_python_program"]
