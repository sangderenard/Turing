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
        inputs[:, 3].reshape((-1, 1, 1)), face_rest[:, 4], face_rest[:, 0], face_rest[:, 1], face_rest[:, 2], face_rest[:, 3], face_material[:, 0].reshape((1, 1, -1)),
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
        "bead_damping_n_s_per_m", "contact_skin_offset_m", "contact_restitution", "friction_coefficient")
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
        default_input[input_index[name]] = (
            1.0 if name == "gas_charge_fraction" else float(abi["parameters"][name]))
    for wheel in wheel_names:
        for axis in "xyz":
            default_input[input_index[f"{wheel}.hub_basis_{axis}_{axis}"]] = 1.0
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
            24, len(input_names), dtype=np.int64).reshape((len(wheel_names), 41)),
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
