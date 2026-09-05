"""Pygame projection of the running validator's actual tensor state."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class PythonValidatorViewer:
    """Draw solver mesh/state without owning or approximating any physics."""

    def __init__(self, model: Mapping[str, Any], *, width: int = 1280,
                 height: int = 800, headless: bool = False) -> None:
        import os
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        pygame.init()
        self.pygame = pygame
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Turing validator — live dually graph")
        self.font = pygame.font.Font(None, 25)
        self.small = pygame.font.Font(None, 19)
        self.clock = pygame.time.Clock()
        self.model = model
        graph = model["mechanical_graph"]
        self.nodes = tuple(graph["nodes"])
        self.node_index = {str(node["identity"]): index
                           for index, node in enumerate(self.nodes)}
        self.graph_edges = tuple(
            (self.node_index[str(edge["nodes"][0])],
             self.node_index[str(edge["nodes"][1])], edge)
            for edge in graph["edges"] if "nodes" in edge
            and str(edge["nodes"][0]) in self.node_index
            and str(edge["nodes"][1]) in self.node_index)
        self.stage_order = tuple(model.get("validator_program", {}).get(
            "stages", ()))
        self.stage_index = {name: index for index, name in enumerate(
            self.stage_order)}
        self.yaw, self.pitch, self.distance = -0.72, -0.23, 5.8
        self.target = np.asarray((0.0, 0.45, 0.0), dtype=np.float64)
        self.dragging = False

    def close(self) -> None:
        self.pygame.quit()

    def events(self) -> bool:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.dragging = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            if event.type == pygame.MOUSEMOTION and self.dragging:
                self.yaw += event.rel[0] * 0.007
                self.pitch = max(-1.2, min(1.2,
                    self.pitch + event.rel[1] * 0.007))
            if event.type == pygame.MOUSEWHEEL:
                self.distance = max(2.0, min(18.0,
                    self.distance * math.exp(-0.10 * event.y)))
        return True

    def _camera(self):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        offset = self.distance * np.asarray((cp * cy, sp, cp * sy))
        camera = self.target + offset
        forward = self.target - camera
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray((0.0, 1.0, 0.0)))
        right /= max(1.0e-12, np.linalg.norm(right))
        up = np.cross(right, forward)
        return camera, right, up, forward

    def _project(self, points: np.ndarray):
        camera, right, up, forward = self._camera()
        relative = points - camera.reshape((1, 3))
        depth = relative @ forward
        focal = 0.78 * min(self.width, self.height)
        safe = np.maximum(depth, 0.05)
        screen = np.column_stack((
            self.width * 0.5 + focal * (relative @ right) / safe,
            self.height * 0.52 - focal * (relative @ up) / safe,
        ))
        return screen, depth

    def _assembly_alpha(self, record: Mapping[str, Any], stage: str,
                        progress: float) -> float:
        installed_at = record.get("assembly_stage")
        if not installed_at or installed_at not in self.stage_index:
            return 1.0
        current = self.stage_index.get(stage, -1)
        target = self.stage_index[str(installed_at)]
        if current < target:
            return 0.0
        if current > target:
            return 1.0
        return max(0.0, min(1.0, float(progress)))

    def _draw_part_geometry(self, center: np.ndarray,
                            node: Mapping[str, Any], color,
                            alpha: float) -> None:
        """Realize renderer-neutral Abstract UI geometry at tensor position."""

        geometry = node.get("geometry") or {}
        primitive = str(geometry.get("primitive", ""))
        if not primitive or primitive.startswith("solver-membrane"):
            return
        pygame = self.pygame
        axis = np.asarray(geometry.get("axis", (0.0, 0.0, 1.0)),
                          dtype=np.float64)
        axis /= max(1.0e-12, float(np.linalg.norm(axis)))
        seed = (np.asarray((0.0, 1.0, 0.0)) if abs(axis[1]) < 0.9 else
                np.asarray((1.0, 0.0, 0.0)))
        radial_a = np.cross(axis, seed)
        radial_a /= max(1.0e-12, float(np.linalg.norm(radial_a)))
        radial_b = np.cross(axis, radial_a)
        shade = tuple(max(0, min(255, int(channel * (0.35 + 0.65 * alpha))))
                      for channel in color)

        def ring(radius: float, offset: float = 0.0, width: int = 2):
            angles = np.linspace(0.0, 2.0 * math.pi, 41)
            points = (center + offset * axis + radius *
                      (np.cos(angles)[:, None] * radial_a
                       + np.sin(angles)[:, None] * radial_b))
            screen, depth = self._project(points)
            if np.all(depth > 0.05):
                pygame.draw.lines(self.screen, shade, True, screen, width)
            return points

        if primitive == "wheel-center-disc":
            radius = float(geometry["radius_m"])
            ring(radius, 0.0, 3)
            spoke_angles = np.linspace(0.0, 2.0 * math.pi, 7)[:-1]
            endpoints = np.asarray([
                center + radius * (math.cos(angle) * radial_a
                                   + math.sin(angle) * radial_b)
                for angle in spoke_angles])
            projected, depth = self._project(np.vstack((center, endpoints)))
            for index in range(len(endpoints)):
                if depth[0] > 0.05 and depth[index + 1] > 0.05:
                    pygame.draw.line(self.screen, shade, projected[0],
                                     projected[index + 1], 2)
        elif primitive == "drop-center-rim":
            radius = float(geometry["radius_m"])
            bead_radius = float(geometry["bead_seat_radius_m"])
            half = 0.5 * float(geometry["width_m"])
            ring(bead_radius, -half, 3)
            ring(bead_radius, half, 3)
            ring(radius * 0.88, 0.0, 2)
        elif primitive == "bead-ring":
            ring(float(geometry["radius_m"]),
                 float(geometry.get("axial_offset_m", 0.0)), 4)
        elif primitive == "bearing-races":
            ring(0.5 * float(geometry["outer_diameter_m"]), 0.0, 4)
            ring(0.5 * float(geometry["bore_m"]), 0.0, 2)
        elif primitive == "wheel-mounting-hub":
            ring(float(geometry["flange_radius_m"]), 0.0, 4)
            ring(float(geometry["barrel_radius_m"]), 0.0, 3)
        elif primitive == "brake-drum":
            half = 0.5 * float(geometry["width_m"])
            ring(float(geometry["radius_m"]), -half, 3)
            ring(float(geometry["radius_m"]), half, 3)
        elif primitive == "axial-structural-casing":
            half = 0.5 * float(geometry["length_m"])
            tube_radius = float(geometry["tube_radius_m"])
            left = ring(tube_radius, -half, 3)
            right = ring(tube_radius, half, 3)
            for angle in np.linspace(0.0, 2.0 * math.pi, 7)[:-1]:
                left_point = (center - half * axis + tube_radius *
                              (math.cos(angle) * radial_a
                               + math.sin(angle) * radial_b))
                right_point = (center + half * axis + tube_radius *
                               (math.cos(angle) * radial_a
                                + math.sin(angle) * radial_b))
                screen, depth = self._project(np.vstack((left_point, right_point)))
                if np.all(depth > 0.05):
                    pygame.draw.line(self.screen, shade, screen[0], screen[1], 2)
            ring(float(geometry["center_radius_m"]), 0.0, 5)

    def draw(self, snapshot: Mapping[str, Any] | None, *, stage: str,
             progress: float, sim_time: float, status: str = "running",
             accepted_time: float = 0.0, substep_dt: float = 0.0,
             substep_index: int = 0, accepted_substeps: int = 0,
             rejected_substeps: int = 0, error_max: float = 0.0,
             error_rms: float = 0.0, error_p95: float = 0.0,
             error_per_wheel: tuple[float, ...] = (),
             error_location: str = "none",
             rule_violation: str = "none") -> None:
        pygame = self.pygame
        self.screen.fill((9, 13, 19))
        if snapshot is None:
            label = self.font.render(
                "Building the authored graph; waiting for its first state…",
                True, (220, 225, 230))
            self.screen.blit(label, (28, 32))
            pygame.display.flip()
            self.clock.tick(60)
            return

        node_position = np.asarray(snapshot["node_position"], dtype=np.float64)
        tire_position = np.asarray(snapshot["tire_position"], dtype=np.float64)
        finite = np.isfinite(node_position).all() and np.isfinite(tire_position).all()
        reasonable = finite and (not tire_position.size or
                     float(np.max(np.abs(tire_position))) < 100.0)

        floor_points = np.asarray([
            (-2.2, -0.75, -2.2), (2.2, -0.75, -2.2),
            (2.2, -0.75, 2.2), (-2.2, -0.75, 2.2)], dtype=np.float64)
        floor_screen, _ = self._project(floor_points)
        pygame.draw.polygon(self.screen, (18, 25, 32), floor_screen, 0)
        pygame.draw.polygon(self.screen, (48, 62, 72), floor_screen, 1)

        if reasonable:
            projected_nodes, node_depth = self._project(node_position)
            node_alpha = tuple(self._assembly_alpha(node, stage, progress)
                               for node in self.nodes)
            for left, right, edge in sorted(
                    self.graph_edges,
                    key=lambda row: float(node_depth[row[0]] + node_depth[row[1]])):
                edge_alpha = min(node_alpha[left], node_alpha[right],
                                 self._assembly_alpha(edge, stage, progress))
                if edge_alpha <= 0.0:
                    continue
                edge_class = edge.get("edge_class")
                color = ((104, 185, 235) if edge_class == "drivetrain" else
                         (94, 206, 193) if edge_class == "pneumatic" else
                         (190, 116, 224) if edge_class == "contact-seal" else
                         (225, 154, 72) if edge_class == "load-bearing-structure" else
                         (132, 143, 151))
                color = tuple(int(channel * (0.35 + 0.65 * edge_alpha))
                              for channel in color)
                pygame.draw.line(self.screen, color,
                                 projected_nodes[left], projected_nodes[right], 3)

            faces = np.asarray(snapshot["tire_faces"], dtype=np.int64)
            face_zones = tuple(snapshot.get("tire_face_zones", ()))
            face_material = np.asarray(snapshot.get(
                "tire_face_material", np.zeros((len(faces), 5))),
                dtype=np.float64)
            camera, _right, _up, _forward = self._camera()
            tire_draw = []
            for wheel, vertices in enumerate(tire_position):
                projected, depth = self._project(vertices)
                for face_index, face in enumerate(faces):
                    face_depth = float(depth[face].mean())
                    if face_depth > 0.05:
                        triangle = vertices[face]
                        outward = np.cross(triangle[1] - triangle[0],
                                           triangle[2] - triangle[0])
                        centroid = triangle.mean(axis=0)
                        exterior = float(np.dot(
                            outward, camera - centroid)) >= 0.0
                        tire_draw.append((face_depth, wheel, face_index,
                                          exterior, projected[face]))
            exterior_colors = {
                "tread": (70, 76, 80),
                "sidewall": (43, 49, 54),
                "bead": (126, 92, 48),
                "rim-closure": (142, 151, 158),
            }
            interior_colors = {
                "tread": (47, 112, 126),
                "sidewall": (54, 137, 145),
                "bead": (92, 151, 142),
                "rim-closure": (92, 106, 116),
            }
            for _depth, wheel, face_index, exterior, polygon in sorted(
                    tire_draw, key=lambda row: row[0], reverse=True):
                zone = (face_zones[face_index]
                        if face_index < len(face_zones) else "sidewall")
                palette = exterior_colors if exterior else interior_colors
                color = palette.get(zone, palette["sidewall"])
                # The same invariant center-surface triangle is drawn once;
                # outward winding selects its exterior or interior palette.
                # Thickness modulates brightness and remains solver material,
                # not a second displaced display surface.
                thickness = (float(face_material[face_index, 0])
                             if face_index < len(face_material) else 0.012)
                scale = max(0.72, min(1.18, thickness / 0.014))
                color = tuple(max(0, min(255, int(channel * scale)))
                              for channel in color)
                pygame.draw.polygon(self.screen, color, polygon)
                outline = ((171, 181, 188) if exterior else (112, 225, 211))
                pygame.draw.polygon(self.screen, outline, polygon, 1)

            pillar_pose = np.asarray(snapshot["pillar_pose"], dtype=np.float64)
            pillar_alpha = np.asarray(snapshot["pillar_alpha"], dtype=np.float64)
            fixture = np.asarray(snapshot["fixture_wheel"], dtype=np.float64)
            anchor = np.asarray(snapshot["roller_anchor"], dtype=np.float64)
            for wheel in range(len(pillar_pose)):
                top = pillar_pose[wheel]
                bottom = np.asarray((top[0], -0.75, top[2]))
                line, _ = self._project(np.stack((bottom, top)))
                color = (220, 173, 61) if pillar_alpha[wheel] > 0.01 else (85, 91, 96)
                pygame.draw.line(self.screen, color, line[0], line[1], 5)
                carriage_y = fixture[wheel, 0]
                roller_points = np.asarray([
                    (anchor[wheel, 0] - 0.18, carriage_y, anchor[wheel, 1]),
                    (anchor[wheel, 0] + 0.18, carriage_y, anchor[wheel, 1]),
                ])
                roller_screen, roller_depth = self._project(roller_points)
                for point, depth_value in zip(roller_screen, roller_depth):
                    if depth_value > 0.05:
                        pygame.draw.circle(self.screen, (196, 202, 207), point, 8, 2)

            for index, node in enumerate(self.nodes):
                alpha = node_alpha[index]
                if alpha <= 0.0:
                    continue
                kind = str(node.get("kind", ""))
                color = ((246, 197, 71) if "bearing" in kind else
                         (205, 71, 74) if "brake" in kind else (210, 217, 223))
                self._draw_part_geometry(node_position[index], node, color, alpha)
                radius = 6 if "bearing" in kind else 4
                if node_depth[index] > 0.05:
                    node_color = tuple(int(channel * (0.35 + 0.65 * alpha))
                                       for channel in color)
                    pygame.draw.circle(self.screen, node_color,
                                       projected_nodes[index], radius)

        title = self.font.render("LIVE PYTHON VALIDATOR — COMMERCIAL DUALLY AXLE",
                                 True, (233, 238, 242))
        self.screen.blit(title, (24, 20))
        fidelity_names = {0.0: "FINE (full deformable mesh)",
                          1.0: "REDUCED (contact-patch integral)",
                          2.0: "WRENCH (hub spring-damper)",
                          3.0: "WRENCH-PER-VERTEX (bead spring-damper)"}
        fidelity_mode = float(snapshot.get("tire_fidelity_mode", 0.0)) if snapshot else 0.0
        fidelity_label = fidelity_names.get(fidelity_mode, f"UNKNOWN ({fidelity_mode})")
        fidelity_line = self.small.render(
            f"tire fidelity: {fidelity_label}", True,
            (255, 214, 92) if fidelity_mode != 0.0 else (156, 177, 190))
        self.screen.blit(fidelity_line, (24, 42))
        stage_line = self.small.render(
            f"{stage}  {progress * 100:5.1f}%   try t={sim_time:10.6f}s   "
            f"dt={substep_dt:.3e}s   attempt={substep_index}   {status}",
            True, ((255, 124, 112) if status == "rejected-substep"
                   else (173, 193, 207)))
        self.screen.blit(stage_line, (24, 62))
        subdivision_line = self.small.render(
            f"accepted t={accepted_time:10.6f}s   "
            f"accepted={accepted_substeps}   rejected={rejected_substeps}",
            True, (156, 177, 190))
        self.screen.blit(subdivision_line, (24, 82))
        error_line = self.small.render(
            f"error matrix: max={error_max:.3e} m @ {error_location}   "
            f"rms={error_rms:.3e} m   p95={error_p95:.3e} m",
            True, (235, 175, 112))
        self.screen.blit(error_line, (24, 102))
        wheel_error_line = self.small.render(
            "wheel maxima m: " + "  ".join(
                f"{value:.3e}" for value in error_per_wheel),
            True, (192, 163, 128))
        self.screen.blit(wheel_error_line, (24, 122))
        rule_line = self.small.render(
            f"violated rule: {rule_violation}",
            True, ((255, 112, 104) if rule_violation != "none"
                   else (130, 195, 145)))
        self.screen.blit(rule_line, (24, 142))
        legend = self.small.render(
            "casing center-surface triangles: exterior tread/sidewall/bead "
            "gray/black/brown | interior liner teal | rim closure steel | drag/orbit, wheel/zoom",
            True, (129, 143, 153))
        self.screen.blit(legend, (24, self.height - 30))
        if not reasonable:
            warning = self.font.render(
                "Solver state exceeded the physical display envelope; frame withheld.",
                True, (255, 102, 94))
            self.screen.blit(warning, (24, 88))
        pressure = np.asarray(snapshot["tire_pressure"], dtype=np.float64)
        pressure_text = self.small.render(
            "pressure kPa: " + "  ".join(
                f"{value / 1000.0:8.1f}" for value in pressure),
            True, (178, 210, 177))
        self.screen.blit(pressure_text, (24, 162))
        pygame.display.flip()
        self.clock.tick(60)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pygame.image.save(self.screen, str(path))


__all__ = ["PythonValidatorViewer"]
