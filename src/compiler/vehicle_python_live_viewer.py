"""Pygame projection of the running validator's actual tensor state."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .abstract_ui_geometry import part_geometry_lines, realize_part_geometry, realize_edge_geometry, fixture_geometry, realize_fixture_geometry


def tire_geometry_triangles(snapshot: Mapping[str, Any], camera: np.ndarray,
                            forward: np.ndarray) -> list:
    """Build depth-ordered center-surface triangles with fill/outline RGB."""
    tire_position = np.asarray(snapshot["tire_position"], dtype=np.float64)
    faces = np.asarray(snapshot["tire_faces"], dtype=np.int64)
    face_zones = tuple(snapshot.get("tire_face_zones", ()))
    face_material = np.asarray(snapshot.get(
        "tire_face_material", np.zeros((len(faces), 5))),
        dtype=np.float64)
    tire_draw = []
    for wheel, vertices in enumerate(tire_position):
        depth = (vertices - camera.reshape((1, 3))) @ forward
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
                                  exterior, triangle))
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
    geometry = []
    for _depth, wheel, face_index, exterior, triangle in sorted(
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
        outline = ((171, 181, 188) if exterior else (112, 225, 211))
        geometry.append((triangle, color, outline))
    return geometry


class PythonValidatorViewer:
    """Draw solver mesh/state without owning or approximating any physics."""

    def __init__(self, model: Mapping[str, Any], *, fixture_plan, width: int = 1280,
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
        self.fixture_plan = fixture_plan
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

    def graph_geometry_objects(self, node_position, node_depth, node_alpha, stage, progress):
        """Return authored graph edges with stage visibility and RGB shading."""
        geometry = []
        for left, right, edge in sorted(
                self.graph_edges,
                key=lambda row: float(node_depth[row[0]] + node_depth[row[1]])):
            edge_alpha = min(node_alpha[left], node_alpha[right],
                             self._assembly_alpha(edge, stage, progress))
            geometry.append(realize_edge_geometry(
                edge, np.stack((node_position[left], node_position[right])), edge_alpha))
        return geometry

    def _draw_part_geometry(self, center: np.ndarray,
                            node: Mapping[str, Any], color,
                            alpha: float) -> None:
        projection = realize_part_geometry(node, center, color, alpha)
        for line in projection.model["geometry_projection"]["polylines"]:
            points = np.asarray(line["positions"], dtype=np.float64)
            shade, width, closed = line["color_rgb"], line["width_px"], line["closed"]
            screen, depth = self._project(points)
            if np.all(depth > 0.05):
                if closed:
                    self.pygame.draw.lines(self.screen, shade, True, screen, width)
                else:
                    self.pygame.draw.line(self.screen, shade, screen[0], screen[1], width)

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
            for edge_object in self.graph_geometry_objects(
                    node_position, node_depth, node_alpha, stage, progress):
                for line in edge_object.model["geometry_projection"]["polylines"]:
                    endpoints = np.asarray(line["positions"], dtype=np.float64)
                    projected, _ = self._project(endpoints)
                    pygame.draw.line(self.screen, line["color_rgb"], projected[0],
                                     projected[1], line["width_px"])

            camera, _right, _up, forward = self._camera()
            for triangle, color, outline in tire_geometry_triangles(snapshot, camera, forward):
                polygon, _depth = self._project(triangle)
                pygame.draw.polygon(self.screen, color, polygon)
                pygame.draw.polygon(self.screen, outline, polygon, 1)

            for pillar_object, roller_object in realize_fixture_geometry(self.fixture_plan, snapshot):
                for support in pillar_object.model["geometry_projection"]["polylines"]:
                    line, _ = self._project(np.asarray(support["positions"]))
                    pygame.draw.line(self.screen, support["color_rgb"], line[0], line[1],
                                     support["width_px"])
                for solid in roller_object.model["geometry_projection"]["solids"]:
                    # Legacy preview projects the same solid surface supplied
                    # to the game; it no longer substitutes a circle marker.
                    vertices = np.asarray(solid["surface"]["positions"]) + solid["position"]
                    screen, depth = self._project(vertices)
                    faces = sorted(solid["surface"]["triangles"],
                                   key=lambda face: float(np.mean(depth[face])), reverse=True)
                    for face in faces:
                        if np.all(depth[face] > 0.05):
                            pygame.draw.polygon(self.screen, solid["material"]["base_color_rgb"], screen[face])

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
