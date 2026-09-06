import numpy as np

from src.compiler.vehicle_python_live_viewer import (
    part_geometry_lines, tire_geometry_triangles, fixture_geometry, PythonValidatorViewer,
)
from src.compiler.abstract_ui import AbstractUI
from src.compiler.abstract_ui_geometry import realize_part_geometry


def test_fixture_objects_use_negotiated_identities_and_actual_mechanical_state():
    from types import SimpleNamespace
    from src.compiler.abstract_ui_geometry import realize_fixture_geometry
    from src.compiler.vehicle_native_assembly import PillarArmPlan, RollerCoveragePlan

    pillar = PillarArmPlan("rig/support", "wheel", "hub", (0, 1, 0), ("wheel",),
                          (1, 2, 3), (4, 5, 6))
    roller = RollerCoveragePlan("rig/mounter", "tire-mounting", "per-wheel-pair",
                                ("wheel",), ("hub",), "vertical-mount-detent", "mount",
                                roller_length_m=0.4)
    plan = SimpleNamespace(wheel_identities=("wheel",), pillars=(pillar,),
                           tire_mounting_rollers=(roller,))
    snapshot = dict(pillar_pose=[[1, 2, 3]], pillar_alpha=[1], roller_anchor=[[5, 6]],
                    fixture_wheel=[[4, 0.25, 120, 8, 180]],
                    fixture_command=[[0, 0, 0, 0, 4.5, 0.5, 0, 1]],
                    pillar_reaction_force=[20])
    support, mount = realize_fixture_geometry(plan, snapshot)[0]
    assert support.identity == pillar.identity and mount.identity == roller.identity
    assert support.model["hub_identity"] == "hub"
    assert mount.model["wheel_identities"] == ("wheel",)
    assert support.model["mechanical_state"]["reaction_force_y_n"] == 20
    state = mount.model["mechanical_state"]
    assert state["command_y"] == 4.5 and state["carriage_y"] == 4
    assert state["actuator_force"] == 120 and state["passive_force"] == 8
    before = mount.model["geometry_projection"]["solids"]
    snapshot["fixture_wheel"][0][0] = 7
    moved = realize_fixture_geometry(plan, snapshot)[0][1]
    assert [p["identity"] for p in before] == [p["identity"] for p in moved.model["geometry_projection"]["solids"]]
    assert moved.model["geometry_projection"]["solids"][0]["position"][1] == 7
    assert before[0]["position"][1] == 4
    assert "markers" not in mount.model["geometry_projection"]
    assert before[0]["form"] == {"primitive": "cylinder", "radius_m": 0.18, "length_m": 0.4}
    surface = before[0]["surface"]
    vertices, normals = np.asarray(surface["positions"]), np.asarray(surface["normals"])
    faces = np.asarray(surface["triangles"])
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [0.36, 0.36, 0.4])
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                     vertices[faces[:, 2]] - vertices[faces[:, 0]])
    assert np.all(np.sum(cross * normals[faces].mean(axis=1), axis=1) > 0)
    from src.compiler.abstract_ui_geometry import fixture_world_objects
    from src.compiler.abstract_ui_world import world_surface_mesh_packet
    objects = fixture_world_objects(((support, mount),), "world")
    packet = world_surface_mesh_packet(objects)
    assert len(objects) == 3 and objects[1].parent == roller.identity
    assert objects[0].physics["state"]["actuator_force"] == 120
    assert objects[0].extensions["mechanical_connections"]["hubs"] == ("hub",)
    assert [s["identity"] for s in packet["object_spans"]] == [s["identity"] for s in before]
    packed = np.asarray(packet["vertices"])
    assert packed.shape == (2 * faces.size, 9)
    np.testing.assert_allclose(packed[:faces.size, :3],
                               vertices[faces.reshape(-1)] + before[0]["position"])
    np.testing.assert_allclose(packed[:faces.size, 3:6], normals[faces.reshape(-1)])
    assert packet["semantic_part_spans"][1]["first_vertex"] == faces.size
    from dataclasses import replace
    rotation = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    rotated = replace(objects[1], transform={"position": [2, 3, 4],
                                             "rotation_matrix": rotation.tolist()})
    rotated_vertices = np.asarray(world_surface_mesh_packet((rotated,))["vertices"])
    np.testing.assert_allclose(rotated_vertices[:, :3], vertices[faces.reshape(-1)] @ rotation.T + [2, 3, 4])
    np.testing.assert_allclose(rotated_vertices[:, 3:6], normals[faces.reshape(-1)] @ rotation.T)


def test_authored_part_geometry_preserves_shape_width_and_world_coordinates():
    center = np.array([2.0, 3.0, 4.0])
    dimensions = dict(axis=(0, 0, 2), radius_m=0.5, bead_seat_radius_m=0.4,
                      width_m=0.2, axial_offset_m=0.1, outer_diameter_m=0.8,
                      bore_m=0.2, flange_radius_m=0.3, barrel_radius_m=0.2,
                      length_m=1.0, tube_radius_m=0.1, center_radius_m=0.2)
    cases = {
        "wheel-center-disc": [3, 2, 2, 2, 2, 2, 2],
        "drop-center-rim": [3, 3, 2],
        "bead-ring": [4],
        "bearing-races": [4, 2],
        "wheel-mounting-hub": [4, 3],
        "brake-drum": [3, 3],
        "axial-structural-casing": [3, 3, 2, 2, 2, 2, 2, 2, 5],
    }
    for primitive, widths in cases.items():
        lines = part_geometry_lines(center, {"geometry": dict(dimensions, primitive=primitive)},
                                    (200, 100, 0), 0.5)
        assert [line[2] for line in lines] == widths
        for points, shade, width, closed in lines:
            assert shade == (135, 67, 0)
            assert np.isfinite(points).all()
            assert points.shape == ((41, 3) if closed else (2, 3))
            if closed:
                np.testing.assert_allclose(points[0], points[-1])
        if primitive == "wheel-center-disc":
            np.testing.assert_allclose(np.linalg.norm(lines[0][0] - center, axis=1), 0.5)
            for points, _, _, _ in lines[1:]:
                np.testing.assert_array_equal(points[0], center)
        if primitive == "axial-structural-casing":
            np.testing.assert_allclose(lines[0][0][:, 2], 3.5)
            np.testing.assert_allclose(lines[1][0][:, 2], 4.5)
    assert part_geometry_lines(center, {}, (1, 2, 3), 1.0) == []
    assert part_geometry_lines(center, {"geometry": {"primitive": "solver-membrane"}},
                               (1, 2, 3), 1.0) == []


def test_tire_geometry_preserves_winding_material_depth_and_center_surface():
    vertices = np.array([
        [0, 0, 2], [1, 0, 2], [0, 1, 2],
        [0, 0, 4], [1, 0, 4], [0, 1, 4],
        [0, 0, -1], [1, 0, -1], [0, 1, -1],
    ], dtype=np.float64)
    faces = np.array([[0, 2, 1], [3, 4, 5], [6, 7, 8]])
    snapshot = dict(tire_position=vertices[None, :, :], tire_faces=faces,
                    tire_face_zones=("tread", "bead", "sidewall"),
                    tire_face_material=np.array([[0.014], [0.028], [0.014]]))
    geometry = tire_geometry_triangles(snapshot, np.zeros(3), np.array([0, 0, 1]))
    assert len(geometry) == 2  # Face behind the camera is withheld.
    far, near = geometry
    np.testing.assert_array_equal(far[0], vertices[faces[1]])
    np.testing.assert_array_equal(near[0], vertices[faces[0]])
    assert far[1:] == ((108, 178, 167), (112, 225, 211))
    assert near[1:] == ((70, 76, 80), (171, 181, 188))
    # Material changes shade, never the invariant center-surface positions.
    snapshot["tire_face_material"][:] = 0
    darker = tire_geometry_triangles(snapshot, np.zeros(3), np.array([0, 0, 1]))
    np.testing.assert_array_equal(darker[0][0], far[0])
    assert darker[0][1] == (66, 108, 102)


def test_part_projection_remains_an_identified_abstract_ui_object():
    node = {"identity": "axle/wheel/rim", "owner": "axle/wheel",
            "fastens_to": "axle/hub", "geometry": {
                "primitive": "drop-center-rim", "radius_m": 0.5,
                "bead_seat_radius_m": 0.4, "width_m": 0.2,
            }}
    first = realize_part_geometry(node, np.zeros(3), (100, 120, 140), 1.0)
    moved = realize_part_geometry(node, np.array([1, 2, 3]), (100, 120, 140), 1.0)
    assert isinstance(first, AbstractUI)
    assert first.identity == moved.identity == node["identity"]
    assert moved.model["fastens_to"] == node["fastens_to"]
    assert moved.model["owner"] == node["owner"]
    assert moved.model["geometry"] == node["geometry"]
    before = first.model["geometry_projection"]["polylines"]
    after = moved.model["geometry_projection"]["polylines"]
    assert [line["identity"] for line in before] == [line["identity"] for line in after]
    for left, right in zip(before, after):
        assert right["owner"] == node["identity"]
        np.testing.assert_allclose(np.asarray(right["positions"]) - left["positions"],
                                   np.tile([1, 2, 3], (41, 1)), atol=1e-14)
    assert "geometry_projection" not in node


def test_fixture_and_graph_geometry_preserve_poses_and_stage_visibility():
    fixtures = fixture_geometry(dict(
        pillar_pose=[[1, 2, 3]], pillar_alpha=[0.5],
        fixture_wheel=[[4]], roller_anchor=[[5, 6]],
    ))
    support, color, width, rollers, marker_color, radius, stroke = fixtures[0]
    np.testing.assert_array_equal(support, [[1, -0.75, 3], [1, 2, 3]])
    np.testing.assert_allclose(rollers, [[4.82, 4, 6], [5.18, 4, 6]])
    assert (color, width, marker_color, radius, stroke) == ((220, 173, 61), 5, (196, 202, 207), 8, 2)
    viewer = PythonValidatorViewer.__new__(PythonValidatorViewer)
    viewer.stage_index = {"install": 0, "run": 1}
    edge = {"identity": "axle/drive", "nodes": ["axle", "wheel"],
            "edge_class": "drivetrain", "assembly_stage": "run"}
    viewer.graph_edges = [(0, 1, edge)]
    positions = np.array([[0, 0, 1], [1, 0, 2]])
    hidden = viewer.graph_geometry_objects(positions, [1, 2], [1, 1], "install", 1)[0]
    visible = viewer.graph_geometry_objects(positions, [1, 2], [1, 1], "run", 0.5)[0]
    assert isinstance(hidden, AbstractUI) and isinstance(visible, AbstractUI)
    assert hidden.identity == visible.identity == edge["identity"]
    assert hidden.model["nodes"] == visible.model["nodes"] == edge["nodes"]
    assert hidden.model["geometry_projection"]["polylines"] == []
    assert hidden.model["geometry_projection"]["visible"] is False
    line = visible.model["geometry_projection"]["polylines"][0]
    np.testing.assert_array_equal(line["positions"], positions)
    assert (line["color_rgb"], line["width_px"]) == ([70, 124, 158], 3)
    assert line["owner"] == edge["identity"]
    assert "geometry_projection" not in edge
