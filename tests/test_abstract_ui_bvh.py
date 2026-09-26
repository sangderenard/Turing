"""Deterministic packed BVH extraction for mesh contact compilation."""

from src.compiler.abstract_ui_bvh import build_packed_triangle_bvh


def test_bvh_uses_spectral_gpu_layout_and_stable_triangle_permutation():
    triangles = [
        ((-2, 0, 0), (-1, 0, 0), (-1, 1, 0)),
        ((2, 0, 0), (3, 0, 0), (2, 1, 0)),
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    ]
    packed = build_packed_triangle_bvh(triangles, leaf_size=1)
    model = packed.to_data()
    assert model["layout"][:4] == ["lo.x", "lo.y", "lo.z", "left"]
    assert len(packed.nodes) == 5
    assert sorted(packed.triangle_order) == [0, 1, 2]
    assert packed.nodes[0][3] == 1
    assert packed.nodes[0][7] == 2
    assert build_packed_triangle_bvh(triangles, leaf_size=1) == packed


def test_empty_mesh_has_no_fake_root_node():
    assert build_packed_triangle_bvh([]).nodes == ()
