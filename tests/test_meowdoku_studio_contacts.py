"""Bounded geometry/translator proof; not a compiled fountain integration test."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from meowdoku import studio_contacts as S
from meowdoku import fountain3d as F
from src.common.tensors import AbstractTensor as T


def vec(values):
    return tuple(T.get_tensor(np.asarray(v, dtype=np.float64)) for v in values)


def array(t):
    return np.asarray(t.data)


def test_face_edge_vertex_both_sides_and_zero_distance():
    p = vec(([.25, .25, 1, -1, .25], [.25, .25, 1, 0, .25], [2, -2, 0, 0, 0]))
    d, n, q, valid = S.closest_triangle(p, vec((0, 0, 0)), vec((1, 0, 0)), vec((0, 1, 0)))
    np.testing.assert_allclose(array(d), [2, 2, np.sqrt(.5), 1, 0])
    np.testing.assert_allclose(np.array([array(v) for v in q]), [[.25,.25,.5,0,.25],[.25,.25,.5,0,.25],[0,0,0,0,0]])
    np.testing.assert_allclose(np.array([array(v) for v in n]), [[0,0,2**-.5,-1,0],[0,0,2**-.5,0,0],[1,-1,0,0,1]])
    assert array(valid) == 1


@pytest.mark.parametrize("vertices", [((0,0,0),(0,0,0),(0,0,0)), ((0,0,0),(1,0,0),(2,0,0))])
def test_collapsed_geometry_is_finite_and_not_a_contact(vertices):
    with np.errstate(all="raise"):
        d, n, q, valid = S.closest_triangle(vec(([0,1],[0,1],[0,0])), *(vec(v) for v in vertices))
    assert array(valid) == 0
    assert all(np.isfinite(array(v)).all() for v in (d, *n, *q))
    assert all((array(v) == 0).all() for v in n)


def test_rigid_rotation_translation_and_winding():
    # Non-axis-aligned orthonormal rotation, then translated far from origin.
    rotation = np.array([[1,2,2],[2,1,-2],[-2,2,-1]], dtype=float) / 3
    shift = np.array([7,-3,11])
    vertices = np.array([[0,0,0],[2,0,0],[0,2,0]], dtype=float)
    points = np.array([[.5,.5,.4],[2,2,-.3],[-1,0,0]])
    expected = S.closest_triangle(vec(points.T), *(vec(v) for v in vertices))
    transformed = S.closest_triangle(vec((points @ rotation.T + shift).T), *(vec(v) for v in (vertices @ rotation.T + shift)))
    np.testing.assert_allclose(array(transformed[0]), array(expected[0]), atol=1e-14)
    np.testing.assert_allclose(np.array([array(v) for v in transformed[1]]), rotation @ np.array([array(v) for v in expected[1]]), atol=1e-14)
    reverse = S.closest_triangle(vec(points.T), *(vec(v) for v in vertices[::-1]))
    np.testing.assert_allclose(array(reverse[0]), array(expected[0]))
    np.testing.assert_allclose(np.array([array(v) for v in reverse[1]]), np.array([array(v) for v in expected[1]]))


def test_shared_diagonal_has_identical_geometry_without_summing_forces():
    p = vec(([.4,.5,.6],[.6,.5,.4],[.2,.2,.2]))
    first = S.closest_triangle(p, vec((0,0,0)), vec((1,0,0)), vec((0,1,0)))
    second = S.closest_triangle(p, vec((1,1,0)), vec((0,1,0)), vec((1,0,0)))
    np.testing.assert_allclose(array(first[0]), array(second[0]))
    np.testing.assert_allclose(np.array([array(v) for v in first[1]]), np.array([array(v) for v in second[1]]))


def test_small_nonzero_triangle_is_not_replaced_by_an_epsilon_surface():
    scale = 1e-7
    d,n,q,valid = S.closest_triangle(vec((scale/4,scale/4,scale)), vec((0,0,0)), vec((scale,0,0)), vec((0,scale,0)))
    assert array(valid) == 1
    np.testing.assert_allclose(array(d), scale, rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.array([array(v) for v in n]).reshape(3), [0,0,1], atol=1e-14)


def test_triangle_names_are_unique_and_do_not_encode_material_physics():
    names = S.triangle_input_names(0) + S.triangle_input_names(1)
    assert len(set(names)) == 18
    assert all(name.startswith("studio_t") for name in names)
    for bad in (-1, True, .5):
        with pytest.raises(ValueError):
            S.triangle_input_names(bad)


def test_room_restores_outside_centres_and_keeps_corner_contacts():
    planes = S.room_planes(vec(([-2, .9], [-3, .9], [-4, .9])), vec((-1,-1,-1)), vec((1,1,1)))
    np.testing.assert_allclose([array(v[0])[0] for v in planes], [-1,3,-2,4,-3,5])
    assert sum(array(d)[1] < .2 for d, _ in planes) == 3
    for axis in range(3):
        assert array(planes[2*axis][1][axis])[0] == 1
        assert array(planes[2*axis+1][1][axis])[0] == -1


def test_stationary_contact_is_the_existing_dem_force_and_torque():
    d, radius, kn, mass, mu, zeta, roll, hertz, coh = vec((-.1,.2,400,1,.3,.2,.01,.5,0))
    n, v, w = vec((0,1,0)), vec((1,-2,3)), vec((.1,.2,.3))
    valid = d * 0 + 1
    actual = S.stationary_contact(d,n,valid,v,w,radius,kn,mass,mu,zeta,roll,hertz,coh)
    expected = F._contact(F._cap(radius-d,6*radius),valid,n,v,F._cross(tuple(radius*x for x in w),n),w,kn,mass,mu,zeta,roll,radius,radius,radius,hertz,coh)
    for a, b in zip(actual, expected):
        for x, y in zip(a,b):
            np.testing.assert_array_equal(array(x),array(y))


def test_triangle_slice_uses_existing_named_input_ssa_webgpu_translator():
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    with scoped_work_contract(PRESETS["fast"]):
        program = SSATensorProgram("studio_triangle")
        names = ("x","y","z") + S.triangle_input_names(0)
        bindings = {name: SSATensorOperations.input(program, (8,) if name in names[:3] else ()) for name in names}
        values = tuple(bindings[n] for n in names)
        with program.activate():
            d, n, q, valid = S.closest_triangle(values[:3],values[3:6],values[6:9],values[9:12])
        outputs = [v.data.value for v in (d,*n,*q,valid)]
        product = webgpu.emit_tensor_program_passes(program.module,"studio_triangle",outputs,count=8)
    assert product.complete, [s.format() for s in product.shortfalls]
    assert all(int(bindings[name].data.value.id) in product.uniform_index for name in S.triangle_input_names(0))
    assert product.dispatches == 1
    assert "sqrt(" in product.source
