"""Bounded authored-part manifold tests, with no live renderer or GPU work."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from meowdoku import studio_contacts as S
from src.common.tensors import AbstractTensor as T


def vec(v):
    return tuple(T.get_tensor(np.asarray(x, dtype=float)) for x in v)


def unpack(value):
    return np.asarray(value.data)


def cube():
    # Each authored quad is wound outward. Face triangles retain their order.
    quads = [
        ((1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)),
        ((-1,-1,1),(-1,1,1),(-1,1,-1),(-1,-1,-1)),
        ((-1,1,-1),(-1,1,1),(1,1,1),(1,1,-1)),
        ((-1,-1,1),(-1,-1,-1),(1,-1,-1),(1,-1,1)),
        ((-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)),
        ((1,-1,-1),(-1,-1,-1),(-1,1,-1),(1,1,-1)),
    ]
    return tuple(tuple(vec(v) for v in tri) for a,b,c,d in quads for tri in ((a,b,c),(a,c,d)))


def test_closed_cube_has_exact_face_edge_vertex_and_interior_exit():
    p = vec(([1.2,1.2,1.2,.8,0],[0,1.2,1.2,0,0],[0,0,1.2,0,0]))
    d,n,q,valid,index = S.convex_part_contact(p,cube(),closed=True)
    np.testing.assert_allclose(unpack(d), [.2,.2*2**.5,.2*3**.5,-.2,-1])
    np.testing.assert_allclose(np.array([unpack(v) for v in n]), [[1,2**-.5,3**-.5,1,1],[0,2**-.5,3**-.5,0,0],[0,0,3**-.5,0,0]])
    assert (unpack(valid) == 1).all()
    assert unpack(index)[-1] == 0  # equal interior exits retain authored identity


def test_cube_random_points_match_independent_analytic_distance():
    points = np.random.default_rng(41).uniform(-2,2,(200,3))
    d,n,q,valid,index = S.convex_part_contact(vec(points.T),cube(),closed=True)
    delta = np.maximum(np.abs(points)-1,0)
    expected = np.linalg.norm(delta,axis=1) + np.minimum(np.max(np.abs(points)-1,axis=1),0)
    np.testing.assert_allclose(unpack(d),expected,atol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(np.array([unpack(v) for v in n]),axis=0),1,atol=2e-15)
    assert (unpack(valid) == 1).all()


def test_sheet_diagonal_is_one_contact_and_retains_original_triangle_identity():
    tris = tuple(tuple(vec(v) for v in t) for t in (
        ((0,0,0),(1,0,0),(1,1,0)), ((0,0,0),(1,1,0),(0,1,0))))
    p = vec(([.5,.2,.8],[.5,.8,.2],[.1,.1,-.1]))
    d,n,q,valid,index = S.convex_part_contact(p,tris,closed=False)
    np.testing.assert_allclose(unpack(d),[.1,.1,.1])
    np.testing.assert_allclose(unpack(index),[0,1,0])
    np.testing.assert_allclose(unpack(n[2]),[1,1,-1])
    repeated = S.convex_part_contact(p,tris+tris,closed=False)
    for a,b in zip((d,*n,*q,valid,index),(repeated[0],*repeated[1],*repeated[2],repeated[3],repeated[4])):
        np.testing.assert_array_equal(unpack(a),unpack(b))


def test_perpendicular_parts_keep_two_independent_contacts():
    horizontal = ((vec((0,0,0)),vec((1,0,0)),vec((0,1,0))),)
    vertical = ((vec((0,0,0)),vec((0,0,1)),vec((0,1,0))),)
    p = vec(([.1],[.1],[.1]))
    contacts = [S.convex_part_contact(p,t,closed=False) for t in (horizontal,vertical)]
    assert sum(float(unpack(c[0])[0]) < .2 for c in contacts) == 2
    np.testing.assert_allclose([unpack(v)[0] for v in contacts[0][1]],[0,0,1])
    np.testing.assert_allclose([unpack(v)[0] for v in contacts[1][1]],[1,0,0])


def test_part_with_degenerate_face_is_explicitly_invalid():
    bad = (vec((0,0,0)),)*3
    d,n,q,valid,index = S.convex_part_contact(vec(([2],[2],[2])),cube()+(bad,),closed=True)
    assert not unpack(valid).any()
    assert all(np.isfinite(unpack(v)).all() for v in (d,*n,*q,index))
    with pytest.raises(ValueError):
        S.convex_part_contact(vec((0,0,0)),(),closed=True)


def test_adjacent_compound_parts_expose_unresolved_cross_part_seam():
    # This is a regression documenting the boundary of the primitive, not a
    # license to add these forces in a deployed compound-track product.
    left = tuple(tuple(vec(v) for v in tri) for tri in (
        ((-1,-1,0),(0,-1,0),(0,1,0)), ((-1,-1,0),(0,1,0),(-1,1,0))))
    right = tuple(tuple(vec(v) for v in tri) for tri in (
        ((0,-1,0),(1,-1,0),(1,1,0)), ((0,-1,0),(1,1,0),(0,1,0))))
    p = vec(([0,.02],[0,0],[.05,.05]))
    a = S.convex_part_contact(p,left,closed=False)
    b = S.convex_part_contact(p,right,closed=False)
    joined = S.convex_part_contact(p,left+right,closed=False)
    # Exactly on the shared edge there are two identical contacts if subparts
    # are treated independently, versus one on the actual convex planar union.
    assert unpack(a[0])[0] == unpack(b[0])[0] == unpack(joined[0])[0]
    assert unpack(a[1][0])[1] > 0  # neighbouring strip's fake lateral edge force
    assert unpack(joined[1][0])[1] == 0
    zero = p[0]*0
    def force(contact):
        d,n,q,valid,index = contact
        return S.stationary_contact(d,n,valid,(zero,)*3,(zero,)*3,
            zero+.1,zero+100,zero+1,zero,zero,zero,zero,zero)[0]
    fa,fb,fj = force(a),force(b),force(joined)
    assert unpack(fa[2]+fb[2])[0] == 2*unpack(fj[2])[0]


def test_planar_nonconvex_union_supports_faces_but_flags_concave_edge_manifold():
    # L footprint: two rectangles share an edge; the missing upper-right
    # quadrant has two different nearest boundary features at the diagonal.
    triangles = tuple(tuple(vec(v) for v in t) for t in (
        ((0,0,0),(2,0,0),(2,1,0)),((0,0,0),(2,1,0),(0,1,0)),
        ((0,1,0),(1,1,0),(1,2,0)),((0,1,0),(1,2,0),(0,2,0))))
    p = vec(([.5,.5,1.1],[.5,1,1.1],[.1,.1,.1]))
    d,n,q,valid,index,supported = S.planar_surface_contact(p,triangles,planar=True,convex_footprint=False)
    np.testing.assert_allclose(unpack(supported),[1,1,0])
    np.testing.assert_allclose(unpack(d)[:2],[.1,.1])
    np.testing.assert_allclose(unpack(n[2])[:2],[1,1])
    # Unsupported geometry is not disguised as invalid/no-contact geometry.
    assert unpack(valid)[2] == 1
    assert unpack(d)[2] < .2


def test_folded_surface_is_rejected_and_floor_rail_ownership_stays_separate():
    floor = ((vec((0,0,0)),vec((1,0,0)),vec((0,1,0))),)
    rail = ((vec((0,0,0)),vec((0,0,1)),vec((0,1,0))),)
    p = vec(([.1],[.1],[.1]))
    with pytest.raises(ValueError,match="folded"):
        S.planar_surface_contact(p,floor+rail,planar=False,convex_footprint=False)
    contacts = [S.planar_surface_contact(p,t,planar=True,convex_footprint=True) for t in (floor,rail)]
    assert all(unpack(c[5])[0] == 1 for c in contacts)
    assert all(unpack(c[0])[0] < .2 for c in contacts)
    assert unpack(contacts[0][1][2])[0] == 1
    assert unpack(contacts[1][1][0])[0] == 1


@pytest.mark.parametrize("closed", [False, True])
def test_bounded_part_contact_and_dem_force_translate_in_existing_pipeline(closed):
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    with scoped_work_contract(PRESETS["fast"]):
        program = SSATensorProgram("studio_part")
        triangle_count = 4 if closed else 2  # tetrahedron versus rectangular sheet
        names = ("x","y","z") + tuple(name for i in range(triangle_count) for name in S.triangle_input_names(i))
        bindings = {name: SSATensorOperations.input(program,(8,) if name in names[:3] else ()) for name in names}
        values = tuple(bindings[n] for n in names)
        tris = tuple(tuple(values[3+9*i+3*j:6+9*i+3*j] for j in range(3)) for i in range(triangle_count))
        with program.activate():
            if closed:
                d,n,q,valid,index = S.convex_part_contact(values[:3],tris,closed=True)
                supported = values[0]*0+1
            else:
                d,n,q,valid,index,supported = S.planar_surface_contact(values[:3],tris,planar=True,convex_footprint=False)
            zero = values[0]*0
            force,torque = S.stationary_contact(d,n,valid,(zero,zero,zero),(zero,zero,zero),zero+.1,zero+100,zero+1,zero+.2,zero+.1,zero+.01,zero,zero)
        outputs = [v.data.value for v in (d,*n,valid,index,supported,*force,*torque)]
        product = webgpu.emit_tensor_program_passes(program.module,"studio_part",outputs,count=8)
    assert product.complete, [s.format() for s in product.shortfalls]
    assert product.dispatches == 1
    assert all(int(bindings[n].data.value.id) in product.uniform_index for n in names[3:])
