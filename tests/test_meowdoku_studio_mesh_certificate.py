import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"examples"))
from meowdoku.studio_mesh_bake import StudioMeshBake
from meowdoku.studio_mesh_certificate import component_seed_certificate


def cube_faces():
    quads=[((1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)),
           ((-1,-1,1),(-1,1,1),(-1,1,-1),(-1,-1,-1)),
           ((-1,1,-1),(-1,1,1),(1,1,1),(1,1,-1)),
           ((-1,-1,1),(-1,-1,-1),(1,-1,-1),(1,-1,1)),
           ((-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)),
           ((1,-1,-1),(-1,-1,-1),(-1,1,-1),(1,1,-1))]
    return [[a,b,c,0] for a,b,c,d in quads for a,b,c in ((a,b,c),(a,c,d))]


def mesh(faces=None):
    faces=cube_faces() if faces is None else faces
    return StudioMeshBake.from_faces(faces,source_ids=[f"cube:{i}" for i in range(len(faces))],frame_id="exact-cube-fixture")


def certify(point,**kw):
    baked=mesh()
    return component_seed_certificate(baked,range(12),point,component_id="cube",closed=True,**kw)


def test_exact_source_parity_certifies_inside_and_outside_with_independent_embedding_evidence():
    inside=certify((0,.2,.1),embedded_boundary_evidence="fixture:axis-aligned-cuboid")
    outside=certify((-2,.2,.1),embedded_boundary_evidence="fixture:axis-aligned-cuboid")
    assert inside["certified"] and inside["position"]=="inside" and inside["parity"]==1
    assert outside["certified"] and outside["position"]=="outside" and outside["parity"]==0
    assert len(inside["intersection_source_ids"])==1
    assert len(outside["intersection_source_ids"])==2
    assert not inside["contact_clearance_certified"] and not inside["native_rounding_certified"]


def test_watertightness_alone_never_claims_embedded_solid():
    result=certify((0,.2,.1))
    assert not result["certified"]
    assert result["position"]=="inside"
    assert result["blockers"][0]["reason"]=="missing-independent-embedded-boundary-proof"


def test_edge_vertex_coplanar_and_boundary_rays_refuse_without_perturbation():
    cases=[((0,.25,.25),"ray-hits-source-edge-or-vertex"),
           ((0,1,.1),"ray-coplanar-with-source-face"),
           ((1,.2,.1),"seed-on-source-boundary")]
    for point,reason in cases:
        result=certify(point,embedded_boundary_evidence="fixture:axis-aligned-cuboid")
        assert not result["certified"]
        assert reason in [b["reason"] for b in result["blockers"]]
        assert all(b["source_ids"] for b in result["blockers"])


def test_open_or_disconnected_boundaries_cannot_pose_as_one_closed_component():
    incomplete=mesh(cube_faces()[:-1])
    result=component_seed_certificate(incomplete,range(11),(0,.2,.1),component_id="open",closed=True)
    assert not result["certified"]
    assert any(b["reason"]=="closed-edge-not-paired-once" for b in result["blockers"])
    original=cube_faces()
    other=[[[p[0]+5,p[1],p[2]] for p in f[:3]]+[0] for f in original]
    disconnected=mesh(original+other)
    result=component_seed_certificate(disconnected,range(24),(0,.2,.1),component_id="two",closed=True)
    assert not result["certified"]
    assert any(b["reason"]=="component-has-disconnected-boundaries" for b in result["blockers"])


def test_open_sheet_explicitly_has_no_volume_certificate():
    baked=mesh(cube_faces()[:1])
    result=component_seed_certificate(baked,[0],(0,.2,.1),component_id="sheet",closed=False)
    assert result["position"]=="sheet-no-volume"
    assert "parity" not in result
