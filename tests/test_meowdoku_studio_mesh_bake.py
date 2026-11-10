"""Exact whole-mesh bake/query checks, without GPU or a second simulator."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"examples"))
from meowdoku.studio_mesh_bake import StudioMeshBake,RegionAdmissionError


def faces():
    return [[[0,0,0],[1,0,0],[0,1,0],7],
            [[4,0,0],[4,1,0],[4,0,1],9],
            [[-4,0,0],[-4,1,0],[-4,0,1],11]]


def bake(data=None,**kwargs):
    data=faces() if data is None else data
    return StudioMeshBake.from_faces(data,source_ids=[f"source:{i}" for i in range(len(data))],frame_id="studio-native-metres:test",**kwargs)


def test_exact_snapshot_digest_preserves_triangle_material_and_frame_identity():
    data=faces()
    a=bake(data,leaf_size=1)
    assert a.triangles[0] == tuple(map(tuple,data[0][:3]))
    assert a.material_ids == (7,9,11)
    assert a.digest == bake(data,leaf_size=2).digest  # BVH layout is not geometry
    data[0][0][0]=.1
    assert a.triangles[0][0][0] == 0  # caller mutation cannot alter held snapshot
    assert bake(data).digest != a.digest
    with pytest.raises(ValueError,match="changed"):
        a.require_digest(bake(data).digest)
    a.require_digest(a.digest)
    data=faces();data[0][3]=8
    assert bake(data).digest != a.digest
    assert StudioMeshBake.from_faces(faces(),source_ids=a.source_ids,frame_id="other").digest != a.digest


def test_sphere_and_swept_candidates_never_truncate():
    a=bake(leaf_size=1)
    assert a.candidates((.2,.2,.1),.2) == (0,)
    assert a.candidates((-5,.1,.1),.2,end=(5,.1,.1)) == (0,1,2)
    with pytest.raises(ValueError,match="candidate capacity"):
        a.candidates((-5,.1,.1),.2,end=(5,.1,.1),max_candidates=2)
    with pytest.raises(ValueError,match="triangle capacity"):
        bake(max_triangles=2)


def test_exact_abstract_tensor_narrow_phase_retains_original_identity():
    hits=bake(leaf_size=1).contact_geometry((.2,.2,.1),.11)
    assert len(hits)==1
    hit=hits[0]
    assert (hit.source_index,hit.source_id,hit.material_id,hit.valid)==(0,"source:0",7,True)
    assert hit.distance==.1
    np.testing.assert_allclose(hit.point,[.2,.2,0])
    np.testing.assert_allclose(hit.normal,[0,0,1])


def test_bvh_candidates_cover_exhaustive_exact_triangle_queries():
    rng=np.random.default_rng(70)
    data=[[*rng.uniform(-3,3,(3,3)).tolist(),i%5] for i in range(30)]
    tree=bake(data,leaf_size=2)
    # One leaf deliberately makes the comparison broad phase exhaustive while
    # retaining exactly the same AbstractTensor narrow-phase implementation.
    exhaustive=bake(data,leaf_size=len(data))
    for point in rng.uniform(-3,3,(20,3)):
        expected=exhaustive.contact_geometry(point,.3)
        actual=tree.contact_geometry(point,.3)
        assert [h.source_index for h in actual]==[h.source_index for h in expected]
        for a,b in zip(actual,expected):
            assert a.distance==b.distance
            assert a.normal==b.normal


def test_invalid_inputs_and_degenerate_faces_are_not_silently_removed():
    with pytest.raises(ValueError,match="finite"):
        bake([[[float("nan"),0,0],[1,0,0],[0,1,0],0]])
    with pytest.raises(ValueError,match="unique"):
        StudioMeshBake.from_faces(faces(),source_ids=["same"]*3,frame_id="frame")
    degenerate=bake([[[0,0,0],[0,0,0],[0,0,0],3]])
    assert len(degenerate.triangles)==1
    hits=degenerate.contact_geometry((0,0,0),.1)
    assert len(hits)==1 and not hits[0].valid
    empty=bake([])
    assert empty.candidates((0,0,0),1)==()


def test_serializable_artifact_and_candidate_payload_preserve_every_source():
    a=bake(leaf_size=1)
    artifact=json.loads(json.dumps(a.to_data()))
    assert artifact["digest"]==a.digest
    assert artifact["triangles"]==[[list(p) for p in tri] for tri in a.triangles]
    assert artifact["bvh"]["triangle_order"]==list(a.bvh.triangle_order)
    payload=a.candidate_payload((.2,.2,.1),.2)
    assert payload["source_indices"]==[0]
    assert payload["source_ids"]==["source:0"]
    assert payload["material_ids"]==[7]
    assert payload["triangles"]==[artifact["triangles"][0]]
    assert payload["complete"] is True and payload["manifold_resolved"] is False


def test_shared_quad_enters_existing_native_environment_as_one_sheet_contact():
    from meowdoku.studio_environment import StudioEnvironment,SCHEMA
    from src.common.tensors import AbstractTensor as T
    data=[[[0,1,0],[1,1,0],[1,1,1],7],[[0,1,0],[1,1,1],[0,1,1],8]]
    a=bake(data)
    admitted=a.native_sheet_part((0,1),part_id="table-top",body_id="table")
    assert admitted["max_coordinate_rounding"]==0
    part=admitted["part"]
    assert part["sourceTriangleIds"]==["source:0","source:1"]
    assert part["materialIds"]==[7,8]
    env=StudioEnvironment({"schema":SCHEMA,"room":{"low":[-2,.25,-2],"high":[2,3,2]},"parts":[part]})
    tensor=lambda v:T.get_tensor(np.asarray(v,dtype=np.float64))
    uniforms={k:tensor(v) for k,v in env.uniforms().items()}
    records=[]
    env.contacts(tuple(tensor([v]) for v in (.5,1.05,.5)),tensor([.1]),tensor([1]),uniforms,
                 lambda overlap,valid,normal:records.append((overlap,valid,normal)))
    # Six room walls, then exactly one shared-top contact across its diagonal.
    assert len(records)==7
    touching=[r for r in records if np.asarray(r[0].data).item()>0]
    assert len(touching)==1
    np.testing.assert_allclose(np.asarray(touching[0][0].data),.05)
    np.testing.assert_allclose([np.asarray(v.data).item() for v in touching[0][2]],[0,1,0])


def test_native_sheet_admission_refuses_folded_and_nonconvex_ownership():
    folded=bake([[[0,0,0],[1,0,0],[0,1,0],0],[[0,0,0],[0,1,0],[0,0,1],0]])
    with pytest.raises(ValueError,match="planar"):
        folded.native_sheet_part((0,1),part_id="fold",body_id="body")
    lshape=bake([[[0,0,0],[2,0,0],[2,1,0],0],[[0,0,0],[2,1,0],[1,1,0],0],
                 [[0,0,0],[1,1,0],[1,2,0],0],[[0,0,0],[1,2,0],[0,2,0],0]])
    with pytest.raises(ValueError,match="convex"):
        lshape.native_sheet_part((0,1,2,3),part_id="L",body_id="body")


def test_region_admission_is_bounded_and_reports_every_unknown_candidate():
    mesh=bake(leaf_size=1)
    sheet=mesh.native_sheet_part((0,),part_id="sheet",body_id="sheet")
    report=mesh.admit_region((.2,.2,.1),.2,admitted_parts=[sheet])
    assert report["admitted"] and report["owners"]=={"source:0":"part:sheet"}
    assert not report["global_scene_supported"]
    assert not report["inside_outside_certified"]
    assert report["unaccounted_source_ids"]==["source:1","source:2"]
    with pytest.raises(RegionAdmissionError) as exc:
        mesh.admit_region((3.9,.2,.2),.2,admitted_parts=[sheet])
    assert exc.value.report["unknown_source_ids"]==["source:1"]
    assert exc.value.report["coverage"]["point"]==[3.9,.2,.2]
    stale=dict(sheet["part"]);stale["meshDigest"]="stale"
    with pytest.raises(ValueError,match="changed"):
        mesh.admit_region((.2,.2,.1),.2,admitted_parts=[{"part":stale}])


def test_region_room_source_claim_requires_exact_inward_plane_and_bounds():
    mesh=bake(leaf_size=1)
    room={"low":[-1,-1,0],"high":[2,2,2]}
    report=mesh.admit_region((.2,.2,.1),.2,room=room,room_source_indices=[0])
    assert report["owners"]=={"source:0":"room:2:0"}
    with pytest.raises(ValueError,match="bounds"):
        mesh.admit_region((.2,.2,.1),.2,room=room,room_source_indices=[1])
    with pytest.raises(ValueError,match="inward"):
        mesh.admit_region((.2,.2,-.1),.2,room={"low":[-1,-1,-2],"high":[2,2,0]},room_source_indices=[0])


@pytest.mark.skipif(shutil.which("node") is None,reason="Node unavailable")
def test_actual_shared_course_faces_bake_as_one_mesh_without_convex_proxy():
    source=(ROOT/"examples/meowdoku/web/marble_studio_courses.js").as_uri()
    script=f"import {{makeStudioCourse}} from {json.dumps(source)}; console.log(JSON.stringify(makeStudioCourse({{kind:'gentle-s',stage:{{width:8,length:6,baseY:0}},materialId:3,railMaterialId:4,segments:16}})));"
    packet=json.loads(subprocess.run(["node","--input-type=module","-e",script],check=True,capture_output=True,text=True,timeout=10).stdout)
    ids=[f"{r['partId']}/triangle/{r['triangleIndex']}" for r in packet["faceRecords"]]
    mesh=StudioMeshBake.from_faces(packet["faces"],source_ids=ids,frame_id="shared-studio-units:test")
    assert len(mesh.triangles)==len(packet["faces"])
    assert mesh.source_ids==tuple(ids)
    for tri,face in zip(mesh.triangles,packet["faces"]):
        assert tri==tuple(map(tuple,face[:3]))
    # A ray/physics snapshot can compare this immutable geometry digest before
    # revealing work. No course contact-readiness flag is changed by a bake.
    assert packet["collisionReady"] is False
    assert mesh.bvh.nodes
    assert sorted(mesh.bvh.triangle_order)==list(range(len(mesh.triangles)))


@pytest.mark.skipif(shutil.which("node") is None,reason="Node unavailable")
def test_shared_room_table_backdrop_and_gobo_builders_enter_one_exact_bake():
    directory=(ROOT/"examples/meowdoku/web").as_uri()
    script=f"""
      import {{studioRoomFaces}} from '{directory}/marble_room.js';
      import {{studioTableFaces,defaultStudioTable,tableFloorDrop}} from '{directory}/marble_studio_table.js';
      import {{backdropFaces,defaultBackdropLayers}} from '{directory}/marble_backdrop.js';
      import {{studioGoboFaces}} from '{directory}/marble_studio_gobos.js';
      const ids={{marble_clear_glass:3,stone_onyx:4}};
      const table=defaultStudioTable();
      const groups={{room:studioRoomFaces({{floorDrop:tableFloorDrop(table),material:()=>4}}),
        table:studioTableFaces(table,ids),
        backdrop:backdropFaces({{count:1,material:'stone_onyx',sweep:true,layers:defaultBackdropLayers()}},ids),
        gobo:studioGoboFaces({{version:1,gobos:[{{id:1,enabled:true,shape:'ellipse',width:2,height:3,x:2,y:1,z:0,pitch:17,yaw:31,roll:9,front:'stone_onyx',back:'marble_clear_glass'}}]}},ids)}};
      console.log(JSON.stringify(groups));
    """
    groups=json.loads(subprocess.run(["node","--input-type=module","-e",script],check=True,capture_output=True,text=True,timeout=10).stdout)
    data=[f for group in groups.values() for f in group]
    identities=[f"{name}:{i}" for name,group in groups.items() for i in range(len(group))]
    mesh=StudioMeshBake.from_faces(data,source_ids=identities,frame_id="shared-studio-units:fixture")
    assert len(mesh.triangles)==sum(map(len,groups.values()))
    for tri,face in zip(mesh.triangles,data):
        assert tri==tuple(map(tuple,face[:3]))
    assert set(mesh.material_ids)=={3,4}
    top=tuple(identities.index(f"table:{i}") for i in (6,7))
    admitted=mesh.native_sheet_part(top,part_id="shared-tabletop",body_id="table")
    assert admitted["part"]["sourceTriangleIds"]==["table:6","table:7"]
    assert admitted["signed_volume"] is False
    slab_indices=tuple(identities.index(f"table:{i}") for i in range(12))
    slab=mesh.native_convex_part(slab_indices,part_id="shared-table-slab",body_id="table",closed=True)
    assert slab["signed_volume"] is True
    assert slab["part"]["sourceTriangleIds"]==[f"table:{i}" for i in range(12)]
    assert slab["max_coordinate_rounding"]>0
    # Actual non-axis-aligned ellipse source, not an invented roundoff case.
    # Strict rational native-f32 admission must refuse warped convex faces,
    # rather than fitting the mesh to a nearby ideal ellipse extrusion.
    gobo_indices=tuple(i for i,name in enumerate(identities) if name.startswith("gobo:"))
    with pytest.raises(ValueError,match="convex"):
        mesh.native_convex_part(gobo_indices,part_id="rotated-gobo",body_id="gobo",closed=True)
