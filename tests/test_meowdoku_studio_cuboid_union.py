import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"examples"))
from meowdoku.studio_mesh_bake import StudioMeshBake
from meowdoku.studio_cuboid_union import build_cuboid_union
from meowdoku.studio_mesh_manifold import NativeMeshBoundary
from src.common.tensors import AbstractTensor as T


def box(low,high,material=3):
    faces=[]
    for axis in range(3):
        u,v=(axis+1)%3,(axis+2)%3
        for side in (0,1):
            points=[]
            for du,dv in ((0,0),(1,0),(1,1),(0,1)):
                p=[0,0,0];p[axis]=(low,high)[side][axis]
                p[u]=(low,high)[du][u];p[v]=(low,high)[dv][v];points.append(p)
            if not side:points.reverse()
            a,b,c,d=points;faces.extend(([a,b,c,material],[a,c,d,material]))
    return faces


def source(groups,bodies=None):
    faces=[];parts=[];ids=[]
    for i,group in enumerate(groups):
        start=len(faces);faces.extend(group);ids.extend(f"part:{i}:tri:{j}" for j in range(len(group)))
        parts.append({"id":f"part:{i}","body_id":"same-body" if bodies is None else bodies[i],
                      "closed":True,"source_indices":tuple(range(start,len(faces)))})
    return StudioMeshBake.from_faces(faces,source_ids=ids,frame_id="native-source-fixture"),parts


def native(union):
    return NativeMeshBoundary(union.vertices,union.triangles,union.triangle_ids,
        closed=True,body_id=union.body_id,embedded_boundary_evidence=union.embedded_boundary_evidence)


def contacts(boundary,p,r=.1):
    p=tuple(T.get_tensor(np.asarray([v],dtype=np.float32)) for v in p)
    result,diag=boundary.query(p)
    scalar=lambda v:np.asarray(v.data).item()
    assert scalar(diag["geometry_valid"])==scalar(diag["nearest_found"])==scalar(diag["finite"])==1
    assert scalar(diag["sign_ambiguous"])==scalar(diag["singular_features"])==0
    return [c for c in result if scalar(c.valid)>0 and scalar(c.distance)<r],{k:scalar(v) for k,v in diag.items()}


def test_touching_cuboids_become_one_exact_outer_boundary_with_internal_provenance():
    mesh,parts=source([box((-1,-1,-1),(1,1,1)),box((1,-1,-1),(3,1,1))])
    union=build_cuboid_union(mesh,parts)
    assert union.exact_volume_ratio==(16,1)
    assert union.embedded_boundary_evidence.startswith("turing.exact-cuboid-arrangement:")
    assert dict(union.proof)["source_boxes_overlap"] is False
    assert union.internal_interfaces
    accounted=set(s for group in union.triangle_provenance for s in group)
    accounted.update(s for row in union.internal_interfaces for s in dict(row)["source_ids"])
    assert accounted==set(mesh.source_ids)
    hits,diag=contacts(native(union),(1,1.05,.2))
    assert len(hits)==1 and diag["inside"]==0  # join is not two independent walls
    inside,diag=contacts(native(union),(1,.2,.1))
    assert len(inside)==1 and diag["inside"]==1
    assert diag["nearest_distance"]<-.7  # no false internal plane at x=1


def test_overlapping_same_body_boxes_use_exact_interval_union_not_double_volume():
    mesh,parts=source([box((-1,-1,-1),(1,1,1)),box((0,-1,-1),(2,1,1))])
    union=build_cuboid_union(mesh,parts)
    assert union.exact_volume_ratio==(12,1)
    assert dict(union.proof)["source_boxes_overlap"] is True
    assert union.internal_interfaces
    native(union)  # outer boundary remains a connected manifold


def test_unrelated_bodies_rotated_boxes_and_ambiguous_materials_refuse():
    mesh,parts=source([box((-1,-1,-1),(1,1,1)),box((1,-1,-1),(3,1,1))],bodies=["a","b"])
    with pytest.raises(ValueError,match="body"):
        build_cuboid_union(mesh,parts)
    mesh,parts=source([box((-1,-1,-1),(1,1,1),3),box((-1,-1,-1),(1,1,1),4)])
    with pytest.raises(ValueError,match="conflicting materials"):
        build_cuboid_union(mesh,parts)
    mixed=box((-1,-1,-1),(1,1,1));mixed[0][3]=9
    mesh,parts=source([mixed])
    with pytest.raises(ValueError,match="material boundary"):
        build_cuboid_union(mesh,parts)
    rotated=[]
    for face in box((-1,-1,-1),(1,1,1)):
        rotated.append([[.6*p[0]-.8*p[1],.8*p[0]+.6*p[1],p[2]] for p in face[:3]]+[3])
    mesh,parts=source([rotated])
    with pytest.raises(ValueError,match="axis-aligned|convex"):
        build_cuboid_union(mesh,parts)


def test_edge_touching_components_do_not_receive_false_embedding_certificate():
    single,single_parts=source([box((-1,-1,-1),(1,1,1))])
    assert build_cuboid_union(single,single_parts,max_cells=1).exact_volume_ratio==(8,1)
    mesh,parts=source([box((-1,-1,-1),(1,1,1)),box((1,1,-1),(3,3,1))])
    with pytest.raises(ValueError,match="edge|manifold|disconnected"):
        build_cuboid_union(mesh,parts)
    mesh,parts=source([box((-1,-1,-1),(1,1,1)),box((1,-1,-1),(3,1,1))])
    with pytest.raises(ValueError,match="capacity"):
        build_cuboid_union(mesh,parts,max_cells=1)


@pytest.mark.skipif(shutil.which("node") is None,reason="Node unavailable")
def test_actual_shared_table_slab_legs_become_one_native_outer_union():
    from meowdoku.studio_scene_prepare import StudioSourcePart
    module=(ROOT/"examples/meowdoku/web/marble_studio_table.js").as_uri()
    script=f"import {{studioTableFaces,defaultStudioTable}} from {json.dumps(module)};console.log(JSON.stringify(studioTableFaces(defaultStudioTable(),{{marble_clear_glass:3}})));"
    faces=json.loads(subprocess.run(["node","--input-type=module","-e",script],check=True,capture_output=True,text=True,timeout=10).stdout)
    assert len(faces)==60
    mesh,parts=source([faces[i:i+12] for i in range(0,60,12)])
    parts=[StudioSourcePart(p["id"],p["body_id"],"closed-cuboid",True,True,p["source_indices"]) for p in parts]
    union=build_cuboid_union(mesh,parts)
    assert len(union.internal_interfaces)==4
    assert set(union.material_ids)=={3}
    assert union.max_coordinate_rounding>0
    for axis in range(3):
        original={float(np.float32(p[axis])) for t in mesh.triangles for p in t}
        assert {p[axis] for p in union.vertices}<=original  # cuts are existing planes only
    body=native(union)
    top,diag=contacts(body,(0,-.95,0))
    assert len(top)==1 and diag["inside"]==0
    corner,diag=contacts(body,(-2.2,-1.18,-1.25))
    assert len(corner)==2 and diag["inside"]==0
    normals=[np.array([np.asarray(v.data).item() for v in c.normal]) for c in corner]
    assert any(n[0]>.99 for n in normals) and any(n[1]<-.99 for n in normals)
    stem,diag=contacts(body,(-2.35,-2,-1.25))
    assert len(stem)==1 and diag["inside"]==1
    assert -.091<diag["nearest_distance"]<-.089
    serialized=union.to_data()
    assert serialized["proof"]["source_accounting"]=="all-source-triangles-exterior-support-or-internal-interface"
