import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"examples"))
from meowdoku.studio_mesh_embedding import verify_native_embedding,NativeEmbeddingError
from meowdoku.studio_mesh_manifold import NativeMeshBoundary


def proof(vertices,triangles,**kwargs):
    return verify_native_embedding(vertices,triangles,[f"source:{i}" for i in range(len(triangles))],**kwargs)


def test_coplanar_and_folded_adjacent_faces_share_only_their_indexed_edge():
    vertices=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
    result=proof(vertices,[(0,1,2),(0,2,3)])
    assert result.exact_pairs==1
    assert result.to_data()["geometric_embedding"] is True
    assert result.to_data()["manifold_topology_certified"] is False
    folded=proof([(0,0,0),(1,0,0),(0,1,0),(0,0,1)],[(0,1,2),(1,0,3)])
    assert folded.evidence_id.startswith("turing.native-f32-embedding:")


@pytest.mark.parametrize("vertices,triangles",[
    ([(0,0,0),(2,0,0),(0,2,0),(.2,.2,-1),(.2,.2,1),(1,1,0)],[(0,1,2),(3,4,5)]),
    ([(0,0,0),(2,0,0),(0,2,0),(.1,.1,0),(.2,.1,0),(.1,.2,0)],[(0,1,2),(3,4,5)]),
    ([(0,0,0),(2,0,0),(0,2,0),(0,0,0),(-1,0,0),(0,-1,0)],[(0,1,2),(3,4,5)]),
    ([(0,0,0),(2,0,0),(0,2,0),(1,1,0)],[(0,1,2),(0,1,3)]),
    ([(0,0,0),(2,0,0),(0,2,0),(1,-1,0),(1,1,0)],[(0,1,2),(0,3,4)]),
])
def test_crossing_coplanar_overlap_or_nonindexed_contact_refuses_with_pair_ids(vertices,triangles):
    with pytest.raises(NativeEmbeddingError) as exc:
        proof(vertices,triangles)
    assert exc.value.report["reason"]=="native-triangles-intersect-outside-shared-topology"
    assert set(exc.value.report["source_ids"])=={"source:0","source:1"}
    assert not exc.value.report["certified"]


def test_exact_indexed_vertex_contact_is_allowed_but_not_a_manifold_claim():
    result=proof([(0,0,0),(1,0,0),(0,1,0),(-1,0,0),(0,-1,0)],[(0,1,2),(0,3,4)])
    assert result.exact_pairs==1
    with pytest.raises(ValueError,match="disconnected"):
        NativeMeshBoundary(result.vertices,result.triangles,result.source_ids,closed=False,body_id="bowtie")


def test_native_rounding_degeneracy_duplicate_triangles_and_pair_budget_refuse():
    with pytest.raises(NativeEmbeddingError,match="degenerate"):
        proof([(1e8,0,0),(1e8+1,0,0),(1e8,1,0)],[(0,1,2)])
    with pytest.raises(NativeEmbeddingError,match="duplicate"):
        proof([(0,0,0),(1,0,0),(0,1,0)],[(0,1,2),(2,1,0)])
    with pytest.raises(NativeEmbeddingError,match="capacity") as exc:
        proof([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],[(0,1,2),(0,2,3)],max_pairs=0)
    assert exc.value.report["examined_pairs"]==1


def test_no_epsilon_erases_representable_gap_and_proof_binds_native_source_identity():
    gap=float(np.nextafter(np.float32(0),np.float32(1)))
    vertices=[(0,0,0),(1,0,0),(0,1,0),(0,0,gap),(1,0,gap),(0,1,gap)]
    triangles=[(0,1,2),(3,4,5)]
    result=proof(vertices,triangles)
    assert result.exact_pairs==0
    changed=verify_native_embedding(vertices,triangles,["different:0","source:1"])
    assert changed.digest!=result.digest
    moved=list(vertices);moved[-1]=(0,2,gap)
    assert proof(moved,triangles).digest!=result.digest


def test_incumbent_bvh_prunes_large_disjoint_pair_set():
    vertices=[];triangles=[]
    for i in range(100):
        start=len(vertices);vertices.extend(((i*3,0,0),(i*3+1,0,0),(i*3,1,0)))
        triangles.append((start,start+1,start+2))
    result=proof(vertices,triangles,max_pairs=500)
    assert result.exact_pairs==0
    assert result.examined_pairs<500<100*99//2


@pytest.mark.skipif(shutil.which("node") is None,reason="Node unavailable")
def test_actual_rotated_192_triangle_gobo_gets_independent_native_embedding_proof():
    module=(ROOT/"examples/meowdoku/web/marble_studio_gobos.js").as_uri()
    script=f"import {{studioGoboFaces}} from {json.dumps(module)}; console.log(JSON.stringify(studioGoboFaces({{version:1,gobos:[{{id:1,enabled:true,shape:'ellipse',width:2,height:3,x:2,y:1,z:0,pitch:17,yaw:31,roll:9,front:'a',back:'b'}}]}},{{a:3,b:4}})));"
    faces=json.loads(subprocess.run(["node","--input-type=module","-e",script],capture_output=True,text=True,check=True,timeout=10).stdout)
    vertices=[];triangles=[];lookup={}
    for face in faces:
        tri=[]
        for point in face[:3]:
            key=tuple(point)
            if key not in lookup:lookup[key]=len(vertices);vertices.append(key)
            tri.append(lookup[key])
        triangles.append(tri)
    ids=[f"gobo:1:{i}" for i in range(len(triangles))]
    result=verify_native_embedding(vertices,triangles,ids,max_pairs=20000)
    assert len(result.triangles)==192
    assert result.examined_pairs<=20000
    assert result.max_coordinate_rounding>0
    boundary=NativeMeshBoundary(result.vertices,result.triangles,result.source_ids,
        closed=True,body_id="gobo:1",embedded_boundary_evidence=result.evidence_id)
    np.testing.assert_array_equal(boundary.vertices,np.asarray(vertices,dtype=np.float32))
    assert boundary.embedded_boundary_evidence==result.evidence_id
