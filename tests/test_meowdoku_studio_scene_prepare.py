"""Actual JS capture -> immutable SI bake -> native admission, CPU only."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"examples")]
from meowdoku.studio_scene_prepare import StudioVisibleBake,StudioSceneAdmissionError


@pytest.fixture(scope="module")
def capture():
    if not shutil.which("node"):
        pytest.skip("Node is required for actual Studio builder capture")
    script="""
      import {packStudioScene} from './examples/meowdoku/web/marble_studio_scene.js';
      import {studioRoomFaces,studioRoomBounds} from './examples/meowdoku/web/marble_room.js';
      import {studioTableFaces,defaultStudioTable} from './examples/meowdoku/web/marble_studio_table.js';
      const parts=[];const collect=prefix=>p=>parts.push({...p,id:prefix+'/'+p.id});
      const room=studioRoomBounds(3,3);
      studioRoomFaces({scale:3,floorDrop:3,material:()=>0,onSurface:collect('room')});
      studioTableFaces(defaultStudioTable(),{marble_clear_glass:0},room.floorY,collect('table'));
      const subject={id:'subject',radius:1,position:[0,0,0],
        model:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],faces:[parts[0].faces[0]],
        motion:{mode:'off',velocityM:[0,0,0],angularVelocity:[0,0,0],age:0}};
      console.log(JSON.stringify({revision:7,state:packStudioScene({channel:'studio',unitM:.01,
        rayOriginM:[0,0,.18],room,parts,subject,pbr:{stride:16,data:Array(16).fill(0)},
        camera:{view:[1,2,3]},authoring:{object:{design:{'base-shape':'marble'}}}})}));
    """
    return json.loads(subprocess.run(["node","--input-type=module","-e",script],cwd=ROOT,
                                    check=True,text=True,capture_output=True).stdout)


def test_real_room_table_capture_preserves_every_triangle_and_native_ray_frame(capture):
    source=copy.deepcopy(capture);bake=StudioVisibleBake.from_capture(source)
    assert bake.revision==7
    assert len(bake.mesh.triangles)==len(capture["state"]["ray"]["material_ids"])
    first=capture["state"]["parts"][0]["faces"][0][0]
    assert bake.mesh.triangles[0][0]==tuple(x*.01+bake.origin_m[i] for i,x in enumerate(first))
    assert bake.mesh.material_ids==tuple(capture["state"]["ray"]["material_ids"])
    assert bake.subject_center_m==bake.origin_m
    assert len([p for p in bake.parts if p.body_id=="table"])==5
    room=[p for p in bake.parts if p.construction=="inward-room-plane"]
    bake._verify_room(room)
    source["state"]["parts"][0]["faces"][0][0][0]=100
    assert bake.capture_json==StudioVisibleBake.from_capture(capture).capture_json
    assert bake.mesh.triangles[0][0][0]!=1


def test_compound_refusal_returns_source_parts_not_partial_geometry(capture):
    bake=StudioVisibleBake.from_capture(capture)
    with pytest.raises(StudioSceneAdmissionError) as raised:
        bake.convex_packet()
    blockers=raised.value.blockers
    assert any(b["part_id"]=="table" and len(b["source_parts"])==5 for b in blockers)
    assert not any(b["part_id"]=="room" for b in blockers)


def test_real_room_table_capture_enters_full_native_feature_environment(capture):
    from meowdoku.studio_mesh_environment import StudioMeshEnvironment
    from src.common.tensors import AbstractTensor as T
    import numpy as np
    bake=StudioVisibleBake.from_capture(capture)
    packet=bake.feature_packet();env=StudioMeshEnvironment(packet)
    assert len(env.parts)==1 and env.parts[0].body_id=="table"
    assert packet["sourceCaptureDigest"]==bake.capture_digest
    provenance=packet["parts"][0]["sourceProvenance"]
    assert provenance["construction"]=="exact-existing-plane-arrangement"
    assert len(provenance["internalInterfaces"])==4
    scalar=lambda v:T.get_tensor(np.asarray(v,dtype=np.float32))
    uniforms={k:scalar(v) for k,v in env.uniforms().items()}
    # The displayed centre remains exactly the existing metre transform. The
    # actual native table top is just below it by the same physical radius.
    p=tuple(scalar([v]) for v in bake.subject_center_m)
    validity=env.contains(p,scalar([bake.radius_m]),uniforms)
    assert np.asarray(validity.data).item()==1


def test_actual_rotated_gobo_enters_feature_packet_with_independent_embedding_proof(capture):
    script="""
      import fs from 'node:fs';
      import {packStudioScene} from './examples/meowdoku/web/marble_studio_scene.js';
      import {studioGoboFaces} from './examples/meowdoku/web/marble_studio_gobos.js';
      const c=JSON.parse(fs.readFileSync(0,'utf8')),s=c.state,parts=s.parts;
      studioGoboFaces({version:1,gobos:[{id:7,enabled:true,shape:'ellipse',width:2,height:3,
        x:2,y:1,z:0,pitch:23,yaw:-67,roll:121,front:'white',back:'white'}]},
        {white:0},p=>parts.push({...p,id:'gobo/'+p.id}));
      console.log(JSON.stringify({revision:c.revision,state:packStudioScene({...s,unitM:s.frame.unitM,
        rayOriginM:s.frame.rayOriginM,parts,pbr:{stride:16,data:s.ray.pbr}})}));
    """
    changed=json.loads(subprocess.run(["node","--input-type=module","-e",script],cwd=ROOT,input=json.dumps(capture),
                                      check=True,text=True,capture_output=True).stdout)
    snapshot=StudioVisibleBake.from_capture(changed);packet=snapshot.feature_packet()
    gobo=next(p for p in packet["parts"] if p["bodyId"]=="gobo-7")
    assert len(gobo["triangles"])==192
    assert gobo["embeddedBoundaryEvidence"].startswith("turing.native-f32-embedding:")
    assert gobo["sourceProvenance"]["maxCoordinateRounding"]>0
    assert gobo["sourceTriangleIds"]==[f"gobo/7:{i}" for i in range(192)]
    assert gobo["sourceProvenance"]["embeddingProof"]["geometric_embedding"] is True


def test_ray_material_vertex_or_identity_drift_refuses_before_bake(capture):
    for change in (
        lambda c:c["state"]["ray"]["vertices"].__setitem__(0,123),
        lambda c:c["state"]["ray"]["material_ids"].__setitem__(0,1),
        lambda c:c["state"]["triangleSources"][0].__setitem__("bodyId","replacement"),
    ):
        changed=copy.deepcopy(capture);change(changed)
        with pytest.raises(ValueError,match="identities differ"):
            StudioVisibleBake.from_capture(changed)


def test_scene_revision_frame_and_profile_remain_bound(capture):
    original=StudioVisibleBake.from_capture(capture)
    changed=copy.deepcopy(capture);changed["revision"]+=1
    other=StudioVisibleBake.from_capture(changed)
    assert other.capture_digest!=original.capture_digest
    assert other.mesh.digest==original.mesh.digest
    changed=copy.deepcopy(capture);changed["state"]["channel"]="exposure"
    with pytest.raises(ValueError,match="not Exposure"):
        StudioVisibleBake.from_capture(changed)
    changed=copy.deepcopy(capture);changed["state"]["room"]=None
    with pytest.raises(ValueError,match="room is absent"):
        StudioVisibleBake.from_capture(changed)
    with pytest.raises(ValueError,match="capacity"):
        StudioVisibleBake.from_capture(capture,max_triangles=20)


def test_room_missing_duplicate_or_reversed_faces_cannot_invent_full_container(capture):
    bake=StudioVisibleBake.from_capture(capture)
    room=[p for p in bake.parts if p.construction=="inward-room-plane"]
    with pytest.raises(ValueError,match="missing|unmatched|cover"):
        bake._verify_room(room[:-1])
    with pytest.raises(ValueError,match="duplicate"):
        bake._verify_room(room+room[:1])
