import sys
from pathlib import Path
import time
import json
import subprocess
import shutil
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"examples"))
from meowdoku.studio_mesh_manifold import NativeMeshBoundary
from meowdoku.studio_contacts import stationary_contact
from src.common.tensors import AbstractTensor as T


def boundary():
    return NativeMeshBoundary([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],[(0,1,2),(0,2,3)],
        ["a","b"],closed=False,body_id="sheet")


def assert_same(mesh,points):
    baked=mesh.bake_static_geometry()
    p=tuple(T.get_tensor(np.asarray(v,dtype=np.float32)) for v in np.asarray(points).T)
    original,diagnostics=mesh.query(p)
    frozen,frozen_diag=mesh.query(p,static_geometry=baked)
    assert diagnostics.keys()==frozen_diag.keys()
    for k in diagnostics:np.testing.assert_array_equal(diagnostics[k].data,frozen_diag[k].data,err_msg=k)
    for a,b in zip(original,frozen):
        assert (a.feature_id,a.kind,a.source_ids)==(b.feature_id,b.kind,b.source_ids)
        for key in ("distance","valid"):
            np.testing.assert_array_equal(getattr(a,key).data,getattr(b,key).data,err_msg=str((a.feature_id,key)))
        for key in ("normal","point"):
            for x,y in zip(getattr(a,key),getattr(b,key)):
                np.testing.assert_array_equal(x.data,y.data,err_msg=str((a.feature_id,key)))
    return baked


def test_frozen_query_is_array_equal_at_face_seam_edge_vertex_and_both_sides():
    baked=assert_same(boundary(),[(.4,.4,.1),(.4,.41,.1),(-.05,-.05,.1),(.5,.5,0),(.5,.2,-.1)])
    assert baked.backend=="AbstractTensor NumPy float32"
    with pytest.raises(TypeError):baked.records[("x",)]=0
    other=NativeMeshBoundary([(0,0,0),(2,0,0),(2,1,0),(0,1,0)],[(0,1,2),(0,2,3)],["a","b"],closed=False,body_id="sheet")
    zero=T.get_tensor(np.asarray([0],dtype=np.float32))
    with pytest.raises(ValueError,match="snapshot"):other.query((zero,)*3,static_geometry=baked)


def emit(mesh,baked):
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations,SSATensorProgram
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS,scoped_work_contract
    started=time.perf_counter()
    with scoped_work_contract(PRESETS["fast"]):
        program=SSATensorProgram("static_geometry_probe")
        p=tuple(SSATensorOperations.input(program,(2,)) for _ in range(3))
        vertices=tuple(tuple(SSATensorOperations.input(program,()) for _ in range(3)) for _ in mesh.vertices)
        with program.activate():
            zero=p[0]*0;radius=zero+.2;force=[zero,zero,zero]
            def wall(overlap,valid,normal):
                f,_=stationary_contact(radius-overlap,normal,valid,(zero,)*3,(zero,)*3,
                    radius,zero+100,zero+1,zero,zero,zero,zero,zero)
                for k in range(3):force[k]=force[k]+f[k]
            diag=mesh.apply(p,radius,zero+1,wall,vertices,static_geometry=baked)
        outputs=[v.data.value for v in (*force,*diag.values())]
        product=webgpu.emit_tensor_program_passes(program.module,"static_geometry_probe",outputs,count=2)
    assert product.complete,[s.format() for s in product.shortfalls]
    return {"bytes":len(product.source.encode()),"lets":product.source.count("let "),
            "atan2":product.source.count("atan2("),"sqrt":product.source.count("sqrt("),
            "seconds":time.perf_counter()-started,"dispatches":product.dispatches}


def test_bounded_frozen_manifold_and_dem_shader_reduces_static_operation_count():
    mesh=boundary();baked=mesh.bake_static_geometry()
    original=emit(mesh,None);frozen=emit(mesh,baked)
    print({"original":original,"frozen":frozen})
    assert original["dispatches"]==frozen["dispatches"]==1
    assert frozen["atan2"]==0<original["atan2"]
    assert frozen["bytes"]<original["bytes"]
    assert frozen["lets"]<original["lets"]


@pytest.fixture(scope="module")
def actual_boundaries():
    if shutil.which("node") is None:pytest.skip("Node unavailable")
    directory=(ROOT/"examples/meowdoku/web").as_uri()
    script=f"""
      import {{studioTableFaces,defaultStudioTable}} from '{directory}/marble_studio_table.js';
      import {{studioGoboFaces}} from '{directory}/marble_studio_gobos.js';
      import {{makeStudioCourse}} from '{directory}/marble_studio_courses.js';
      const tableState=defaultStudioTable();
      console.log(JSON.stringify({{
        table:studioTableFaces(tableState,{{[tableState.material]:3}}),
        gobo:studioGoboFaces({{version:1,gobos:[{{id:1,enabled:true,shape:'ellipse',width:2,height:3,x:2,y:1,z:0,pitch:17,yaw:31,roll:9,front:'a',back:'b'}}]}},{{a:3,b:4}}),
        course:makeStudioCourse({{kind:'gentle-s',stage:{{width:8,length:6,baseY:0}},materialId:3,segments:16}})
      }}));
    """
    data=json.loads(subprocess.run(["node","--input-type=module","-e",script],capture_output=True,text=True,check=True,timeout=10).stdout)
    from meowdoku.studio_mesh_bake import StudioMeshBake
    from meowdoku.studio_cuboid_union import build_cuboid_union
    from meowdoku.studio_mesh_embedding import verify_native_embedding
    faces=data["table"]
    bake=StudioMeshBake.from_faces(faces,source_ids=[f"table:{i}" for i in range(len(faces))],frame_id="static-test")
    parts=[SimpleNamespace(id=f"table:{i//12}",body_id="table",source_indices=tuple(range(i,i+12)),closed=True) for i in range(0,60,12)]
    union=build_cuboid_union(bake,parts)
    table=NativeMeshBoundary(union.vertices,union.triangles,union.triangle_ids,closed=True,body_id=union.body_id,
        embedded_boundary_evidence=union.embedded_boundary_evidence)
    result={"table":(table,[(0,-.95,0),(-2.2,-1.18,-1.25),(-2.35,-2,-1.25),(0,-1,0)])}
    for name,faces in (("gobo",data["gobo"]),("course",data["course"]["faces"])):
        vertices=[];lookup={};triangles=[]
        for face in faces:
            tri=[]
            for point in face[:3]:
                key=tuple(point)
                if key not in lookup:lookup[key]=len(vertices);vertices.append(key)
                tri.append(lookup[key])
            triangles.append(tri)
        proof=verify_native_embedding(vertices,triangles,[f"{name}:{i}" for i in range(len(faces))])
        mesh=NativeMeshBoundary(proof.vertices,proof.triangles,proof.source_ids,closed=True,body_id=name,
            embedded_boundary_evidence=proof.evidence_id)
        if name=="gobo":
            n=np.array(faces[0][4][0]);center=np.array([2,1,0])
            points=[center,center+.2*n,center-.2*n,proof.vertices[0]]
        else:
            center=np.array(data["course"]["route"][8]);tx,tz=5.76,-1.2*np.pi
            side=np.array([-tz,0,tx])/np.hypot(tx,tz)
            points=[center+[0,.025,0],center+.28*side+[0,.025,0],center-[0,.01,0],proof.vertices[0]]
        result[name]=(mesh,points)
    return result


@pytest.mark.parametrize("name",["table","gobo","course"])
def test_actual_immutable_mesh_all_finite_contacts_and_diagnostics_are_array_equal(actual_boundaries,name):
    mesh,points=actual_boundaries[name]
    assert_same(mesh,points)
