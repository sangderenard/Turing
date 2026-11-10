import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"examples"))
from meowdoku.studio_mesh_manifold import NativeMeshBoundary
from meowdoku.studio_contacts import stationary_contact
from src.common.tensors import AbstractTensor as T


def tensor(x):
    return T.get_tensor(np.asarray(x,dtype=np.float32))


def vec(p):
    return tuple(tensor([x]) for x in p)


def value(x):
    return np.asarray(x.data).item()


def boundary(faces,closed=False,**kw):
    vertices=[];lookup={};triangles=[]
    for face in faces:
        tri=[]
        for p in face[:3]:
            key=tuple(p)
            if key not in lookup:
                lookup[key]=len(vertices);vertices.append(key)
            tri.append(lookup[key])
        triangles.append(tri)
    return NativeMeshBoundary(vertices,triangles,[f"source:{i}" for i in range(len(faces))],
        closed=closed,body_id="body",embedded_boundary_evidence="fixture:authored-simple-boundary" if closed else None,**kw)


def quad(a,b,c,d):
    return [(a,b,c),(a,c,d)]


def cube():
    quads=[((1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)),
           ((-1,-1,1),(-1,1,1),(-1,1,-1),(-1,-1,-1)),
           ((-1,1,-1),(-1,1,1),(1,1,1),(1,1,-1)),
           ((-1,-1,1),(-1,-1,-1),(1,-1,-1),(1,-1,1)),
           ((-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)),
           ((1,-1,-1),(-1,-1,-1),(-1,1,-1),(1,1,-1))]
    return [tri for q in quads for tri in quad(*q)]


def touching(mesh,p,r=.2):
    contacts,diagnostics=mesh.query(vec(p))
    assert value(diagnostics["geometry_valid"])==1
    assert value(diagnostics["nearest_found"])==1
    assert value(diagnostics["sign_ambiguous"])==0
    assert value(diagnostics["singular_features"])==0
    return [c for c in contacts if value(c.valid)>0 and value(c.distance)<r],diagnostics


def test_coplanar_seam_and_vertex_have_one_contact_not_duplicate_force():
    mesh=boundary(quad((0,0,0),(1,0,0),(1,1,0),(0,1,0)))
    for p,kind in [((.4,.4,.1),"edge"),((.4,.41,.1),"face"),((-.05,-.05,.1),"vertex")]:
        contacts,_=touching(mesh,p)
        assert len(contacts)==1 and contacts[0].kind==kind
        assert set(contacts[0].source_ids)<={"source:0","source:1"}
    records=[];zero=tensor([0]);radius=tensor([.2])
    def wall(overlap,valid,normal):
        force,_=stationary_contact(radius-overlap,normal,valid,(zero,)*3,(zero,)*3,
            radius,zero+100,zero+1,zero,zero,zero,zero,zero)
        records.append(force)
    mesh.apply(vec((.4,.4,.1)),radius,zero+1,wall)
    force=[sum(value(f[k]) for f in records) for k in range(3)]
    np.testing.assert_allclose(force,[0,0,10],atol=2e-6)


def test_concave_floor_rail_corner_keeps_two_faces_and_suppresses_diagonal_edge():
    floor=quad((0,0,-1),(0,0,1),(1,0,1),(1,0,-1))
    rail=quad((0,0,1),(0,0,-1),(0,1,-1),(0,1,1))
    mesh=boundary(floor+rail)
    contacts,_=touching(mesh,(.1,.1,0))
    assert len(contacts)==2 and all(c.kind=="face" for c in contacts)
    normals=sorted(tuple(round(value(v),5) for v in c.normal) for c in contacts)
    assert normals==[(0,1,0),(1,0,0)]
    convex,_=touching(mesh,(-.1,-.1,0))
    assert len(convex)==1 and convex[0].kind=="edge"
    np.testing.assert_allclose([value(v) for v in convex[0].normal],[-2**-.5,-2**-.5,0],atol=2e-6)


def test_closed_boundary_inside_exit_outside_edge_and_vertex():
    mesh=boundary(cube(),closed=True)
    inside,diag=touching(mesh,(.9,.2,.1))
    assert value(diag["inside"])==1 and len(inside)==1
    assert value(inside[0].distance)<0
    np.testing.assert_allclose([value(v) for v in inside[0].normal],[1,0,0],atol=2e-6)
    for p,kind in [((1.1,.2,.1),"face"),((1.1,1.1,.1),"edge"),((1.1,1.1,1.1),"vertex")]:
        contacts,diag=touching(mesh,p)
        assert value(diag["inside"])==0
        assert len(contacts)==1 and contacts[0].kind==kind
    with pytest.raises(ValueError,match="outward"):
        boundary([tuple(reversed(t)) for t in cube()],closed=True)


def test_concave_vertex_retains_three_faces_not_one_false_radial_contact():
    faces=quad((0,0,0),(0,1,0),(0,1,1),(0,0,1))
    faces+=quad((0,0,1),(1,0,1),(1,0,0),(0,0,0))
    faces+=quad((0,0,0),(1,0,0),(1,1,0),(0,1,0))
    mesh=boundary(faces)
    contacts,_=touching(mesh,(.1,.15,.12),.2)
    assert len(contacts)==3 and all(c.kind=="face" for c in contacts)
    assert sorted(tuple(round(value(v),5) for v in c.normal) for c in contacts)==[(0,0,1),(0,1,0),(1,0,0)]
    opposite,_=touching(mesh,(-.1,-.1,-.1),.2)
    assert len(opposite)==1 and opposite[0].kind=="vertex"


def test_tiny_real_native_float32_concave_fold_is_not_flattened():
    epsilon=2**-18
    left=quad((-1,0,-1),(0,0,-1),(0,0,1),(-1,0,1))
    right=quad((0,0,-1),(1,epsilon,-1),(1,epsilon,1),(0,0,1))
    mesh=boundary(left+right)
    contacts,_=touching(mesh,(-epsilon*.05,.1,.13))
    assert len(contacts)==2  # actual crease has two distinct local minima
    assert all(c.kind=="face" for c in contacts)
    assert mesh.max_coordinate_rounding==0
    normals=[tuple(value(v) for v in c.normal) for c in contacts]
    assert normals[0]!=normals[1]


def test_unresolved_compound_table_soup_is_not_relabelled_as_one_boundary():
    shifted=[tuple((p[0]+2,p[1],p[2]) for p in t) for t in cube()]
    with pytest.raises(ValueError,match="edge|manifold|disconnected"):
        boundary(cube()+shifted,closed=True)


@pytest.mark.skipif(shutil.which("node") is None,reason="Node unavailable")
def test_actual_rotated_gobo_float32_boundary_retains_geometry_and_inside_exit():
    module=(ROOT/"examples/meowdoku/web/marble_studio_gobos.js").as_uri()
    script=f"import {{studioGoboFaces}} from {json.dumps(module)}; console.log(JSON.stringify(studioGoboFaces({{version:1,gobos:[{{id:1,enabled:true,shape:'ellipse',width:2,height:3,x:2,y:1,z:0,pitch:17,yaw:31,roll:9,front:'a',back:'b'}}]}},{{a:3,b:4}})));"
    faces=json.loads(subprocess.run(["node","--input-type=module","-e",script],capture_output=True,text=True,check=True,timeout=10).stdout)
    mesh=boundary(faces,closed=True)
    assert len(mesh.triangles)==len(faces)==192
    for triangle,face in zip(mesh.triangles,faces):
        np.testing.assert_array_equal(np.array([mesh.vertices[i] for i in triangle]),np.asarray(face[:3],dtype=np.float32))
    contacts,diag=touching(mesh,(2,1,0))
    assert value(diag["inside"])==1 and len(contacts)==1
    assert -.061<value(contacts[0].distance)<-.059
    outward=np.asarray(faces[0][4][0])
    contacts,diag=touching(mesh,np.array([2,1,0])+.2*outward,.2)
    assert value(diag["inside"])==0 and contacts
    assert all(value(c.distance)>=0 for c in contacts)
    assert all(np.dot([value(v) for v in c.normal],outward)>0 for c in contacts)


@pytest.mark.skipif(shutil.which("node") is None,reason="Node unavailable")
def test_actual_course_exterior_join_and_floor_bank_keep_correct_contact_ownership():
    module=(ROOT/"examples/meowdoku/web/marble_studio_courses.js").as_uri()
    script=f"import {{makeStudioCourse}} from {json.dumps(module)}; console.log(JSON.stringify(makeStudioCourse({{kind:'gentle-s',stage:{{width:8,length:6,baseY:0}},materialId:3,segments:16}})));"
    course=json.loads(subprocess.run(["node","--input-type=module","-e",script],capture_output=True,text=True,check=True,timeout=10).stdout)
    mesh=boundary(course["faces"],closed=True)
    assert len(mesh.triangles)==len(course["faces"])==472
    p=list(course["route"][8]);p[1]+=.025
    join,diag=touching(mesh,p,.08)
    assert len(join)==1 and value(diag["inside"])==0
    # The authored section at t=.5 uses tangent (width*.72,-span*.20*pi).
    tx,tz=5.76,-1.2*math.pi;length=math.hypot(tx,tz)
    side=(-tz/length,0,tx/length)
    p=[course["route"][8][k]+.28*side[k] for k in range(3)];p[1]+=.025
    corner,diag=touching(mesh,p,.08)
    assert len(corner)==2 and all(c.kind=="face" for c in corner)
    normals=[np.array([value(v) for v in c.normal]) for c in corner]
    assert any(n[1]>.99 for n in normals)
    assert any(np.dot(n,side)<-.9 for n in normals)
    # No internal construction prism faces entered this outer-boundary query.
    assert all(record["role"]!="construction" for record in course["faceRecords"])


def test_native_feature_manifold_and_existing_dem_wall_emit_in_existing_translator():
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations,SSATensorProgram
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS,scoped_work_contract
    mesh=boundary(quad((0,0,0),(1,0,0),(1,1,0),(0,1,0)))
    with scoped_work_contract(PRESETS["fast"]):
        program=SSATensorProgram("studio_boundary_contact")
        p=tuple(SSATensorOperations.input(program,(2,)) for _ in range(3))
        vertices=tuple(tuple(SSATensorOperations.input(program,()) for _ in range(3)) for _ in mesh.vertices)
        with program.activate():
            zero=p[0]*0;radius=zero+.2;force=[zero,zero,zero]
            def wall(overlap,valid,normal):
                f,_=stationary_contact(radius-overlap,normal,valid,(zero,)*3,(zero,)*3,
                    radius,zero+100,zero+1,zero,zero,zero,zero,zero)
                for k in range(3):force[k]=force[k]+f[k]
            diag=mesh.apply(p,radius,zero+1,wall,vertices)
        outputs=[v.data.value for v in (*force,*diag.values())]
        product=webgpu.emit_tensor_program_passes(program.module,"studio_boundary_contact",outputs,count=2)
    assert product.complete,[s.format() for s in product.shortfalls]
    assert product.dispatches==1
    assert all(int(v.data.value.id) in product.uniform_index for vertex in vertices for v in vertex)
