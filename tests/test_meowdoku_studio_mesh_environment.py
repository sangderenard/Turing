"""Full existing particle law with opt-in native mesh-feature environment."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"examples")]
from meowdoku import fountain3d as F,studio_physics_build as B
from meowdoku.studio_environment import StudioEnvironment,SCHEMA
from meowdoku.studio_mesh_environment import StudioMeshEnvironment,CONTACT_MODEL
from src.common.tensors import AbstractTensor as T


def packet(corner=False):
    vertices=[[0,.2,-.2],[0,.2,.2],[.2,.2,.2],[.2,.2,-.2]]
    triangles=[[0,1,2],[0,2,3]]
    if corner:
        vertices.extend([[0,.4,-.2],[0,.4,.2]])
        triangles.extend([[1,0,4],[1,4,5]])
    return {"schema":SCHEMA,"contact_model":CONTACT_MODEL,"frame":{"unitM":.01,"originM":[0,.05,0]},
        "room":{"low":[-.5,.001,-.5],"high":[.5,1,.5]},
        "parts":[{"id":"surface","bodyId":"surface","closed":False,"convex":not corner,
            "vertices":vertices,"triangles":triangles,"materialIds":[3]*len(triangles),
            "sourceTriangleIds":[f"surface:{i}" for i in range(len(triangles))]}]}


def columns(environment,p=(.1,.209,0)):
    values=F.initial_columns(2,active=1)
    values.update(environment.uniforms());values.update(environment.grid_uniforms(.01))
    for k,v in zip("xyz",p):values[k][0]=v
    values["age"][:]=1;values["rs"][0]=.01
    for k in ("grav_x","grav_y","grav_z","fl_rho","fl_mu","spin_drag","temp","emit_rate"):
        values[k]=F._f32(0)
    for i in range(4):values[f"r{i}"]=F._f32(.01);values[f"swell{i}"]=F._f32(0)
    return values


def step(env,c):
    produced=env.law(2,1)(*(T.get_tensor(np.asarray(c[k],dtype=np.float32)) for k in env.arguments))
    return {name:np.asarray(v.data) for name,v in zip(F.OUTPUTS,produced)}


def test_full_law_keeps_two_real_corner_contacts_and_no_old_chamber():
    env=StudioMeshEnvironment(packet(True));c=columns(env,(.009,.209,0));out=step(env,c)
    assert out["vx_next"][0]>0 and out["vy_next"][0]>0
    assert out["n_out"][0]==0 and out["n_nan"][0]==0 and out["n_live"][0]==1
    changed=dict(c);changed["lid"]=F._f32(-1000);changed["sdf_on"]=F._f32(1)
    again=step(env,changed)
    for name in F.OUTPUTS:np.testing.assert_array_equal(out[name],again[name])


def test_native_feature_seam_matches_existing_single_sheet_force():
    source=packet();features=StudioMeshEnvironment(source)
    legacy=copy.deepcopy(source);legacy.pop("contact_model");convex=StudioEnvironment(legacy)
    for p in ((.1,.209,0),(.11,.209,0)):
        a=step(features,columns(features,p));b=step(convex,columns(convex,p))
        for name in F.OUTPUTS:np.testing.assert_allclose(a[name],b[name],rtol=2e-5,atol=1e-7,err_msg=name)


def test_mutated_bake_uniforms_are_not_silently_admitted():
    env=StudioMeshEnvironment(packet());values=columns(env)
    values["studio_p0_v0_y"]=F._f32(.201)
    out=step(env,values)
    assert out["n_out"][0]>0


def test_model_topology_and_geometry_identity_are_explicit():
    source=packet();env=B.environment_for(source)
    assert isinstance(env,StudioMeshEnvironment)
    changed=copy.deepcopy(source);changed["parts"][0]["vertices"][0][1]+=.0001
    assert B.digest(env,2,1)!=B.digest(B.environment_for(changed),2,1)
    repeated=copy.deepcopy(source);repeated["parts"].append({**repeated["parts"][0],"id":"second"})
    with pytest.raises(ValueError,match="intersect|unique"):
        B.environment_for(repeated)
    with pytest.raises(ValueError,match="unsupported"):
        B.environment_for({**source,"contact_model":"proxy"})


def test_full_feature_particle_product_uses_existing_translator(tmp_path):
    manifest=B.build(packet(),tmp_path,radius=.01,pad=2,active=1,wasm=False)
    rec=manifest["sizes"]["studio"]
    assert (tmp_path/rec["wgsl"]).is_file()
    assert rec["wgsl_abi"]["dispatches"]>0
    assert manifest["geometry_contract"]["contact_model"]==CONTACT_MODEL
    assert manifest["geometry_contract"]["boundaries"][0]["sourceTriangleIds"]==["surface:0","surface:1"]
    assert set(F.STATE)<=set(rec["wgsl_abi"]["feed_slots"])
    assert "studio_p0_v0_y" in rec["wgsl_abi"]["uniform_index"]


def test_same_body_disjoint_open_boundaries_are_proved_not_relabelled():
    source=packet();second=copy.deepcopy(source["parts"][0]);second["id"]="emitter"
    second["vertices"]=[[x,y+.1,z] for x,y,z in second["vertices"]]
    source["parts"].append(second)
    env=StudioMeshEnvironment(source)
    assert len(env.parts)==2 and len({p.body_id for p in env.parts})==1
    proof=env.geometry_contract["disjointBodyBoundaries"][0]
    assert proof["partIds"]==["surface","emitter"] and proof["embedding"]["geometric_embedding"]


def test_nested_closed_component_cannot_be_called_disjoint_exterior():
    import importlib.util
    spec=importlib.util.spec_from_file_location("studio_manifold_fixture",ROOT/"tests/test_meowdoku_studio_mesh_manifold.py")
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    cube=fixture.cube
    def record(scale,ident):
        vertices=[];lookup={};triangles=[]
        for tri in cube():
            ids=[]
            for p in tri:
                p=tuple(v*scale for v in p)
                if p not in lookup:lookup[p]=len(vertices);vertices.append(p)
                ids.append(lookup[p])
            triangles.append(ids)
        return {"id":ident,"bodyId":"same","closed":True,"vertices":vertices,"triangles":triangles,
            "materialIds":[0]*len(triangles),"sourceTriangleIds":[f"{ident}:{i}" for i in range(len(triangles))],
            "embeddedBoundaryEvidence":"test:verified-convex-cuboid"}
    source=packet();source["parts"]=[record(.1,"outer"),record(.05,"inner")]
    with pytest.raises(ValueError,match="inside|uncertified"):
        StudioMeshEnvironment(source)
