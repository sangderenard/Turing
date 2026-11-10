import copy
import numpy as np
import pytest
from examples.meowdoku.scene_material_engines import validate_scene_engines

def fixture():
    a=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)*.01
    v=a[np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])]
    p=np.zeros((1,16));p[0,:3]=.9;p[0,5]=.42;p[0,6]=1.48;p[0,7]=.79
    c={'version':1,'groups':[{'name':'stage','engine':'dielectric','ranges':[{'start':0,'count':8}]},{'name':'room','engine':'legacy','ranges':[]}],'volumes':[{'start':0,'count':8,'material':0}], 'optical_media':{'version':1,'materials':[{'material':0,'wavelengths_nm':[450,650],'transmittance':[.8,.8],'reference_thickness_mm':1,'concentration':1}]}}
    return c,v,np.zeros(8,int),p

def test_scene_contract_preserves_authored_material_and_projects_unique_volume():
    c,v,i,p=fixture();before=p.copy()
    assert validate_scene_engines(c,v,i,p)==c
    from camera_designer.studio_scene_engines import project_scene_engines
    ii,pp,b,o=project_scene_engines({'vertices':v.flatten().tolist(),'material_ids':i.tolist(),'pbr':p.flatten().tolist(),'material_engines':c})
    assert np.array_equal(p,before);assert np.array_equal(pp[0],p[0].astype(np.float32))
    assert set(ii)=={1};assert pp[1,7]==0;assert b['ranges'][0]['negative']==1
    assert o['materials'][0]['transmittance']==[.8,.8]

def test_missing_volume_never_falls_back_and_open_winding_rejected():
    c,v,i,p=fixture();c['volumes']=[]
    with pytest.raises(ValueError,match='no legacy fallback'):validate_scene_engines(c,v,i,p)
    c,v,i,p=fixture();v[0]=v[0,::-1]
    with pytest.raises(ValueError,match='winding'):validate_scene_engines(c,v,i,p)

def test_legacy_absent_stays_absent():
    assert validate_scene_engines(None,[],[],[]) is None

def test_scene_volume_reaches_native_matbuf_and_explicit_boundary_tags():
    c,v,i,p=fixture();validate_scene_engines(c,v,i,p)
    from camera_designer.studio_scene_engines import project_scene_engines
    from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
    ii,pp,b,o=project_scene_engines({'material_ids':i,'pbr':p,'material_engines':c})
    scene=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.zeros(3),np.ones(3)),np.zeros((0,12),np.float32))
    packed=append_saved_studio_mesh_to_gpu_scene(scene,v,ii,pp,scale_m=1,center_m=[0,0,0],wavelengths_um=[.55],boundary_media=b,optical_media=o)
    assert np.all(packed[0][:,3]==1)
    assert np.all(packed[1][:,7]==0)
    native_id=int(packed[0][0,7])-2
    row=packed[2].reshape(-1,32,12)[native_id,0]
    assert row[7]==pytest.approx(1.48)
    assert np.exp(-4*np.pi*row[8]/(.55e-6)*.001)==pytest.approx(.8,rel=1e-5)

def test_mixed_opaque_scene_and_optical_volumes_share_one_append():
    c,v,i,p=fixture();validate_scene_engines(c,v,i,p)
    from camera_designer.studio_scene_engines import project_scene_engines
    from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
    opaque=p[0].copy();opaque[5]=0;opaque[7]=1
    p=np.vstack([p,opaque]);v=np.concatenate([v,v[:1]+.1]);i=np.r_[i,1]
    c['groups'][1]['ranges']=[{'start':8,'count':1}]
    validate_scene_engines(c,v,i,p)
    ii,pp,b,o=project_scene_engines({'material_ids':i,'pbr':p,'material_engines':c})
    scene=(np.zeros((0,16),np.float32),np.zeros((0,16),np.float32),np.zeros((0,12),np.float32),np.zeros((0,3,3),np.float32),np.zeros((0,8),np.float32),{},(np.zeros(3),np.ones(3)),np.zeros((0,12),np.float32))
    packed=append_saved_studio_mesh_to_gpu_scene(scene,v,ii,pp,scale_m=1,center_m=[0,0,0],wavelengths_um=[.55],boundary_media=b,optical_media=o)
    assert packed[0][-1,3]==0 and packed[1][-1,7]==1
    assert np.all(packed[0][:-1,3]==1)

def _rotated_slabs(offset):
    points=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    a=points*np.array([.003,.003,.0002]);b=a+np.array([offset,0,0])
    angle=.63;c=np.cos(angle);s=np.sin(angle)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    return (a@rotation.T)[faces],(b@rotation.T)[faces]


def test_rotated_separate_filter_slabs_with_overlapping_world_bounds_admit():
    a,b=_rotated_slabs(.0065)
    assert np.all(np.minimum(a.max((0,1)),b.max((0,1)))>np.maximum(a.min((0,1)),b.min((0,1))))
    c,v,i,p=fixture();v=np.concatenate([a,b]);i=np.zeros(24,int)
    c['groups'][0]['ranges']=[{'start':0,'count':24}]
    c['volumes']=[{'start':0,'count':12,'material':0},{'start':12,'count':12,'material':0}]
    assert validate_scene_engines(c,v,i,p)==c


def test_rotated_overlapping_filter_slabs_still_reject():
    a,b=_rotated_slabs(.005)
    c,v,i,p=fixture();v=np.concatenate([a,b]);i=np.zeros(24,int)
    c['groups'][0]['ranges']=[{'start':0,'count':24}]
    c['volumes']=[{'start':0,'count':12,'material':0},{'start':12,'count':12,'material':0}]
    with pytest.raises(ValueError,match='overlapping scene volume'):
        validate_scene_engines(c,v,i,p)


def test_separation_rejects_float32_ambiguous_gap_and_contains_no_false_separation():
    from examples.meowdoku.scene_material_engines import separated_on_face_axis
    a,b=_rotated_slabs(.006+1e-10)
    assert not separated_on_face_axis(a,b)
    assert not separated_on_face_axis(a,a*.5)
    a,b=_rotated_slabs(.0065)
    shift=np.array([.12,-.3,.08])
    assert separated_on_face_axis(a+shift,b+shift)

