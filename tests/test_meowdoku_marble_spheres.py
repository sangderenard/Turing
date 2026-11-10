"""Exact-region admission must be authored, authenticated, and fail closed."""
import copy
import numpy as np
import pytest
from examples.meowdoku.marble_spheres import saved_analytic_spheres, preflight_triangle_budget


@pytest.fixture
def sphere():
    # Same full-v1 shell counts and outward orientation, independently built.
    theta=np.linspace(0,np.pi,97);phi=np.arange(128)*2*np.pi/128
    points=np.stack(np.broadcast_arrays(np.sin(theta[:,None])*np.cos(phi),
        np.sin(theta[:,None])*np.sin(phi),np.cos(theta[:,None])),axis=-1)
    faces=[]
    for i in range(96):
        for j in range(128):
            k=(j+1)%128
            if i:faces.append([points[i,j],points[i+1,j],points[i,k]])
            if i<95:faces.append([points[i,k],points[i+1,j],points[i+1,k]])
    vertices=np.asarray(faces,np.float32)
    assert len(vertices)==24320
    regions={'version':1,'spheres':[{'start':0,'count':24320,'center':[0,0,0],'radius':1}]}
    saved={'generator':'meowdoku-marble-object-v1','materials':{'index':{'glass':0}},
           'design':{'base-shape':'marble','coverage':0,'band-count':0,'bubble-count':0,
                     'shell':'glass','analytic_spheres':regions}}
    return saved,vertices,np.zeros(len(vertices),np.int32),np.zeros((1,16),np.float32)


def test_declared_uniform_sphere_admitted_without_mutating_mechanical_mesh(sphere):
    saved,v,ids,pbr=sphere;before=v.copy()
    result=saved_analytic_spheres(saved,v,ids,pbr,{'sphere_coverage':'exact'})
    assert result==saved['design']['analytic_spheres'] and result is not saved['design']['analytic_spheres']
    np.testing.assert_array_equal(v,before)


@pytest.mark.parametrize('key,value',[('coverage',.1),('band-count',1),('base-shape','dice')])
def test_unsupported_design_rejected_before_geometry_processing(sphere,key,value):
    saved,v,ids,pbr=sphere;saved['design'][key]=value
    with pytest.raises(ValueError,match='uncoated spherical marble'):
        saved_analytic_spheres(saved,v,ids,pbr,{'sphere_coverage':'exact'})


def test_legacy_mesh_does_not_require_or_infer_analytic_metadata(sphere):
    saved,v,ids,pbr=sphere;saved['design'].pop('analytic_spheres')
    assert saved_analytic_spheres(saved,v,ids,pbr,{}) is None
    with pytest.raises(ValueError,match='Save this design again'):
        saved_analytic_spheres(saved,v,ids,pbr,{'sphere_coverage':'exact'})


def test_bad_saved_descriptor_and_emission_fail_closed(sphere):
    saved,v,ids,pbr=sphere
    saved['design']['analytic_spheres']['spheres'][0]['radius']=.9
    with pytest.raises(ValueError):saved_analytic_spheres(saved,v,ids,pbr,{'sphere_coverage':'exact'})
    saved['design']['analytic_spheres']['spheres'][0]['radius']=1;pbr[0,8]=1
    with pytest.raises(ValueError,match='nonemissive'):
        saved_analytic_spheres(saved,v,ids,pbr,{'sphere_coverage':'exact'})


def test_budget_lower_bound_excludes_only_declared_exact_shell(sphere):
    saved,v,ids,pbr=sphere
    config={'triangle_budget':1000,'scale_m':.01,'object_center_m':[0,0,.18]}
    with pytest.raises(ValueError,match='exposure budget is 1000'):preflight_triangle_budget(v,config)
    assert preflight_triangle_budget(v,config,saved['design']['analytic_spheres'])==0


def test_region_bubble_exact_spheres_survive_full_decode(sphere, tmp_path):
    from tests.test_meowdoku_marble_photo import _saved_object, _wire, objects, photos
    _, outer, _, _ = sphere
    vertices = np.concatenate((outer * .1, outer))
    ids = np.zeros(len(vertices), np.uint32)
    normals = vertices / np.linalg.norm(vertices, axis=2, keepdims=True)
    saved = _saved_object(vertices, normals, ids)
    saved['design'].update(shell='glass', **{'bubble-count': 1})
    n = len(outer)
    saved['design']['region_media'] = {'version': 1, 'regions': [
        {'start': n, 'count': n, 'mode': 'host', 'material': 0},
        {'start': 0, 'count': n, 'mode': 'air_bubble', 'material': -1}]}
    saved['design']['optical_media'] = {'version': 1, 'materials': [
        {'material': 0, 'wavelengths_nm': [380, 780], 'transmittance': [.99, .99],
         'reference_thickness_mm': 1, 'concentration': 1}]}
    saved['design']['analytic_spheres'] = {'version': 1, 'spheres': [
        {'start': 0, 'count': n, 'center': [0, 0, 0], 'radius': .1},
        {'start': n, 'count': n, 'center': [0, 0, 0], 'radius': 1}]}
    store = objects.Store(tmp_path / 'region-spheres')
    digest = store.put(saved)['digest']
    config = photos.decode_upload(_wire(digest, vertices, normals, ids, sphere_coverage='exact'), store)[3]
    assert config['boundary_transport'] == 'studio-region-media-v1'
    assert config['analytic_spheres'] == saved['design']['analytic_spheres']
    assert 'boundary_media' not in config
    live_config = photos.decode_upload(_wire(digest,vertices,normals,ids,transport='fixed:3',
        live={'session_id':'a'*32},sphere_coverage='exact'),store)[3]
    assert live_config['region_media'] == config['region_media']
    assert live_config['optical_media'] == config['optical_media']
    assert live_config['analytic_spheres'] == config['analytic_spheres']
    # A matching mesh span is not enough: the region must really own air.
    saved['design']['region_media']['regions'][1].update(mode='injection_dye', material=0)
    with pytest.raises(ValueError, match='explicit volume region'):
        saved_analytic_spheres(saved, vertices, ids, np.asarray(saved['materials']['tables']['pbr']['data']).reshape(-1,16), {'sphere_coverage':'exact'})
