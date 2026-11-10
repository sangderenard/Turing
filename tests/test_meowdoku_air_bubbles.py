"""Saved air-bubble media admission; canonical MRP1/GPU receipts are separate."""
import copy
import numpy as np
import pytest
from examples.meowdoku.marble_media import saved_boundary_media


def fixture():
    # Closed spherical-vertex octahedra are enough to test ownership admission;
    # end-to-end tests use the actual full canonical JS generator and MRP1 bytes.
    points=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],np.float32)
    faces=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    outer=points[faces];vertices=np.concatenate((outer*.05,outer))
    ids=np.zeros(len(vertices),np.int32)
    pbr=np.zeros((1,16),np.float32);pbr[0,:3]=1;pbr[0,5]=1;pbr[0,6]=1.5
    saved={'design':{'base-shape':'marble','kind':'clear','shell':'glass','bubble-count':1,
                    'boundary_media':{'version':1,'ranges':[
                        {'start':0,'count':8,'positive':0,'negative':-1},
                        {'start':8,'count':8,'positive':-1,'negative':0}]}},
           'physical':{'radius-mm':10},'materials':{'index':{'glass':0}}}
    return saved,vertices,ids,pbr


def test_legacy_design_has_no_new_optical_contract():
    assert saved_boundary_media({'design':{}},None,None,None) is None


def test_saved_map_is_copied_and_validated_without_source_mutation():
    saved,vertices,ids,pbr=fixture();before=copy.deepcopy(saved)
    result=saved_boundary_media(saved,vertices,ids,pbr)
    assert result==saved['design']['boundary_media'];assert saved==before
    result['ranges'][0]['positive']=-1
    assert saved==before


@pytest.mark.parametrize('field,value',[('bubble-count',True),('bubble-count',65),('bubble-count',1.5),
    ('base-shape','d6'),('coverage',1),('band-count',1)])
def test_unsupported_count_shape_coating_rejected(field,value):
    s,v,i,p=fixture();s['design'][field]=value
    with pytest.raises(ValueError):saved_boundary_media(s,v,i,p)


def test_axial_overlap_and_small_physical_radius_rejected():
    s,v,i,p=fixture();s['design'].update(kind='cat_eye',strands=2,**{'ribbon-layout':'axial'})
    with pytest.raises(ValueError,match='overlap'):saved_boundary_media(s,v,i,p)
    s,v,i,p=fixture();s['physical']['radius-mm']=.5
    with pytest.raises(ValueError,match='1 mm'):saved_boundary_media(s,v,i,p)


@pytest.mark.parametrize('col,value',[(4,.1),(5,0),(6,1),(7,1),(8,1)])
def test_host_must_be_a_transparent_nonmetal_nonemissive_dielectric(col,value):
    s,v,i,p=fixture();p[0,col]=value
    with pytest.raises(ValueError):saved_boundary_media(s,v,i,p)


def test_triangle_membership_and_winding_rejected_if_tampered():
    s,v,i,p=fixture();s['design']['boundary_media']['ranges'][0]['start']=1
    with pytest.raises(ValueError):saved_boundary_media(s,v,i,p)
    s,v,i,p=fixture();v[0]=v[0][::-1]
    with pytest.raises(ValueError,match='winding'):saved_boundary_media(s,v,i,p)


def test_overlapping_air_spheres_rejected():
    s,v,i,p=fixture();s['design']['bubble-count']=2
    v=np.concatenate((v[:8],v));i=np.zeros(len(v),np.int32)
    s['design']['boundary_media']['ranges']=[
        {'start':0,'count':8,'positive':0,'negative':-1},
        {'start':8,'count':8,'positive':0,'negative':-1},
        {'start':16,'count':8,'positive':-1,'negative':0}]
    with pytest.raises(ValueError,match='overlap'):saved_boundary_media(s,v,i,p)
