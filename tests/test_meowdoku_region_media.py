import copy
import numpy as np
import pytest
from tests.test_meowdoku_air_bubbles import fixture
from examples.meowdoku.marble_region_media import saved_region_media


def region_fixture():
    s,v,i,p=fixture();s['design'].pop('boundary_media');s['design']['optical_media']={'version':1,'materials':[]}
    # Two coincident dye volumes and an air bubble intentionally cohabit.
    outer=v[8:];v=np.concatenate((outer*.3,outer*.25,outer*.05,outer))
    p=np.concatenate((p,p,p));p[:,6]=1.52
    i=np.repeat([1,2,0,0],8)
    s['design']['region_media']={'version':1,'regions':[
      {'start':24,'count':8,'mode':'host','material':0},
      {'start':0,'count':8,'mode':'injection_dye','material':1},
      {'start':8,'count':8,'mode':'injection_dye','material':2},
      {'start':16,'count':8,'mode':'air_bubble','material':-1}]}
    return s,v,i,p


def test_dyes_and_air_can_cohabit_with_real_closed_boundaries():
    s,v,i,p=region_fixture();before=copy.deepcopy(s)
    assert saved_region_media(s,v,i,p)==s['design']['region_media']
    assert s==before


def test_ambiguous_rods_are_not_given_hidden_precedence():
    s,v,i,p=region_fixture();s['design']['region_media']['regions'][1]['mode']='glass_rod'
    with pytest.raises(ValueError,match='ownership'):saved_region_media(s,v,i,p)
    s['design']['region_media']['regions'][2]['mode']='glass_rod'
    with pytest.raises(ValueError,match='ownership'):saved_region_media(s,v,i,p)
    s['design']['region_media']['regions'][2]['material']=1;i[8:16]=1
    assert saved_region_media(s,v,i,p)


def test_dye_cannot_silently_have_a_distinct_refractive_index():
    s,v,i,p=region_fixture();p[1,6]=1.54
    with pytest.raises(ValueError,match='host IOR'):saved_region_media(s,v,i,p)


def test_each_cut_volume_needs_closed_outward_geometry():
    s,v,i,p=region_fixture();v[:8]=v[:8,::-1,:]
    with pytest.raises(ValueError,match='outward'):saved_region_media(s,v,i,p)


def test_legacy_design_has_no_region_activation():
    assert saved_region_media({'design':{}},None,None,None) is None
