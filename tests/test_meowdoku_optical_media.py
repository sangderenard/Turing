"""Opt-in no-bubble spectral media admission, with legacy definitions untouched."""
import copy
import numpy as np
import pytest
from examples.meowdoku.marble_media import saved_boundary_media
from tests.test_meowdoku_air_bubbles import fixture


def optical_fixture():
    s,v,i,p=fixture();s['design']['bubble-count']=0
    s['design']['optical_media']={'version':1,'materials':[]}
    p=np.concatenate((p,p));p[1,6]=1.54
    i[:8]=1
    s['design']['boundary_media']['ranges'][0].update(positive=0,negative=1)
    return s,v,i,p


def test_no_bubble_explicit_inclusion_admitted_without_mutation():
    s,v,i,p=optical_fixture();before=copy.deepcopy(s)
    assert saved_boundary_media(s,v,i,p)==s['design']['boundary_media']
    assert s==before


def test_legacy_no_bubble_map_cannot_silently_opt_in():
    s,v,i,p=optical_fixture();del s['design']['optical_media']
    with pytest.raises(ValueError,match='nonzero bubble'):saved_boundary_media(s,v,i,p)


def test_overlapping_inclusions_rejected_instead_of_guessing_owner():
    s,v,i,p=optical_fixture()
    v=np.concatenate((v[:8],v));i=np.concatenate((i[:8],i))
    s['design']['boundary_media']['ranges']=[
      {'start':0,'count':8,'positive':0,'negative':1},
      {'start':8,'count':8,'positive':0,'negative':1},
      {'start':16,'count':8,'positive':-1,'negative':0}]
    with pytest.raises(ValueError,match='ownership'):saved_boundary_media(s,v,i,p)
    v[:8, :, 0]-=.2;v[8:16,:,0]+=.2
    assert saved_boundary_media(s,v,i,p)==s['design']['boundary_media']


def test_inclusion_must_be_contained():
    s,v,i,p=optical_fixture();v[:8,:,0]+=1
    with pytest.raises(ValueError,match='strictly inside'):saved_boundary_media(s,v,i,p)



@pytest.mark.parametrize('change',['reverse_all','reverse_one','open'])
def test_bad_surface_topology_rejected(change):
    s,v,i,p=optical_fixture()
    if change=='reverse_all':v[:8]=v[:8,::-1,:]
    elif change=='reverse_one':v[0]=v[0,::-1,:]
    else:v[0]=v[1]
    with pytest.raises(ValueError,match='winding|closed'):saved_boundary_media(s,v,i,p)


def test_actual_canonical_ribbon_closed_and_outward():
    import json,subprocess
    from pathlib import Path
    from examples.meowdoku.optical_topology import require_closed_outward
    root=Path(__file__).resolve().parents[1]
    script="""import {catEyeMarble} from './examples/meowdoku/web/marble_mesh.js';
const mesh=catEyeMarble({glass:0,eye:1,accent:2},{rings:24,segments:48});
console.log(JSON.stringify(mesh.ribbons.map(r=>mesh.interior.slice(r.start,r.start+r.count).map(f=>f.slice(0,3)))));"""
    result=subprocess.run(['node','--input-type=module','-e',script],cwd=root,check=True,capture_output=True,text=True,timeout=15)
    for triangles in json.loads(result.stdout):require_closed_outward(np.asarray(triangles,np.float32))
