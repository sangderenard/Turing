"""Additive chips retain ordinary closed-volume and disjointness admission."""
import copy,json,subprocess
from pathlib import Path
import numpy as np
import pytest
from examples.meowdoku.marble_region_media import saved_region_media

def test_axial_ribbons_with_additive_confetti_are_admitted_without_extending_bundle_members():
    script="""import {catEyeMarble} from './examples/meowdoku/web/marble_mesh.js';
import {regionMediaFromGeometry} from './examples/meowdoku/web/marble_optical_authoring.js';
const mesh=catEyeMarble({glass:0,eye:1,accent:2,confetti:3,ribbons:[1,2,4]},
 {kind:'cat_eye',ribbonLayout:'axial',strandCount:3,rings:12,segments:24,confettiEnabled:true,confettiCount:100});
console.log(JSON.stringify({regions:regionMediaFromGeometry(mesh,0,'glass_rod'),faces:[...mesh.interior,...mesh.glass]}));"""
    result=json.loads(subprocess.run(['node','--input-type=module','-e',script],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True,timeout=10).stdout)
    design={'base-shape':'marble','kind':'cat_eye','ribbon-layout':'axial','strands':3,'length':75,'curve':18,'twist':90,'phase':0,'orientation-yaw':0,'orientation-pitch':0,'orientation-roll':0,'shell':'glass','bubble-count':0,'confetti-enabled':True,'confetti-count':100,'optical_media':{'version':1,'materials':[]},'region_media':result['regions']}
    saved={'design':design,'materials':{'index':{'glass':0}},'physical':{'radius-mm':10}}
    vertices=np.asarray([f[:3] for f in result['faces']],np.float32)
    ids=np.asarray([f[3] for f in result['faces']]);pbr=np.zeros((5,16),np.float32);pbr[:,5]=1;pbr[:,6]=1.52
    admitted=saved_region_media(saved,vertices,ids,pbr)
    assert admitted['axial_bundle']['regions']==[1,2,3]
    assert len(admitted['regions'])==104
    bad=copy.deepcopy(saved);bad['design']['confetti-enabled']=False
    with pytest.raises(ValueError):saved_region_media(bad,vertices,ids,pbr)
    bad=copy.deepcopy(saved);bad['design']['confetti-count']=99
    with pytest.raises(ValueError):saved_region_media(bad,vertices,ids,pbr)

