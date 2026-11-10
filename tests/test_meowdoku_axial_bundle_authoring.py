"""Generated axial-cut authoring remains separate from runtime activation."""
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from examples.meowdoku.marble_region_media import saved_axial_bundle, saved_region_media


@pytest.fixture(scope='module')
def authored_cases():
    script="""import {catEyeMarble} from './examples/meowdoku/web/marble_mesh.js';
import {regionMediaFromGeometry} from './examples/meowdoku/web/marble_optical_authoring.js';
const result=[];
const cases=[];
for(const angles of [[0,0,0],[35,-21,68],[0,0,90]])for(const n of [2,3,4,7])cases.push({angles,n,rings:12,segments:24});
cases.push({angles:[35,-21,68],n:2,rings:96,segments:128});
cases.push({angles:[-30,90,180],n:2,rings:128,segments:12});
for(const {angles,n,rings,segments} of cases) {
  const [yaw,pitch,roll]=angles;
  const mesh=catEyeMarble({glass:0,eye:1,accent:2},{kind:'cat_eye',ribbonLayout:'axial',strandCount:n,
    rings,segments,eyeLength:.85,eyeCurve:.28,ribbonTwist:-2.7,ribbonPhase:75*Math.PI/180,
    orientationYaw:yaw*Math.PI/180,orientationPitch:pitch*Math.PI/180,orientationRoll:roll*Math.PI/180});
  result.push({design:{'base-shape':'marble',kind:'cat_eye','ribbon-layout':'axial',strands:n,
    length:85,curve:28,twist:-270,phase:75,'orientation-yaw':yaw,'orientation-pitch':pitch,
    'orientation-roll':roll,shell:'glass','bubble-count':0,optical_media:{version:1,materials:[]},
    region_media:regionMediaFromGeometry(mesh,0,'glass_rod')},
    vertices:[...mesh.interior,...mesh.glass].map(f=>f.slice(0,3)),
    material_ids:[...mesh.interior,...mesh.glass].map(f=>f[3])});
}
console.log(JSON.stringify(result));"""
    run=subprocess.run(['node','--input-type=module','-e',script],
                       cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True,text=True,timeout=20)
    return json.loads(run.stdout)


def case_parts(case):
    d=copy.deepcopy(case['design'])
    return d,d['region_media']['regions'],np.asarray(case['vertices'],np.float32)


def test_real_generator_frames_and_symmetric_rods_round_trip_with_explicit_validated_admission(authored_cases):
    for case in authored_cases:
        d,r,v=case_parts(case);before=copy.deepcopy(d)
        result=saved_axial_bundle(d,r,v)
        assert result==d['region_media']['axial_bundle']
        result['frame']['origin'][0]+=1
        assert d==before
        pbr=np.zeros((3,16),np.float32);pbr[:,5]=1;pbr[:,6]=[1.52,1.54,1.56]
        saved={'design':d,'materials':{'index':{'glass':0}},'physical':{'radius-mm':10}}
        assert saved_region_media(saved,v,np.asarray(case['material_ids']),pbr)==d['region_media']


@pytest.mark.parametrize('field,value',[
    ('regions',[2,1]),('regions',[1,1]),('regions',[True,2]),('regions',[1,2,3]),
    ('eye_length',.75),('eye_curve',0),('twist',1),('phase',0),('phase',float('nan')),
    ('unexpected',1)])
def test_bad_bundle_fields_rejected(authored_cases,field,value):
    d,r,v=case_parts(authored_cases[0]);d['region_media']['axial_bundle'][field]=value
    with pytest.raises(ValueError):saved_axial_bundle(d,r,v)


@pytest.mark.parametrize('field,value',[
    ('origin',[.1,0,0]),('origin',[0,0]),('origin',[False,0,0]),
    ('axes',[[1,0,0],[0,1,0],[0,0,-1]]),('scale',.5),('scale',0),('scale',True),('scale',float('inf')),
    ('extra',0)])
def test_bad_generator_frame_rejected(authored_cases,field,value):
    d,r,v=case_parts(authored_cases[0]);d['region_media']['axial_bundle']['frame'][field]=value
    with pytest.raises(ValueError):saved_axial_bundle(d,r,v)


@pytest.mark.parametrize('field,value',[
    ('kind','hybrid'),('ribbon-layout','separated'),('strands',3),('strands',True),
    ('orientation-yaw',5),('orientation-roll','0'),('length',80)])
def test_recipe_changes_cannot_relabel_existing_bundle(authored_cases,field,value):
    d,r,v=case_parts(authored_cases[0]);d[field]=value
    with pytest.raises(ValueError):saved_axial_bundle(d,r,v)


def test_nonuniform_or_reordered_geometry_cannot_claim_common_bundle(authored_cases):
    for mutation in ('shape','order','mode','range','other_shape'):
        d,r,v=case_parts(authored_cases[1])
        if mutation=='shape':v[r[2]['start']+10,0,0]+=.01
        if mutation=='order':
            a,b=r[1]['start'],r[2]['start'];n=r[1]['count']
            v[a:a+n],v[b:b+n]=v[b:b+n].copy(),v[a:a+n].copy()
        if mutation=='mode':r[2]['mode']='injection_dye'
        if mutation=='range':r[2]['start']+=1
        if mutation=='other_shape':r.append({'start':0,'count':4,'mode':'glass_rod','material':1})
        with pytest.raises(ValueError):saved_axial_bundle(d,r,v)


@pytest.mark.parametrize('field,recipe,value,factor',[('twist','twist',100,.01),('phase','phase',45,np.pi/180)])
def test_matching_recipe_edits_cannot_reorient_cuts_without_rebuilding_geometry(authored_cases,field,recipe,value,factor):
    d,r,v=case_parts(authored_cases[0]);d[recipe]=value;d['region_media']['axial_bundle'][field]=value*factor
    with pytest.raises(ValueError,match='phase or twist'):
        saved_axial_bundle(d,r,v)


def test_absent_bundle_preserves_legacy_contract():
    assert saved_axial_bundle({},[],None) is None
    with pytest.raises(ValueError,match='descriptor'):
        saved_axial_bundle({'region_media':{'axial_bundle':None}},[],None)
