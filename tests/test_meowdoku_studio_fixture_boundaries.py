import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
from meowdoku.studio_mesh_bake import StudioMeshBake
from meowdoku.studio_scene_prepare import StudioSourcePart
from meowdoku.studio_fixture_boundaries import prepare_fixture_boundaries
from meowdoku.studio_mesh_environment import StudioMeshEnvironment,CONTACT_MODEL
from meowdoku.studio_environment import SCHEMA


@pytest.fixture(scope='module')
def fixtures():
    if not shutil.which('node'):pytest.skip('Node required')
    return json.loads(subprocess.run(['node','tests/meowdoku_studio_fixture_components_check.mjs','--json'],
        cwd=ROOT,check=True,capture_output=True,text=True,timeout=20).stdout)['parts']


def inputs(metadata):
    faces=[[*[tuple(v*.01+(.05 if k==1 else 0) for k,v in enumerate(p)) for p in f[:3]],f[3]] for f in metadata['faces']]
    mesh=StudioMeshBake.from_faces(faces,source_ids=[metadata['id']+':'+str(i) for i in range(len(faces))],frame_id='test-native')
    part=StudioSourcePart(metadata['id'],metadata['bodyId'],'fixture',False,False,tuple(range(len(faces))))
    return mesh,part


def environment(records):
    return StudioMeshEnvironment({'schema':SCHEMA,'contact_model':CONTACT_MODEL,
        'room':{'low':[-1,-1,-1],'high':[1,1,1]},'parts':records})


def test_actual_fixture_roles_measured_admission(fixtures):
    outcomes={}
    for metadata in fixtures:
        before=copy.deepcopy(metadata);mesh,part=inputs(metadata)
        try:
            records=prepare_fixture_boundaries(mesh,part,metadata)
            assert {r['bodyId'] for r in records}=={part.body_id}
            accounted=[i for r in records for i in r['sourceProvenance']['sourceTriangleIndices']]
            assert sorted(accounted)==list(part.source_indices)
            environment(records)
            outcomes[part.id]='admitted'
        except ValueError as error:
            outcomes[part.id]=str(error)
        assert metadata==before
    print(json.dumps(outcomes,indent=2))
    assert len(outcomes)==10
    assert all(outcomes[name]=='admitted' for name in ('key-left','key-right','fill-lower','fill-upper','kicker-left','kicker-right','background','underlight'))
    for name in ('table-left','table-right'):
        assert 'native-triangles-intersect-outside-shared-topology' in outcomes[name]


@pytest.mark.parametrize('change',[
    lambda m:m['components'].pop(),
    lambda m:m['components'][0].__setitem__('triangleCount',1),
    lambda m:m['components'][1].__setitem__('firstTriangle',0),
    lambda m:m['componentGroups'][0].__setitem__('triangleCount',9),
    lambda m:m['components'][0].__setitem__('bodyId','wrong'),
    lambda m:m['faces'][0].__setitem__(3,1000),
])
def test_bad_source_ownership_is_never_partial_success(fixtures,change):
    original=fixtures[0];mesh,part=inputs(original);changed=copy.deepcopy(original);change(changed)
    with pytest.raises(ValueError):prepare_fixture_boundaries(mesh,part,changed)


def test_actual_axis_box_verified_but_rotated_or_incomplete_box_refuses(fixtures):
    source=next(m for m in fixtures if m['id']=='table-left')
    component=copy.deepcopy(next(c for c in source['components'] if c['id']=='modifier/stand-base'))
    first=component['firstTriangle'];component['firstTriangle']=0
    metadata={'id':'isolated-base','bodyId':source['bodyId'],
        'faces':copy.deepcopy(source['faces'][first:first+12]),'components':[component],'componentGroups':[]}
    mesh,part=inputs(metadata)
    records=prepare_fixture_boundaries(mesh,part,metadata)
    assert len(records)==1 and records[0]['closed'] is True
    environment(records)
    rotated=copy.deepcopy(metadata)
    for face in rotated['faces']:
        for p in face[:3]:p[0],p[2]=(p[0]-p[2])*.7071067811865476,(p[0]+p[2])*.7071067811865476
    mesh,part=inputs(rotated)
    with pytest.raises(ValueError):prepare_fixture_boundaries(mesh,part,rotated)
    missing=copy.deepcopy(metadata);missing['faces'].pop();missing['components'][0]['triangleCount']=11
    mesh,part=inputs(missing)
    with pytest.raises(ValueError):prepare_fixture_boundaries(mesh,part,missing)


def test_exact_shared_rim_retains_original_group_and_corner_provenance(fixtures):
    for name in ('background','table-right'):
        source=next(m for m in fixtures if m['id']==name)
        mesh,part=inputs(source);records=prepare_fixture_boundaries(mesh,part,source)
        joined=next(r for r in records if r['id'].endswith('/housing+shroud'))
        provenance=joined['sourceProvenance']
        groups=[g for g in source['componentGroups'] if g['id'] in ('housing','shroud')]
        assert provenance['originalGroups']==groups
        indices=sorted(i for g in groups for i in range(g['firstTriangle'],g['firstTriangle']+g['triangleCount']))
        assert provenance['sourceTriangleIndices']==indices
        assert joined['sourceTriangleIds']==[mesh.source_ids[i] for i in indices]
        import numpy as np
        for source_i,triangle,permutation in zip(indices,joined['triangles'],provenance['cornerPermutations']):
            expected=np.asarray(mesh.triangles[source_i],dtype=np.float32).astype(np.float64)
            assert [joined['vertices'][v] for v in triangle]==[list(expected[k]) for k in permutation]


def test_point_only_join_refuses_without_tolerance_weld():
    faces=[[[0,0,0],[1,0,0],[0,1,0],0],[[0,0,0],[-1,0,0],[0,-1,1],0]]
    components=[{'id':name,'bodyId':'body','firstTriangle':i,'triangleCount':1} for i,name in enumerate(('housing','shroud'))]
    groups=[{**c,'sourceConstruction':'boxHousing','intendedShape':'open-front-box'} for c in components]
    metadata={'id':'point-only','bodyId':'body','faces':faces,'components':components,'componentGroups':groups}
    mesh,part=inputs(metadata)
    with pytest.raises(ValueError,match='disconnected'):prepare_fixture_boundaries(mesh,part,metadata)
