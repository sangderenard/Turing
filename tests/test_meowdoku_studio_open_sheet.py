import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
from meowdoku.studio_open_sheet import orient_open_sheet


def orient(faces,**kwargs):
    return orient_open_sheet(faces,[f'source:{i}' for i in range(len(faces))],
                             body_id='sheet',closed=False,two_sided=True,**kwargs)


def prove_retained(faces,result):
    assert result.corner_permutations[0]==(0,1,2)
    assert result.material_ids==tuple(f[3] for f in faces)
    assert result.embedding.source_ids==result.source_ids
    for face,tri,perm in zip(faces,result.triangles,result.corner_permutations):
        expected=np.asarray(face[:3],dtype=np.float32).astype(np.float64)
        assert tuple(result.vertices[i] for i in tri)==tuple(tuple(expected[k]) for k in perm)


def test_fold_and_reversed_face_retain_corners_materials_and_anchor():
    faces=[[[0,0,0],[1,0,0],[0,1,0],3],[[0,0,0],[1,0,0],[0,0,1],7]]
    before=copy.deepcopy(faces);result=orient(faces)
    assert result.corner_permutations==((0,1,2),(0,2,1))
    prove_retained(faces,result);assert faces==before
    faces[1][1],faces[1][2]=faces[1][2],faces[1][1]
    consistent=orient(faces)
    assert consistent.corner_permutations==((0,1,2),(0,1,2))
    prove_retained(faces,consistent)


def test_real_fixture_open_housings():
    if not shutil.which('node'):pytest.skip('Node required for actual fixture sources')
    data=json.loads(subprocess.run(['node','tests/meowdoku_studio_fixture_components_check.mjs','--json'],
                    cwd=ROOT,check=True,capture_output=True,text=True,timeout=20).stdout)
    checked=flipped=0
    for part in data['parts']:
        for group in part['componentGroups']:
            if group['intendedShape'] not in ('open-front-box','open-front-polygonal-tube','open-ended-polygonal-tube'):continue
            first=group['firstTriangle'];faces=copy.deepcopy(part['faces'][first:first+group['triangleCount']])
            # Native SI frame, no tolerance welding or render-face mutation.
            for face in faces:
                for p in face[:3]:
                    for k in range(3):p[k]=p[k]*.01+(0.05 if k==1 else 0)
            before=copy.deepcopy(faces);result=orient(faces)
            prove_retained(faces,result);assert faces==before
            checked+=1;flipped+=sum(p==(0,2,1) for p in result.corner_permutations)
    assert checked==15 and flipped>0


def test_mobius_nonorientable_cycle_refuses():
    n=5;points=[]
    for i in range(n):
        t=2*math.pi*i/n
        for side in (-.2,.2):points.append([(1+side*math.cos(t/2))*math.cos(t),(1+side*math.cos(t/2))*math.sin(t),side*math.sin(t/2)])
    faces=[]
    for i in range(n):
        a,b=2*i,2*i+1;c,d=(2*i+2,2*i+3) if i+1<n else (1,0)
        faces.extend([[points[a],points[b],points[c],0],[points[b],points[d],points[c],0]])
    with pytest.raises(ValueError,match='nonorientable'):orient(faces)


def test_refuses_nonmanifold_degenerate_closed_and_unbounded():
    f=[[[0,0,0],[1,0,0],[0,1,0],0]]
    with pytest.raises(ValueError,match='nonmanifold'):orient(f*3)
    with pytest.raises(ValueError,match='degenerate'):orient([[[0,0,0]]*3+[0]])
    with pytest.raises(ValueError,match='open sheets'):
        orient_open_sheet(f,['a'],body_id='solid',closed=True,two_sided=True)
    with pytest.raises(ValueError,match='capacity'):orient(f,max_triangles=0)
    with pytest.raises(ValueError,match='closed topology'):orient(f+[[f[0][0],f[0][2],f[0][1],0]])


def test_existing_embedding_and_feature_budgets_still_gate_orientation():
    # Topologically orientable but geometrically overlapping, not a valid sheet.
    faces=[[[0,0,0],[1,0,0],[0,1,0],0],[[0,0,0],[1,0,0],[.2,.2,0],1]]
    with pytest.raises(ValueError):orient(faces)
    faces[1][2]=[0,0,1]
    with pytest.raises(ValueError,match='feature capacity'):orient(faces,max_features=1)
    with pytest.raises(ValueError):orient(faces,max_pairs=0)
