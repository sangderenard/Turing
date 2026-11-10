import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'examples')]
from meowdoku.studio_sheet_solid_ownership import (certify_sheet_solid_ownership,
    make_sheet_solid_certificate,validate_sheet_solid_certificate)
from meowdoku.studio_open_sheet import orient_open_sheet
from meowdoku.studio_mesh_bake import StudioMeshBake
from meowdoku.studio_cuboid_union import build_cuboid_union
from meowdoku.studio_mesh_manifold import NativeMeshBoundary
from tests.test_meowdoku_studio_cuboid_union import box
from tests.test_meowdoku_studio_mesh_manifold import vec,value,quad,boundary
from tests.test_meowdoku_studio_fixture_boundaries import fixtures,inputs
from meowdoku.studio_fixture_boundaries import prepare_fixture_boundaries


def setup_case():
    sheet_faces=[[*t,0] for t in quad((-2,-2,0),(2,-2,0),(2,2,0),(-2,2,0))]
    faces=sheet_faces+box((0,-1,-1),(1,1,1))
    mesh=StudioMeshBake.from_faces(faces,source_ids=[f'source:{i}' for i in range(len(faces))],frame_id='native-test')
    s=orient_open_sheet(sheet_faces,mesh.source_ids[:2],body_id='body',closed=False,two_sided=True)
    sheet={'id':'sheet','bodyId':'body','closed':False,'vertices':s.vertices,'triangles':s.triangles,
        'materialIds':s.material_ids,'sourceTriangleIds':s.source_ids,'embeddedBoundaryEvidence':s.embedding.evidence_id,
        'sourceProvenance':{'relation':'two-sided-open-sheet-corner-permutation','meshDigest':mesh.digest,
            'sourceTriangleIndices':[0,1],'cornerPermutations':s.corner_permutations}}
    boxes=[{'id':'box','body_id':'body','closed':True,'source_indices':tuple(range(2,14))}]
    u=build_cuboid_union(mesh,boxes)
    solid={'id':'solid','bodyId':'body','closed':True,'vertices':u.vertices,'triangles':u.triangles,
        'materialIds':u.material_ids,'sourceTriangleIds':u.triangle_ids,'embeddedBoundaryEvidence':u.embedded_boundary_evidence,
        'sourceProvenance':u.to_data()}
    return mesh,sheet,solid,boxes


def test_source_bound_certificate_rebuilds_actual_union_and_sheet():
    mesh,sheet,solid,boxes=setup_case()
    cert=make_sheet_solid_certificate(mesh,sheet,solid,boxes)
    verified=validate_sheet_solid_certificate(cert,[solid,sheet])
    assert (verified['sheet_index'],verified['solid_index'])==(1,0)
    assert verified['boxes']==(((0.,-1.,-1.),(1.,1.,1.)),)
    assert certify_sheet_solid_ownership(mesh,sheet,solid,boxes).contains((0,1,1))


@pytest.mark.parametrize('field',['vertices','triangles','materialIds','bodyId'])
def test_stale_actual_packet_is_rejected(field):
    mesh,sheet,solid,boxes=setup_case();cert=make_sheet_solid_certificate(mesh,sheet,solid,boxes)
    changed=json.loads(json.dumps(solid))
    if field=='bodyId':changed[field]='other'
    elif field=='vertices':changed[field][0][0]+=.1
    elif field=='triangles':changed[field][0].reverse()
    else:changed[field][0]+=1
    with pytest.raises(ValueError):validate_sheet_solid_certificate(cert,[sheet,changed])


def test_rehashed_forged_bounds_and_source_geometry_cannot_grant_ownership():
    mesh,sheet,solid,boxes=setup_case();original=make_sheet_solid_certificate(mesh,sheet,solid,boxes)
    for change in (lambda c:c['sourceBoxes'][0]['high'].__setitem__(0,10),
                   lambda c:c['sourceBoxGeometry'][0]['faces'][0][0].__setitem__(0,10)):
        cert=copy.deepcopy(original);change(cert);cert.pop('digest')
        cert['digest']=hashlib.sha256(json.dumps(cert,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        with pytest.raises(ValueError):validate_sheet_solid_certificate(cert,[sheet,solid])


def test_native_clipped_edges_corners_and_inside_keep_nearest_coverage():
    mesh,sheet,solid,boxes=setup_case();ownership=certify_sheet_solid_ownership(mesh,sheet,solid,boxes)
    original=NativeMeshBoundary(sheet['vertices'],sheet['triangles'],sheet['sourceTriangleIds'],closed=False,body_id='body')
    volume=NativeMeshBoundary(solid['vertices'],solid['triangles'],solid['sourceTriangleIds'],closed=True,body_id='body',embedded_boundary_evidence=solid['embeddedBoundaryEvidence'])
    # Exact independent clipped reference, used ONLY as a test oracle. It is
    # never returned as render geometry or substituted into the actual law.
    faces=[];xs=(-2,0,1,2);ys=(-2,-1,1,2)
    for i in range(3):
        for j in range(3):
            if (i,j)==(1,1):continue
            faces+=quad((xs[i],ys[j],0),(xs[i+1],ys[j],0),(xs[i+1],ys[j+1],0),(xs[i],ys[j+1],0))
    clipped=boundary(faces)
    def distances(b,p,mask=False):
        contacts,diag=b.query(vec(p))
        assert value(diag['geometry_valid'])==value(diag['finite'])==value(diag['nearest_found'])==1
        assert value(diag['singular_features'])==value(diag['sign_ambiguous'])==0
        return [value(c.distance) for c in contacts if value(c.valid)>0 and
                not (mask and ownership.contains(tuple(value(v) for v in c.point)))]
    for p in [(-.1,0,.1),(-.1,1.1,.1),(.1,1.1,.1),(1.1,1.1,.1),(.5,0,1.1),
              (0,1,.1),(0,1,0),(.5,0,.2),(.5,0,0),(1,1,1)]:
        solid_dist=distances(volume,p)
        actual=distances(original,p,True)+solid_dist
        reference=distances(clipped,p)+solid_dist
        assert actual and reference
        np.testing.assert_allclose(min(actual),min(reference),atol=2e-6,rtol=2e-6)


def test_both_actual_table_emitters_get_source_bound_certificates(fixtures):
    for metadata in fixtures:
        if metadata['id'] not in ('table-left','table-right'):continue
        mesh,part=inputs(metadata);records=prepare_fixture_boundaries(mesh,part,metadata)
        sheet=next(r for r in records if r['id'].endswith('/emitter'))
        solid=next(r for r in records if '/modifier-union/' in r['id'])
        boxes=[{'id':part.id+'/'+c['id'],'body_id':part.body_id,'closed':True,
            'source_indices':tuple(part.source_indices[i] for i in range(c['firstTriangle'],c['firstTriangle']+c['triangleCount']))}
            for c in metadata['components'] if c.get('sourceConstruction')=='solidBox']
        cert=make_sheet_solid_certificate(mesh,sheet,solid,boxes)
        verified=validate_sheet_solid_certificate(cert,records)
        assert len(verified['boxes'])==2
        # The post/base union has empty space inside its global AABB. That
        # space must NOT become solid through a proxy bounding-box mask.
        ownership=certify_sheet_solid_ownership(mesh,sheet,solid,boxes)
        lows=[b[0] for b in verified['boxes']];highs=[b[1] for b in verified['boxes']]
        p=[min(v[0] for v in lows),max(v[1] for v in highs),min(v[2] for v in lows)]
        assert not ownership.contains(p)
