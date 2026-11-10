import numpy as np
import pytest
from tests.test_meowdoku_studio_mesh_static import boundary
from tests.test_meowdoku_studio_mesh_manifold import boundary as from_faces, vec, value


def test_identity_filter_default_parity_and_frozen_geometry():
    mesh=boundary();p=vec((.4,.4,.1))
    original,diag=mesh.query(p)
    for bake in (None,mesh.bake_static_geometry()):
        filtered,fd=mesh.query(p,static_geometry=bake,contact_filter=lambda q:q[0]==q[0])
        for key in diag:np.testing.assert_array_equal(diag[key].data,fd[key].data)
        for a,b in zip(original,filtered):
            for key in ('distance','valid'):np.testing.assert_array_equal(getattr(a,key).data,getattr(b,key).data)
            for key in ('normal','point'):
                for x,y in zip(getattr(a,key),getattr(b,key)):np.testing.assert_array_equal(x.data,y.data)


def test_exclusion_uses_feature_point_and_precedes_nearest_and_singular():
    mesh=boundary();p=vec((.4,.4,.1))
    contacts,d=mesh.query(p,contact_filter=lambda q:q[0]>.5)
    assert all(value(c.valid)==0 for c in contacts)
    assert value(d['nearest_found'])==0
    assert value(d['nearest_distance'])==0
    for key in ('finite','geometry_valid'):assert value(d[key])==1
    for key in ('inside','sign_ambiguous','singular_features'):assert value(d[key])==0
    # Closest point on the outside edge has x=0, unlike particle x=-.1.
    contacts,d=mesh.query(vec((-.1,.4,.1)),contact_filter=lambda q:q[0]>=0)
    assert sum(value(c.valid) for c in contacts)==1
    assert value(d['nearest_found'])==1
    calls=[]
    mesh.apply(p,p[0]*0+.2,p[0]*0+1,lambda dl,valid,n:calls.append(value(valid)),contact_filter=lambda q:q[0]>.5)
    assert calls and not any(calls)


def test_closed_filter_refused_before_callback():
    faces=[((0,0,0),(0,1,0),(1,0,0)),((0,0,0),(1,0,0),(0,0,1)),((0,0,0),(0,0,1),(0,1,0)),((1,0,0),(0,1,0),(0,0,1))]
    mesh=from_faces(faces,closed=True)
    with pytest.raises(ValueError,match='open sheets'):
        mesh.query(vec((2,2,2)),contact_filter=lambda q:pytest.fail('must not evaluate'))


def test_filtered_nearest_reselects_remaining_corner_face():
    from tests.test_meowdoku_studio_mesh_manifold import quad
    floor=quad((0,0,-1),(0,0,1),(1,0,1),(1,0,-1))
    rail=quad((0,0,1),(0,0,-1),(0,1,-1),(0,1,1))
    mesh=from_faces(floor+rail)
    p=vec((.1,.2,0))
    _,before=mesh.query(p)
    contacts,after=mesh.query(p,contact_filter=lambda q:q[0]>0)
    assert value(before['nearest_distance'])==np.float32(.1)
    assert value(after['nearest_distance'])==np.float32(.2)
    retained=[c for c in contacts if value(c.valid)]
    assert len(retained)==1 and all(value(c.point[1])==0 for c in retained)
