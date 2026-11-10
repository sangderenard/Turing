"""Named original source references survive actual MRP1 admission."""
import pytest
from tests.test_meowdoku_marble_photo import photo_payload,_wire,photos


def test_named_emitters_survive_decode_with_exact_source_revision(photo_payload):
    from camera_designer.studio_named_emitters import named_emitter
    store,digest,vertices,normals,ids=photo_payload
    d,_=named_emitter('fluorescent_cool_white')
    reference={'material':0,'profile':d['key'],'revision':d['revision'],'gain':.5,'filter':[1,1,1]}
    p=[0.]*16;p[6]=1.5;p[7]=1
    fixture={'vertices':vertices[0].ravel().tolist(),'material_ids':[0],'pbr':p,'emitter_profiles':{'version':1,'records':[reference]}}
    result=photos.decode_upload(_wire(digest,vertices,normals,ids,fixtures=fixture),store)
    assert result[3]['fixtures']['emitter_profiles']==fixture['emitter_profiles']
    reference['revision']='0'*64
    with pytest.raises(photos.PhotoError,match='revision'):
        photos.decode_upload(_wire(digest,vertices,normals,ids,fixtures=fixture),store)

@pytest.mark.parametrize('field,value',[('gain',10**400),('filter',[10**400,1,1])])
def test_named_numeric_overflow_returns_photo_error(photo_payload,field,value):
    from camera_designer.studio_named_emitters import named_emitter
    store,digest,vertices,normals,ids=photo_payload
    d,_=named_emitter('fluorescent_cool_white')
    reference={'material':0,'profile':d['key'],'revision':d['revision'],'gain':.5,'filter':[1,1,1]}
    reference[field]=value
    p=[0.]*16;p[6]=1.5;p[7]=1
    fixture={'vertices':vertices[0].ravel().tolist(),'material_ids':[0],'pbr':p,'emitter_profiles':{'version':1,'records':[reference]}}
    with pytest.raises(photos.PhotoError,match='gain/filter'):
        photos.decode_upload(_wire(digest,vertices,normals,ids,fixtures=fixture),store)
