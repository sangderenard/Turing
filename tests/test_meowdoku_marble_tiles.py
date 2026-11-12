from pathlib import Path
import os
import pytest
from tests.test_meowdoku_marble_photo import photo_payload, _wire, _FakeProcess
from meowdoku import marble_photo as photos

SPEC=Path(os.environ.get('STUDIO_TEST_SPECTRAL_ROOT',Path(__file__).resolve().parents[2]/'spectral-analyzer'))


def test_large_and_supersampled_config(photo_payload):
    store,digest,vertices,normals,ids=photo_payload
    _,_,_,config,_=photos.decode_upload(_wire(digest,vertices,normals,ids,
        transport='fixed:3',camera={'resolution':8192},oversampling=2,tiled={'tile_size':4096}),store,SPEC)
    plan=photos._tile_contract(str(SPEC)).tile_plan(config)
    assert (plan['width'],plan['output_width'],plan['total_tiles'])==(16384,8192,16)
    _,_,_,auto,_=photos.decode_upload(_wire(digest,vertices,normals,ids,
        transport='fixed:3',camera={'width':8001,'height':5001}),store,SPEC)
    assert auto['camera']['width']==8001 and auto['tiled']=={'tile_size':1024}


def test_uniform_sample_target_is_owned_config(photo_payload):
    store,digest,v,n,ids=photo_payload
    _,_,_,config,_=photos.decode_upload(_wire(digest,v,n,ids,transport='fixed:3',
        samples_per_output_pixel=64,oversampling=2),store,SPEC)
    assert config['samples_per_output_pixel']==64
    assert config['tiled']=={'tile_size':1024}
    assert photos._tile_contract(str(SPEC)).tile_plan(config)['target_samples_per_pixel']==16
    _,_,_,legacy,_=photos.decode_upload(_wire(digest,v,n,ids,transport='fixed:3'),store,SPEC)
    assert 'samples_per_output_pixel' not in legacy and 'tiled' not in legacy


@pytest.mark.parametrize('target',[None,True,0,-1,1.5,'64',65537])
def test_bad_uniform_sample_target_rejected(photo_payload,target):
    store,digest,v,n,ids=photo_payload
    with pytest.raises(photos.PhotoError,match='samples_per_output_pixel'):
        photos.decode_upload(_wire(digest,v,n,ids,transport='fixed:3',samples_per_output_pixel=target),store,SPEC)


@pytest.mark.parametrize('options',[
 {'transport':'fixed:1','camera':{'resolution':8192}},
 {'transport':'fixed:3','camera':{'resolution':8192},'client_intersections':True},
 {'transport':'fixed:3','camera':{'resolution':8192},'orbit_frames':8},
 {'transport':'fixed:3','camera':{'resolution':8192},'oversampling':True},
 {'transport':'fixed:3','camera':{'width':8192}},
 {'transport':'fixed:3','camera':{'resolution':8192},'tiled':{'tile_size':8192}},
])
def test_invalid_tiled_requests_rejected(photo_payload,options):
    store,digest,v,n,ids=photo_payload
    with pytest.raises(photos.PhotoError):photos.decode_upload(_wire(digest,v,n,ids,**options),store,SPEC)


def test_low_disk_rejected_before_worker(tmp_path,photo_payload,monkeypatch):
    store,digest,v,n,ids=photo_payload
    jobs=photos.PhotoJobs(tmp_path/'jobs',SPEC)
    class Disk:free=1
    monkeypatch.setattr(photos._tile_contract(str(SPEC)).shutil,'disk_usage',lambda p:Disk())
    monkeypatch.setattr(photos.subprocess,'Popen',lambda *a,**k:pytest.fail('must not start a worker'))
    try:
        with pytest.raises(photos.PhotoError,match='free disk') as error:
            jobs.start(_wire(digest,v,n,ids,transport='fixed:3',camera={'resolution':8192}),store)
        assert error.value.status==507
    finally:jobs.close()
