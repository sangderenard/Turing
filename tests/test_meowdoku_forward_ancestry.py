"""Explicit request admission for queued ancestry; no live requests."""
import pytest
from tests.test_meowdoku_camera_pair_delivery import request_config, spectral_root
from tests.test_meowdoku_marble_photo import photo_payload


def test_queue_optin_default_and_explicit(photo_payload,spectral_root):
    default=request_config(photo_payload,spectral_root)
    assert 'light_field_reactive_queue' not in default
    assert request_config(photo_payload,spectral_root,light_field_reactive_queue='native-ancestry-v1')==dict(default,light_field_reactive_queue='native-ancestry-v1')


@pytest.mark.parametrize('mode',[None,True,0,{},[],'bad'])
def test_bad_queue_mode(photo_payload,spectral_root,mode):
    with pytest.raises(ValueError):request_config(photo_payload,spectral_root,light_field_reactive_queue=mode)

from tests.test_meowdoku_camera_pair_delivery import pair_job
from meowdoku import marble_photo as photos
import importlib.util
import json


def test_v3_actual_artifact_route_and_explicit_queue_mode(pair_job,spectral_root):
    jobs,digest,directory,config,status,receipt,_,save=pair_job
    spec=importlib.util.spec_from_file_location('ancestry_transport_fixture',spectral_root/'tests/test_studio_forward_ancestry_transport.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    store,metadata,rows=module.fixture();store.consume_forward(metadata,rows)
    manifest,payload=store.snapshot()
    receipt.update(schema=manifest['schema'],metadata=manifest,byte_length=len(payload),sha256=manifest['payload_sha256'])
    (directory/receipt['name']).write_bytes(payload)
    config['light_field_reactive_queue']='native-ancestry-v1'
    del config['light_field_camera_records']  # Empty v3 camera section needs no camera opt-in.
    save()
    with jobs.open_field(digest,receipt['name']) as stream:assert stream.read()==payload
    del config['light_field_reactive_queue'];save()
    with pytest.raises(photos.PhotoError):jobs.open_field(digest,receipt['name'])
    config['light_field_reactive_queue']='native-ancestry-v1'
    receipt['metadata']['contexts'][0]['dispatch']['native_queue']['generation']=9;save()
    with pytest.raises(photos.PhotoError):jobs.open_field(digest,receipt['name'])


def test_queue_mode_requires_information(photo_payload,spectral_root):
    from tests.test_meowdoku_forward_field import emitter_request
    from tests.test_meowdoku_marble_photo import _wire
    store,digest,vertices,normals,ids=photo_payload
    options=emitter_request();options.update(light_field_internal_media=True,light_field_reactive_queue='native-ancestry-v1')
    with pytest.raises(photos.PhotoError):photos.decode_upload(_wire(digest,vertices,normals,ids,**options),store,spectral_root)
