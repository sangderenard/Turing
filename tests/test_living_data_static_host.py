"""Static adapter preserves the actual compiled map and renderer assets."""
from pathlib import Path
import hashlib
import pytest
from tools.add_living_data_static_host import patch_static_host

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/dot-oriented-arena-candidate-20261110/index.html'


def _model(document):
 marker=document.index(b'id="abstract-ui-model"');start=document.index(b'>',marker)+1
 return document[start:document.index(b'</script>',start)]


@pytest.mark.skipif(not SOURCE.exists(),reason='Exact compiled Living Data fixture is not materialized')
def test_static_adapter_retains_original_model_and_is_bound_to_source():
 source=SOURCE.read_bytes();digest=hashlib.sha256(source).hexdigest()
 out,receipt=patch_static_host(source,expected_source_sha256=digest)
 assert _model(out)==_model(source)
 assert receipt['source_recovered_exactly'] is True
 assert b'Physics is unavailable in the static map host' in out
 assert b'Game updates are unavailable in the static map host' in out
 assert b'navigationRuntime.ready = Promise.resolve()' in out
 assert out.count(b'globalThis.livingDataStaticHost = Object.freeze')==1
 with pytest.raises(ValueError):patch_static_host(source,expected_source_sha256='0'*64)
 with pytest.raises(ValueError):patch_static_host(out,expected_source_sha256=hashlib.sha256(out).hexdigest())
