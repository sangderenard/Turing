"""Auto mode is independent of the preserved manual 1..200 limit."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
from meowdoku.marble_photo import _config, PhotoError
SAVED={'digest':'a'*64,'geometry_sha256':'b'*64,'physical':{'radius-mm':25}}

def config(**values):
 return _config({'object_digest':SAVED['digest'],**values},SAVED)[0]

def test_auto_default_and_legacy_explicit_manual_count():
 assert config()['bounce_mode']=='auto'
 assert config(max_bounces=37)['bounce_mode']=='manual'
 result=config(max_bounces=37,bounce_mode='auto')
 assert result['bounce_mode']=='auto' and result['max_bounces']==37
 assert config(max_bounces=200,bounce_mode='manual')['max_bounces']==200

@pytest.mark.parametrize('mode',['automatic',None,True,1000,{},[]])
def test_unknown_modes_are_refused(mode):
 with pytest.raises(PhotoError,match='bounce_mode'):
  config(bounce_mode=mode)

def test_probe_depth_does_not_expand_render_limit():
 with pytest.raises(PhotoError,match='max_bounces'):
  config(bounce_mode='auto',max_bounces=1000)
