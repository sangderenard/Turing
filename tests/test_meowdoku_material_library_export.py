from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
from meowdoku.material_library_export import export_native_material_library

def test_library_exports_original_source_and_exact_native_chunks():
    result=export_native_material_library()
    assert len(result['records'])==47
    assert result['self_contained_runtime_buffers'] is False
    assert sum(r['native_compiles'] for r in result['records'])==38
    for r in result['records']:
        if r['native_compiles']:
            assert len(r['compiled']['pbr'])==16
            assert isinstance(r['compiled']['n_bands'],int)
            assert isinstance(r['profile_dependencies'],dict)
            assert len(r['compiled']['raymat_compat'])==20
        else:
            assert 'unresolved color_profile_name' in r['error']
            assert 'compiled' not in r
        assert r['source']['key']==r['key']
    json.dumps(result,allow_nan=False)
