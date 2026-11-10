from examples.meowdoku.client_triangle_build import build, source, original_kernel
import hashlib

def test_original_triangle_kernel_and_artifacts(tmp_path):
    assert original_kernel() in source()
    assert '#define T_SELF 1e-4' in original_kernel()
    m=build(tmp_path)
    assert m['scope']=='triangle-only-distance-query'
    for a in m['artifacts'].values():
        data=(tmp_path/a['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==a['sha256']
        assert b'#version 300 es' in data
