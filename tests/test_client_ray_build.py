from examples.meowdoku.client_ray_build import build, source
from camera_designer.parametric_surfaces import SphericalSurface
import hashlib

def test_original_kernel_and_manifest(tmp_path):
    original = SphericalSurface(R=1.0, r_max=1.0).glsl_intercept_fn('original_sphere_hit', min_t_argument=True)
    assert original in source()
    manifest = build(tmp_path)
    assert manifest['scope'] == 'sphere-only-distance-query'
    for artifact in manifest['artifacts'].values():
        data = (tmp_path / artifact['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact['sha256']
        assert b'#version 300 es' in data
