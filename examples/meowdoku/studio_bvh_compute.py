"""Artifact identity for exact laws plus explicit temporary host traversal policy."""
import hashlib,json
from pathlib import Path
from examples.meowdoku.studio_triangle_compute import build as triangle_build,studio_source
from examples.meowdoku.studio_slab_compute import build as slab_build
from examples.meowdoku.client_ray_build import SPECTRAL

def query_identity():
    text=(SPECTRAL/'demo_pluck_gl.py').read_text(encoding='utf-8')
    text=text[text.index('bool hit_tri(vec3 ro, vec3 rd, TriGeom t, out float hit_t)'):]
    laws={}
    for name,prefix in [('hit_tri','bool hit_tri('),('slab_axis','bool slab_axis('),('hit_aabb','bool hit_aabb('),('nearest_filtered','int nearest_filtered(')]:
        start=text.index(prefix);end=text.index('\n}',start)+2;laws[name]=text[start:end]
    record={'schema':'studio.bvh-query.v1','traversal':'original-lifo-host-v1','scope':'triangle-candidate-only; analytic sphere competition remains in original server continuation','epsilon':1e-7,'laws':laws}
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest(),record

def build(directory):
    triangle_build(directory);slab_build(directory)
    digest,record=query_identity();p=Path(directory)/'studio_bvh_query.json';p.write_text(json.dumps({'query_kernel_sha256':digest,**record},indent=2));return digest
if __name__=='__main__':print(build('build/dot-client-ray-20261109/studio-compute'))
