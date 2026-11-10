"""Exact composed region primary triangle law; ownership stays on the server."""
import hashlib,json,sys
from pathlib import Path
from examples.meowdoku.client_ray_build import SPECTRAL
from examples.meowdoku.studio_triangle_compute import build as triangle_build
from examples.meowdoku.studio_slab_compute import build as slab_build

def extract(text,prefix):
    start=text.index(prefix);end=text.index('\n}',start)+2
    return text[start:end]

def region_sources():
    sys.path.insert(0,str(SPECTRAL))
    import demo_pluck_gl as renderer
    source=renderer._GPU_SENSOR_REGION_CS
    return {name:extract(source,prefix) for name,prefix in [('hit_tri','bool hit_tri(vec3'),('slab_axis','bool slab_axis('),('hit_aabb','bool hit_aabb('),('nearest_filtered','int nearest_filtered('),('studio_region_epsilon','float studio_region_epsilon(vec3 p) {')]}

def query_identity():
    laws=region_sources()
    record={'schema':'studio.bvh-query.v1','traversal':'original-lifo-host-v1','scope':'region-primary-triangle-candidate-only; filter must be -1; sphere and partition competition remains server-side','query_variant':'region-bvh','epsilon':1e-7,'laws':laws}
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest(),record

def build(directory):
    digest,record=query_identity();laws=record['laws']
    triangle_build(directory,source=laws['hit_tri'],pure_helpers={'studio_region_epsilon':laws['studio_region_epsilon']});slab_build(directory)
    (Path(directory)/'studio_bvh_query.json').write_text(json.dumps({'query_kernel_sha256':digest,**record},indent=2))
    return digest
if __name__=='__main__':print(build('build/dot-client-ray-20261109/studio-region-compute'))
