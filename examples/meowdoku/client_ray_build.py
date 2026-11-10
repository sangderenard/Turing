"""Compile the original Spectral spherical-intersection kernel for client queries.

No substitute intersection law: SphericalSurface supplies the numerical GLSL;
Turing project IR owns the WebGL2 translation. This is sphere-only geometry,
not the Studio spectral transport engine or a server workload delegation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
SPECTRAL = ROOT.parent / 'spectral-analyzer'
for directory in (ROOT, SPECTRAL):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from camera_designer.parametric_surfaces import SphericalSurface
from src.compiler.glsl_project_ir import compile_glsl_project_to_webgl

VERTEX = '''#version 430 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
'''


def source():
    original = SphericalSurface(R=1.0, r_max=1.0).glsl_intercept_fn('original_sphere_hit', min_t_argument=True)
    return '''#version 430 core
uniform sampler2D uOrigins;
uniform sampler2D uDirections;
uniform sampler2D uSpheres;
uniform int uSphereCount;
uniform int uRayCount;
uniform int uWidth;
layout(location=0) out vec4 hitRecord;
layout(location=1) out vec4 hitNormal;
''' + original + '''
void main() {
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    int ri=pixel.y*uWidth+pixel.x;
    hitRecord=vec4(-1.0,-1.0,0.0,0.0);
    hitNormal=vec4(0.0);
    if(ri>=uRayCount) return;
    vec4 origin=texelFetch(uOrigins,pixel,0);
    vec4 direction=texelFetch(uDirections,pixel,0);
    vec3 rd=normalize(direction.xyz);
    float closest=direction.w;
    for(int si=0;si<uSphereCount;si++) {
        vec4 sphere=texelFetch(uSpheres,ivec2(si,0),0);
        vec4 owner=texelFetch(uSpheres,ivec2(si,1),0);
        vec3 ro=(origin.xyz-sphere.xyz)/sphere.w+vec3(0.0,0.0,1.0);
        float localT; vec3 normal;
        if(original_sphere_hit(ro,rd,origin.w/sphere.w,localT,normal)) {
            float distance=localT*sphere.w;
            if(distance<closest) {
                closest=distance;
                hitRecord=vec4(distance,owner.x,1.0,float(si));
                hitNormal=vec4(normal,0.0);
            }
        }
    }
}
'''


def build(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    src=source()
    artifacts={}
    for stage,text in [('vert',VERTEX),('frag',src)]:
        shader=compile_glsl_project_to_webgl(text,source_name='client_sphere_query_'+stage,stage_hint='vertex' if stage=='vert' else 'fragment')
        if not shader.complete:
            raise RuntimeError(str(shader.diagnostics))
        name=f'client_ray_sphere.{stage}.glsl'
        data=shader.source.encode('utf-8'); (directory/name).write_bytes(data)
        artifacts[stage]={'file':name,'sha256':hashlib.sha256(data).hexdigest()}
    manifest={'schema':'turing.client-sphere-query.v1','scope':'sphere-only-distance-query',
        'source_owner':'Spectral SphericalSurface.glsl_intercept_fn','source_sha256':hashlib.sha256(src.encode()).hexdigest(),
        'compiler':'src.compiler.glsl_project_ir.compile_glsl_project_to_webgl','numeric':'WebGL2 highp float32',
        'maximum_spheres':65,'maximum_batch':4096,'ray_stride':8,
        'ray_fields':['origin_x','origin_y','origin_z','min_distance','direction_x','direction_y','direction_z','max_distance'],
        'hit_fields':['distance','owner_id','valid','sphere_index'],'normal':'face-forward geometric normal',
        'unit':'caller-declared common length unit; normalized direction internally','artifacts':artifacts}
    (directory/'client_ray_sphere.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().out),indent=2))
