"""Compile the original Spectral triangle law, without modifying its epsilon."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from examples.meowdoku.client_ray_build import SPECTRAL, VERTEX
from src.compiler.glsl_project_ir import compile_glsl_project_to_webgl

def original_kernel():
    text=(SPECTRAL/'csrc/shaders/ray_bvh_intersect.comp.glsl').read_text(encoding='utf-8')
    start=text.index('float moller_trumbore(')
    end=text.index('\n}',start)+2
    epsilon=re.search(r'^#define T_SELF\s+([^\s]+)',text,re.M).group(1)
    return '#define T_SELF '+epsilon+'\n'+text[start:end]

def source():
    return '''#version 430 core
uniform sampler2D uOrigins;
uniform sampler2D uDirections;
uniform sampler2D uTriangles;
uniform int uTriangleCount;
uniform int uRayCount;
uniform int uWidth;
layout(location=0) out vec4 hitRecord;
layout(location=1) out vec4 hitCoordinates;
'''+original_kernel()+'''
void main() {
 ivec2 pixel=ivec2(gl_FragCoord.xy);
 hitRecord=vec4(-1.0,-1.0,0.0,0.0);hitCoordinates=vec4(0.0);
 if(pixel.y*uWidth+pixel.x>=uRayCount)return;
 vec4 org=texelFetch(uOrigins,pixel,0),direction=texelFetch(uDirections,pixel,0);
 vec3 rd=normalize(direction.xyz);float closest=direction.w;
 for(int ti=0;ti<uTriangleCount;ti++) {
  vec4 a=texelFetch(uTriangles,ivec2(ti,0),0);
  vec3 e1=texelFetch(uTriangles,ivec2(ti,1),0).xyz;
  vec3 e2=texelFetch(uTriangles,ivec2(ti,2),0).xyz;
  float u,v;float t=moller_trumbore(a.xyz,e1,e2,org.xyz,rd,u,v);
  if(t>org.w && t<closest) {
   closest=t;hitRecord=vec4(t,a.w,1.0,float(ti));
   hitCoordinates=vec4(1.0-u-v,u,v,dot(cross(e1,e2),rd)<0.0?1.0:-1.0);
  }
 }
}
'''

def build(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    src=source();artifacts={}
    for stage,text in [('vert',VERTEX),('frag',src)]:
        shader=compile_glsl_project_to_webgl(text,source_name='client_triangle_'+stage,stage_hint='vertex' if stage=='vert' else 'fragment')
        if not shader.complete:raise RuntimeError(str(shader.diagnostics))
        name='client_ray_triangle.'+stage+'.glsl';data=shader.source.encode();(directory/name).write_bytes(data)
        artifacts[stage]={'file':name,'sha256':hashlib.sha256(data).hexdigest()}
    m={'schema':'turing.client-triangle-query.v1','scope':'triangle-only-distance-query','source_owner':'Spectral ray_bvh_intersect.comp.glsl moller_trumbore','source_sha256':hashlib.sha256(src.encode()).hexdigest(),'maximum_triangles':4096,'maximum_batch':4096,'artifacts':artifacts,'epsilon':'Original T_SELF=1e-4 and determinant epsilon=1e-9, unchanged','ray_stride':8,'hit_fields':['distance','owner','valid','triangle_index'],'coordinate_fields':['b0','b1','b2','front_facing_sign'],'limitations':'No BVH; O(rays*triangles). No spectral transport or unsupported parametric geometry approximation.'}
    (directory/'client_ray_triangle.manifest.json').write_text(json.dumps(m,indent=2));return m
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);print(json.dumps(build(p.parse_args().out),indent=2))
