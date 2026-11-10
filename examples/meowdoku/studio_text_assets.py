"""Two explicit text products using the existing Spectral font/mesh owners.

The cookie retains original Bezier outlines. The optional glyph is an ordinary
tessellated mesh from the existing tile glyph builder, never an exact curve
collider. This module does not install either asset into the current scene.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

SPECTRAL_ROOT=Path(__file__).resolve().parents[2].parent/'spectral-analyzer'
if str(SPECTRAL_ROOT) not in sys.path:
    sys.path.insert(0,str(SPECTRAL_ROOT))
from camera_designer.curve_cookie import text_cookie


def build_text_assets(text, *, family='DejaVu Sans', weight='bold', height_m=.02,
                      border_m=.002, glyph=False, depth_ratio=.2):
    cookie=text_cookie(text,family=family,weight=weight,height_m=height_m,border_m=border_m)
    result={'schema':'turing.studio-text-assets.v1','cookie':cookie.to_dict()}
    if glyph:
        if not np.isfinite(depth_ratio) or not 0<depth_ratio<=2:
            raise ValueError('Glyph depth ratio must be positive and at most two')
        from examples.meowdoku.phong_tiles import token_mesh
        vertices,normals=token_mesh(text,family,weight,depth_ratio=depth_ratio,
                                    outline_subdivisions=2,profile_segments=1,cap_grid=4)
        if vertices.ndim!=3 or vertices.shape[1:]!=(3,3) or len(vertices)==0 or not np.isfinite(vertices).all():
            raise ValueError('Existing glyph builder returned no finite triangles')
        extent=float(np.ptp(vertices[:,:,1]))
        if extent<=0:raise ValueError('Glyph mesh has no visible height')
        vertices=np.asarray(vertices*(height_m/extent),np.float32)
        result['glyph']={'kind':'tessellated-glyph-mesh','unit':'metres',
            'vertices':vertices,'normals':normals,
            'source_owner':'examples.meowdoku.phong_tiles.token_mesh -> scene_orders._glyph_triangles',
            'font':cookie.source['font'],'depth_ratio':float(depth_ratio),
            'outline_subdivisions':2,'profile_segments':1,'cap_grid':4,
            'contact_admission':'not certified; ordinary rendered mesh only'}
    return result


def write_text_assets(directory, **kwargs):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    result=build_text_assets(**kwargs)
    paths={}
    path=directory/'curve-cookie.json'
    path.write_text(json.dumps(result['cookie'],indent=2),encoding='utf-8');paths['cookie']=str(path)
    if 'glyph' in result:
        glyph=dict(result['glyph']);vertices=glyph.pop('vertices');normals=glyph.pop('normals')
        path=directory/'glyph-mesh.npz'
        np.savez_compressed(path,vertices=vertices,normals=normals,metadata=json.dumps(glyph))
        paths['glyph']=str(path)
    return paths


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text');parser.add_argument('--output',required=True)
    parser.add_argument('--glyph',action='store_true')
    parser.add_argument('--height-m',type=float,default=.02)
    parser.add_argument('--depth-ratio',type=float,default=.2)
    args=parser.parse_args()
    print(json.dumps(write_text_assets(args.output,text=args.text,glyph=args.glyph,
                     height_m=args.height_m,depth_ratio=args.depth_ratio),indent=2))

