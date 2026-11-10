"""Explicit scene-group engine admission; no inferred optical material values."""
from __future__ import annotations
import copy
import numpy as np
from .optical_topology import require_closed_outward



def separated_on_face_axis(left, right):
    """Conservative separation proof for rotated closed slabs.

    A disjoint projection proves separation even for nonconvex triangle soups.
    Failure to find one is NOT proof of intersection and remains rejected. Face
    normals suffice for parallel grid-cell slabs; this is not a general CSG or
    convex-intersection solver. Require a float32-scale positive gap rather than
    accepting a numerically ambiguous contact on a rotated separating plane.
    """
    a=np.asarray(left,dtype=np.float64).reshape(-1,3,3)
    b=np.asarray(right,dtype=np.float64).reshape(-1,3,3)
    origin=(a.mean(axis=(0,1))+b.mean(axis=(0,1)))*.5
    av=(a-origin).reshape(-1,3);bv=(b-origin).reshape(-1,3)
    scale=max(float(np.abs(a).max()),float(np.abs(b).max()),
              float(np.ptp(av,axis=0).max()),float(np.ptp(bv,axis=0).max()),1e-12)
    tolerance=16*np.finfo(np.float32).eps*scale
    world_gap=np.maximum(bv.min(axis=0)-av.max(axis=0),av.min(axis=0)-bv.max(axis=0))
    if np.any(world_gap>tolerance):return True
    for mesh in (a,b):
        normals=np.cross(mesh[:,1]-mesh[:,0],mesh[:,2]-mesh[:,0])
        lengths=np.linalg.norm(normals,axis=1)
        for normal,length in zip(normals,lengths):
            if not np.isfinite(length) or length<=0:continue
            axis=normal/length;pa=av@axis;pb=bv@axis
            gap=max(float(pb.min()-pa.max()),float(pa.min()-pb.max()))
            if gap>tolerance:return True
    return False


def validate_scene_engines(value, vertices, material_ids, pbr):
    """Preserve authored PBR; validate closed volume ownership for optical groups.

    Surface-only legacy fixtures have no metadata and remain byte-for-byte
    unchanged. This contract does not turn artistic milk glass into scattering.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'version','groups','volumes','optical_media'} or type(value['version']) is not int or value['version'] != 1:
        raise ValueError('invalid scene material engine contract')
    groups=value['groups'];v=np.asarray(vertices,float).reshape(-1,3,3)
    ids=np.asarray(material_ids,int);p=np.asarray(pbr,float).reshape(-1,16)
    ownership=np.full(len(v),-1,int)
    if not isinstance(groups,list) or len(groups)!=2:
        raise ValueError('stage and room engine groups are required')
    names=set()
    for gi,g in enumerate(groups):
        if not isinstance(g,dict) or set(g)!={'name','engine','ranges'} or g['name'] not in ('stage','room') or g['name'] in names or g['engine'] not in ('legacy','dielectric'):
            raise ValueError('invalid scene engine group')
        names.add(g['name'])
        if not isinstance(g['ranges'],list) or len(g['ranges'])>5000:raise ValueError('invalid scene group ranges')
        for r in g['ranges']:
            if not isinstance(r,dict) or set(r)!={'start','count'} or any(type(r[k]) is not int for k in r):
                raise ValueError('invalid scene group range')
            a,n=r['start'],r['count']
            if a<0 or n<1 or a+n>len(v) or np.any(ownership[a:a+n]!=-1):
                raise ValueError('scene group ranges overlap or exceed geometry')
            ownership[a:a+n]=gi
    if np.any(ownership<0):raise ValueError('engine groups must cover every scene triangle')
    if not isinstance(value['volumes'],list) or len(value['volumes'])>1024:raise ValueError('invalid scene volume count')
    covered=np.zeros(len(v),bool);required=set();bounds=[]
    for r in value['volumes']:
        if not isinstance(r,dict) or set(r)!={'start','count','material'} or any(type(r[k]) is not int for k in r):
            raise ValueError('invalid scene volume range')
        a,n,m=r['start'],r['count'],r['material']
        if a<0 or n<4 or a+n>len(v) or not 0<=m<len(p) or np.any(covered[a:a+n]) or np.any(ids[a:a+n]!=m):
            raise ValueError('scene volume ownership disagrees with geometry')
        if len(set(ownership[a:a+n]))!=1 or groups[ownership[a]]['engine']!='dielectric':
            raise ValueError('optical volume must belong to one dielectric group')
        tri=v[a:a+n];extent=float(np.ptp(tri,axis=(0,1)).max())
        if extent<=0:raise ValueError('empty optical scene volume')
        require_closed_outward((tri-tri.mean(axis=(0,1)))/extent)
        lo,hi=tri.min(axis=(0,1)),tri.max(axis=(0,1))
        for blo,bhi,btri in bounds:
            if np.all(np.minimum(hi,bhi)>np.maximum(lo,blo)) and not separated_on_face_axis(tri,btri):
                raise ValueError('overlapping scene volume bounds require explicit ownership support')
        bounds.append((lo,hi,tri));covered[a:a+n]=True;required.add(m)
    for gi,g in enumerate(groups):
        if g['engine']=='dielectric':
            transmissive=(p[ids,5]>0)|(p[ids,7]<.999)
            if np.any((ownership==gi)&transmissive&~covered):
                raise ValueError('dielectric scene surfaces require closed volume metadata; no legacy fallback')
    # Validate the same native reference-thickness contract used by the subject.
    # Opacity/transmission are adapter flags only; never edit saved source rows.
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]/'spectral-analyzer'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from camera_designer.studio_optical_media import validate_optical_media
    projected=p.copy()
    for m in required:projected[m,5]=1;projected[m,7]=0
    if required:validate_optical_media(value['optical_media'],projected,required)
    elif value['optical_media']!={'version':1,'materials':[]}:
        raise ValueError('unused scene optical definitions')
    return copy.deepcopy(value)
