"""Saved, digest-covered air-boundary admission for still and live exposures."""
from __future__ import annotations
import copy
import numpy as np

def saved_boundary_media(saved, vertices, material_ids, pbr):
    design=saved.get('design',{})
    count=design.get('bubble-count',0)
    if not isinstance(count,int) or isinstance(count,bool) or not 0<=count<=64:
        raise ValueError('bubble count must be a whole number from 0 to 64')
    explicit_optics = 'optical_media' in design
    if count==0 and not explicit_optics:
        if 'boundary_media' in design:raise ValueError('boundary media requires a nonzero bubble count')
        return None
    if (design.get('base-shape')!='marble' or float(design.get('coverage',0))!=0 or
            float(design.get('band-count',0))!=0):
        raise ValueError('air bubbles require an uncoated spherical marble')
    if (design.get('kind') in ('cat_eye','hybrid') and design.get('ribbon-layout','separated')=='axial'
            and int(design.get('strands',1))>1):
        raise ValueError('air bubbles require disjoint interiors; centered axial ribbons overlap')
    if float(saved.get('physical',{}).get('radius-mm',0))<1.:
        raise ValueError('air bubbles require a marble radius of at least 1 mm')
    shell=saved['materials']['index'][design['shell']]
    row=pbr[shell]
    if not (row[4]==0 and row[5]>0 and row[7]<.999 and row[6]>1 and np.all(row[8:11]==0)):
        raise ValueError('air bubbles require a transparent nonmetal, nonemissive glass host with IOR greater than one')
    value=design.get('boundary_media')
    if not isinstance(value,dict) or set(value)!={'version','ranges'} or value['version']!=1:
        raise ValueError('saved bubbles have no version 1 boundary media')
    ranges=value['ranges']
    if not isinstance(ranges,list) or not ranges or len(ranges)>1024:
        raise ValueError('invalid saved boundary ranges')
    cursor=0;outer=0;bubbles=[];inclusions=[];inclusion_bounds=[]
    for item in ranges:
        if not isinstance(item,dict) or set(item)!={'start','count','positive','negative'}:
            raise ValueError('invalid saved boundary range')
        if any(not isinstance(v,int) or isinstance(v,bool) for v in item.values()):
            raise ValueError('boundary indices must be whole numbers')
        start,size,positive,negative=(item[k] for k in ('start','count','positive','negative'))
        if start!=cursor or size<=0 or start+size>len(vertices) or any(v< -1 or v>=len(pbr) for v in (positive,negative)):
            raise ValueError('boundary ranges must cover saved triangles once in their original order')
        cursor+=size
        face_ids=material_ids[start:cursor]
        points=vertices[start:cursor].reshape(-1,3)
        if explicit_optics:
            from .optical_topology import require_closed_outward
            require_closed_outward(vertices[start:cursor])
        if positive==-1 and negative==shell:
            outer+=1
            if not np.all(face_ids==shell) or not np.allclose(np.linalg.norm(points,axis=1),1.,atol=3e-6,rtol=0):
                raise ValueError('outer glass boundary differs from the saved unit sphere')
        elif positive==shell and negative==-1:
            if not np.all(face_ids==shell):raise ValueError('bubble boundary must use the glass host surface')
            center=(points.min(axis=0)+points.max(axis=0))/2
            distances=np.linalg.norm(points-center,axis=1);radius=float(np.mean(distances))
            if not .01799<=radius<=.08001 or np.ptp(distances)>3e-6 or np.linalg.norm(center)+radius>.940003:
                raise ValueError('bubble must be a contained sphere in the supported size range')
            tri=vertices[start:cursor]
            cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            valid=np.linalg.norm(cross,axis=1)>1e-12
            if np.any(np.einsum('ij,ij->i',cross[valid],tri[valid,0]-center)<=0):
                raise ValueError('bubble boundary winding must point from air toward glass')
            bubbles.append((center,radius))
        elif positive==shell and negative>=0:
            if not np.all(face_ids==negative):raise ValueError('interior boundary material differs from saved triangles')
            mat=pbr[negative]
            if mat[7]<.999 and (mat[4]!=0 or mat[5]<=0 or mat[6]<1 or np.any(mat[8:11]!=0)):
                raise ValueError('transmissive interiors must be nonmetal, nonemissive dielectrics')
            center=(points.min(axis=0)+points.max(axis=0))/2
            inclusions.append((center,float(np.max(np.linalg.norm(points-center,axis=1)))))
            if explicit_optics:
                if np.any(np.linalg.norm(points,axis=1)>=1.):
                    raise ValueError('optical inclusion must be strictly inside the glass host')
                inclusion_bounds.append((points.min(axis=0),points.max(axis=0)))
        else:raise ValueError('saved medium pair is not a supported shell, inclusion or air bubble')
    if cursor!=len(vertices) or outer!=1 or len(bubbles)!=count:
        raise ValueError('saved boundary membership does not match the requested bubbles and shell')
    for i,(center,radius) in enumerate(bubbles):
        for other,other_radius in inclusions+bubbles[:i]:
            if np.linalg.norm(center-other)<radius+other_radius+.004-4e-6:
                raise ValueError('air bubbles overlap another bubble or an existing inclusion')
    if explicit_optics:
        # Conservative admission, not an invented ownership rule. Overlapping
        # boxes may contain disjoint solids; those wait for exact region proof.
        for j,(lo,hi) in enumerate(inclusion_bounds):
            for other_lo,other_hi in inclusion_bounds[:j]:
                if not np.any((hi<other_lo)|(other_hi<lo)):
                    raise ValueError('optical inclusion bounds overlap; exact cut-volume ownership is required')
    return copy.deepcopy(value)
