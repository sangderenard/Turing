"""Digest-covered rod/dye/air region admission for the shared photo shaders."""
from __future__ import annotations
import copy
import math
import numpy as np
from .optical_topology import require_closed_outward, centered_insphere_radius


def saved_axial_bundle(design,regions,vertices):
    """Validate optional authored cuts without enabling their runtime transport.

    Axes are the shell-local directions of the generator X/Y/Z basis. This
    record describes one common frame, not per-rod transforms or material
    precedence. Rod ordering is the generator's evenly spaced axial ordering.
    """
    media=design.get('region_media',{})
    if not isinstance(media,dict):raise ValueError('invalid region media descriptor')
    if 'axial_bundle' not in media:return None
    value=media['axial_bundle']
    if not isinstance(value,dict) or set(value)!={'regions','eye_length','eye_curve','twist','phase','frame'}:
        raise ValueError('invalid axial bundle descriptor')
    count=design.get('strands')
    if (design.get('kind')!='cat_eye' or design.get('ribbon-layout')!='axial' or
            type(count) is not int or not 2<=count<=7):
        raise ValueError('axial bundle requires the symmetric axial cat-eye recipe')
    ids=value['regions']
    if (not isinstance(ids,list) or any(type(i) is not int for i in ids) or
            ids!=list(range(1,count+1)) or len(regions)<=count):
        raise ValueError('axial bundle regions must follow authored ribbon order')
    host=regions[0]
    if (not isinstance(host,dict) or host.get('mode')!='host' or
            type(host.get('count')) is not int or host['count']<4):
        raise ValueError('axial bundle requires an explicit host')
    for index in ids:
        region=regions[index]
        if (not isinstance(region,dict) or region.get('mode')!='glass_rod' or
                any(type(region.get(key)) is not int for key in ('start','count')) or
                region.get('start')!=(index-1)*host['count'] or region.get('count')!=host['count']):
            raise ValueError('axial bundle does not match generated ribbon ranges')
    # Explicit additive confetti is outside the axial ownership bundle. It
    # keeps ordinary closed-volume admission and pairwise disjoint AABB checks
    # below; only the authored ribbon IDs receive the axial overlap exception.
    chips = design.get('confetti-count', 0) if design.get('confetti-enabled') is True else 0
    if type(chips) is not int or not 0 <= chips <= 500:
        raise ValueError('invalid additive confetti count')
    chip_regions = regions[count+1:count+1+chips]
    if len(chip_regions) != chips or any(not isinstance(r,dict) or
            r.get('mode') != 'glass_rod' or r.get('count') != 4 or
            r.get('start') != count*host['count']+4*i for i,r in enumerate(chip_regions)):
        raise ValueError('axial bundle confetti does not match generated chip ranges')
    if any(not isinstance(r,dict) or r.get('mode')!='air_bubble' for r in regions[count+1+chips:]):
        raise ValueError('axial bundle cannot include other rod or dye shapes')
    def finite_number(x):return type(x) in (int,float) and math.isfinite(x)
    for field,recipe,factor,lo,hi in (
            ('eye_length','length',.01,.4,.85),('eye_curve','curve',.01,-.28,.28),
            ('twist','twist',.01,-3.2,3.2),('phase','phase',math.pi/180,0,2*math.pi)):
        number=value[field];authored=design.get(recipe)
        if (not finite_number(number) or not lo<=number<=hi or not finite_number(authored) or
                not math.isclose(number,authored*factor,rel_tol=0,abs_tol=1e-12)):
            raise ValueError('axial bundle parameters disagree with the saved recipe')
    frame=value['frame']
    if not isinstance(frame,dict) or set(frame)!={'origin','axes','scale'}:
        raise ValueError('invalid axial bundle frame')
    origin,axes,scale=frame['origin'],frame['axes'],frame['scale']
    if (not isinstance(origin,list) or len(origin)!=3 or not all(map(finite_number,origin)) or
            not isinstance(axes,list) or len(axes)!=3 or
            any(not isinstance(row,list) or len(row)!=3 or not all(map(finite_number,row)) for row in axes) or
            not finite_number(scale) or not 0<scale<=1):
        raise ValueError('invalid axial bundle frame')
    angles=[design.get('orientation-'+axis) for axis in ('yaw','pitch','roll')]
    if any(not finite_number(x) or abs(x)>180 for x in angles):
        raise ValueError('invalid saved axial bundle orientation')
    yaw,pitch,roll=[x*math.pi/180 for x in angles]
    cy,sy,cx,sx,cz,sz=math.cos(yaw),math.sin(yaw),math.cos(pitch),math.sin(pitch),math.cos(roll),math.sin(roll)
    # Same Y, then X, then Z rotation as marble_mesh.js; rows are basis axes.
    expected=np.array([[cz*cy-sz*sx*sy,sz*cy+cz*sx*sy,-cx*sy],
                       [-sz*cx,cz*cx,sx],
                       [cz*sy+sz*sx*cy,sz*sy-cz*sx*cy,cx*cy]])
    axes=np.asarray(axes,np.float64);origin=np.asarray(origin,np.float64)
    if not np.allclose(axes,expected,rtol=0,atol=2e-12):
        raise ValueError('axial bundle axes disagree with the generator orientation')
    points=np.asarray(vertices[:count*host['count']],np.float64)
    if points.shape!=(count*host['count'],3,3) or not np.isfinite(points).all():
        raise ValueError('invalid axial bundle geometry')
    points=points.reshape(count,-1,3)
    # orient rotates about the triangle-weighted common centroid, then fits
    # around the shell origin. Its final centroid is s*c, hence o = mean-R*mean.
    centroid=points.reshape(-1,3).mean(axis=0)
    if not np.allclose(origin,centroid-centroid@axes,rtol=0,atol=3e-6):
        raise ValueError('axial bundle origin disagrees with the common generator centroid')
    if not any(angles):
        if scale!=1 or np.any(origin!=0):raise ValueError('unoriented axial bundle has no fit transform')
    else:
        radius=float(np.linalg.norm(points,axis=2).max())
        if radius>0.94+3e-6 or (scale<1 and not math.isclose(radius,.94,rel_tol=0,abs_tol=3e-6)):
            raise ValueError('axial bundle fit disagrees with the generator envelope')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        local=((points-origin)@axes.T)/scale
    if not np.isfinite(local).all():raise ValueError('invalid axial bundle inverse frame')
    length=value['eye_length'];tolerance=3e-6/scale
    if not (np.allclose(local[:,:,1].min(axis=1),-length,rtol=0,atol=tolerance) and
            np.allclose(local[:,:,1].max(axis=1),length,rtol=0,atol=tolerance)):
        raise ValueError('axial bundle length disagrees with the saved geometry')
    first=local[0];center=value['eye_curve']*(1-(first[:,1]/length)**2)
    radial_x=first[:,0]-center
    # The generator's j=0 ring vertex lies on each ribbon's positive wide
    # axis. Follow that existing vertex identity, rather than infer the cut
    # orientation from material colors or merely trust changed recipe text.
    triangles=first.reshape(-1,3,3)
    top_cap=np.abs(triangles[:,0,1]-length)<=tolerance
    segments=int(np.count_nonzero(top_cap));face_count=host['count']
    if (not 12<=segments<=256 or not np.all(top_cap[:segments]) or
            np.any(top_cap[segments:]) or face_count%(2*segments)):
        raise ValueError('axial bundle does not have generated ribbon topology')
    rings=face_count//(2*segments)+1
    if not 8<=rings<=128:raise ValueError('invalid axial bundle longitudinal tessellation')
    for ring in range(1,rings):
        point=triangles[segments+(ring-1)*2*segments,0]
        y=length*math.cos(math.pi*ring/rings)
        angle=value['phase']+value['twist']*point[1]/length
        x=point[0]-value['eye_curve']*(1-(point[1]/length)**2)
        along=x*math.cos(angle)+point[2]*math.sin(angle)
        across=-x*math.sin(angle)+point[2]*math.cos(angle)
        if abs(point[1]-y)>tolerance or abs(across)>tolerance or along<=0:
            raise ValueError('axial bundle phase or twist disagrees with generated ribbon axes')
    for index in range(1,count):
        angle=math.pi*index/count;c,s=math.cos(angle),math.sin(angle)
        target=np.column_stack((center+radial_x*c-first[:,2]*s,first[:,1],radial_x*s+first[:,2]*c))
        if not np.allclose(local[index],target,rtol=0,atol=tolerance):
            raise ValueError('axial bundle geometry is not a common symmetric ribbon family')
    return copy.deepcopy(value)


def saved_region_media(saved,vertices,material_ids,pbr):
    design=saved.get('design',{})
    if 'region_media' not in design:return None
    native=design.get('canonical_volume_optics')
    if native is not None:
        if native!={'version':1,'source':'native_spectral_bands'} or type(native.get('version')) is not int or 'optical_media' in design or 'authoring' not in saved.get('materials',{}):
            raise ValueError('native volume optics requires canonical sources and no competing optical media')
    if ('optical_media' not in design and native is None) or 'boundary_media' in design:
        raise ValueError('region media require optical definitions and replace fixed boundary pairs')
    if design.get('base-shape') not in ('marble','d4','d6','d8','d10','d12','d20') or float(design.get('coverage',0))!=0 or float(design.get('band-count',0))!=0:
        raise ValueError('region media require an uncoated generated host')
    if float(saved.get('physical',{}).get('radius-mm',0))<1:
        raise ValueError('region media require radius of at least 1 mm')
    value=design['region_media']
    if (not isinstance(value,dict) or not {'version','regions'}<=set(value) or
            set(value)-{'version','regions','axial_bundle'} or type(value['version']) is not int or value['version']!=1):
        raise ValueError('expected version 1 region media')
    regions=value['regions'];modes=('host','glass_rod','injection_dye','air_bubble')
    if not isinstance(regions,list) or not 1<=len(regions)<=1024:raise ValueError('invalid region count')
    owners=np.full(len(vertices),-1,np.int32);bounds=[];bubbles=0
    shell=saved['materials']['index'][design['shell']]
    die=design.get('base-shape')!='marble'
    if die and design.get('bubble-count',0):raise ValueError('optical die bubbles are not yet supported')
    host_radius=1.0
    for ri,r in enumerate(regions):
        if not isinstance(r,dict) or set(r)!={'start','count','mode','material'} or r['mode'] not in modes:
            raise ValueError('invalid region descriptor')
        start,count,mat=r['start'],r['count'],r['material'];mode=r['mode']
        if any(type(x) is not int for x in (start,count,mat)) or start<0 or count<4 or start+count>len(vertices):
            raise ValueError('region outside saved geometry')
        if (ri==0)!=(mode=='host') or (ri==0 and mat!=shell):raise ValueError('region zero must be the glass host')
        if (mode=='air_bubble' and mat!=-1) or (mode!='air_bubble' and not 0<=mat<len(pbr)):
            raise ValueError('invalid region material')
        if np.any(owners[start:start+count]>=0):raise ValueError('overlapping region triangle membership')
        owners[start:start+count]=ri
        tris=vertices[start:start+count];points=tris.reshape(-1,3)
        require_closed_outward(tris)
        expected=shell if mode=='air_bubble' else mat
        if not np.all(material_ids[start:start+count]==expected):raise ValueError('region material disagrees with saved faces')
        if mode=='host':
            if die:
                host_radius=centered_insphere_radius(tris)
            elif not np.allclose(np.linalg.norm(points,axis=1),1,atol=3e-6,rtol=0):raise ValueError('host is not the saved spherical shell')
        elif np.any(np.linalg.norm(points,axis=1)>=host_radius-3e-6):raise ValueError('region must fit strictly within the verified host insphere')
        if mode=='air_bubble':bubbles+=1
        else:
            row=pbr[mat]
            if row[4]!=0 or row[5]<=0 or row[7]!=0 or row[6]<1 or np.any(row[8:11]!=0):
                raise ValueError('region requires a nonmetal nonemissive dielectric with opacity zero')
            if mode=='injection_dye' and row[6]!=pbr[shell,6]:
                raise ValueError('injected dye uses host IOR; use glass rod for a distinct index')
        bounds.append((points.min(axis=0),points.max(axis=0)))
    if np.any(owners<0):raise ValueError('region map must cover saved geometry')
    count=design.get('bubble-count',0)
    if type(count) is not int or count!=bubbles:raise ValueError('region bubbles disagree with saved design')
    bundle_members=set()
    if 'axial_bundle' in value:
        if value['axial_bundle'] is None:raise ValueError('invalid axial bundle descriptor')
        bundle=saved_axial_bundle(design,regions,vertices)
        bundle_members=set(bundle['regions'])
    for i,a in enumerate(regions[1:],1):
        for j in range(1,i):
            b=regions[j]
            ambiguous=(a['mode']=='glass_rod' and b['mode']=='glass_rod' and a['material']!=b['material']) or ({a['mode'],b['mode']}=={'glass_rod','injection_dye'})
            if ambiguous and not (i in bundle_members and j in bundle_members and a['mode']==b['mode']=='glass_rod'):
                lo,hi=bounds[i];other_lo,other_hi=bounds[j]
                if not np.any((hi<other_lo)|(other_hi<lo)):
                    raise ValueError('overlapping rod materials need authored cut ownership or injection target')
    return copy.deepcopy(value)
