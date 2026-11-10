"""Complete native fixture boundaries from exact captured component ranges.

No metadata shape assertion certifies geometry. Sheets use native orientation
and embedding proofs; boxes use the incumbent exact cuboid union admission.
Same-body separation remains independently checked by StudioMeshEnvironment.
"""
from meowdoku.studio_open_sheet import orient_open_sheet
from meowdoku.studio_cuboid_union import build_cuboid_union
from meowdoku.studio_mesh_manifold import NativeMeshBoundary
import copy
import numpy as np


def prepare_fixture_boundaries(mesh,part,metadata):
    indices=tuple(part.source_indices);count=len(indices)
    if (not count or len(set(indices))!=count or any(type(i) is not int or not 0<=i<len(mesh.triangles) for i in indices)
            or metadata.get('id')!=part.id or metadata.get('bodyId')!=part.body_id
            or len(metadata.get('faces',()))!=count):
        raise ValueError('fixture source identity/range does not match its whole mesh')
    components=metadata.get('components');groups=metadata.get('componentGroups')
    if not isinstance(components,list) or not components or not isinstance(groups,list):
        raise ValueError('complete fixture component ownership is required')
    def span(record):
        first,n=record.get('firstTriangle'),record.get('triangleCount')
        if type(first) is not int or type(n) is not int or first<0 or n<1 or first+n>count:
            raise ValueError('fixture component range is invalid')
        return tuple(range(first,first+n))
    ownership={};component_ranges={}
    for component in components:
        name=component.get('id')
        if not isinstance(name,str) or not name or name in component_ranges or component.get('bodyId')!=part.body_id:
            raise ValueError('fixture component identity is invalid')
        selected=span(component);component_ranges[name]=selected
        for i in selected:
            if i in ownership:raise ValueError('fixture component source overlap')
            ownership[i]=name
    if set(ownership)!=set(range(count)):raise ValueError('fixture component source coverage is incomplete')
    for local,index in enumerate(indices):
        face=metadata['faces'][local]
        if len(face)<4 or face[3]!=mesh.material_ids[index]:
            raise ValueError('fixture source material identity differs from whole mesh')
    selected_groups=[];group_owned=set();group_names=set()
    for group in groups:
        name=group.get('id');selected=span(group)
        if not isinstance(name,str) or not name or name in group_names or group_owned.intersection(selected):
            raise ValueError('fixture group identity/range overlap')
        group_names.add(name);group_owned.update(selected)
        selected_set=set(selected)
        if any(selected_set.intersection(r) and not set(r)<=selected_set for r in component_ranges.values()):
            raise ValueError('fixture group splits an authored component')
        selected_groups.append((group,selected))
    records=[];accounted=set();boxes=[];open_groups=[]
    def sheet(name,local,original_groups=()):
        global_indices=tuple(indices[i] for i in local)
        faces=[[*mesh.triangles[i],mesh.material_ids[i]] for i in global_indices]
        ids=[mesh.source_ids[i] for i in global_indices]
        result=orient_open_sheet(faces,ids,body_id=part.body_id,closed=False,two_sided=True)
        records.append({'id':part.id+'/'+name,'bodyId':part.body_id,'closed':False,'convex':False,
            'vertices':[list(p) for p in result.vertices],'triangles':[list(t) for t in result.triangles],
            'materialIds':list(result.material_ids),'sourceTriangleIds':list(result.source_ids),
            'embeddedBoundaryEvidence':result.embedding.evidence_id,
            'sourceProvenance':{'relation':'two-sided-open-sheet-corner-permutation','sourcePartIds':[part.id],
                'meshDigest':mesh.digest,'sourceTriangleIndices':list(global_indices),
                'originalGroups':copy.deepcopy(list(original_groups)),
                'sourceTriangleIds':ids,'cornerPermutations':[list(p) for p in result.corner_permutations],
                'zeroDistanceOrientationAnchor':ids[0],'embeddingProof':result.embedding.to_data(),
                'maxCoordinateRounding':result.max_coordinate_rounding}})
        accounted.update(local)
    for group,local in selected_groups:
        shape=group.get('intendedShape');construction=group.get('sourceConstruction')
        if (shape,construction) in {('open-front-box','boxHousing'),('open-front-polygonal-tube','tubeHousing'),
                                   ('open-ended-polygonal-tube','snootTube')}:
            open_groups.append((group,local))
        elif group['id']=='modifier':
            # Individual box validation below, never a claimed group hull.
            if any(not (c.get('intendedShape')=='cuboid' and c.get('sourceConstruction')=='solidBox')
                   for c in components if set(component_ranges[c['id']])<=set(local)):
                raise ValueError('unsupported fixture modifier component')
        else:raise ValueError('unsupported fixture component group '+group['id'])
    # Authored group boundaries are not necessarily physical boundaries. Join
    # only exact shared native points, never proximity or overlapping AABBs.
    # A point-only join enters the existing connected-fan/manifold gate and
    # refuses; crossings without shared vertices remain separate and must
    # still pass the independent same-body embedding guard.
    native_points=[{tuple(float(np.float32(v)) for v in p)
                    for i in local for p in mesh.triangles[indices[i]]}
                   for _,local in open_groups]
    remaining=set(range(len(open_groups)))
    while remaining:
        members={min(remaining)};pending=list(members);remaining-=members
        while pending:
            a=pending.pop()
            for b in sorted(remaining):
                if native_points[a]&native_points[b]:
                    remaining.remove(b);members.add(b);pending.append(b)
        ordered=sorted(members,key=lambda i:open_groups[i][1][0])
        originals=[open_groups[i][0] for i in ordered]
        local=tuple(sorted(i for member in ordered for i in open_groups[member][1]))
        sheet('+'.join(g['id'] for g in originals),local,originals)
    for component in components:
        local=component_ranges[component['id']]
        if set(local)<=accounted:continue
        if component.get('intendedShape')=='cuboid' and component.get('sourceConstruction')=='solidBox':
            box={'id':part.id+'/'+component['id'],'body_id':part.body_id,'closed':True,
                 'source_indices':tuple(indices[i] for i in local)}
            # Real native convex admission, eight-corner axis alignment, all
            # six side coverage and material checks precede any overlap test.
            single=build_cuboid_union(mesh,[box])
            low=tuple(min(v[k] for v in single.vertices) for k in range(3))
            high=tuple(max(v[k] for v in single.vertices) for k in range(3))
            boxes.append((box,low,high,local));accounted.update(local)
        elif component['id']=='emitter' and component.get('sourceConstruction') in ('circleFace/reversed','quadFace/reversed'):
            sheet(component['id'],local)
        else:raise ValueError('unaccounted or unsupported fixture component '+component['id'])
    # Closed AABB overlap/touch graph is exact only after all boxes passed
    # axis-aligned geometry admission above. Separate graph components stay
    # separate boundaries with the SAME body identity.
    remaining=set(range(len(boxes)))
    while remaining:
        members={min(remaining)};pending=list(members);remaining-=members
        while pending:
            a=pending.pop()
            for b in sorted(remaining):
                if all(boxes[a][1][k]<=boxes[b][2][k] and boxes[b][1][k]<=boxes[a][2][k] for k in range(3)):
                    remaining.remove(b);members.add(b);pending.append(b)
        ordered=sorted(members);union=build_cuboid_union(mesh,[boxes[i][0] for i in ordered])
        record={'id':part.id+'/modifier-union/'+str(ordered[0]),'bodyId':part.body_id,'closed':True,'convex':False,
            'vertices':[list(v) for v in union.vertices],'triangles':[list(t) for t in union.triangles],
            'materialIds':list(union.material_ids),'sourceTriangleIds':list(union.triangle_ids),
            'embeddedBoundaryEvidence':union.embedded_boundary_evidence,'sourceProvenance':union.to_data()}
        record['sourceProvenance']['meshDigest']=mesh.digest
        record['sourceProvenance']['sourceTriangleIndices']=[i for j in ordered for i in boxes[j][0]['source_indices']]
        NativeMeshBoundary(record['vertices'],record['triangles'],record['sourceTriangleIds'],closed=True,
            body_id=part.body_id,embedded_boundary_evidence=union.embedded_boundary_evidence)
        records.append(record)
    if accounted!=set(range(count)):raise ValueError('fixture boundary source accounting is incomplete')
    if len({r['id'] for r in records})!=len(records):raise ValueError('fixture output boundary identity collision')
    return records
