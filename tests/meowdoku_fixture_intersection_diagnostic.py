"""Bounded CPU receipt: exact real fixture intersection semantics, no edits."""
import json
from fractions import Fraction as Q
from time import perf_counter
from test_meowdoku_studio_fixture_boundaries import fixtures,inputs
from meowdoku.studio_fixture_boundaries import prepare_fixture_boundaries
from meowdoku.studio_mesh_embedding import verify_native_embedding,_sub,_dot,_cross,_segment_triangle
from meowdoku.studio_mesh_certificate import component_seed_certificate
from meowdoku.studio_mesh_bake import StudioMeshBake


def combined(records):
    vertices=[];triangles=[];ids=[]
    for r in records:
        offset=len(vertices);vertices.extend(r['vertices'])
        triangles.extend(tuple(v+offset for v in t) for t in r['triangles'])
        ids.extend(r['id']+'/'+s for s in r['sourceTriangleIds'])
    return vertices,triangles,ids


def pair_receipt(records,a_id,b_id):
    def find(source):
        for r in records:
            if source in r['sourceTriangleIds']:
                t=r['triangles'][r['sourceTriangleIds'].index(source)]
                return tuple(tuple(Q(v) for v in r['vertices'][i]) for i in t)
        raise ValueError(source)
    a,b=find(a_id),find(b_id)
    na=_cross(_sub(a[1],a[0]),_sub(a[2],a[0]));nb=_cross(_sub(b[1],b[0]),_sub(b[2],b[0]))
    points=set()
    for source,target,normal in ((a,b,nb),(b,a,na)):
        for u,v in zip(source,(*source[1:],source[0])):points.update(_segment_triangle(u,v,target,normal))
    sign=lambda v:0 if v==0 else 1 if v>0 else -1
    signs_a=[sign(_dot(_sub(p,b[0]),nb)) for p in a]
    signs_b=[sign(_dot(_sub(p,a[0]),na)) for p in b]
    coplanar=not any(_cross(na,nb)) and not any(signs_a)
    interior=[]
    if points:
        mid=tuple(sum(p[k] for p in points)/len(points) for k in range(3))
        for tri,n in ((a,na),(b,nb)):
            interior.append(all(_dot(_cross(_sub(v,u),_sub(mid,u)),n)>0 for u,v in zip(tri,(*tri[1:],tri[0]))))
    return {'sources':[a_id,b_id],'coplanar':coplanar,'intersection_point_count':len(points),
        'intersection_extremes_are_exact_shared_vertices':points==set(a)&set(b),
        'plane_side_signs':[signs_a,signs_b],'intersection_midpoint_strict_triangle_interiors':interior,
        'shared_native_vertices':len(set(a)&set(b)),
        'intersection_points_exact':[[str(v) for v in p] for p in sorted(points)]}


def main():
    receipt={}
    for meta in fixtures.__wrapped__():
        name=meta['id']
        if name not in ('underlight','background','table-left','table-right'):continue
        mesh,part=inputs(meta);records=prepare_fixture_boundaries(mesh,part,meta)
        if name=='underlight':
            args=combined(records);start=perf_counter()
            try:
                proof=verify_native_embedding(*args,max_pairs=50000)
                receipt[name]={'result':'proved','triangles':len(args[1]),'vertices':len(args[0]),
                    'examined_pairs':proof.examined_pairs,'exact_pairs':proof.exact_pairs,'bvh_nodes':proof.bvh_nodes,
                    'seconds':perf_counter()-start,'max_pairs':50000,'evidence_id':proof.evidence_id}
            except ValueError as error:
                receipt[name]={'result':'refused','seconds':perf_counter()-start,'max_pairs':50000,
                    'error':str(error),'report':getattr(error,'report',None)}
        elif name in ('background','table-right'):
            pair=('background:51','background:14') if name=='background' else ('table-right:57','table-right:19')
            receipt[name]=pair_receipt(records,*pair)
            def source_group(group_name):
                group=next(g for g in meta['componentGroups'] if g['id']==group_name)
                ids={mesh.source_ids[i] for i in range(group['firstTriangle'],group['firstTriangle']+group['triangleCount'])}
                owner=next(r for r in records if ids<=set(r['sourceTriangleIds']))
                chosen=[i for i,s in enumerate(owner['sourceTriangleIds']) if s in ids]
                return {**owner,'triangles':[owner['triangles'][i] for i in chosen],
                    'sourceTriangleIds':[owner['sourceTriangleIds'][i] for i in chosen]}
            housing=source_group('housing');shroud=source_group('shroud')
            classes={};examples={};tested=0;all_shared=True
            for ai,at in enumerate(shroud['triangles']):
                av=[shroud['vertices'][v] for v in at]
                for bi,bt in enumerate(housing['triangles']):
                    bv=[housing['vertices'][v] for v in bt]
                    if not all(min(p[k] for p in av)<=max(p[k] for p in bv) and min(p[k] for p in bv)<=max(p[k] for p in av) for k in range(3)):continue
                    tested+=1
                    report=pair_receipt(records,shroud['sourceTriangleIds'][ai],housing['sourceTriangleIds'][bi])
                    n=report['intersection_point_count']
                    if not n:continue
                    all_shared=all_shared and report['intersection_extremes_are_exact_shared_vertices']
                    kind=('coplanar' if report['coplanar'] else 'noncoplanar')+('/point' if n==1 else '/segment-or-area')
                    if all(report['intersection_midpoint_strict_triangle_interiors']):kind+='/both-interiors'
                    classes[kind]=classes.get(kind,0)+1;examples.setdefault(kind,report)
            receipt[name]['complete_cross_boundary_scan']={'total_pairs':len(shroud['triangles'])*len(housing['triangles']),
                'bbox_overlap_pairs_exactly_tested':tested,'intersection_classes':classes,'examples':examples,
                'all_intersections_are_exact_shared_vertices_or_edges':all_shared}
        else:
            emitter=next(r for r in records if r['id'].endswith('/emitter'))
            stand=next(r for r in records if '/modifier-union/' in r['id'])
            faces=[[*[stand['vertices'][v] for v in t],stand['materialIds'][i]] for i,t in enumerate(stand['triangles'])]
            bake=StudioMeshBake.from_faces(faces,source_ids=stand['sourceTriangleIds'],frame_id='native')
            samples={}
            points=list(emitter['vertices'])
            points.extend([sum(emitter['vertices'][v][k] for v in t)/3 for k in range(3)] for t in emitter['triangles'])
            for point in points:
                for direction in ((1,0,0),(0,1,0),(0,0,1),(1,2,3),(2,3,5)):
                    cert=component_seed_certificate(bake,tuple(range(len(faces))),point,component_id=stand['id'],
                        closed=True,embedded_boundary_evidence=stand['embeddedBoundaryEvidence'],direction=direction)
                    if cert['certified']:
                        samples.setdefault(cert['position'],{'point':point,'direction':direction});break
            receipt[name]={'emitter_points_against_closed_stand':samples,
                'intersection':pair_receipt(records,'table-left:45','fixture-table-left/outer/0/2/1/1/1/0')}
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':main()
