"""Bounded exact self-intersection proof for the actual native-f32 mesh.

The incumbent Spectral-compatible packed BVH prunes disjoint triangle boxes.
Remaining intersections use exact integer/rational predicates, including
coplanar overlap. This proves geometric embedding only: the native boundary
gate still owns closedness, orientation and manifold topology checks.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import combinations
import struct

import numpy as np

from src.compiler.abstract_ui_bvh import build_packed_triangle_bvh


def _sub(a,b):return tuple(x-y for x,y in zip(a,b))
def _dot(a,b):return sum(x*y for x,y in zip(a,b))
def _cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def _at(a,b,t):return tuple(a[k]+t*(b[k]-a[k]) for k in range(3))


def _segment_triangle(a,b,triangle,normal):
    """Exact endpoints of segment intersection with a closed triangle."""
    da=_dot(normal,_sub(a,triangle[0]));db=_dot(normal,_sub(b,triangle[0]))
    if (da>0 and db>0) or (da<0 and db<0):return ()
    if da or db:
        t=Fraction(da,da-db)
        if not 0<=t<=1:return ()
        p=_at(a,b,t)
        if all(_dot(_cross(_sub(v,u),_sub(p,u)),normal)>=0
               for u,v in zip(triangle,(*triangle[1:],triangle[0]))):return (p,)
        return ()
    # Both endpoints lie on the plane. Clip its segment parameter against
    # the triangle's three exact inward halfspaces, retaining touching points.
    low,high=Fraction(0),Fraction(1)
    for u,v in zip(triangle,(*triangle[1:],triangle[0])):
        edge=_sub(v,u)
        fa=_dot(_cross(edge,_sub(a,u)),normal)
        fb=_dot(_cross(edge,_sub(b,u)),normal)
        slope=fb-fa
        if not slope:
            if fa<0:return ()
        elif slope>0:
            low=max(low,Fraction(-fa,slope))
        else:
            high=min(high,Fraction(-fa,slope))
        if low>high:return ()
    return (_at(a,b,low),_at(a,b,high))


class NativeEmbeddingError(ValueError):
    def __init__(self,reason,*,source_ids=(),triangle_indices=(),examined_pairs=0):
        self.report={"reason":reason,"source_ids":list(source_ids),
                     "triangle_indices":list(triangle_indices),"examined_pairs":examined_pairs,
                     "certified":False}
        super().__init__(reason+(": "+", ".join(source_ids) if source_ids else ""))


@dataclass(frozen=True)
class NativeEmbeddingProof:
    vertices: tuple
    triangles: tuple
    source_ids: tuple
    digest: str
    evidence_id: str
    examined_pairs: int
    exact_pairs: int
    bvh_nodes: int
    max_coordinate_rounding: float

    def to_data(self):
        return {"schema":"turing.native-mesh-embedding-proof.v1","digest":self.digest,
                "evidence_id":self.evidence_id,"coordinate_storage":"float32",
                "predicate":"exact-dyadic-integer-and-rational-segment-triangle-clipping",
                "allowed_intersections":"common-indexed-edge-or-vertex-only",
                "source_ids":list(self.source_ids),"examined_pairs":self.examined_pairs,
                "exact_pairs":self.exact_pairs,"bvh_nodes":self.bvh_nodes,
                "max_coordinate_rounding":self.max_coordinate_rounding,
                "geometric_embedding":True,"manifold_topology_certified":False}


def verify_native_embedding(vertices,triangles,source_ids,*,max_triangles=4096,max_pairs=20000):
    """Prove all native-f32 triangle pairs intersect only at shared topology.

    No input assertion can grant this proof. Pair budget includes leaf pairs
    even when tighter individual AABBs subsequently exclude them. Overflow
    refuses with an incomplete-proof report. Float64 originals are rounded
    exactly once to the same float32 coordinates used by NativeMeshBoundary.
    """
    if type(max_triangles) is not int or max_triangles<1 or type(max_pairs) is not int or max_pairs<0:
        raise ValueError("positive triangle and nonnegative pair capacities are required")
    original=np.asarray(vertices,dtype=np.float64)
    if original.ndim!=2 or original.shape[1]!=3 or not len(original) or not np.isfinite(original).all():
        raise ValueError("embedding proof needs finite XYZ vertices")
    with np.errstate(over="ignore",invalid="ignore"):
        native=original.astype("<f4")
    if not np.isfinite(native).all():raise ValueError("native coordinates exceed float32 range")
    verts=tuple(map(tuple,native.astype(np.float64)))
    tris=tuple(tuple(t) for t in triangles);ids=tuple(source_ids)
    if not tris or len(tris)>max_triangles:raise ValueError("embedding proof exceeds triangle capacity")
    if len(ids)!=len(tris) or len(set(ids))!=len(ids) or any(not isinstance(s,str) or not s for s in ids):
        raise ValueError("one unique source identity is required per triangle")
    if any(len(t)!=3 or len(set(t))!=3 or any(type(i) is not int or not 0<=i<len(verts) for i in t) for t in tris):
        raise ValueError("invalid embedding triangle indices")
    ratios=[tuple(float(x).as_integer_ratio() for x in v) for v in verts]
    denominator=max(d for v in ratios for _,d in v)  # every denominator is a power of two
    exact=tuple(tuple(n*(denominator//d) for n,d in v) for v in ratios)
    shapes=tuple(tuple(exact[i] for i in tri) for tri in tris)
    normals=tuple(_cross(_sub(b,a),_sub(c,a)) for a,b,c in shapes)
    for i,n in enumerate(normals):
        if not any(n):raise NativeEmbeddingError("degenerate-native-triangle",source_ids=(ids[i],),triangle_indices=(i,))
    float_shapes=tuple(tuple(verts[i] for i in tri) for tri in tris)
    bvh=build_packed_triangle_bvh(float_shapes,leaf_size=4)
    bounds=tuple((tuple(min(p[k] for p in t) for k in range(3)),
                  tuple(max(p[k] for p in t) for k in range(3))) for t in float_shapes)
    overlap=lambda a,b:all(a[0][k]<=b[1][k] and b[0][k]<=a[1][k] for k in range(3))
    node_bounds=lambda n:(n[:3],n[4:7])
    stack=[(0,0)];examined=0;tested=0
    while stack:
        ai,bi=stack.pop();a,b=bvh.nodes[ai],bvh.nodes[bi]
        if not overlap(node_bounds(a),node_bounds(b)):continue
        ac,bc=int(a[9]),int(b[9])
        if ac and bc:
            aa=bvh.triangle_order[int(a[8]):int(a[8])+ac]
            bb=bvh.triangle_order[int(b[8]):int(b[8])+bc]
            pairs=combinations(aa,2) if ai==bi else ((i,j) for i in aa for j in bb)
            for i,j in pairs:
                examined+=1
                if examined>max_pairs:
                    raise NativeEmbeddingError("embedding-pair-capacity-exceeded",source_ids=(ids[i],ids[j]),
                                               triangle_indices=(i,j),examined_pairs=examined)
                if not overlap(bounds[i],bounds[j]):continue
                tested+=1
                common=set(tris[i])&set(tris[j])
                if len(common)==3:
                    raise NativeEmbeddingError("duplicate-native-triangle",source_ids=(ids[i],ids[j]),
                                               triangle_indices=(i,j),examined_pairs=examined)
                # Distinct noncoplanar planes sharing an indexed edge intersect
                # only on its line; two nondegenerate triangles span exactly
                # that common edge on the line, so no clipping is necessary.
                if len(common)==2 and any(_cross(normals[i],normals[j])):continue
                allowed=tuple(exact[v] for v in sorted(common))
                def permitted(p):
                    if not allowed:return False
                    if len(allowed)==1:return p==allowed[0]
                    u,v=allowed;edge=_sub(v,u);delta=_sub(p,u)
                    return not any(_cross(delta,edge)) and 0<=_dot(delta,edge)<=_dot(edge,edge)
                # The intersection of two closed triangles is convex. Its
                # extreme points occur on a boundary edge of either triangle;
                # clipping all six edges therefore proves the entire
                # intersection is contained in the allowed shared point/edge.
                # This also covers containment, area overlap and zero-area
                # touching in the coplanar case, without an epsilon or ray.
                for source,target in ((i,j),(j,i)):
                    triangle=shapes[source]
                    for u,v in zip(triangle,(*triangle[1:],triangle[0])):
                        for point in _segment_triangle(u,v,shapes[target],normals[target]):
                            if not permitted(point):
                                raise NativeEmbeddingError("native-triangles-intersect-outside-shared-topology",
                                    source_ids=(ids[i],ids[j]),triangle_indices=(i,j),examined_pairs=examined)
        elif ai==bi:
            left,right=int(a[3]),int(a[7])
            stack.extend(((left,left),(left,right),(right,right)))
        elif not ac:
            stack.extend(((int(a[3]),bi),(int(a[7]),bi)))
        else:
            stack.extend(((ai,int(b[3])),(ai,int(b[7]))))
    h=hashlib.sha256(b"turing.native-f32-embedding.v1\0")
    h.update(struct.pack("<QQ",len(verts),len(tris)));h.update(native.tobytes())
    for tri,ident in zip(tris,ids):
        h.update(struct.pack("<3Q",*tri));raw=ident.encode();h.update(struct.pack("<Q",len(raw)));h.update(raw)
    digest=h.hexdigest()
    return NativeEmbeddingProof(verts,tris,ids,digest,"turing.native-f32-embedding:"+digest,
                               examined,tested,len(bvh.nodes),float(np.max(np.abs(native.astype(np.float64)-original))))
