"""Computational orientation of explicitly two-sided OPEN native sheets.

Never repairs render faces/normals or infers a closed solid. Native open-sheet
distance forces are unsigned and orientation independent away from exact zero
distance. At exact zero the existing pseudonormal fallback uses the consistent
orientation propagated from source triangle zero, whose winding is retained.
"""
from dataclasses import dataclass
from collections import deque
import numpy as np

from meowdoku.studio_mesh_embedding import verify_native_embedding
from meowdoku.studio_mesh_manifold import NativeMeshBoundary


@dataclass(frozen=True)
class OrientedOpenSheet:
    vertices: tuple
    triangles: tuple
    source_ids: tuple
    material_ids: tuple
    corner_permutations: tuple
    max_coordinate_rounding: float
    embedding: object
    boundary: object


def orient_open_sheet(faces, source_ids, *, body_id, closed, two_sided,
                      max_triangles=1024, max_features=4096, max_pairs=20000):
    """Copy exact f32 corners, orient by edge parity, then prove admission.

    Output triangle i retains source i and material i. Its corner permutation
    maps each output corner to that source face's corner, with no interpolation,
    tolerance weld, triangle removal, source normal change or material merge.
    One connected sheet is required; separate components need separate calls.
    """
    if closed is not False or two_sided is not True:
        raise ValueError("only explicitly two-sided open sheets may be oriented")
    if type(max_triangles) is not int or max_triangles < 1:
        raise ValueError("positive triangle capacity required")
    if not isinstance(faces,(list,tuple)) or not 0<len(faces)<=max_triangles:
        raise ValueError("open sheet missing or exceeds triangle capacity")
    ids=tuple(source_ids)
    if len(ids)!=len(faces) or any(not isinstance(s,str) or not s for s in ids) or len(set(ids))!=len(ids):
        raise ValueError("distinct source identities required for every triangle")
    vertices=[];lookup={};original=[];materials=[];edges={};rounding=0.
    for ti,face in enumerate(faces):
        if len(face)<4 or type(face[3]) is not int or face[3]<0:
            raise ValueError("every source triangle needs its material identity")
        values=np.asarray(face[:3],dtype=np.float64)
        if values.shape!=(3,3) or not np.isfinite(values).all():
            raise ValueError("finite source triangle coordinates required")
        with np.errstate(over='ignore',invalid='ignore'):
            native=values.astype(np.float32)
        if not np.isfinite(native).all():raise ValueError("native coordinate overflow")
        rounding=max(rounding,float(np.max(np.abs(native.astype(np.float64)-values))))
        tri=[]
        for point in native:
            key=tuple(float(v) for v in point)
            if key not in lookup:lookup[key]=len(vertices);vertices.append(key)
            tri.append(lookup[key])
        if len(set(tri))!=3:raise ValueError("degenerate native triangle")
        original.append(tuple(tri));materials.append(face[3])
        for u,v in zip(tri,(*tri[1:],tri[0])):
            key=tuple(sorted((u,v)));uses=edges.setdefault(key,[])
            uses.append((ti,u<v))
            if len(uses)>2:raise ValueError("nonmanifold edge incidence")
    if not any(len(uses)==1 for uses in edges.values()):
        raise ValueError("closed topology is not an open sheet")
    adjacency=[[] for _ in faces]
    for uses in edges.values():
        if len(uses)==2:
            (a,da),(b,db)=uses
            adjacency[a].append((b,da==db));adjacency[b].append((a,da==db))
    flips={0:False};pending=deque([0])
    while pending:
        a=pending.popleft()
        for b,parity in adjacency[a]:
            expected=flips[a]^parity
            if b in flips:
                if flips[b]!=expected:raise ValueError("nonorientable sheet cycle")
            else:flips[b]=expected;pending.append(b)
    if len(flips)!=len(faces):raise ValueError("disconnected open sheet")
    permutations=tuple((0,2,1) if flips[i] else (0,1,2) for i in range(len(faces)))
    triangles=tuple(tuple(tri[k] for k in perm) for tri,perm in zip(original,permutations))
    vertices=tuple(vertices)
    # Existing admission owns degeneracy, vertex fans, feature capacity and
    # exact embedding; orientation alone is deliberately not a certificate.
    boundary=NativeMeshBoundary(vertices,triangles,ids,closed=False,body_id=body_id,max_features=max_features)
    proof=verify_native_embedding(vertices,triangles,ids,max_triangles=max_triangles,max_pairs=max_pairs)
    return OrientedOpenSheet(vertices,triangles,ids,tuple(materials),permutations,rounding,proof,boundary)
