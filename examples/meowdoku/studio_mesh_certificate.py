"""Bounded exact source-topology/seed parity evidence, not a contact solver.

Binary source floats are exact rationals here, as in studio_environment's
admission gate. No ray epsilon, perturbation or guessed mesh repair is used.
Closed edge topology alone does not establish absence of self intersections:
only independently authored embedded-boundary evidence enables a solid seed
certificate. Without it the result stays blocked even if ray parity exists.
"""
from fractions import Fraction


def _sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def _dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def component_seed_certificate(bake, indices, point, *, component_id, closed,
                               embedded_boundary_evidence=None,
                               direction=(1,0,0), max_triangles=4096):
    """Return explicit source IDs, exact parity, or the precise blocker.

    This uses original source geometry, not rounded native-product geometry.
    The caller must supply one authored component; geometry is never grouped
    by proximity. ``embedded_boundary_evidence`` is a producer's independent
    construction-proof identifier, not a claim derived from this edge check.
    A string is retained as provenance, never generated or inferred here.
    Empty evidence keeps closed solids blocked. Open sheets claim no volume.
    Surface-edge/vertex, coplanar-ray and on-boundary seeds are refused without
    choosing a perturbed ray. A clear outside centre still needs radius/contact
    clearance and swept-envelope admission before any warm step.
    """
    indices=tuple(indices)
    if not isinstance(component_id,str) or not component_id or type(closed) is not bool:
        raise ValueError("component identity and authored closed flag are required")
    if type(max_triangles) is not int or max_triangles<1 or len(indices)>max_triangles:
        raise ValueError("component exceeds bounded seed-certificate triangle capacity")
    if not indices or len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<len(bake.triangles) for i in indices):
        raise ValueError("component needs distinct source triangle indices")
    if embedded_boundary_evidence is not None and (not isinstance(embedded_boundary_evidence,str) or not embedded_boundary_evidence):
        raise ValueError("embedding evidence must be an independent nonempty construction-proof ID")
    def exact(v):
        if len(v)!=3:
            raise ValueError("seed and ray direction need three finite coordinates")
        return tuple(Fraction(float(x)) for x in v)
    origin,ray=exact(point),exact(direction)
    if not any(ray):
        raise ValueError("seed ray must have a nonzero direction")
    result={"schema":"turing.studio-component-seed.v1","digest":bake.digest,
            "frame_id":bake.frame_id,"component_id":component_id,
            "source_ids":[bake.source_ids[i] for i in indices],
            "point":list(point),"direction":list(direction),"closed":closed,
            "embedded_boundary_evidence":embedded_boundary_evidence,
            "certified":False,"position":None,"blockers":[],
            "contact_clearance_certified":False,"native_rounding_certified":False}
    def block(reason,source_indices=()):
        result["blockers"].append({"reason":reason,"source_ids":[bake.source_ids[i] for i in source_indices]})
    triangles=[tuple(exact(v) for v in bake.triangles[i]) for i in indices]
    edges={}
    normals=[]
    vertex_faces={}
    for local,(source,tri) in enumerate(zip(indices,triangles)):
        a,b,c=tri
        normal=_cross(_sub(b,a),_sub(c,a))
        normals.append(normal)
        if not any(normal):
            block("degenerate-source-triangle",[source])
        for vertex in tri:
            vertex_faces.setdefault(vertex,set()).add(local)
        for u,v in zip(tri,(*tri[1:],tri[0])):
            edges.setdefault((u,v),[]).append(local)
    if result["blockers"]:
        return result
    if not closed:
        result.update(certified=True,position="sheet-no-volume")
        return result
    adjacency=[set() for _ in triangles]
    for edge,owners in edges.items():
        reverse=edges.get((edge[1],edge[0]),[])
        if len(owners)!=1 or len(reverse)!=1:
            block("closed-edge-not-paired-once",[indices[j] for j in owners+reverse])
        else:
            adjacency[owners[0]].add(reverse[0])
    if result["blockers"]:
        return result
    def reached(start,allowed):
        seen=set();todo=[start]
        while todo:
            item=todo.pop()
            if item in seen:
                continue
            seen.add(item)
            todo.extend(adjacency[item]&allowed-seen)
        return seen
    if len(reached(0,set(range(len(triangles)))))!=len(triangles):
        block("component-has-disconnected-boundaries",indices)
    for incident in vertex_faces.values():
        if reached(next(iter(incident)),incident)!=incident:
            block("nonmanifold-vertex-link",[indices[j] for j in incident])
            break
    volume6=sum(_dot(a,_cross(b,c)) for a,b,c in triangles)
    if volume6<=0:
        block("nonpositive-outward-volume",indices)
    if result["blockers"]:
        return result
    intersections=[]
    for source,(a,b,c),normal in zip(indices,triangles,normals):
        denominator=_dot(normal,ray)
        numerator=_dot(normal,_sub(a,origin))
        if not denominator:
            if not numerator:
                block("ray-coplanar-with-source-face",[source])
            continue
        t=numerator/denominator
        if t<0:
            continue
        q=tuple(origin[k]+t*ray[k] for k in range(3))
        sides=[_dot(_cross(_sub(v,u),_sub(q,u)),normal) for u,v in ((a,b),(b,c),(c,a))]
        if min(sides)<0:
            continue
        if not t:
            block("seed-on-source-boundary",[source])
        elif min(sides)==0:
            block("ray-hits-source-edge-or-vertex",[source])
        else:
            intersections.append(source)
    result["intersection_source_ids"]=[bake.source_ids[i] for i in intersections]
    if result["blockers"]:
        return result
    result["parity"]=len(intersections)%2
    result["position"]="inside" if result["parity"] else "outside"
    if embedded_boundary_evidence is None:
        block("missing-independent-embedded-boundary-proof",indices)
    else:
        result["certified"]=True
    return result
