"""Native AbstractTensor local-feature contacts for one authored outer boundary.

Unlike a convex proxy, this retains every actual float32 triangle and its
concave features. It does not turn overlapping construction solids into a
union: the caller must supply their single conforming exterior boundary.
Table slab+leg touching/internal faces need that boundary first. Course
exterior faces may supply it if their exact indexed topology passes admission.

Faces own strict interior projections; unique edges/vertices own their local
distance minima. A concave crease has two face minima and no false radial edge
contact. An exactly coplanar triangulation seam has one edge contact, not two
face forces. Microscopic noncoplanar float32 folds remain real geometry.
"""
from dataclasses import dataclass
import hashlib
import numpy as np

from src.common.tensors import AbstractTensor as T
from meowdoku.entities import as_float
from meowdoku.fountain3d import _cap
from meowdoku.studio_contacts import _sub, _dot, _cross, _safe_positive, _select


@dataclass(frozen=True)
class FeatureContact:
    feature_id: int
    kind: str
    source_ids: tuple
    distance: object
    normal: tuple
    point: tuple
    valid: object


class NativeMeshBoundary:
    """Baked topology, actual native-f32 coordinates, incumbent wall-call ABI.

    Closed boundaries require independent embedding provenance for the rounded
    native boundary. Paired edges
    and vertex fans are checked here, but do not prove no self intersections.
    Updating vertex uniforms requires rebaking/revalidating this snapshot.
    There is no force merging across unrelated bodies or a hidden integrator.
    """
    def __init__(self, vertices, triangles, source_ids, *, closed,
                 body_id, embedded_boundary_evidence=None, max_features=4096):
        if type(closed) is not bool or not isinstance(body_id,str) or not body_id:
            raise ValueError("authored closed flag and body identity are required")
        if closed and (not isinstance(embedded_boundary_evidence,str) or not embedded_boundary_evidence):
            raise ValueError("closed boundary requires independent embedded-boundary provenance")
        values=np.asarray(vertices,dtype=np.float64)
        if values.ndim!=2 or values.shape[1]!=3 or not len(values) or not np.isfinite(values).all():
            raise ValueError("boundary needs finite XYZ vertices")
        with np.errstate(over="ignore",invalid="ignore"):
            native=values.astype(np.float32)
        if not np.isfinite(native).all():
            raise ValueError("boundary exceeds native float32 coordinates")
        self.vertices=tuple(map(tuple,native.astype(np.float64)))
        self.max_coordinate_rounding=float(np.max(np.abs(native.astype(np.float64)-values)))
        self.triangles=tuple(tuple(t) for t in triangles)
        self.source_ids=tuple(source_ids)
        if not self.triangles or len(self.source_ids)!=len(self.triangles) or len(set(self.source_ids))!=len(self.source_ids):
            raise ValueError("one distinct source ID is required per boundary triangle")
        if any(not isinstance(s,str) or not s for s in self.source_ids):
            raise ValueError("source identities must be nonempty strings")
        directed={}; edges={}; incident=[[] for _ in self.vertices]
        volume=0.0
        for ti,triangle in enumerate(self.triangles):
            if len(triangle)!=3 or len(set(triangle))!=3 or any(type(i) is not int or not 0<=i<len(native) for i in triangle):
                raise ValueError("invalid boundary triangle indices")
            a,b,c=native[list(triangle)]
            with np.errstate(over="ignore",under="ignore",invalid="ignore"):
                n=np.cross(b-a,c-a); area2=np.dot(n,n)
            if not np.isfinite(area2) or area2<=0:
                raise ValueError("boundary triangle degenerates in native float32 arithmetic")
            aa,bb,cc=(np.asarray(self.vertices[i]) for i in triangle)
            volume+=float(np.dot(aa,np.cross(bb,cc)))
            for v in triangle:
                incident[v].append(ti)
            for u,v in zip(triangle,(*triangle[1:],triangle[0])):
                if (u,v) in directed:
                    raise ValueError("duplicate or inconsistently wound boundary edge")
                directed[u,v]=ti
                edges.setdefault(tuple(sorted((u,v))),[]).append(ti)
        if any(not fan for fan in incident):
            raise ValueError("boundary contains unreferenced vertices")
        if any(len(fan)>2 or (closed and len(fan)!=2) for fan in edges.values()):
            raise ValueError("boundary must be a conforming manifold, not construction-part soup")
        if closed and (any((v,u) not in directed for u,v in directed) or volume<=0):
            raise ValueError("closed boundary must be outward wound with positive volume")
        adjacency=[set() for _ in self.triangles]
        for fan in edges.values():
            if len(fan)==2:
                a,b=fan;adjacency[a].add(b);adjacency[b].add(a)
        def connected(allowed):
            todo=[next(iter(allowed))];seen=set()
            while todo:
                i=todo.pop()
                if i not in seen:
                    seen.add(i);todo.extend(adjacency[i]&allowed-seen)
            return seen==allowed
        if not connected(set(range(len(self.triangles)))) or any(not connected(set(fan)) for fan in incident):
            raise ValueError("boundary component or vertex fan is disconnected")
        self.edges=tuple((edge,tuple(fan)) for edge,fan in sorted(edges.items()))
        self.incident=tuple(tuple(fan) for fan in incident)
        self.neighbours=tuple(tuple(sorted({v for edge in edges if i in edge for v in edge if v!=i})) for i in range(len(native)))
        if type(max_features) is not int or max_features<1 or len(self.triangles)+len(self.edges)+len(native)>max_features:
            raise ValueError("boundary exceeds declared native feature capacity")
        self.closed=closed;self.body_id=body_id
        self.embedded_boundary_evidence=embedded_boundary_evidence
        h=hashlib.sha256(native.tobytes())
        h.update(repr((self.triangles,self.source_ids,body_id,closed)).encode())
        self.digest=h.hexdigest()

    def bake_static_geometry(self):
        """Capture immutable geometry through the existing NumPy-f32 tensors."""
        from meowdoku.studio_mesh_static import bake_static_geometry
        return bake_static_geometry(self)

    def query(self, p, vertex_values=None, *, static_geometry=None, contact_filter=None):
        """Return (feature contacts, diagnostics) on AbstractTensor particle spans.

        Optional named vertex inputs must represent this validated snapshot.
        By default all normals, edge cones and angle-weighted vertex
        pseudonormals are calculated on the tensor substrate per query. The
        optional static_geometry captures those same expressions once through
        NumPy-f32 AbstractTensor for this immutable snapshot; GPU transcendental
        bit identity is not asserted. Particle-dependent equations are shared.
        Closed inside/outside follows the oriented pseudonormal at the closest
        feature of this one embedded boundary. Inside gets one shortest outward
        exit; outside retains all outward-facing local minima, including two
        sides of a concave corner. Undefined nearest sign is diagnostic, not an
        invented inside/outside answer. Callers must stop on diagnostics.
        For open sheets only, contact_filter(point) may supply independently
        proved exterior ownership. It masks candidate features before nearest
        and singular accounting. If all are excluded, nearest_found is zero;
        finite and geometry_valid remain meaningful. It does not construct new
        clipped edges and is not an embedding or ownership certificate itself.
        """
        # Exclude existing sheet features, never synthesize clipped edges.
        if contact_filter is not None and self.closed:
            raise ValueError("contact_filter is supported only for open sheets")
        zero=p[0]*0.0
        if static_geometry is not None and static_geometry.boundary_digest!=self.digest:
            raise ValueError("static geometry bake does not match this boundary snapshot")
        def static(key,produce):
            return produce() if static_geometry is None else static_geometry.value(key,produce,zero)
        finite=as_float((p[0]-p[0])==0.0)
        for component in p[1:]:
            finite=finite*as_float((component-component)==0.0)
        vertices=vertex_values if vertex_values is not None else tuple(tuple(zero+float(x) for x in v) for v in self.vertices)
        face_normals=[];raw=[];all_valid=zero+1.0
        for ti,triangle in enumerate(self.triangles):
            a,b,c=(vertices[i] for i in triangle)
            normal=static(("face-raw",ti),lambda:_cross(_sub(b,a),_sub(c,a)))
            length2=static(("face-length2",ti),lambda:_dot(normal,normal))
            all_valid=all_valid*as_float(length2>0.0)*as_float((length2-length2)==0.0)
            def face_unit():
                length=_safe_positive(length2).sqrt()
                return tuple(v/length for v in normal)
            face_normals.append(static(("face-unit",ti),face_unit));raw.append(normal)
        vertex_pseudo=[]
        for vi,fan in enumerate(self.incident):
            def vertex_normal():
                accumulated=[zero,zero,zero]
                for ti in fan:
                    tri=self.triangles[ti];at=tri.index(vi)
                    u=_sub(vertices[tri[(at+1)%3]],vertices[vi]);v=_sub(vertices[tri[(at+2)%3]],vertices[vi])
                    cross=_cross(u,v)
                    angle=_dot(cross,cross).sqrt().atan2(_dot(u,v))
                    for k in range(3):
                        accumulated[k]=accumulated[k]+angle*face_normals[ti][k]
                return tuple(accumulated)
            vertex_pseudo.append(static(("vertex-pseudo",vi),vertex_normal))
        features=[]
        def feature(kind,fan,q,mask,pseudo):
            delta=_sub(p,q);distance=_dot(delta,delta).sqrt()
            fi=len(features)
            plen2=static(("feature-pseudo-length2",fi),lambda:_dot(pseudo,pseudo))
            def normalized_pseudo():
                plen=_safe_positive(plen2).sqrt()
                return tuple(pseudo[k]/plen for k in range(3))
            pseudo_unit=static(("feature-pseudo-unit",fi),normalized_pseudo)
            normal=tuple(T.where(distance>0.0,delta[k]/_safe_positive(distance),zero+pseudo_unit[k]) for k in range(3))
            orientation=_dot(delta,pseudo)
            valid=as_float(mask)
            if contact_filter is not None:
                valid=valid*as_float(contact_filter(q))
            features.append((kind,tuple(self.source_ids[i] for i in fan),q,valid,distance,normal,orientation,as_float(plen2>0.0)))
        # Strict face interiors leave all tessellation boundaries to one indexed
        # edge/vertex, even when materials change across the same plane.
        for ti,tri in enumerate(self.triangles):
            a,b,c=(vertices[i] for i in tri);normal=raw[ti]
            divisor=static(("face-divisor",ti),lambda:_safe_positive(_dot(normal,normal)))
            offset=_dot(_sub(p,a),normal)/divisor
            q=tuple(p[k]-offset*normal[k] for k in range(3))
            mask=zero==0.0
            for ei,(u,v) in enumerate(((a,b),(b,c),(c,a))):
                edge=static(("face-edge",ti,ei),lambda:_sub(v,u))
                mask=mask&(_dot(_cross(edge,_sub(q,u)),normal)>0.0)
            feature("face",(ti,),q,mask,face_normals[ti])
        for ei,((u,v),fan) in enumerate(self.edges):
            a,b=vertices[u],vertices[v]
            edge=static(("edge-vector",ei),lambda:_sub(b,a))
            divisor=static(("edge-divisor",ei),lambda:_safe_positive(_dot(edge,edge)))
            t=_dot(_sub(p,a),edge)/divisor
            q=tuple(a[k]+t*edge[k] for k in range(3));delta=_sub(p,q)
            mask=(t>0.0)&(t<1.0)
            for ti in fan:
                w=next(i for i in self.triangles[ti] if i not in (u,v))
                def cone():
                    toward=_sub(vertices[w],a)
                    along=_dot(toward,edge)/divisor
                    return tuple(toward[k]-along*edge[k] for k in range(3))
                inward=static(("edge-cone",ei,ti),cone)
                mask=mask&(_dot(delta,inward)<=0.0)
            pseudo=static(("edge-pseudo",ei),lambda:tuple(sum((face_normals[ti][k] for ti in fan),zero) for k in range(3)))
            feature("edge",fan,q,mask,pseudo)
        for vi,fan in enumerate(self.incident):
            q=vertices[vi];delta=_sub(p,q);mask=zero==0.0
            for neighbour in self.neighbours[vi]:
                edge=static(("vertex-edge",vi,neighbour),lambda:_sub(vertices[neighbour],q))
                mask=mask&(_dot(delta,edge)<=0.0)
            feature("vertex",fan,q,mask,vertex_pseudo[vi])
        best=zero;best_distance=zero;best_orientation=zero;has=zero
        for fi,(_,_,_,valid,distance,_,orientation,_) in enumerate(features):
            take=(valid>0.0)&((has<1.0)|(distance<best_distance))
            best=T.where(take,zero+float(fi),best)
            best_distance=T.where(take,distance,best_distance)
            best_orientation=T.where(take,orientation,best_orientation)
            has=has+as_float(take);has=has.clamp(0.0,1.0)
        inside=as_float(best_orientation<0.0) if self.closed else zero
        sign_ambiguous=as_float((best_distance>0.0)&(best_orientation==0.0)) if self.closed else zero
        output=[];singular=zero
        for fi,(kind,sources,q,valid,distance,normal,orientation,pseudo_valid) in enumerate(features):
            winner=as_float(best==float(fi))
            eligible=valid*((1.0-inside)*as_float(orientation>=0.0)+inside*winner) if self.closed else valid
            singular=singular+eligible*(1.0-pseudo_valid)
            normal=tuple(v*(1.0-2.0*inside) for v in normal)
            signed=distance*(1.0-2.0*inside)
            output.append(FeatureContact(fi,kind,sources,signed,normal,q,
                eligible*all_valid*pseudo_valid*(1.0-sign_ambiguous)))
        return tuple(output),{"geometry_valid":all_valid,"nearest_found":has,
                              "sign_ambiguous":sign_ambiguous,"singular_features":singular,
                              "inside":inside,"nearest_feature":best,"finite":finite,
                              "nearest_distance":best_distance*(1.0-2.0*inside)}

    def apply(self,p,radius,live,wall,vertex_values=None,*,static_geometry=None,contact_filter=None):
        """Use the incumbent fountain wall callback; never integrate or relocate."""
        contacts,diagnostics=self.query(p,vertex_values,static_geometry=static_geometry,contact_filter=contact_filter)
        for contact in contacts:
            wall(_cap(radius-contact.distance,6.0*radius),live*contact.valid,contact.normal)
        return diagnostics
