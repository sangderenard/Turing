"""Exact authored feature boundaries in the incumbent fountain environment ABI.

This is an opt-in native-law environment, never a default-law change. The
complete mesh snapshot is admitted before compilation. Vertex inputs remain
named ABI fields, but changing this validated snapshot requires rebaking.
Geometry/sign/feature failures enter the existing n_out diagnostic rather
than silently dropping a contact and declaring a hidden first frame ready.
"""
from meowdoku import fountain3d as F
from meowdoku.studio_contacts import room_planes
from meowdoku.studio_environment import StudioEnvironment,Part,ROOM_NAMES,SCHEMA,_points
from meowdoku.studio_mesh_manifold import NativeMeshBoundary
import copy

CONTACT_MODEL="native-local-features-v1"
BODY_BOUNDARY_PAIR_BUDGET=50000


def disjoint_body_boundaries(parts,boundaries):
    """Prove separate boundaries of one static body cannot double a surface.

    Merely having different part IDs is insufficient. Boundaries may neither
    cross nor touch; closed volumes may not contain another component. Exact
    alternative ray directions change only the proof query, never a vertex or
    seed. Ambiguity at every bounded direction refuses the compound.
    """
    from meowdoku.studio_mesh_embedding import verify_native_embedding
    from meowdoku.studio_mesh_bake import StudioMeshBake
    from meowdoku.studio_mesh_certificate import component_seed_certificate
    groups={};proofs=[]
    for i,part in enumerate(parts):groups.setdefault(part.body_id,[]).append(i)
    for body,indices in groups.items():
        if len(indices)<2:continue
        vertices=[];triangles=[];ids=[]
        for i in indices:
            boundary=boundaries[i];offset=len(vertices)
            vertices.extend(boundary.vertices)
            triangles.extend(tuple(v+offset for v in tri) for tri in boundary.triangles)
            ids.extend(f"{parts[i].id}/{source}" for source in boundary.source_ids)
        # The measured complete 1,208-triangle underlight needs 46,544 BVH
        # leaf pairs (13,996 exact pairs). This explicit bounded allowance
        # executes every pair; no intersection is skipped to fit a budget.
        proof=verify_native_embedding(vertices,triangles,ids,max_pairs=BODY_BOUNDARY_PAIR_BUDGET)
        containment=[]
        for i in indices:
            boundary=boundaries[i]
            if not boundary.closed:continue
            faces=[[*[boundary.vertices[v] for v in tri],parts[i].materials[t]] for t,tri in enumerate(boundary.triangles)]
            bake=StudioMeshBake.from_faces(faces,source_ids=boundary.source_ids,frame_id="native-compound:"+proof.digest)
            for j in indices:
                if i==j:continue
                certificate=None
                for direction in ((1,0,0),(0,1,0),(0,0,1),(1,2,3),(2,3,5)):
                    result=component_seed_certificate(bake,tuple(range(len(faces))),boundaries[j].vertices[0],
                        component_id=parts[i].id,closed=True,embedded_boundary_evidence=boundary.embedded_boundary_evidence,
                        direction=direction)
                    if result["certified"]:certificate=result;break
                if certificate is None or certificate["position"]!="outside":
                    raise ValueError(f"{body}: separate boundary {parts[j].id} is inside or uncertified against {parts[i].id}; exact union required")
                containment.append({"container":parts[i].id,"component":parts[j].id,
                    "position":"outside","direction":certificate["direction"]})
        proofs.append({"bodyId":body,"partIds":[parts[i].id for i in indices],
                       "embedding":proof.to_data(),"containment":containment})
    return proofs


class StudioMeshEnvironment(StudioEnvironment):
    def __init__(self,packet):
        if packet.get("schema")!=SCHEMA or packet.get("contact_model")!=CONTACT_MODEL:
            raise ValueError("explicit native feature-mesh scene contract required")
        bounds=_points([packet["room"]["low"],packet["room"]["high"]],"room bounds")
        if any(a>=b for a,b in zip(*bounds)):
            raise ValueError("room bounds must strictly increase")
        self.bounds=tuple(map(tuple,bounds))
        parts=[];boundaries=[]
        for record in packet.get("parts",()):
            ident=record.get("id");body=record.get("bodyId",ident)
            if not isinstance(ident,str) or not ident:
                raise ValueError("a feature boundary needs a stable part identity")
            boundary=NativeMeshBoundary(record.get("vertices"),record.get("triangles"),record.get("sourceTriangleIds",()),
                closed=record.get("closed"),body_id=body,
                embedded_boundary_evidence=record.get("embeddedBoundaryEvidence"))
            materials=record.get("materialIds")
            if not isinstance(materials,(list,tuple)) or len(materials)!=len(boundary.triangles) or any(type(m) is not int or m<0 for m in materials):
                raise ValueError("every feature triangle must retain its source material identity")
            parts.append(Part(ident,body,boundary.closed,boundary.vertices,boundary.triangles,tuple(materials)))
            boundaries.append(boundary)
        if len({p.id for p in parts})!=len(parts):
            raise ValueError("native boundary part IDs must be unique")
        compound_proofs=disjoint_body_boundaries(parts,boundaries)
        self.parts=tuple(parts);self.boundaries=tuple(boundaries)
        self.input_names=ROOM_NAMES+tuple(self.vertex_name(i,j,k) for i,p in enumerate(parts) for j in range(len(p.vertices)) for k in range(3))
        self.geometry_contract={"contact_model":CONTACT_MODEL,"snapshot_vertices":"immutable-native-f32",
            "bodyBoundaryPairBudget":BODY_BOUNDARY_PAIR_BUDGET,
            "disjointBodyBoundaries":compound_proofs,
            "boundaries":[{"id":p.id,"bodyId":p.body_id,"digest":b.digest,
                           "sourceTriangleIds":list(b.source_ids),"embeddedBoundaryEvidence":b.embedded_boundary_evidence,
                           "maxCoordinateRounding":b.max_coordinate_rounding,
                           "sourceProvenance":copy.deepcopy(record.get("sourceProvenance"))}
                          for p,b,record in zip(self.parts,self.boundaries,packet.get("parts",()))]}

    def _vertices(self,i,a):
        return tuple(tuple(a[self.vertex_name(i,j,k)] for k in range(3)) for j in range(len(self.parts[i].vertices)))

    def contacts(self,p,radius,live,a,wall):
        low,high=tuple(a[k] for k in ROOM_NAMES[:3]),tuple(a[k] for k in ROOM_NAMES[3:])
        for distance,normal in room_planes(p,low,high):
            wall(F._cap(radius-distance,6.0*radius),live,normal)
        for i,boundary in enumerate(self.boundaries):
            boundary.apply(p,radius,live,wall,self._vertices(i,a))

    def contains(self,p,radius,a):
        inside=p[0]*0.0+1.0
        for k in range(3):
            inside=inside*F.as_float(p[k]>=a[ROOM_NAMES[k]]-radius)*F.as_float(p[k]<=a[ROOM_NAMES[k+3]]+radius)
        # Admission is for these exact rounded vertices. Runtime setters may
        # not alter a validated surface while an old product is still in use.
        for i,(part,boundary) in enumerate(zip(self.parts,self.boundaries)):
            for j,vertex in enumerate(part.vertices):
                for k,value in enumerate(vertex):
                    inside=inside*F.as_float(a[self.vertex_name(i,j,k)]==float(value))
            _,diag=boundary.query(p,self._vertices(i,a))
            inside=inside*diag["finite"]*diag["geometry_valid"]*diag["nearest_found"]
            inside=inside*F.as_float(diag["sign_ambiguous"]==0.0)*F.as_float(diag["singular_features"]==0.0)
            inside=inside*F.as_float(diag["nearest_distance"]>=-radius)
        return inside
