"""Bind the actual displayed Studio capture to its existing native mesh bake.

No UI reads, Exposure reconstruction, proxy geometry, background job, or GPU
launch lives here. This is the request-side ownership/admission boundary for
an eventual hidden product. Unsupported contact ownership is an explicit
error with source parts; no partial scene is returned as a usable product.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from meowdoku.studio_mesh_bake import StudioMeshBake
from meowdoku.studio_environment import StudioEnvironment, SCHEMA


def _vector(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError(f"{name} requires {count} finite numbers")
    return tuple(float(v) for v in value)


class StudioSceneAdmissionError(ValueError):
    def __init__(self, blockers):
        self.blockers = tuple(blockers)
        super().__init__("Studio native contact admission is incomplete: " + "; ".join(
            f"{item['part_id']}: {item['reason']}" for item in blockers))


@dataclass(frozen=True)
class StudioSourcePart:
    id: str
    body_id: str
    construction: str
    closed: bool
    convex: bool
    source_indices: tuple


@dataclass(frozen=True)
class StudioVisibleBake:
    revision: int
    capture_json: bytes
    capture_digest: str
    unit_m: float
    origin_m: tuple
    ray_origin_m: tuple
    room: tuple
    radius_m: float
    subject_center_m: tuple
    parts: tuple
    mesh: StudioMeshBake

    @classmethod
    def from_capture(cls, capture, *, max_triangles=100000):
        # Canonical detached ownership. Later caller edits cannot mutate a bake.
        raw = json.dumps(capture, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        value = json.loads(raw)
        state = value.get("state", {})
        revision = value.get("revision")
        if type(revision) is not int or revision < 0 or state.get("schema") != "turing.studio-visible-scene.v1" or state.get("channel") != "studio":
            raise ValueError("a versioned displayed Studio capture is required, not Exposure")
        unit = state.get("frame", {}).get("unitM")
        if type(unit) not in (int, float) or not math.isfinite(unit) or unit <= 0:
            raise ValueError("the displayed Studio metre reference is missing")
        ray_origin = _vector(state["frame"].get("rayOriginM"), 3, "ray origin")
        bounds = state.get("room")
        if not isinstance(bounds, dict):
            raise ValueError("the rendered containing room is absent")
        low = _vector([bounds.get(k) for k in ("leftX", "floorY", "backZ")], 3, "room low")
        high = _vector([bounds.get(k) for k in ("rightX", "ceilingY", "frontZ")], 3, "room high")
        if any(a >= b for a, b in zip(low, high)):
            raise ValueError("invalid displayed room bounds")
        origin = (0., (1-low[1])*unit, 0.)
        native = lambda p: tuple(v*unit+origin[k] for k, v in enumerate(p))
        subject = state.get("subject", {})
        radius = subject.get("radius")
        if type(radius) not in (int, float) or not .0001 <= radius*unit <= .2:
            raise ValueError("displayed physical marble radius must be 0.1..200 mm")
        if state.get("authoring", {}).get("object", {}).get("design", {}).get("base-shape") != "marble":
            raise ValueError("the native spherical law requires an authored marble")
        center = native(_vector(subject.get("position"), 3, "displayed subject position"))
        model = _vector(subject.get("model"), 16, "displayed subject model")
        if tuple(model[12:15]) != tuple(subject["position"]):
            raise ValueError("displayed subject centre differs from its actual model")
        columns=[tuple(model[4*j+k]/radius for k in range(3)) for j in range(3)]
        dot=lambda a,b:sum(x*y for x,y in zip(a,b))
        a,b,c=columns
        cross=(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
        tolerance=32*2**-23
        if (model[3],model[7],model[11],model[15])!=(0,0,0,1) or dot(cross,c)<=0 or any(
                abs(dot(x,y)-(1 if i==j else 0))>tolerance for i,x in enumerate(columns) for j,y in enumerate(columns)):
            raise ValueError("displayed subject model is not a proper uniform rigid transform")
        ray = state.get("ray", {})
        pbr = ray.get("pbr")
        if not isinstance(pbr, list) or not pbr or len(pbr) % 16 or any(type(v) not in (int,float) or not math.isfinite(v) for v in pbr):
            raise ValueError("complete finite Studio material rows are required")
        faces, sources, parts, seen = [], [], [], set()
        expected_vertices, expected_normals, expected_ids, expected_sources = [], [], [], []
        for part in state.get("parts", ()):
            ident, body = part.get("id"), part.get("bodyId", part.get("id"))
            if not isinstance(ident,str) or not ident or ident in seen or not isinstance(body,str) or not body:
                raise ValueError("authored parts need distinct stable identities")
            seen.add(ident)
            if type(part.get("closed")) is not bool or type(part.get("convex")) is not bool:
                raise ValueError(f"{ident}: authored topology metadata is missing")
            indices = []
            if not part.get("faces"):
                raise ValueError(f"{ident}: empty source component")
            for ti, face in enumerate(part["faces"]):
                if len(faces) >= max_triangles:
                    raise ValueError("complete scene exceeds declared triangle capacity")
                if len(face) < 5 or type(face[3]) is not int or not 0 <= face[3] < len(pbr)//16:
                    raise ValueError(f"{ident}: missing triangle material or normal identity")
                points = tuple(_vector(p,3,"source vertex") for p in face[:3])
                if len(face[4]) != 3:
                    raise ValueError("every rendered triangle needs three authored normals")
                for normal in face[4]:
                    expected_normals.extend(_vector(normal,3,"source normal"))
                indices.append(len(faces));faces.append([*(native(p) for p in points),face[3]])
                sources.append(f"{ident}:{ti}")
                expected_vertices.extend(v*unit+ray_origin[k] for p in points for k,v in enumerate(p))
                expected_ids.append(face[3]);expected_sources.append({"partId":ident,"bodyId":body,"triangle":ti})
            if part.get("firstTriangle") != indices[0] or part.get("triangleCount") != len(indices):
                raise ValueError(f"{ident}: source triangle ranges differ from displayed order")
            parts.append(StudioSourcePart(ident,body,part.get("construction",""),part["closed"],part["convex"],tuple(indices)))
        if ray.get("vertices") != expected_vertices or ray.get("normals") != expected_normals or ray.get("material_ids") != expected_ids or state.get("triangleSources") != expected_sources:
            raise ValueError("ray geometry/material/source identities differ from the captured Studio faces")
        subject_faces=subject.get("faces")
        if not isinstance(subject_faces,list) or not subject_faces or len(subject_faces)+len(faces)>max_triangles:
            raise ValueError("complete subject geometry is missing or exceeds scene capacity")
        subject_vertices,subject_ids=[],[]
        for face in subject_faces:
            if len(face)<5 or type(face[3]) is not int or not 0<=face[3]<len(pbr)//16 or len(face[4])!=3:
                raise ValueError("complete subject triangle material/normal identity is required")
            for point in face[:3]:
                x,y,z=_vector(point,3,"subject vertex")
                world=(model[0]*x+model[4]*y+model[8]*z+model[12],
                       model[1]*x+model[5]*y+model[9]*z+model[13],
                       model[2]*x+model[6]*y+model[10]*z+model[14])
                subject_vertices.extend(v*unit+ray_origin[k] for k,v in enumerate(world))
            for normal in face[4]:
                _vector(normal,3,"subject normal")
            subject_ids.append(face[3])
        if subject.get("ray",{}).get("vertices")!=subject_vertices or subject.get("ray",{}).get("material_ids")!=subject_ids:
            raise ValueError("subject ray geometry differs from the displayed source mesh and model")
        frame_data = {"unitM":unit,"originM":origin}
        frame_id = "studio-native:" + hashlib.sha256(json.dumps(frame_data,sort_keys=True).encode()).hexdigest()
        mesh = StudioMeshBake.from_faces(faces,source_ids=sources,frame_id=frame_id,max_triangles=max_triangles)
        return cls(revision,raw,hashlib.sha256(raw).hexdigest(),unit,origin,ray_origin,
                   (native(low),native(high)),radius*unit,center,tuple(parts),mesh)

    def convex_packet(self):
        """Real existing native product input; refuses any omitted component.

        Feature-manifold and authored compound-union admission can extend this
        boundary later. A BVH bake alone never implies physics is ready.
        """
        admissions, blockers, bodies = [], [], {}
        room_parts = []
        for part in self.parts:
            if part.construction == "inward-room-plane":
                room_parts.append(part);continue
            bodies.setdefault(part.body_id,[]).append(part.id)
            try:
                if not part.convex:
                    raise ValueError("requires exact native feature-manifold admission")
                admission = self.mesh.native_convex_part(part.source_indices,part_id=part.id,body_id=part.body_id,closed=part.closed)
                admissions.append(admission["part"])
            except ValueError as error:
                blockers.append({"part_id":part.id,"source_ids":[self.mesh.source_ids[i] for i in part.source_indices],"reason":str(error)})
        for body, ids in bodies.items():
            if len(ids)>1:
                blockers.append({"part_id":body,"source_parts":ids,"reason":"authored compound needs exterior contact ownership"})
        try:
            self._verify_room(room_parts)
        except ValueError as error:
            blockers.append({"part_id":"room","reason":str(error)})
        if blockers:
            raise StudioSceneAdmissionError(blockers)
        packet={"schema":SCHEMA,"frame":{"unitM":self.unit_m,"originM":list(self.origin_m)},
                "room":{"low":list(self.room[0]),"high":list(self.room[1])},"parts":admissions}
        StudioEnvironment(packet)
        return packet

    def feature_packet(self):
        """Admit complete current-source bodies through native local features.

        Verified axis-aligned furniture uses its exact exterior union. Other
        boundaries undergo an independent exact self-intersection proof on
        their actual native-f32 coordinates, then manifold admission. Open
        connected sheets use their actual indexed feature boundary. All
        blockers are collected and no partial usable product is returned.
        """
        from meowdoku.studio_cuboid_union import build_cuboid_union
        from meowdoku.studio_mesh_environment import StudioMeshEnvironment,CONTACT_MODEL
        from meowdoku.studio_mesh_manifold import NativeMeshBoundary
        from meowdoku.studio_mesh_embedding import verify_native_embedding
        import numpy as np
        groups={};room_parts=[];records=[];blockers=[]
        source_records={record["id"]:record for record in json.loads(self.capture_json)["state"]["parts"]}
        for part in self.parts:
            if part.construction=="inward-room-plane":room_parts.append(part)
            else:groups.setdefault(part.body_id,[]).append(part)
        try:self._verify_room(room_parts)
        except ValueError as error:blockers.append({"part_id":"room","reason":str(error)})
        for body,parts in groups.items():
            try:
                if len(parts)==1 and parts[0].construction=="fixture-housing-emitter-modifier":
                    from meowdoku.studio_fixture_boundaries import prepare_fixture_boundaries
                    fixture=prepare_fixture_boundaries(self.mesh,parts[0],source_records[parts[0].id])
                    # The adapter preserves component ownership; the environment
                    # independently proves the complete shared-body relationship.
                    StudioMeshEnvironment({"schema":SCHEMA,"contact_model":CONTACT_MODEL,
                        "room":{"low":self.room[0],"high":self.room[1]},"parts":fixture})
                    records.extend(fixture)
                    continue
                if all(p.construction=="cuboid" for p in parts):
                    union=build_cuboid_union(self.mesh,parts)
                    record={"id":body,"bodyId":body,"closed":True,"convex":False,
                        "vertices":[list(v) for v in union.vertices],"triangles":[list(t) for t in union.triangles],
                        "materialIds":list(union.material_ids),"sourceTriangleIds":list(union.triangle_ids),
                        "embeddedBoundaryEvidence":union.embedded_boundary_evidence,
                        "sourceProvenance":union.to_data()}
                else:
                    if len({p.closed for p in parts})!=1:
                        raise ValueError("mixed closed/open components need explicit contact ownership")
                    closed=parts[0].closed
                    indices=tuple(i for part in parts for i in part.source_indices)
                    vertices=[];lookup={};triangles=[];rounding=0.0
                    for index in indices:
                        tri=[]
                        for point in self.mesh.triangles[index]:
                            # Identify equal native coordinates exactly, not a
                            # tolerance weld. The source bake remains unchanged.
                            rounded=tuple(float(np.float32(v)) for v in point)
                            rounding=max(rounding,*(abs(v-w) for v,w in zip(point,rounded)))
                            if rounded not in lookup:lookup[rounded]=len(vertices);vertices.append(rounded)
                            tri.append(lookup[rounded])
                        triangles.append(tri)
                    source_ids=[self.mesh.source_ids[i] for i in indices]
                    proof=verify_native_embedding(vertices,triangles,source_ids)
                    record={"id":body,"bodyId":body,"closed":closed,"convex":all(p.convex for p in parts),
                        "vertices":[list(v) for v in vertices],"triangles":triangles,
                        "materialIds":[self.mesh.material_ids[i] for i in indices],
                        "sourceTriangleIds":source_ids,"embeddedBoundaryEvidence":proof.evidence_id,
                        "sourceProvenance":{"sourcePartIds":[p.id for p in parts],"meshDigest":self.mesh.digest,
                            "sourceTriangleIndices":list(indices),"relation":"unchanged-source-triangles-at-native-f32",
                            "maxCoordinateRounding":rounding,"embeddingProof":proof.to_data()}}
                # Admission now, before the expensive compiler or device work.
                NativeMeshBoundary(record["vertices"],record["triangles"],record["sourceTriangleIds"],
                    closed=record["closed"],body_id=body,embedded_boundary_evidence=record["embeddedBoundaryEvidence"])
                records.append(record)
            except ValueError as error:
                blockers.append({"part_id":body,"source_parts":[p.id for p in parts],
                    "source_ids":[self.mesh.source_ids[i] for p in parts for i in p.source_indices],"reason":str(error)})
        if blockers:raise StudioSceneAdmissionError(blockers)
        packet={"schema":SCHEMA,"contact_model":CONTACT_MODEL,
            "frame":{"unitM":self.unit_m,"originM":list(self.origin_m)},
            "room":{"low":list(self.room[0]),"high":list(self.room[1])},"parts":records,
            "sourceCaptureDigest":self.capture_digest,"sourceMeshDigest":self.mesh.digest,"sourceRevision":self.revision}
        StudioMeshEnvironment(packet)
        return packet

    def _verify_room(self, parts):
        # Validate the complete actual tiling before replacing it with the
        # incumbent six inward room planes. Exact shared edges and positive
        # projected area rule out omitted/duplicated cells or hidden openings.
        low,high=self.room;area={};edges={}
        for part in parts:
            for index in part.source_indices:
                tri=self.mesh.triangles[index]
                if any(not low[k]<=p[k]<=high[k] for p in tri for k in range(3)):
                    raise ValueError("room source exceeds declared bounds")
                matches=[(k,s) for k in range(3) for s,b in enumerate((low[k],high[k])) if all(p[k]==b for p in tri)]
                if len(matches)!=1:
                    raise ValueError("room source is not a declared boundary plane")
                k,side=matches[0];u,v=(k+1)%3,(k+2)%3
                a,b,c=[tuple(Fraction(x) for x in p) for p in tri]
                cross=(b[u]-a[u])*(c[v]-a[v])-(b[v]-a[v])*(c[u]-a[u])
                if cross*(1 if side==0 else -1)<=0:
                    raise ValueError("room source is not inward wound")
                area[k,side]=area.get((k,side),Fraction(0))+abs(cross)
                for x,y in zip(tri,(*tri[1:],tri[0])):
                    if (x,y) in edges:
                        raise ValueError("duplicate room triangle boundary")
                    edges[x,y]=True
        if not edges or any((y,x) not in edges for x,y in edges):
            raise ValueError("the rendered room has a missing or unmatched boundary")
        for k in range(3):
            u,v=(k+1)%3,(k+2)%3
            expected=2*(Fraction(high[u])-Fraction(low[u]))*(Fraction(high[v])-Fraction(low[v]))
            if any(area.get((k,s))!=expected for s in (0,1)):
                raise ValueError("the rendered room does not cover every declared wall exactly once")


def build_visible_scene(capture,root,*,wasm=False):
    """Explicit isolated capture -> whole bake -> admitted native product.

    This does not execute physics, install renderers, publish files, or report
    a ready scene. The caller owns cancellation/job scheduling and must still
    warm the actual product and stage both readers. Each complete capture has
    its own directory, so a new frame/radius/material cannot overwrite a live
    manifest with the same reusable kernel topology. Existing directories are
    not overwritten, retried, or mistaken for a successful build.
    """
    from meowdoku.studio_physics_build import build
    snapshot=StudioVisibleBake.from_capture(capture)
    packet=snapshot.feature_packet()  # complete admission before compilation
    directory=Path(root)/snapshot.capture_digest
    directory.mkdir(parents=True,exist_ok=False)
    try:
        manifest=build(packet,directory,radius=snapshot.radius_m,pad=2,active=1,wasm=wasm)
        (directory/"scene.capture.json").write_bytes(snapshot.capture_json)
        (directory/"scene.mesh.json").write_text(json.dumps(snapshot.mesh.to_data(),allow_nan=False),encoding="utf-8")
        (directory/"scene.rigid.json").write_text(json.dumps(packet,allow_nan=False),encoding="utf-8")
        record=manifest["sizes"]["studio"]
        receipt={"schema":"turing.studio-compiled-candidate.v1","stage":"compiled-not-executed",
            "sourceRevision":snapshot.revision,"captureDigest":snapshot.capture_digest,"meshDigest":snapshot.mesh.digest,
            "manifest":f"studio_physics_{record['source_digest']}.manifest.json",
            "rigidScene":"scene.rigid.json","capture":"scene.capture.json","wholeMesh":"scene.mesh.json",
            "nativeStepRequired":True,"rendererStagingRequired":True,"revisionCheckRequired":True}
        # These are the actual persisted bytes, not a second JSON canonicalizer
        # in the browser. A stale HTTP response or mutated local artifact must
        # not pair one capture with another product during hidden preparation.
        names=[receipt[key] for key in ("manifest","rigidScene","capture","wholeMesh")]+[record["wgsl"]]
        if wasm:names.extend([record["wasm"],record["wasm_abi"]])
        receipt["files"]={name:{"sha256":hashlib.sha256((directory/name).read_bytes()).hexdigest(),
                                "bytes":(directory/name).stat().st_size} for name in names}
        # The last-written completion receipt is never an activation signal.
        (directory/"compiled.json").write_text(json.dumps(receipt,allow_nan=False),encoding="utf-8")
        return {"directory":directory,"receipt":receipt,"manifest":manifest,"rigid_scene":packet}
    except Exception as error:
        (directory/"failed.json").write_text(json.dumps({"error":str(error),"stage":"compile-failed"}),encoding="utf-8")
        raise
