"""Immutable whole-mesh spatial bake using Turing's existing Spectral BVH.

This is an exact triangle broad phase and an AbstractTensor geometry bridge,
not a voxel/SDF replacement, contact manifold, integrator or deployed product.
Every input face is retained, including nonconvex and disconnected geometry.
Coordinates must already be in one declared shared frame. There is no welding,
simplification, implicit material conversion, or silently truncated candidate
list. Native product traversal/sign/manifold integration is still required.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct

import numpy as np

from src.compiler.abstract_ui_bvh import build_packed_triangle_bvh
from src.common.tensors import AbstractTensor
from meowdoku.studio_contacts import closest_triangle


def _point(value, name):
    if len(value) != 3:
        raise ValueError(f"{name} needs three coordinates")
    point = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in point):
        raise ValueError(f"{name} coordinates must be finite")
    return point


@dataclass(frozen=True)
class MeshContactGeometry:
    source_index: int
    source_id: str
    material_id: int
    distance: float
    normal: tuple
    point: tuple
    valid: bool


class RegionAdmissionError(ValueError):
    """A bounded query envelope includes unresolved source geometry."""
    def __init__(self, report):
        self.report=report
        super().__init__("unresolved Studio geometry in coverage envelope: "+", ".join(report["unknown_source_ids"]))


@dataclass(frozen=True)
class StudioMeshBake:
    """Read-only snapshot; source revision stays separate from content digest."""
    triangles: tuple
    source_ids: tuple
    material_ids: tuple
    frame_id: str
    digest: str
    bvh: object

    @classmethod
    def from_faces(cls, faces, *, source_ids, frame_id, max_triangles=100000, leaf_size=4):
        """Bake exact shared Studio [a,b,c,material,...] faces, in source order.

        This does not read current UI state. The snapshot owner must supply the
        same face snapshot used by rendering, with stable per-triangle identity
        and the common coordinate-frame identity. Frame IDs are content names,
        not claims that this function has performed a coordinate transform.
        """
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("a shared coordinate frame identity is required")
        if type(max_triangles) is not int or max_triangles < 0:
            raise ValueError("max_triangles must be a nonnegative integer")
        if type(leaf_size) is not int or leaf_size < 1:
            raise ValueError("leaf_size must be a positive integer")
        triangles, materials = [], []
        for face in faces:
            if len(triangles) >= max_triangles:
                raise ValueError("whole mesh exceeds the declared bake triangle capacity")
            if len(face) < 4 or type(face[3]) is not int or face[3] < 0:
                raise ValueError("each shared face needs three vertices and a material ID")
            triangle = tuple(_point(p, "triangle vertex") for p in face[:3])
            # Geometry is kept even for degenerate source faces. closest_triangle
            # reports their validity explicitly; no source identity disappears.
            triangles.append(triangle)
            materials.append(face[3])
        identities = tuple(source_ids)
        if len(identities) != len(triangles) or any(not isinstance(s,str) or not s for s in identities):
            raise ValueError("one stable source identity is required per triangle")
        if len(set(identities)) != len(identities):
            raise ValueError("source triangle identities must be unique")
        h = hashlib.sha256()
        def text(value):
            raw = value.encode("utf-8")
            h.update(struct.pack("<Q",len(raw)))
            h.update(raw)
        text("turing.studio-whole-mesh-bvh.v1")
        text(frame_id)
        for tri, material, ident in zip(triangles,materials,identities):
            text(ident)
            text(str(material))
            h.update(struct.pack("<9d",*(v for point in tri for v in point)))
        return cls(tuple(triangles),identities,tuple(materials),frame_id,h.hexdigest(),
                   build_packed_triangle_bvh(triangles,leaf_size=leaf_size))

    def require_digest(self, digest):
        """Reject stale work before first-frame reveal or geometry publication."""
        if digest != self.digest:
            raise ValueError("Studio mesh changed since this bake snapshot")

    def to_data(self):
        """Serializable bake artifact; BVH permutation never renumbers sources."""
        return {"schema":"turing.studio-whole-mesh-bake.v1", "digest":self.digest,
                "frame_id":self.frame_id, "coordinate_storage":"float64",
                "triangles":[[list(p) for p in tri] for tri in self.triangles],
                "source_ids":list(self.source_ids),"material_ids":list(self.material_ids),
                "bvh":self.bvh.to_data(),
                "contract":{"geometry":"unmodified-source-triangles",
                            "geometry_displacement":0,"broad_phase":"conservative-aabb",
                            "numeric_contract":"float64-source; native conversion must be declared",
                            "closed_volume_sign":False,"contact_manifold":False}}

    def candidate_payload(self, point, radius, *, end=None, max_candidates=4096):
        """Native-adapter input, not a claim that the current product uploads it."""
        indices=self.candidates(point,radius,end=end,max_candidates=max_candidates)
        return {"schema":"turing.studio-mesh-candidates.v1","digest":self.digest,
                "frame_id":self.frame_id,"source_indices":list(indices),
                "source_ids":[self.source_ids[i] for i in indices],
                "material_ids":[self.material_ids[i] for i in indices],
                "triangles":[[list(p) for p in self.triangles[i]] for i in indices],
                "query":{"point":list(point),"radius":float(radius),
                         "end":list(point if end is None else end)},
                "complete":True,"manifold_resolved":False}

    def native_sheet_part(self, indices, *, part_id, body_id):
        """Admit an explicitly owned planar patch through the existing gate.

        The caller selects one authored body/surface, never the whole scene or
        arbitrary candidate set. Exact coordinate equality builds shared edge
        indices; no tolerance welding or coplanarity guess is used. Existing
        studio_environment._part checks native-f32 planarity, filled connected
        convex footprint, winding, edge ownership and nondegeneracy. Thus a
        table-top quad can join across its diagonal, while an L-shaped patch,
        folded group or closed solid is refused. This is a two-sided sheet,
        NOT whole-table volume containment. Return the unchanged source record
        and measured native-coordinate rounding, with triangle identity intact.
        """
        return self.native_convex_part(indices,part_id=part_id,body_id=body_id,closed=False)

    def native_convex_part(self, indices, *, part_id, body_id, closed=True):
        """Existing native admission for an explicit complete convex boundary.

        closed=True retains the incumbent solid inside/nearest-exit semantics.
        Exact source indexing is preserved. Native float32 rounding may destroy
        exact convexity or planarity on rotated geometry; the incumbent gate
        then refuses rather than silently fitting or repairing the boundary.
        """
        from meowdoku.studio_environment import _part
        if type(closed) is not bool:
            raise TypeError("closed must be an authored Boolean topology flag")
        indices=tuple(indices)
        if not indices or len(set(indices))!=len(indices) or any(type(i) is not int or i<0 or i>=len(self.triangles) for i in indices):
            raise ValueError("select distinct source triangle indices for one authored surface")
        vertices,lookup,triangles=[],{},[]
        for index in indices:
            triangle=[]
            for point in self.triangles[index]:
                if point not in lookup:
                    lookup[point]=len(vertices)
                    vertices.append(point)
                triangle.append(lookup[point])
            triangles.append(triangle)
        record={"id":part_id,"bodyId":body_id,"closed":closed,"convex":True,
                "vertices":[list(p) for p in vertices],"triangles":triangles,
                "materialIds":[self.material_ids[i] for i in indices],
                "sourceTriangleIndices":list(indices),
                "sourceTriangleIds":[self.source_ids[i] for i in indices],
                "meshDigest":self.digest,"frameId":self.frame_id}
        admitted=_part(record)
        error=max(abs(float(native)-original) for native_point,source_point in zip(admitted.vertices,vertices)
                  for native,original in zip(native_point,source_point))
        return {"part":record,"native_coordinate_storage":"float32",
                "max_coordinate_rounding":error,"signed_volume":closed}

    def admit_region(self, point, radius, *, admitted_parts=(), room=None,
                     room_source_indices=(), end=None, max_candidates=4096):
        """Require complete source coverage only within this query envelope.

        Parts must be admitted by the existing native gate and reproduce the
        claimed source triangles exactly. Room coverage requires each claimed
        source triangle to lie on one inward-wound bounded room plane. This
        certifies candidate accounting, NOT time-of-impact or future motion.
        Moving beyond the envelope requires another admission. Unknown IDs are
        returned in RegionAdmissionError.report, never silently discarded.
        """
        from meowdoku.studio_environment import _part
        candidates=self.candidates(point,radius,end=end,max_candidates=max_candidates)
        owners={}
        bodies=set()
        for admission in admitted_parts:
            record=admission["part"]
            self.require_digest(record.get("meshDigest"))
            if record.get("frameId")!=self.frame_id:
                raise ValueError("admitted part frame does not match the mesh bake")
            admitted=_part(record)
            if admitted.body_id in bodies:
                raise ValueError("compound part coverage requires resolved shared body ownership")
            bodies.add(admitted.body_id)
            indices=record.get("sourceTriangleIndices",())
            identities=record.get("sourceTriangleIds",())
            if len(indices)!=len(record["triangles"]) or len(identities)!=len(indices):
                raise ValueError("admitted part lacks its complete source triangle map")
            for j,index in enumerate(indices):
                if type(index) is not int or index<0 or index>=len(self.triangles):
                    raise ValueError("admitted part claims an invalid source triangle")
                triangle=tuple(tuple(record["vertices"][i]) for i in record["triangles"][j])
                if (triangle!=self.triangles[index] or identities[j]!=self.source_ids[index]
                        or record["materialIds"][j]!=self.material_ids[index]):
                    raise ValueError("admitted part differs from claimed source geometry or identity")
                if index in owners:
                    raise ValueError("source triangle has more than one contact owner")
                owners[index]="part:"+admitted.id
        room_indices=tuple(room_source_indices)
        if room_indices:
            if room is None:
                raise ValueError("room source coverage requires exact shared bounds")
            low,high=_point(room["low"],"room low"),_point(room["high"],"room high")
            if any(a>=b for a,b in zip(low,high)):
                raise ValueError("room bounds must strictly increase")
            for index in room_indices:
                if type(index) is not int or index<0 or index>=len(self.triangles) or index in owners:
                    raise ValueError("invalid or multiply-owned room source triangle")
                tri=self.triangles[index]
                if any(not low[k]<=p[k]<=high[k] for p in tri for k in range(3)):
                    raise ValueError("room source triangle exceeds shared room bounds")
                normal=np.cross(np.subtract(tri[1],tri[0]),np.subtract(tri[2],tri[0]))
                matches=[(axis,side) for axis in range(3) for side,bound in enumerate((low[axis],high[axis]))
                         if all(p[axis]==bound for p in tri) and normal[axis]*(1 if side==0 else -1)>0]
                if len(matches)!=1:
                    raise ValueError("room source triangle is not an inward shared-boundary face")
                owners[index]=f"room:{matches[0][0]}:{matches[0][1]}"
        unknown=[i for i in candidates if i not in owners]
        unaccounted=[i for i in range(len(self.triangles)) if i not in owners]
        report={"schema":"turing.studio-mesh-region-admission.v1","digest":self.digest,"frame_id":self.frame_id,
                "coverage":{"point":list(point),"radius":float(radius),"end":list(point if end is None else end),
                            "kind":"radius-expanded-segment-aabb"},
                "candidate_source_indices":list(candidates),
                "owners":{self.source_ids[i]:owners[i] for i in candidates if i in owners},
                "unknown_source_ids":[self.source_ids[i] for i in unknown],
                "unaccounted_source_ids":[self.source_ids[i] for i in unaccounted],
                "inside_outside_certified":not unaccounted,
                "admitted":not unknown,"global_scene_supported":False}
        if unknown:
            raise RegionAdmissionError(report)
        return report

    def candidates(self, point, radius, *, end=None, max_candidates=4096):
        """Conservative sphere or swept-sphere broad phase, in source order.

        Traverses the incumbent packed BVH with a radius-expanded segment AABB.
        False positives are intentional; no contacting triangle may be omitted.
        This is NOT a swept time-of-impact test. Capacity overflow is explicit,
        never a partial list that lets particles pass through missing geometry.
        """
        point = _point(point,"query point")
        end = point if end is None else _point(end,"query endpoint")
        radius = float(radius)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("query radius must be finite and nonnegative")
        if type(max_candidates) is not int or max_candidates < 0:
            raise ValueError("max_candidates must be a nonnegative integer")
        low = tuple(min(a,b)-radius for a,b in zip(point,end))
        high = tuple(max(a,b)+radius for a,b in zip(point,end))
        if not all(math.isfinite(v) for v in (*low,*high)):
            raise ValueError("query bounds exceed finite coordinate range")
        stack, found = ([0] if self.bvh.nodes else []), []
        while stack:
            node = self.bvh.nodes[stack.pop()]
            if any(node[k]>high[k] or node[4+k]<low[k] for k in range(3)):
                continue
            count = int(node[9])
            if count:
                start = int(node[8])
                found.extend(self.bvh.triangle_order[start:start+count])
                if len(found)>max_candidates:
                    raise ValueError("mesh broad phase exceeds declared candidate capacity")
            else:
                stack.extend((int(node[7]),int(node[3])))
        return tuple(sorted(found))

    def contact_geometry(self, point, radius, *, max_candidates=4096):
        """CPU geometry proof on the same AbstractTensor primitive as the law.

        Returns all narrow-phase touching triangles with source identities,
        including invalid degenerate triangles (valid=False). This is NOT a
        force sum: seams, closed-solid sign and contact manifold ownership must
        be resolved before a caller can use these records as particle contacts.
        """
        indices = self.candidates(point,radius,max_candidates=max_candidates)
        if not indices:
            return ()
        vertices = np.asarray([self.triangles[i] for i in indices],dtype=np.float64)
        p = tuple(AbstractTensor.get_tensor(np.full(len(indices),v,dtype=np.float64)) for v in point)
        tri = tuple(tuple(AbstractTensor.get_tensor(vertices[:,i,k]) for k in range(3)) for i in range(3))
        distance,normal,closest,valid = closest_triangle(p,*tri)
        ds = np.asarray(distance.data).reshape(-1)
        ns = np.array([np.asarray(v.data).reshape(-1) for v in normal]).T
        qs = np.array([np.asarray(v.data).reshape(-1) for v in closest]).T
        vs = np.asarray(valid.data).reshape(-1)
        return tuple(MeshContactGeometry(index,self.source_ids[index],self.material_ids[index],float(ds[j]),
            tuple(ns[j]),tuple(qs[j]),bool(vs[j])) for j,index in enumerate(indices) if ds[j]<=radius)
