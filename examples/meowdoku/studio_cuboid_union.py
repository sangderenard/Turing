"""Exact native boundary arrangement for same-body axis-aligned cuboids.

Cuts are existing authored native-f32 face coordinates, not a sampled voxel
resolution. Every interval cell is wholly inside or outside each verified box.
Only faces between occupied and empty cells survive. This removes the table's
slab/leg internal interfaces without moving or approximating its exterior.
Rotated boxes and material boundaries needing further clipping are refused.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import hashlib

from meowdoku.studio_environment import _part


@dataclass(frozen=True)
class CuboidUnion:
    body_id: str
    vertices: tuple
    triangles: tuple
    material_ids: tuple
    triangle_ids: tuple
    triangle_provenance: tuple
    internal_interfaces: tuple
    source_part_ids: tuple
    max_coordinate_rounding: float
    exact_volume_ratio: tuple
    digest: str
    embedded_boundary_evidence: str
    proof: tuple

    def to_data(self):
        return {"schema":"turing.studio-cuboid-outer-union.v1","bodyId":self.body_id,
                "closed":True,"contactModel":"native-local-features-v1",
                "construction":"exact-existing-plane-arrangement","coordinateStorage":"float32",
                "vertices":[list(v) for v in self.vertices],"triangles":[list(t) for t in self.triangles],
                "materialIds":list(self.material_ids),"triangleIds":list(self.triangle_ids),
                "triangleProvenance":[list(s) for s in self.triangle_provenance],
                "provenanceRelation":"coplanar-original-source-face-support",
                "internalInterfaces":[dict(row) for row in self.internal_interfaces],
                "sourcePartIds":list(self.source_part_ids),"digest":self.digest,
                "maxCoordinateRounding":self.max_coordinate_rounding,
                "exactVolumeRatio":list(self.exact_volume_ratio),
                "embeddedBoundaryEvidence":self.embedded_boundary_evidence,"proof":dict(self.proof)}


def build_cuboid_union(mesh, parts, *, max_cells=8192):
    """Return one conforming exterior with explicit original-source accounting.

    parts uses StudioSourcePart's id/body_id/source_indices/closed fields (or
    dictionaries with those names). Each must describe an entire authored
    axis-aligned cuboid. This never infers object membership by proximity.
    Triangle provenance is supporting original side-face triangles, not a
    fabricated one-to-one mapping after exact tessellation subdivision.
    """
    parts=tuple(parts)
    if not parts or type(max_cells) is not int or max_cells<1:
        raise ValueError("a bounded nonempty authored cuboid group is required")
    field=lambda p,k:p[k] if isinstance(p,dict) else getattr(p,k)
    body_ids={field(p,"body_id") for p in parts}
    if len(body_ids)!=1 or any(not isinstance(v,str) or not v for v in body_ids):
        raise ValueError("cuboid union only accepts one explicit shared body identity")
    body_id=next(iter(body_ids))
    if len({field(p,"id") for p in parts})!=len(parts):
        raise ValueError("source part identities must be unique")
    boxes=[];all_sources=set();rounding=0.0
    for part in parts:
        if field(part,"closed") is not True:
            raise ValueError("cuboid union requires closed authored parts")
        ident=field(part,"id");indices=tuple(field(part,"source_indices"))
        if all_sources.intersection(mesh.source_ids[i] for i in indices):
            raise ValueError("source triangles cannot belong to multiple cuboid parts")
        admission=mesh.native_convex_part(indices,part_id=ident,body_id=body_id,closed=True)
        native=_part(admission["part"])
        rounding=max(rounding,admission["max_coordinate_rounding"])
        vertices=native.vertices
        low=tuple(min(p[k] for p in vertices) for k in range(3))
        high=tuple(max(p[k] for p in vertices) for k in range(3))
        if len(vertices)!=8 or set(vertices)!=set(product(*zip(low,high))) or len(native.triangles)!=12:
            raise ValueError(f"{ident}: union only supports exact axis-aligned eight-corner cuboids")
        sides={}
        for ti,triangle in enumerate(native.triangles):
            points=[vertices[i] for i in triangle]
            matches=[(axis,side) for axis in range(3) for side,bound in enumerate((low[axis],high[axis]))
                     if all(p[axis]==bound for p in points)]
            if len(matches)!=1:
                raise ValueError(f"{ident}: source face is not an axis-aligned cuboid side")
            sides.setdefault(matches[0],[]).append((mesh.source_ids[indices[ti]],native.materials[ti]))
        if len(sides)!=6 or any(len(s)!=2 for s in sides.values()):
            raise ValueError(f"{ident}: every cuboid side must have its two complete source triangles")
        if any(len({material for _,material in s})!=1 for s in sides.values()):
            raise ValueError(f"{ident}: side material boundary needs exact source-triangle clipping")
        boxes.append((ident,low,high,sides))
        all_sources.update(mesh.source_ids[i] for i in indices)
    cuts=tuple(tuple(sorted({bound[k] for _,low,high,_ in boxes for bound in (low,high)})) for k in range(3))
    shape=tuple(len(axis)-1 for axis in cuts)
    if shape[0]*shape[1]*shape[2]>max_cells:
        raise ValueError("authored plane arrangement exceeds declared cell capacity")
    occupied=set()
    for cell in product(*(range(n) for n in shape)):
        if any(all(low[k]<=cuts[k][cell[k]] and cuts[k][cell[k]+1]<=high[k] for k in range(3))
               for _,low,high,_ in boxes):
            occupied.add(cell)
    def support(axis,coordinate,cell,side=None):
        result=[]
        for ident,low,high,sides in boxes:
            if not all(low[k]<=cuts[k][cell[k]] and cuts[k][cell[k]+1]<=high[k] for k in range(3) if k!=axis):
                continue
            for candidate,bound in enumerate((low[axis],high[axis])):
                if coordinate==bound and (side is None or side==candidate):
                    result.extend((ident,source,material) for source,material in sides[axis,candidate])
        return tuple(result)
    vertices=[];vertex_ids={};triangles=[];materials=[];triangle_ids=[];provenance=[]
    internal=[];seen_internal=set();accounted=set()
    for cell in sorted(occupied):
        for axis in range(3):
            u,v=(axis+1)%3,(axis+2)%3
            for side in (0,1):
                neighbour=list(cell);neighbour[axis]+=1 if side else -1
                plane=cell[axis]+side;coordinate=cuts[axis][plane]
                ownership=support(axis,coordinate,cell)
                if tuple(neighbour) in occupied:
                    key=(axis,plane,cell[u],cell[v])
                    if ownership and key not in seen_internal:
                        seen_internal.add(key)
                        ids=tuple(sorted({source for _,source,_ in ownership}));accounted.update(ids)
                        internal.append((('axis',axis),('coordinate',coordinate),
                            ('u_interval',(cuts[u][cell[u]],cuts[u][cell[u]+1])),
                            ('v_interval',(cuts[v][cell[v]],cuts[v][cell[v]+1])),('source_ids',ids)))
                    continue
                ownership=support(axis,coordinate,cell,side)
                if not ownership:
                    raise ValueError("outer union face lacks original authored side support")
                face_materials={material for _,_,material in ownership}
                if len(face_materials)!=1:
                    raise ValueError("coincident exterior cuboid faces have conflicting materials")
                material=next(iter(face_materials))
                source_ids=tuple(sorted({source for _,source,_ in ownership}));accounted.update(source_ids)
                corners=[]
                for du,dv in ((0,0),(1,0),(1,1),(0,1)):
                    point=[None,None,None];point[axis]=coordinate
                    point[u]=cuts[u][cell[u]+du];point[v]=cuts[v][cell[v]+dv]
                    point=tuple(point)
                    if point not in vertex_ids:
                        vertex_ids[point]=len(vertices);vertices.append(point)
                    corners.append(vertex_ids[point])
                if not side:
                    corners=list(reversed(corners))
                a,b,c,d=corners
                for half,triangle in enumerate(((a,b,c),(a,c,d))):
                    triangles.append(triangle);materials.append(material);provenance.append(source_ids)
                    triangle_ids.append(f"{body_id}/outer/{axis}/{plane}/{cell[u]}/{cell[v]}/{side}/{half}")
    if accounted!=all_sources:
        raise ValueError("union arrangement failed to account for every original source triangle")
    volume=sum((Fraction(cuts[0][i+1])-Fraction(cuts[0][i]))*
               (Fraction(cuts[1][j+1])-Fraction(cuts[1][j]))*
               (Fraction(cuts[2][k+1])-Fraction(cuts[2][k])) for i,j,k in occupied)
    h=hashlib.sha256(repr((mesh.digest,body_id,vertices,triangles,materials,provenance)).encode()).hexdigest()
    evidence="turing.exact-cuboid-arrangement:"+h
    # Cell interiors are disjoint Cartesian intervals; retained quads separate
    # one occupied cell from empty space. Their interiors cannot intersect.
    # Reject disconnected/edge-only/vertex-only unions by checking the resulting
    # conforming boundary incidence and vertex fans, not by assuming every box
    # arrangement is an embedded manifold. No force query is needed here.
    from meowdoku.studio_mesh_manifold import NativeMeshBoundary
    NativeMeshBoundary(vertices,triangles,triangle_ids,closed=True,body_id=body_id,
                       embedded_boundary_evidence=evidence)
    overlapping=any(all(min(a[2][k],b[2][k])>max(a[1][k],b[1][k]) for k in range(3))
                    for i,a in enumerate(boxes) for b in boxes[i+1:])
    proof=(("cell_interiors","pairwise-disjoint-existing-plane-intervals"),
           ("exterior_rule","occupied-to-empty-face-only"),
           ("boundary_incidence","paired-edges-connected-vertex-fans"),
           ("source_accounting","all-source-triangles-exterior-support-or-internal-interface"),
           ("source_boxes_overlap",overlapping),("occupied_cells",len(occupied)),
           ("arrangement_cells",shape[0]*shape[1]*shape[2]))
    return CuboidUnion(body_id,tuple(vertices),tuple(triangles),tuple(materials),tuple(triangle_ids),
        tuple(provenance),tuple(internal),tuple(field(p,"id") for p in parts),rounding,
        (volume.numerator,volume.denominator),h,evidence,proof)
