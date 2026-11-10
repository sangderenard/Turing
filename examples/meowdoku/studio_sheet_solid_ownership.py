"""Source-bound ownership evidence for a two-sided sheet and cuboid solid.

This certifies geometry, not an activated runtime or a new force law. For a
point outside V, a segment to a sheet point inside/on V meets the solid boundary
no farther away. Every boundary introduced by clipping S outside V lies on the
solid boundary. Thus S outside V plus the existing solid surface retains
nearest-contact coverage of S union V without generating clipped render faces.
The existing multi-feature force sum and diagnostics require runtime tests.
"""
from dataclasses import dataclass
import hashlib
import json

from meowdoku.studio_cuboid_union import build_cuboid_union
from meowdoku.studio_open_sheet import orient_open_sheet
from meowdoku.studio_mesh_bake import StudioMeshBake
from meowdoku.studio_mesh_manifold import NativeMeshBoundary
from meowdoku.studio_mesh_embedding import verify_native_embedding


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


@dataclass(frozen=True)
class SheetSolidOwnership:
    body_id: str
    sheet_id: str
    solid_id: str
    source_mesh_digest: str
    sheet_boundary_digest: str
    union_digest: str
    boxes: tuple
    digest: str

    def contains(self,point):
        """Diagnostic exact inclusive source-volume predicate, not union AABB."""
        if len(point)!=3:raise ValueError('point needs XYZ')
        return any(all(low[k]<=point[k]<=high[k] for k in range(3)) for _,low,high,_ in self.boxes)

    def to_data(self):
        return {'schema':'turing.native-sheet-solid-ownership.v1','bodyId':self.body_id,
            'sheetPartId':self.sheet_id,'solidPartId':self.solid_id,'sourceMeshDigest':self.source_mesh_digest,
            'sheetBoundaryDigest':self.sheet_boundary_digest,'solidUnionDigest':self.union_digest,
            'sourceBoxes':[{'sourcePartId':name,'low':list(low),'high':list(high),'sourceTriangleIds':list(ids)}
                           for name,low,high,ids in self.boxes],
            'pointRule':'suppress-sheet-feature-closest-point-inside-or-on-any-source-box',
            'solidRule':'retain-incumbent-closed-boundary-contacts',
            'coverageProof':'clipped-sheet-boundary-subset-of-solid-boundary;segment-first-solid-intersection',
            'runtimeIntegrated':False,'digest':self.digest}


def certify_sheet_solid_ownership(mesh,sheet,solid,source_boxes):
    """Rebuild BOTH proofs from actual source triangles before deriving bounds.

    A caller-supplied digest/AABB/embedded-evidence label grants no admission.
    The sheet must be the exact oriented copy of its source, and the owner the
    exact same-body exterior produced from the supplied verified source boxes.
    """
    body=sheet.get('bodyId')
    if (not isinstance(body,str) or not body or solid.get('bodyId')!=body or sheet.get('closed') is not False
            or solid.get('closed') is not True or not sheet.get('id') or not solid.get('id') or sheet['id']==solid['id']):
        raise ValueError('distinct same-body open sheet and closed solid required')
    provenance=sheet.get('sourceProvenance',{})
    indices=provenance.get('sourceTriangleIndices')
    if (provenance.get('relation')!='two-sided-open-sheet-corner-permutation' or provenance.get('meshDigest')!=mesh.digest
            or not isinstance(indices,list) or not indices or len(set(indices))!=len(indices)
            or any(type(i) is not int or not 0<=i<len(mesh.triangles) for i in indices)):
        raise ValueError('exact two-sided sheet source ownership required')
    rebuilt=orient_open_sheet([[*mesh.triangles[i],mesh.material_ids[i]] for i in indices],
        [mesh.source_ids[i] for i in indices],body_id=body,closed=False,two_sided=True)
    for key,expected in [('vertices',rebuilt.vertices),('triangles',rebuilt.triangles),
                         ('materialIds',rebuilt.material_ids),('sourceTriangleIds',rebuilt.source_ids),
                         ('embeddedBoundaryEvidence',rebuilt.embedding.evidence_id)]:
        if _canonical(sheet.get(key))!=_canonical(expected):raise ValueError('sheet boundary differs from rebuilt source proof: '+key)
    if _canonical(provenance.get('cornerPermutations'))!=_canonical(rebuilt.corner_permutations):
        raise ValueError('sheet source corner receipt differs')
    source_boxes=tuple(source_boxes)
    union=build_cuboid_union(mesh,source_boxes)
    if union.body_id!=body:raise ValueError('source boxes belong to a different body')
    for key,expected in [('vertices',union.vertices),('triangles',union.triangles),
                         ('materialIds',union.material_ids),('sourceTriangleIds',union.triangle_ids),
                         ('embeddedBoundaryEvidence',union.embedded_boundary_evidence)]:
        if _canonical(solid.get(key))!=_canonical(expected):raise ValueError('solid boundary differs from rebuilt source union: '+key)
    if solid.get('sourceProvenance',{}).get('digest')!=union.digest:
        raise ValueError('solid union receipt differs from rebuilt source')
    boxes=[];solid_sources=set()
    field=lambda p,k:p[k] if isinstance(p,dict) else getattr(p,k)
    for source in source_boxes:
        # The producer independently verifies each complete eight-corner
        # axis-aligned solid, including all six faces and native winding.
        one=build_cuboid_union(mesh,[source])
        low=tuple(min(v[k] for v in one.vertices) for k in range(3))
        high=tuple(max(v[k] for v in one.vertices) for k in range(3))
        ids=tuple(mesh.source_ids[i] for i in field(source,'source_indices'))
        solid_sources.update(ids);boxes.append((field(source,'id'),low,high,ids))
    if solid_sources&set(rebuilt.source_ids):raise ValueError('sheet and solid cannot reuse source triangles')
    boxes=tuple(boxes)
    digest=hashlib.sha256(_canonical((body,sheet['id'],solid['id'],mesh.digest,rebuilt.boundary.digest,union.digest,boxes)).encode()).hexdigest()
    return SheetSolidOwnership(body,sheet['id'],solid['id'],mesh.digest,rebuilt.boundary.digest,union.digest,boxes,digest)


_GEOMETRY=('id','bodyId','closed','vertices','triangles','materialIds','sourceTriangleIds')


def make_sheet_solid_certificate(mesh,sheet,solid,source_boxes):
    """Create a self-contained, independently checkable packet certificate."""
    source_boxes=tuple(source_boxes)
    verified=certify_sheet_solid_ownership(mesh,sheet,solid,source_boxes)
    field=lambda p,k:p[k] if isinstance(p,dict) else getattr(p,k)
    payload=[]
    for box in source_boxes:
        indices=tuple(field(box,'source_indices'))
        payload.append({'id':field(box,'id'),'bodyId':field(box,'body_id'),
            'faces':[[*mesh.triangles[i],mesh.material_ids[i]] for i in indices],
            'sourceTriangleIds':[mesh.source_ids[i] for i in indices]})
    data=verified.to_data()
    data['sourceBoxGeometry']=payload
    data['sheetGeometry']={k:sheet[k] for k in _GEOMETRY}
    data['solidGeometry']={k:solid[k] for k in _GEOMETRY}
    # Detach caller-owned lists; no later mutation can change a signed receipt.
    data=json.loads(_canonical(data))
    data.pop('digest')
    data['digest']=hashlib.sha256(_canonical(data).encode()).hexdigest()
    return data


def validate_sheet_solid_certificate(certificate,records):
    """Return actual packet indices, exact boxes and validated geometry receipt.

    Reconstructs source solids, never trusts sourceProvenance assertions or an
    AABB label. Changing box coordinates without changing the complete actual
    solid boundary cannot pass this equality and manifold/embedding proof.
    """
    data=json.loads(_canonical(certificate))
    supplied=data.pop('digest',None)
    if (data.get('schema')!='turing.native-sheet-solid-ownership.v1' or
            supplied!=hashlib.sha256(_canonical(data).encode()).hexdigest()):
        raise ValueError('sheet-solid certificate digest/schema mismatch')
    if (data.get('pointRule')!='suppress-sheet-feature-closest-point-inside-or-on-any-source-box'
            or data.get('solidRule')!='retain-incumbent-closed-boundary-contacts'):
        raise ValueError('unsupported sheet-solid ownership rule')
    def record(ident):
        found=[(i,r) for i,r in enumerate(records) if r.get('id')==ident]
        if len(found)!=1:raise ValueError('ownership needs one actual packet boundary per ID')
        return found[0]
    si,sheet=record(data.get('sheetPartId'));vi,solid=record(data.get('solidPartId'))
    body=data.get('bodyId')
    if si==vi or sheet.get('bodyId')!=body or solid.get('bodyId')!=body or sheet.get('closed') is not False or solid.get('closed') is not True:
        raise ValueError('ownership requires distinct same-body sheet and solid')
    for key,actual in [('sheetGeometry',sheet),('solidGeometry',solid)]:
        if _canonical(data.get(key))!=_canonical({k:actual.get(k) for k in _GEOMETRY}):
            raise ValueError('ownership geometry differs from actual packet '+key)
    boundary=NativeMeshBoundary(sheet['vertices'],sheet['triangles'],sheet['sourceTriangleIds'],closed=False,body_id=body)
    verify_native_embedding(boundary.vertices,boundary.triangles,boundary.source_ids)
    if boundary.digest!=data.get('sheetBoundaryDigest'):raise ValueError('sheet boundary identity differs')
    source=data.get('sourceBoxGeometry')
    if not isinstance(source,list) or not 1<=len(source)<=64:raise ValueError('bounded source box geometry required')
    faces=[];ids=[];parts=[]
    for box in source:
        if box.get('bodyId')!=body or not isinstance(box.get('faces'),list) or len(box['faces'])!=12 or len(box.get('sourceTriangleIds',()))!=12:
            raise ValueError('every source box needs all12 source triangles and same body')
        first=len(faces);faces.extend(box['faces']);ids.extend(box['sourceTriangleIds'])
        parts.append({'id':box.get('id'),'body_id':body,'closed':True,'source_indices':tuple(range(first,len(faces)))})
    if set(ids)&set(boundary.source_ids):raise ValueError('sheet/solid source triangle identities overlap')
    mesh=StudioMeshBake.from_faces(faces,source_ids=ids,frame_id='packet-sheet-solid-source',max_triangles=768)
    union=build_cuboid_union(mesh,parts)
    for key,expected in [('vertices',union.vertices),('triangles',union.triangles),('materialIds',union.material_ids),('sourceTriangleIds',union.triangle_ids)]:
        if _canonical(solid[key])!=_canonical(expected):raise ValueError('actual solid is not the exact source-box union: '+key)
    boxes=[]
    for box,part in zip(source,parts):
        one=build_cuboid_union(mesh,[part])
        low=tuple(min(v[k] for v in one.vertices) for k in range(3));high=tuple(max(v[k] for v in one.vertices) for k in range(3))
        boxes.append({'sourcePartId':part['id'],'low':list(low),'high':list(high),'sourceTriangleIds':box['sourceTriangleIds']})
    if _canonical(boxes)!=_canonical(data.get('sourceBoxes')):raise ValueError('mask boxes differ from independently verified source solids')
    data['digest']=supplied
    return {'sheet_index':si,'solid_index':vi,'boxes':tuple((tuple(b['low']),tuple(b['high'])) for b in boxes),
            'receipt':data}
