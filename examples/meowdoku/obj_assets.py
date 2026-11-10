"""Immutable owner-scoped OBJ sources, parsed by Spectral's existing importer."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import numpy as np

MAX_BYTES = 16 * 1024 * 1024
MAX_TRIANGLES = 500000
_DIGEST = re.compile(r'^[0-9a-f]{64}$')

class AssetError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status

class Store:
    def __init__(self, directory, accounts, spectral_root):
        self.directory = Path(directory)
        self.accounts = accounts
        self.spectral_root = Path(spectral_root)
        self.lock = threading.Lock()
        self._loader = None

    def _parse(self, obj, mtl):
        # Bound allocations before the existing importer triangulates polygons.
        triangles = 0
        for line in obj.splitlines():
            fields = line.split()
            if fields and fields[0].lower() == 'f':
                triangles += max(0, len(fields) - 3)
                if triangles > MAX_TRIANGLES:
                    raise AssetError('OBJ exceeds the triangle limit', 413)
        # Text inputs only: mtllib is never resolved against a server path.
        if self._loader is None:
            spec = importlib.util.spec_from_file_location('studio_obj_loader', self.spectral_root / 'camera_designer' / 'obj_loader.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._loader = module.load_obj
        try:
            vertices, ids, pbr, names = self._loader(obj, mtl)
        except (ValueError, IndexError, OverflowError) as exc:
            raise AssetError(str(exc)) from exc
        if not 0 < len(vertices) <= MAX_TRIANGLES or not np.isfinite(vertices).all() or not np.isfinite(pbr).all():
            raise AssetError('OBJ exceeds the triangle limit or contains nonfinite values')
        tri = vertices.astype(np.float64)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if np.any(np.linalg.norm(cross, axis=1) == 0):
            raise AssetError('OBJ contains a degenerate triangle')
        return {'vertices': vertices.reshape(-1).tolist(), 'material_ids': ids.tolist(),
                'pbr': pbr.reshape(-1).tolist(), 'material_names': list(names)}

    def _record(self, digest):
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise AssetError('OBJ asset unavailable', 404)
        try:
            value = json.loads((self.directory / (digest + '.json')).read_text(encoding='utf-8'))
        except FileNotFoundError:
            raise AssetError('OBJ asset unavailable', 404) from None
        source = {k: value[k] for k in ('owner', 'name', 'obj', 'mtl')}
        if self._digest(source) != digest:
            raise AssetError('Stored OBJ identity mismatch', 500)
        if self.accounts.owner_of('obj_asset', digest) != value['owner']:
            raise AssetError('OBJ asset unavailable', 404)
        return value

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()

    def get(self, digest, account_id=None, *, geometry=True):
        value = self._record(digest)
        if value['visibility'] != 'public' and account_id != value['owner']:
            raise AssetError('OBJ asset unavailable', 404)
        out = {'digest': digest, 'name': value['name'], 'visibility': value['visibility'],
               'owned': account_id == value['owner'], 'readOnly': True,
               'source_sha256': hashlib.sha256(value['obj'].encode('utf-8')).hexdigest(),
               'has_mtl': value['mtl'] is not None, 'attribution': value.get('attribution', ''),
               'mechanicsCertified': False}
        if geometry:
            out['geometry'] = self._parse(value['obj'], value['mtl'])
        return out

    def source(self, digest, account_id=None, *, mtl=False):
        self.get(digest, account_id, geometry=False)
        value = self._record(digest)['mtl' if mtl else 'obj']
        if value is None:
            raise AssetError('OBJ has no MTL source', 404)
        return value.encode('utf-8')

    def entries(self, account_id=None):
        out = []
        for path in self.directory.glob('*.json'):
            try:
                out.append(self.get(path.stem, account_id, geometry=False))
            except AssetError as exc:
                if exc.status != 404:
                    raise
        return sorted(out, key=lambda item: item['name'].casefold())

    def put(self, value, account_id):
        if account_id is None:
            raise AssetError('Log in to add an OBJ asset', 401)
        if not isinstance(value, dict) or set(value) not in ({'name', 'obj'}, {'name', 'obj', 'mtl'}):
            raise AssetError('Expected name, obj and optional mtl text')
        if not isinstance(value['name'], str) or not 1 <= len(value['name'].strip()) <= 80:
            raise AssetError('Name must be 1..80 characters')
        if not isinstance(value['obj'], str) or not value['obj'] or ('mtl' in value and not isinstance(value['mtl'], str)):
            raise AssetError('OBJ and MTL must be text')
        if len(value['obj'].encode('utf-8')) + len(value.get('mtl', '').encode('utf-8')) > MAX_BYTES:
            raise AssetError('OBJ source is too large', 413)
        self._parse(value['obj'], value.get('mtl'))
        source = {'owner': account_id, 'name': value['name'].strip(), 'obj': value['obj'], 'mtl': value.get('mtl')}
        digest = self._digest(source)
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / (digest + '.json')
            duplicate = path.exists()
            if not duplicate:
                if len(self.accounts.assets_for_user(account_id, 'obj_asset')) >= 128:
                    raise AssetError('OBJ library is full', 413)
                # Existing account registry is the ownership authority. A failed
                # claim is never interpreted as authorization for a different user.
                self.accounts.claim(account_id, 'obj_asset', digest)
                if self.accounts.owner_of('obj_asset', digest) != account_id:
                    raise AssetError('Cannot claim OBJ asset', 403)
                self._write(path, {**source, 'visibility': 'private'})
            self.get(digest, account_id, geometry=False)
        return {'digest': digest, 'duplicate': duplicate}

    def _write(self, path, value):
        fd, name = tempfile.mkstemp(prefix='.obj-', dir=self.directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
                json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    def publish(self, digest, verified_owner_id, *, attribution):
        """Operator-only publication after explicitly verifying the real owner.

        No HTTP visibility mutation exists. Public assets remain immutable.
        """
        with self.lock:
            value = self._record(digest)
            if verified_owner_id != value['owner']:
                raise AssetError('Verified owner does not own this OBJ', 403)
            if not isinstance(attribution, str) or not attribution.strip():
                raise AssetError('Public asset attribution is required')
            value.update(visibility='public', attribution=attribution)
            self._write(self.directory / (digest + '.json'), value)
        return self.get(digest, verified_owner_id, geometry=False)


def subject_packet(asset, shell):
    """Use the supplied float32 face order and geometric flat normals."""
    positions = np.asarray(asset['geometry']['vertices'], dtype='<f4').reshape(-1, 9)
    tri = positions.astype(np.float64).reshape(-1, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.asarray([math.hypot(*row) for row in cross])
    normals = np.repeat((cross / lengths[:, None])[:, None, :], 3, axis=1).astype('<f4').reshape(-1, 9)
    packed = np.empty(len(positions), dtype=[('positions', '<f4', (9,)), ('normals', '<f4', (9,)), ('id', '<u4')])
    packed['positions'], packed['normals'], packed['id'] = positions, normals, shell
    return packed
