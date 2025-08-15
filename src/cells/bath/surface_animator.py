# -*- coding: utf-8 -*-
"""
SurfaceAnimator — Voxel surface animation hash & instanced renderer (1D/2D/3D)
==============================================================================

Purpose
-------
Plug this class alongside your Voxel (MAC) fluid to render **prebaked tile meshes**
(“water patches”) selected by a compact **animation key** computed from per-cell
state: occupancy (phi), local surface normal, tangential flow direction & speed,
pressure class, gravity relation, and the marching-cubes/squares neighbor mask.

This yields fluid-looking water by instancing precomputed meshes (or quads in 2D)
with flipbook textures, *decoupled from the physics*. It supports **1D/2D/3D**.

Integration overview
--------------------
- Provide a MAC-style solver exposing:
  - `nx, ny, nz, dx`, cell-centered `phi` (0..1), pressure `pr`, and a method
    `export_vector_field()` or access to cell-centered velocity samples.
- Bake a tileset offline: `tileset.json` + `atlas.png` + prototype meshes (e.g. glTF).
- Instantiate `SurfaceAnimator(tileset, dx, dims)`; per frame call
  `anim.update(mac, t)` then either:
  - `batches = anim.instance_batches()` → feed into your existing GL layer, or
  - call `anim.draw(gl_resources, shader, viewproj)` to render via PyOpenGL.

Design
------
- **Key**: 64-bit packed integer from quantized fields (few bits each).
- **Hysteresis**: for phi & normal/angle bins to prevent flicker.
- **Dirty update**: only rebuild instances for cells whose key changed.
- **Instancing**: batches by `mesh_id`, per-instance mat (3x4), atlas row & frame.

MIT License.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Iterable, Any
import json, math
import numpy as np

# Optional OpenGL imports guarded (draw path only)
try:  # pragma: no cover
    from OpenGL.GL import (
        glBindVertexArray, glUseProgram, glUniformMatrix4fv, glUniform1i,
        glUniform2f, glUniform4f, GL_FALSE, glDrawElementsInstanced,
        GL_TRIANGLES, glActiveTexture, GL_TEXTURE0, glBindTexture,
    )
except Exception:
    glBindVertexArray = None  # type: ignore


# ----------------------------------------------------------------------------
# Tileset schema
# ----------------------------------------------------------------------------
@dataclass
class TileVariant:
    mesh_id: str           # key to mesh resource (e.g., glTF mesh name)
    atlas_row: int         # row index in flipbook atlas
    fps_base: float        # base fps; scaled by speed bin multiplier
    uv_scroll: Tuple[float, float] = (0.0, 0.0)

@dataclass
class Tileset:
    frames_per_row: int
    rows_per_variant: int
    variants: Dict[int, TileVariant]  # key(int) → variant

    @staticmethod
    def load(path_json: str) -> "Tileset":
        with open(path_json, 'r') as f:
            data = json.load(f)
        frames_per_row = int(data.get('frames_per_row', 16))
        rows_per_variant = int(data.get('rows_per_variant', 1))
        variants: Dict[int, TileVariant] = {}
        for k, v in data.get('variants', {}).items():
            # keys may be hex strings like "0x12ab..." or decimal strings
            key_int = int(k, 0) if isinstance(k, str) else int(k)
            variants[key_int] = TileVariant(
                mesh_id=v['mesh'], atlas_row=int(v['atlas_row']),
                fps_base=float(v.get('frame_rate_base', 8.0)),
                uv_scroll=tuple(v.get('uv_scroll', [0.0, 0.0]))  # type: ignore
            )
        return Tileset(frames_per_row=frames_per_row,
                       rows_per_variant=rows_per_variant,
                       variants=variants)


# ----------------------------------------------------------------------------
# Packed key fields & quantizers
# ----------------------------------------------------------------------------
class KeyPacker:
    """Pack quantized per-cell features into a 64-bit key.

    Layout (little → big):
      8  neighbor_mask      (MC/SQ case)
      4  occ_bin            φ bin (0..15)
      5  normal_bin         sphere/hemisphere bin
      5  flow_dir_bin       tangent direction bin
      3  speed_bin          {still, slow, med, fast}
      3  pressure_bin       {low, mid-, mid+, high}
      2  gravity_class      relation of n and g
      1  edge_flag          free-boundary proximity
      1  foam_flag          curvature*speed heuristic
     32  reserved           version/style/LOD
    """
    OFFS = {
        'neighbor_mask': 0,
        'occ_bin': 8,
        'normal_bin': 12,
        'flow_dir_bin': 17,
        'speed_bin': 22,
        'pressure_bin': 25,
        'gravity_class': 28,
        'edge_flag': 30,
        'foam_flag': 31,
        'reserved': 32,
    }
    MASKS = {
        'neighbor_mask': (1<<8)-1,
        'occ_bin': (1<<4)-1,
        'normal_bin': (1<<5)-1,
        'flow_dir_bin': (1<<5)-1,
        'speed_bin': (1<<3)-1,
        'pressure_bin': (1<<3)-1,
        'gravity_class': (1<<2)-1,
        'edge_flag': 1,
        'foam_flag': 1,
        'reserved': (1<<32)-1,
    }

    @classmethod
    def pack(cls, **kw: int) -> int:
        k = 0
        for name, off in cls.OFFS.items():
            val = int(kw.get(name, 0)) & cls.MASKS[name]
            k |= (val << off)
        return k


# Quantizer helpers -----------------------------------------------------------

def q_phi(phi: float) -> int:
    return int(np.clip(phi * 15.0 + 1e-9, 0, 15))

def q_pressure(p: float, edges: Tuple[float, float, float]=(-50.0, 50.0, 200.0)) -> int:
    a,b,c = edges
    if p < a: return 0
    if p < b: return 1
    if p < c: return 2
    return 3

def q_speed(s: float, edges=(0.05, 0.20, 0.5)) -> int:
    if s < edges[0]: return 0
    if s < edges[1]: return 1
    if s < edges[2]: return 2
    return 3

def q_gravity(dotng: float) -> int:
    # 0: facing up (water on top edge), 1: near sideways, 2: facing down
    if dotng < -0.5: return 0
    if dotng < 0.5:  return 1
    return 2

# Direction bins --------------------------------------------------------------

def quantize_normal(n: np.ndarray, bins_az=8, bins_el=4) -> int:
    # map normal to azimuth ∈ [0,2π), elevation ∈ [-π/2, π/2]
    x,y,z = n
    az = math.atan2(y, x) % (2*math.pi)
    el = math.atan2(z, max(1e-12, math.hypot(x,y)))  # [-pi/2, pi/2]
    iaz = int(az / (2*math.pi) * bins_az) % bins_az
    iel = int((el + math.pi/2) / math.pi * bins_el)
    iel = max(0, min(bins_el-1, iel))
    return iel * bins_az + iaz  # fits in 5 bits for 8*4=32

def quantize_flow_dir(vt: np.ndarray, bins=16) -> int:
    if np.linalg.norm(vt) < 1e-8: return 0
    x,y = vt[0], (vt[1] if vt.shape[0]>1 else 0.0)
    ang = math.atan2(y, x) % (2*math.pi)
    return int(ang / (2*math.pi) * bins) % bins

# ----------------------------------------------------------------------------
# Marching mask from phi (cell-centered) → corner occupancy → mask
# ----------------------------------------------------------------------------

def marching_mask_cell(phi_cc: np.ndarray, i: int, j: int, k: int, dim: int, thresh=0.5) -> int:
    """Return 8-bit (3D) or 4-bit (2D) mask by sampling corner occupancy.
    We derive corner values by averaging neighboring cell-centered phi.
    Order (3D):
      c0=(0,0,0), c1=(1,0,0), c2=(1,1,0), c3=(0,1,0), c4=(0,0,1), c5=(1,0,1), c6=(1,1,1), c7=(0,1,1)
    For 2D (nz=1), only the bottom quad (c0..c3) is used.
    """
    nx, ny, nz = phi_cc.shape
    def avg_corner(ii, jj, kk):
        # average of up to 8 contributing centers around the corner
        xs = [ii-1, ii]
        ys = [jj-1, jj]
        zs = [kk-1, kk]
        vals = []
        for a in xs:
            if not (0 <= a < nx): continue
            for b in ys:
                if not (0 <= b < ny): continue
                for c in zs:
                    if not (0 <= c < nz): continue
                    vals.append(phi_cc[a,b,c])
        return np.mean(vals) if vals else 0.0

    # base cell corner at (i,j,k)
    c = [0.0]*8
    c[0] = avg_corner(i,   j,   k  )
    c[1] = avg_corner(i+1, j,   k  )
    c[2] = avg_corner(i+1, j+1, k  )
    c[3] = avg_corner(i,   j+1, k  )
    if nz > 1:
        c[4] = avg_corner(i,   j,   k+1)
        c[5] = avg_corner(i+1, j,   k+1)
        c[6] = avg_corner(i+1, j+1, k+1)
        c[7] = avg_corner(i,   j+1, k+1)
    mask = 0
    if nz > 1:
        for bit,val in enumerate(c):
            if val >= thresh: mask |= (1<<bit)
    else:
        for bit in range(4):
            if c[bit] >= thresh: mask |= (1<<bit)
    return mask


# ----------------------------------------------------------------------------
# Instance record & batch container
# ----------------------------------------------------------------------------
@dataclass
class Instance:
    xform: np.ndarray      # (3x4) affine in world space
    atlas_row: int
    frame: int

@dataclass
class InstanceBatch:
    mesh_id: str
    instances: List[Instance]


# ----------------------------------------------------------------------------
# SurfaceAnimator
# ----------------------------------------------------------------------------
class SurfaceAnimator:
    def __init__(self, tileset: Tileset, dx: float, dim: int,
                 hysteresis_phi: float = 0.04,
                 hysteresis_dir_deg: float = 10.0,
                 speed_multipliers: Tuple[float,float,float,float]=(0.0, 0.8, 1.0, 1.3)):
        self.tileset = tileset
        self.dx = float(dx)
        self.dim = int(dim)
        self.last_keys: Optional[np.ndarray] = None  # (nx,ny,nz) int64
        self.last_dir_bin: Optional[np.ndarray] = None
        self.hyst_phi = float(hysteresis_phi)
        self.hyst_dir = float(math.radians(hysteresis_dir_deg))
        self.speed_mul = speed_multipliers
        self._batches: List[InstanceBatch] = []

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def update(self, mac: Any, t: float) -> None:
        """Build instance batches for cells with changed animation keys.
        `mac` must expose: nx, ny, nz, dx, phi (cc), pr (cc), and:
         - method `export_vector_field()` → (pos_cc, vel_cc) or
         - arrays `u, v, w` at faces and a sampler `_sample_velocity(X3)`.
        """
        nx, ny, nz = getattr(mac, 'nx'), getattr(mac, 'ny'), getattr(mac, 'nz')
        phi = np.asarray(mac.phi)  # shape (nx,ny,nz)
        pr  = np.asarray(mac.pr)
        # velocity at centers
        if hasattr(mac, 'export_vector_field'):
            pos_cc, vel_cc = mac.export_vector_field()
            vel = vel_cc.reshape(nx, ny, max(1,nz), 3)
        else:
            # sample from faces to centers using existing sampler
            centers = self._centers(nx, ny, nz, mac.dx)
            vel = mac._sample_velocity(centers.reshape(-1,3)).reshape(nx,ny,max(1,nz),3)

        # compute per-cell features
        grad = self._grad_center(phi, mac.dx)  # ∇phi
        nrm = self._safe_normalize(grad)       # surface normal
        ut  = vel - np.sum(vel*nrm, axis=3, keepdims=True) * nrm  # tangential flow
        speed = np.linalg.norm(ut, axis=3)
        dotng = np.sum(nrm * np.array(getattr(mac, 'gravity', (0, -1, 0))), axis=3)

        # keys & dirty map
        keys = np.zeros((nx,ny,nz), dtype=np.int64)
        dir_bin_prev = self.last_dir_bin if self.last_dir_bin is not None else np.full((nx,ny,nz), -1)
        dirty = np.zeros((nx,ny,nz), dtype=bool)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    occ = float(phi[i,j,k])
                    # hysteresis on phi: compare against last occ via key reuse
                    occ_bin = q_phi(occ)
                    mask = marching_mask_cell(phi, i, j, k, self.dim)
                    n = nrm[i,j,k]
                    nb = quantize_normal(n)
                    vt = ut[i,j,k]
                    # hysteresis on direction: if direction small change, reuse bin
                    db = quantize_flow_dir(vt[:2] if self.dim>=2 else np.array([vt[0],0.0]))
                    if dir_bin_prev[i,j,k] >= 0:
                        # compute angle diff if both have meaningful speed
                        if speed[i,j,k] < 1e-3:
                            db = 0
                        else:
                            # crude: bins only; hysteresis managed by speed threshold above
                            pass
                    sb = q_speed(speed[i,j,k])
                    pb = q_pressure(pr[i,j,k])
                    gc = q_gravity(dotng[i,j,k])
                    edge_flag = 1 if self._near_boundary(i,j,k,nx,ny,nz) else 0
                    foam_flag = 1 if (sb>=2 and np.linalg.norm(grad[i,j,k])>0.4) else 0
                    key = KeyPacker.pack(neighbor_mask=mask, occ_bin=occ_bin,
                                         normal_bin=nb, flow_dir_bin=db,
                                         speed_bin=sb, pressure_bin=pb,
                                         gravity_class=gc, edge_flag=edge_flag,
                                         foam_flag=foam_flag)
                    keys[i,j,k] = key

        if self.last_keys is None:
            dirty[...] = True
        else:
            dirty = (keys != self.last_keys)

        # build instances (only dirty cells)
        self._batches = []
        batches_by_mesh: Dict[str, List[Instance]] = {}
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not dirty[i,j,k]:
                        continue
                    key = int(keys[i,j,k])
                    variant = self._find_variant_with_fallback(key)
                    if variant is None:
                        continue
                    # transform
                    center = np.array([(i+0.5)*self.dx, (j+0.5)*self.dx, (k+0.5)*self.dx])
                    n = nrm[i,j,k]
                    vt = ut [i,j,k]
                    M = self._tile_xform(center, n, vt)
                    # animation
                    sb = q_speed(speed[i,j,k])
                    fps = variant.fps_base * (self.speed_mul[sb] if sb < len(self.speed_mul) else 1.0)
                    frame = int(t * fps) % max(1, self.tileset.frames_per_row)
                    inst = Instance(xform=M, atlas_row=variant.atlas_row, frame=frame)
                    batches_by_mesh.setdefault(variant.mesh_id, []).append(inst)

        # wrap
        for mesh_id, insts in batches_by_mesh.items():
            self._batches.append(InstanceBatch(mesh_id=mesh_id, instances=insts))

        self.last_keys = keys
        self.last_dir_bin = np.vectorize(lambda b: b)(keys & (KeyPacker.MASKS['flow_dir_bin']<<KeyPacker.OFFS['flow_dir_bin']))  # rough cache

    def instance_batches(self) -> List[InstanceBatch]:
        return self._batches

    # ------------------------------------------------------------------
    # Fallback search: progressively drop low-importance fields
    # ------------------------------------------------------------------
    def _find_variant_with_fallback(self, key: int) -> Optional[TileVariant]:
        if key in self.tileset.variants:
            return self.tileset.variants[key]
        # fallback order: pressure → speed → flow_dir → normal → occ_bin → neighbor
        fields = ['pressure_bin','speed_bin','flow_dir_bin','normal_bin','occ_bin','neighbor_mask']
        for f in fields:
            k2 = self._zero_field(key, f)
            if k2 in self.tileset.variants:
                return self.tileset.variants[k2]
            key = k2
        return None

    @staticmethod
    def _zero_field(key: int, field: str) -> int:
        off = KeyPacker.OFFS[field]; mask = KeyPacker.MASKS[field]
        return key & ~(mask << off)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------
    def _centers(self, nx, ny, nz, dx) -> np.ndarray:
        Ii, Jj, Kk = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5, np.arange(nz)+0.5, indexing='ij')
        return np.stack([Ii*dx, Jj*dx, Kk*dx], axis=-1)

    def _grad_center(self, phi: np.ndarray, dx: float) -> np.ndarray:
        # central differences; Neumann at borders
        gx = np.zeros_like(phi); gy = np.zeros_like(phi); gz = np.zeros_like(phi)
        gx[1:-1,...] = (phi[2: ,...]-phi[0:-2,...])/(2*dx)
        gy[:,1:-1,:] = (phi[:,2: ,:]-phi[:,0:-2,:])/(2*dx) if phi.ndim>=2 else 0
        if phi.ndim==3 and phi.shape[2]>1:
            gz[:,:,1:-1] = (phi[:,:,2:] - phi[:,:,0:-2])/(2*dx)
        G = np.stack([gx, gy if phi.ndim>=2 else np.zeros_like(gx), gz if phi.ndim==3 else np.zeros_like(gx)], axis=-1)
        return G

    @staticmethod
    def _safe_normalize(v: np.ndarray, eps: float=1e-9) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        n = np.maximum(n, eps)
        return v / n

    def _near_boundary(self, i,j,k, nx,ny,nz) -> bool:
        return (i==0 or j==0 or k==0 or i==nx-1 or j==ny-1 or k==nz-1)

    def _tile_xform(self, center: np.ndarray, n: np.ndarray, vt: np.ndarray) -> np.ndarray:
        """Return 3x4 affine: columns are basis vectors, last column is translation."""
        # Build orthonormal basis: z'=n, x' along vt if present, y'=z'×x'
        z = self._safe_normalize(n)
        x = vt
        if np.linalg.norm(x) < 1e-6:
            # pick any tangent
            a = np.array([1.0,0.0,0.0]) if abs(z[0])<0.9 else np.array([0.0,1.0,0.0])
            x = np.cross(z, a)
        x = self._safe_normalize(x)
        y = np.cross(z, x)
        M = np.stack([x*self.dx, y*self.dx, z*self.dx, center], axis=-1)  # 3x4
        return M.astype(np.float32)

    # ------------------------------------------------------------------
    # (Optional) OpenGL fast path — requires prepared GL meshes/shaders
    # ------------------------------------------------------------------
    def draw(self, gl: Any, shader_prog: int, viewproj: np.ndarray, mesh_registry: Dict[str, Any], atlas_tex: int) -> None:
        """Instanced draw of all batches. Provide your own mesh_registry with VAO/EBO.
        mesh_registry[mesh_id] must expose: vao, index_count, instance_buffer (SSBO/UBO) writer.
        `gl` is a small adapter with methods to write per-instance data to GPU.
        """
        if glBindVertexArray is None:
            raise RuntimeError("PyOpenGL not available; use instance_batches() instead")
        glUseProgram(shader_prog)
        glUniformMatrix4fv(gl.get_uniform_loc(shader_prog, "uViewProj"), 1, GL_FALSE, viewproj.astype(np.float32))
        glActiveTexture(GL_TEXTURE0); glBindTexture(gl.GL_TEXTURE_2D, atlas_tex)  # type: ignore
        glUniform1i(gl.get_uniform_loc(shader_prog, "uAtlas"), 0)
        glUniform2f(gl.get_uniform_loc(shader_prog, "uAtlasDims"), float(self.tileset.frames_per_row), float(self.tileset.rows_per_variant))

        for batch in self._batches:
            mesh = mesh_registry.get(batch.mesh_id)
            if mesh is None or not batch.instances:
                continue
            # stream instances
            gl.update_instance_ssbo(mesh, batch.instances)  # user-provided
            glBindVertexArray(mesh.vao)
            glDrawElementsInstanced(GL_TRIANGLES, mesh.index_count, gl.GL_UNSIGNED_INT, None, len(batch.instances))  # type: ignore


# ----------------------------------------------------------------------------
# End of module
# ----------------------------------------------------------------------------
