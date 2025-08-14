import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .hierarchy import Cell


def _inside_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0.0:
        return False
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u >= 0.0) and (v >= 0.0) and (w >= 0.0)


def _vertex_triangle_penetration(v: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return 0.0, None
    n = n / norm
    dist = np.dot(v - a, n)
    if dist >= 0.0:
        return dist, None
    proj = v - dist * n
    if _inside_triangle(proj, a, b, c):
        return dist, n
    return dist, None


def _resolve_pair(v: np.ndarray, tri_verts: List[np.ndarray], normal: np.ndarray, depth: float):
    v += -depth * normal


def build_self_contacts_spatial_hash(
    X: np.ndarray,
    faces: np.ndarray,
    cell_ids: np.ndarray,
    voxel_size: float,
    *,
    # memory & batching
    max_vox_entries: int = 8_000_000,    # ~face-voxel expansions per face-chunk
    vbatch: int = 250_000,               # vertices per vertex-chunk
    ram_limit_bytes: int | None = None,  # cap temporary arrays; shrink chunks if needed
    # sparsification
    vertex_sample: float = 1.0,          # process this fraction of vertices globally (0..1]
    keep_prob: float = 1.0,              # thin candidate (vi,fi) inside each batch (0..1]
    rng_seed: int | None = 0,
    # adjacency
    adjacency: str = "self",             # "none" | "self"
) -> np.ndarray:
    """
    Streaming broad-phase vertex–triangle candidate generator with sub-batching
    and sparsification. Returns unique (vi, fi) pairs.

    - Sub-batches faces by *expanded voxel count* so we never allocate the full
      face->voxel expansion at once.
    - Sub-batches vertices so the 27-neighborhood key set is bounded.
    - `vertex_sample < 1` skips a fraction of vertices deterministically.
    - `keep_prob < 1` randomly thins candidates (per-pair Bernoulli).
    - `adjacency="self"` removes pairs where the vertex is one of the face's 3 vertices.
    - Pairs are deduped incrementally via 64-bit hashing to stay NumPy-only.

    Notes:
      * Temporary array sizes are estimated from ``max_vox_entries`` and
        ``vbatch``.  If ``ram_limit_bytes`` is given, both chunk sizes are
        aggressively halved until the estimate fits the budget.  This prevents
        rare out-of-memory spikes on large meshes.
    """
    n = X.shape[0]
    m = faces.shape[0]
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=np.int32)

    inv_vox = 1.0 / max(voxel_size, 1e-12)
    face_cell = cell_ids[faces[:, 0]]

    # --- Estimate chunk memory & shrink to fit budget -------------------------
    def _estimate(face_vox: int, vchunk: int) -> tuple[int, int]:
        """Return (face_bytes, vert_bytes) for the memory model."""
        # face expansion arrays: face_for_vox, kt, vx, vy, vz, order_t
        face_bytes = face_vox * (8 * 5 + 4)
        # vertex arrays: idx_v, V, vvox, kv, vv_idx and a few temporaries
        vert_bytes = vchunk * (3 * 8 + 3 * 8 + 27 * (8 + 4))
        return face_bytes, vert_bytes

    if ram_limit_bytes is not None:
        fb, vb = _estimate(max_vox_entries, vbatch)
        while fb + vb > ram_limit_bytes and (max_vox_entries > 1 or vbatch > 1):
            if fb >= vb and max_vox_entries > 1:
                max_vox_entries = max(1, max_vox_entries // 2)
            elif vbatch > 1:
                vbatch = max(1, vbatch // 2)
            fb, vb = _estimate(max_vox_entries, vbatch)

    # --- Precompute face voxel AABBs + per-face expansion counts (vectorized) ---
    P = X[faces]  # (m, 3, 3)
    mn = np.floor(P.min(axis=1) * inv_vox).astype(np.int64)   # (m,3)
    mx = np.floor(P.max(axis=1) * inv_vox).astype(np.int64)   # (m,3)
    spans = (mx - mn + 1)                                     # (m,3)
    counts = spans[:, 0] * spans[:, 1] * spans[:, 2]          # voxels per face
    if counts.sum() == 0:
        return np.empty((0, 2), dtype=np.int32)

    # --- Build face-chunk boundaries by cumulative voxel expansions ------------
    cum_counts = np.cumsum(counts)
    thresholds = np.arange(max_vox_entries, cum_counts[-1], max_vox_entries)
    f_starts = np.concatenate(([0], np.searchsorted(cum_counts, thresholds, side="left"), [m]))

    # --- Vertex downsampling mask (deterministic) ------------------------------
    if vertex_sample < 1.0:
        # hash-like deterministic mask: keep roughly vertex_sample fraction
        step = max(1, int(round(1.0 / vertex_sample)))
        v_keep_mask = (np.arange(n, dtype=np.int64) % step) == 0
    else:
        v_keep_mask = np.ones(n, dtype=bool)

    rng = np.random.default_rng(rng_seed)

    codes_seen = np.empty(0, dtype=np.int64)
    mask32 = np.int64((1 << 32) - 1)

    # --- Iterate face-chunks ---------------------------------------------------
    for fs, fe in zip(f_starts[:-1], f_starts[1:]):
        F_idx = np.arange(fs, fe, dtype=np.int64)
        if F_idx.size == 0:
            continue

        spans_f = spans[F_idx]
        mn_f = mn[F_idx]

        # Expand face voxels for this chunk (vectorized, same trick as before)
        sx, sy, sz = spans_f.T
        counts_f = sx * sy * sz
        total = int(counts_f.sum())
        off = np.empty(len(F_idx) + 1, dtype=np.int64); off[0] = 0
        np.cumsum(counts_f, out=off[1:])

        face_for_vox = np.repeat(F_idx, counts_f)
        t_all = np.arange(total, dtype=np.int64) - off[face_for_vox - F_idx[0]]

        sy_ = sy[face_for_vox - F_idx[0]]
        sz_ = sz[face_for_vox - F_idx[0]]
        yz = sy_ * sz_
        dx = t_all // yz
        rem = t_all - dx * yz
        dy = rem // sz_
        dz = rem - dy * sz_

        vx = mn_f[face_for_vox - F_idx[0], 0] + dx
        vy = mn_f[face_for_vox - F_idx[0], 1] + dy
        vz = mn_f[face_for_vox - F_idx[0], 2] + dz

        # hash voxel indices (int64, works with negatives)
        kt = (vx * 73856093) ^ (vy * 19349663) ^ (vz * 83492791)
        order_t = np.argsort(kt, kind="mergesort")
        kt_sorted = kt[order_t]
        face_sorted = face_for_vox[order_t].astype(np.int32, copy=False)

        # --- Iterate vertex-chunks --------------------------------------------
        for v0 in range(0, n, vbatch):
            v1 = min(v0 + vbatch, n)
            # apply global vertex downsampling in this chunk
            sel = v_keep_mask[v0:v1]
            if not sel.any():
                continue
            idx_v = np.arange(v0, v1, dtype=np.int32)[sel]
            V = X[idx_v]

            # 27-neighborhood voxel keys for these vertices
            vvox = np.floor(V * inv_vox).astype(np.int64)  # (k,3)
            offs = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                                        indexing="ij")).reshape(3, -1).T  # (27,3)
            nei = vvox[:, None, :] + offs[None, :, :]      # (k,27,3)
            kv = (nei[..., 0].ravel() * 73856093) ^ \
                 (nei[..., 1].ravel() * 19349663) ^ \
                 (nei[..., 2].ravel() * 83492791)          # (27k,)
            vv_idx = np.repeat(idx_v.astype(np.int32, copy=False), 27)

            # intersect with face voxels for this face-chunk
            L = np.searchsorted(kt_sorted, kv, side="left")
            R = np.searchsorted(kt_sorted, kv, side="right")
            lens = R - L
            valid = lens > 0
            if not valid.any():
                continue

            L = L[valid]
            lens = lens[valid]
            verts_sel = vv_idx[valid]

            csum = np.cumsum(lens, dtype=np.int64)
            starts = csum - lens
            base = np.repeat(L, lens)
            within = np.arange(csum[-1], dtype=np.int64) - np.repeat(starts, lens)
            tri_pos = base + within

            fi_all = face_sorted[tri_pos]                # candidate faces
            vi_all = np.repeat(verts_sel, lens)          # matching vertices

            # same-cell filter
            same = (cell_ids[vi_all] == face_cell[fi_all])
            if not same.any():
                continue
            vi_all = vi_all[same]; fi_all = fi_all[same]

            # optional thinning of pairs
            if keep_prob < 1.0 and vi_all.size:
                keep = rng.random(vi_all.size) < keep_prob
                if not keep.any():
                    continue
                vi_all = vi_all[keep]; fi_all = fi_all[keep]

            # adjacency filter ("self"): drop if vi in faces[fi]
            if adjacency == "self" and vi_all.size:
                # gather 3 verts of each candidate face and compare
                tri_verts = faces[fi_all]                      # (q,3)
                bad = (tri_verts[:, 0] == vi_all) | \
                      (tri_verts[:, 1] == vi_all) | \
                      (tri_verts[:, 2] == vi_all)
                if bad.any():
                    keep = ~bad
                    vi_all = vi_all[keep]; fi_all = fi_all[keep]

            if vi_all.size:
                # discard vertices exactly coplanar with the triangle
                tri = faces[fi_all]
                a = X[tri[:, 0]]
                b = X[tri[:, 1]]
                c = X[tri[:, 2]]
                n = np.cross(b - a, c - a)
                dist = np.einsum("ij,ij->i", X[vi_all] - a, n)
                mask = dist != 0.0
                if not mask.any():
                    continue
                vi_all = vi_all[mask]; fi_all = fi_all[mask]

            if vi_all.size:
                codes = (vi_all.astype(np.int64) << 32) | fi_all.astype(np.int64)
                codes_seen = np.unique(np.concatenate([codes_seen, codes]))

    if codes_seen.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    out = np.empty((codes_seen.size, 2), dtype=np.int32)
    out[:, 0] = (codes_seen >> 32).astype(np.int32)
    out[:, 1] = (codes_seen & mask32).astype(np.int32)
    return out


def _closest_point_on_tri_batch(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ab = b - a
    ac = c - a
    ap = p - a
    dot00 = np.einsum("ij,ij->i", ab, ab)
    dot01 = np.einsum("ij,ij->i", ab, ac)
    dot11 = np.einsum("ij,ij->i", ac, ac)
    dot02 = np.einsum("ij,ij->i", ab, ap)
    dot12 = np.einsum("ij,ij->i", ac, ap)
    denom = dot00 * dot11 - dot01 * dot01
    denom = np.where(denom == 0.0, 1e-20, denom)
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    w = 1.0 - u - v
    interior = (u >= 0.0) & (v >= 0.0) & (w >= 0.0)
    q0 = a + u[:, None] * ab + v[:, None] * ac
    bc0 = np.stack([w, u, v], axis=1)
    t_ab = np.clip(dot02 / np.where(dot00 == 0.0, 1e-20, dot00), 0.0, 1.0)
    q1 = a + t_ab[:, None] * ab
    bc1 = np.stack([1.0 - t_ab, t_ab, np.zeros_like(t_ab)], axis=1)
    bc_vec = c - b
    pb = p - b
    dotbb = np.einsum("ij,ij->i", bc_vec, bc_vec)
    t_bc = np.clip(
        np.einsum("ij,ij->i", pb, bc_vec) / np.where(dotbb == 0.0, 1e-20, dotbb),
        0.0,
        1.0,
    )
    q2 = b + t_bc[:, None] * bc_vec
    bc2 = np.stack([np.zeros_like(t_bc), 1.0 - t_bc, t_bc], axis=1)
    ca = a - c
    pc = p - c
    dotcc = np.einsum("ij,ij->i", ca, ca)
    t_ca = np.clip(
        np.einsum("ij,ij->i", pc, ca) / np.where(dotcc == 0.0, 1e-20, dotcc),
        0.0,
        1.0,
    )
    q3 = c + t_ca[:, None] * ca
    bc3 = np.stack([t_ca, np.zeros_like(t_ca), 1.0 - t_ca], axis=1)
    d0 = np.sum((p - q0) ** 2, axis=1)
    d1 = np.sum((p - q1) ** 2, axis=1)
    d2 = np.sum((p - q2) ** 2, axis=1)
    d3 = np.sum((p - q3) ** 2, axis=1)
    d0 = np.where(interior, d0, np.inf)
    D = np.stack([d0, d1, d2, d3], axis=1)
    pick = np.argmin(D, axis=1)
    q = np.where(
        (pick == 0)[:, None],
        q0,
        np.where((pick == 1)[:, None], q1, np.where((pick == 2)[:, None], q2, q3)),
    )
    bc = np.where(
        (pick == 0)[:, None],
        bc0,
        np.where((pick == 1)[:, None], bc1, np.where((pick == 2)[:, None], bc2, bc3)),
    )
    dvec = p - q
    d = np.linalg.norm(dvec, axis=1)
    n = dvec / (d[:, None] + 1e-20)
    face_n = np.cross(ab, ac)
    face_n /= np.linalg.norm(face_n, axis=1, keepdims=True) + 1e-20
    n = np.where((d <= 1e-12)[:, None], face_n, n)
    return q, bc, n, d


def project_self_contacts_streamed(
    X: np.ndarray,
    faces: np.ndarray,
    invm: np.ndarray,
    cell_ids: np.ndarray,
    voxel_size: float,
    dt: float,
    *,
    min_separation: float = 0.0,
    compliance: float = 0.0,
    iters: int = 2,
    max_vox_entries: int = 8_000_000,
    vbatch: int = 250_000,
    ram_limit_bytes: int | None = None,
    vertex_sample: float = 1.0,
    keep_prob: float = 1.0,
    rng_seed: int | None = 0,
    adjacency: str = "self",
) -> None:
    """Stream self-contact detection and projection into ``X``.

    Sub-batches of vertex–triangle candidates are generated, narrowed and
    projected immediately, so no global pair list is stored and peak memory is
    bounded by ``max_vox_entries`` and ``vbatch``.
    """
    n = X.shape[0]
    m = faces.shape[0]
    if n == 0 or m == 0:
        return

    inv_vox = 1.0 / max(voxel_size, 1e-12)
    face_cell = cell_ids[faces[:, 0]]
    rng = np.random.default_rng(rng_seed)

    def _estimate(face_vox: int, vchunk: int) -> tuple[int, int]:
        face_bytes = face_vox * (8 * 5 + 4)
        vert_bytes = vchunk * (3 * 8 + 3 * 8 + 27 * (8 + 4))
        return face_bytes, vert_bytes

    if ram_limit_bytes is not None:
        fb, vb = _estimate(max_vox_entries, vbatch)
        while fb + vb > ram_limit_bytes and (max_vox_entries > 1 or vbatch > 1):
            if fb >= vb and max_vox_entries > 1:
                max_vox_entries = max(1, max_vox_entries // 2)
            elif vbatch > 1:
                vbatch = max(1, vbatch // 2)
            fb, vb = _estimate(max_vox_entries, vbatch)

    P = X[faces]
    mn = np.floor(P.min(axis=1) * inv_vox).astype(np.int64)
    mx = np.floor(P.max(axis=1) * inv_vox).astype(np.int64)
    spans = (mx - mn + 1)
    counts = spans[:, 0] * spans[:, 1] * spans[:, 2]
    if counts.sum() == 0:
        return

    cum = np.cumsum(counts)
    thresholds = np.arange(max_vox_entries, cum[-1], max_vox_entries)
    f_starts = np.concatenate(([0], np.searchsorted(cum, thresholds, "left"), [m]))

    if vertex_sample < 1.0:
        step = max(1, int(round(1.0 / vertex_sample)))
        v_keep_mask = (np.arange(n, dtype=np.int64) % step) == 0
    else:
        v_keep_mask = np.ones(n, dtype=bool)

    alpha = compliance / (dt * dt)

    for _ in range(iters):
        for fs, fe in zip(f_starts[:-1], f_starts[1:]):
            if fs == fe:
                continue
            F_idx = np.arange(fs, fe, dtype=np.int64)
            spans_f = spans[F_idx]
            mn_f = mn[F_idx]

            sx, sy, sz = spans_f.T
            counts_f = sx * sy * sz
            total = int(counts_f.sum())
            off = np.empty(len(F_idx) + 1, dtype=np.int64)
            off[0] = 0
            np.cumsum(counts_f, out=off[1:])

            face_for_vox = np.repeat(F_idx, counts_f)
            t_all = np.arange(total, dtype=np.int64) - off[face_for_vox - F_idx[0]]

            sy_ = sy[face_for_vox - F_idx[0]]
            sz_ = sz[face_for_vox - F_idx[0]]
            yz = sy_ * sz_
            dx = t_all // yz
            rem = t_all - dx * yz
            dy = rem // sz_
            dz = rem - dy * sz_

            vx = mn_f[face_for_vox - F_idx[0], 0] + dx
            vy = mn_f[face_for_vox - F_idx[0], 1] + dy
            vz = mn_f[face_for_vox - F_idx[0], 2] + dz

            kt = (vx * 73856093) ^ (vy * 19349663) ^ (vz * 83492791)
            order_t = np.argsort(kt, kind="mergesort")
            kt_sorted = kt[order_t]
            face_sorted = face_for_vox[order_t].astype(np.int32, copy=False)

            for v0 in range(0, n, vbatch):
                v1 = min(v0 + vbatch, n)
                sel = v_keep_mask[v0:v1]
                if not sel.any():
                    continue
                idx_v = np.arange(v0, v1, dtype=np.int32)[sel]
                V = X[idx_v]

                vvox = np.floor(V * inv_vox).astype(np.int64)
                offs = np.array(
                    np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
                ).reshape(3, -1).T
                nei = vvox[:, None, :] + offs[None, :, :]
                kv = (
                    (nei[..., 0].ravel() * 73856093)
                    ^ (nei[..., 1].ravel() * 19349663)
                    ^ (nei[..., 2].ravel() * 83492791)
                )
                vv_idx = np.repeat(idx_v.astype(np.int32, copy=False), 27)

                L = np.searchsorted(kt_sorted, kv, "left")
                R = np.searchsorted(kt_sorted, kv, "right")
                lens = R - L
                valid = lens > 0
                if not valid.any():
                    continue

                L = L[valid]
                lens = lens[valid]
                verts_sel = vv_idx[valid]

                csum = np.cumsum(lens, dtype=np.int64)
                starts = csum - lens
                base = np.repeat(L, lens)
                within = np.arange(csum[-1], dtype=np.int64) - np.repeat(starts, lens)
                tri_pos = base + within

                fi_all = face_sorted[tri_pos]
                vi_all = np.repeat(verts_sel, lens)

                same = cell_ids[vi_all] == face_cell[fi_all]
                if not same.any():
                    continue
                vi_all = vi_all[same]
                fi_all = fi_all[same]

                if keep_prob < 1.0 and vi_all.size:
                    keep = rng.random(vi_all.size) < keep_prob
                    if not keep.any():
                        continue
                    vi_all = vi_all[keep]
                    fi_all = fi_all[keep]

                if adjacency == "self" and vi_all.size:
                    tv = faces[fi_all]
                    bad = (
                        (tv[:, 0] == vi_all)
                        | (tv[:, 1] == vi_all)
                        | (tv[:, 2] == vi_all)
                    )
                    if bad.any():
                        keep = ~bad
                        vi_all = vi_all[keep]
                        fi_all = fi_all[keep]

                if vi_all.size == 0:
                    continue

                tri = faces[fi_all]
                ai, bi, ci = tri[:, 0], tri[:, 1], tri[:, 2]

                pv = X[vi_all]
                pa = X[ai]
                pb = X[bi]
                pc = X[ci]

                q, bc, nrm, dist = _closest_point_on_tri_batch(pv, pa, pb, pc)

                C = min_separation - dist
                mask = C > 0.0
                if not np.any(mask):
                    continue

                vi_m = vi_all[mask]
                ai_m = ai[mask]
                bi_m = bi[mask]
                ci_m = ci[mask]
                n_m = nrm[mask]
                C_m = C[mask]
                bc_m = bc[mask]

                wv = invm[vi_m]
                wa = invm[ai_m]
                wb = invm[bi_m]
                wc = invm[ci_m]
                wsum = wv + (bc_m[:, 0] ** 2) * wa + (bc_m[:, 1] ** 2) * wb + (
                    bc_m[:, 2] ** 2
                ) * wc
                valid_mass = wsum > 0.0
                if not np.any(valid_mass):
                    continue

                vi_m = vi_m[valid_mass]
                ai_m = ai_m[valid_mass]
                bi_m = bi_m[valid_mass]
                ci_m = ci_m[valid_mass]
                n_m = n_m[valid_mass]
                C_m = C_m[valid_mass]
                bc_m = bc_m[valid_mass]
                wv = wv[valid_mass]
                wa = wa[valid_mass]
                wb = wb[valid_mass]
                wc = wc[valid_mass]
                wsum = wsum[valid_mass]

                dl = C_m / (wsum + alpha)

                np.add.at(X, vi_m, (wv * dl)[:, None] * n_m)
                np.add.at(X, ai_m, -(wa * (bc_m[:, 0] * dl))[:, None] * n_m)
                np.add.at(X, bi_m, -(wb * (bc_m[:, 1] * dl))[:, None] * n_m)
                np.add.at(X, ci_m, -(wc * (bc_m[:, 2] * dl))[:, None] * n_m)


def resolve_membrane_collisions(
    cells: List["Cell"], min_separation: float = 0.0, iters: int = 10
):
    """Naively separate vertex-triangle penetrations within/between cells.

    Runs a handful of relaxation iterations; each pass scatters small
    corrections to penetrating vertex/triangle pairs.  This is an
    intentionally simple placeholder for a future broad-phase + XPBD contact
    solver but already prevents visible interpenetration in small meshes.
    """
    for _ in range(iters):
        changed = False
        for idx, cell in enumerate(cells):
            V = cell.X
            F = cell.faces
            for vi, v in enumerate(V):
                for f in F:
                    if vi in f:
                        continue
                    a, b, c = V[f[0]], V[f[1]], V[f[2]]
                    dist, n = _vertex_triangle_penetration(v, a, b, c)
                    if n is not None and dist < min_separation:
                        _resolve_pair(v, [a, b, c], n, dist - min_separation)
                        changed = True

            for other in cells[idx + 1 :]:
                Vo = other.X
                Fo = other.faces
                for v in V:
                    for f in Fo:
                        a, b, c = Vo[f[0]], Vo[f[1]], Vo[f[2]]
                        dist, n = _vertex_triangle_penetration(v, a, b, c)
                        if n is not None and dist < min_separation:
                            _resolve_pair(v, [a, b, c], n, dist - min_separation)
                            changed = True
                for v in Vo:
                    for f in F:
                        a, b, c = V[f[0]], V[f[1]], V[f[2]]
                        dist, n = _vertex_triangle_penetration(v, a, b, c)
                        if n is not None and dist < min_separation:
                            _resolve_pair(v, [a, b, c], n, dist - min_separation)
                            changed = True
        if not changed:
            break
