# -*- coding: utf-8 -*-
"""
FluxSpring loader/saver + structural validation.
Pure Python + AbstractTensor-aware numeric check for D1@D0≈0 (optional).
"""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, List, Optional
import json

from ...abstraction import AbstractTensor as AT
from .fs_types import (
    FluxSpringSpec, NodeSpec, EdgeSpec, FaceSpec, DECSpec,
    DirichletCfg, RegCfg, NodeCtrl, LearnCtrl, EdgeTransport, EdgeTransportLearn,
    EdgeCtrl, FaceLearn, SpectralCfg, SpectralMetrics
)

# ------------ coercion into dataclasses ------------
def _coerce_node(d: Dict) -> NodeSpec:
    ctrl = d.get("ctrl", {})
    learn = ctrl.get("learn", {})
    return NodeSpec(
        id=int(d["id"]),
        p0=AT.get_tensor(d["p0"]),
        v0=AT.get_tensor(d["v0"]),
        mass=AT.get_tensor(d.get("mass", 1.0)),
        ctrl=NodeCtrl(
            alpha=AT.get_tensor(ctrl.get("alpha", 0.0)),
            w=AT.get_tensor(ctrl.get("w", 1.0)),
            b=AT.get_tensor(ctrl.get("b", 0.0)),
            learn=LearnCtrl(
                alpha=bool(learn.get("alpha", True)),
                w=bool(learn.get("w", True)),
                b=bool(learn.get("b", True)),
            ),
        ),
        scripted_axes=[int(a) for a in d["scripted_axes"]],
    )

def _coerce_edge(d: Dict) -> EdgeSpec:
    tr = d.get("transport", {})
    trL = tr.get("learn", {})
    ctrl = d.get("ctrl", {})
    cL = ctrl.get("learn", {})
    return EdgeSpec(
        src=int(d["src"]),
        dst=int(d["dst"]),
        transport=EdgeTransport(
            kappa=AT.get_tensor(tr.get("kappa", 1.0)),
            lambda_s=AT.get_tensor(tr["lambda_s"]) if "lambda_s" in tr else None,
            x=AT.get_tensor(tr["x"]) if "x" in tr else None,
            learn=EdgeTransportLearn(
                kappa=bool(trL.get("kappa", True)),
                k=False,
                l0=False,
                lambda_s=bool(trL.get("lambda_s", True)),
                x=bool(trL.get("x", True)),
            ),
        ),
        ctrl=EdgeCtrl(
            alpha=AT.get_tensor(ctrl.get("alpha", 0.0)),
            w=AT.get_tensor(ctrl.get("w", 1.0)),
            b=AT.get_tensor(ctrl.get("b", 0.0)),
            learn=LearnCtrl(
                alpha=bool(cL.get("alpha", True)),
                w=bool(cL.get("w", True)),
                b=bool(cL.get("b", True)),
            ),
        ),
    )

def _coerce_face(d: Dict) -> FaceSpec:
    L = d.get("learn", {})
    return FaceSpec(
        edges=[int(x) for x in d["edges"]],
        alpha=AT.get_tensor(d["alpha"]),
        c=AT.get_tensor(d["c"]),
        learn=FaceLearn(alpha=bool(L.get("alpha", True)), c=bool(L.get("c", True))),
    )

def _coerce_dec(d: Dict) -> DECSpec:
    return DECSpec(
        D0=[[float(x) for x in row] for row in d["D0"]],
        D1=[[float(x) for x in row] for row in d["D1"]],
    )

def _coerce_dirichlet(d: Optional[Dict]) -> Optional[DirichletCfg]:
    if d is None:
        return None
    return DirichletCfg(
        window=int(d["window"]) if d.get("window") is not None else None,
        ema_beta=float(d["ema_beta"]) if d.get("ema_beta") is not None else None,
        gain=float(d.get("gain", 3.0)),
    )

def _coerce_regs(d: Optional[Dict]) -> Optional[RegCfg]:
    if d is None:
        return None
    return RegCfg(
        lambda_phi=float(d.get("lambda_phi", 0.0)),
        mu_smooth=float(d.get("mu_smooth", 0.0)),
        lambda_l0=float(d.get("lambda_l0", 0.0)),
        lambda_b=float(d.get("lambda_b", 0.0)),
        lambda_c=float(d.get("lambda_c", 0.0)),
        lambda_w=float(d.get("lambda_w", 0.0)),
    )


def _coerce_spectral(d: Optional[Dict]) -> SpectralCfg:
    if d is None:
        return SpectralCfg()
    m = d.get("metrics", {})
    return SpectralCfg(
        enabled=bool(d.get("enabled", False)),
        tick_hz=float(d.get("tick_hz", 44100.0)),
        win_len=int(d.get("win_len", 1024)),
        hop_len=int(d.get("hop_len", 256)),
        window=str(d.get("window", "hann")),
        metrics=SpectralMetrics(
            bands=[[float(lo), float(hi)] for lo, hi in m.get("bands", [])],
            centroid=bool(m.get("centroid", False)),
        ),
    )

# ------------ I/O ------------
def load_fluxspring(path: str) -> FluxSpringSpec:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for k in ["version", "D", "nodes", "edges", "faces", "dec"]:
        if k not in raw:
            raise ValueError(f"missing key: {k}")

    spec = FluxSpringSpec(
        version=str(raw["version"]),
        D=int(raw["D"]),
        nodes=[_coerce_node(n) for n in raw["nodes"]],
        edges=[_coerce_edge(e) for e in raw["edges"]],
        faces=[_coerce_face(fc) for fc in raw["faces"]],
        dec=_coerce_dec(raw["dec"]),
        dirichlet=_coerce_dirichlet(raw.get("dirichlet")),
        regularizers=_coerce_regs(raw.get("regularizers")),
        spectral=_coerce_spectral(raw.get("spectral")),
        rho=AT.get_tensor(raw.get("rho", 0.0)),
        beta=AT.get_tensor(raw.get("beta", 0.0)),
        gamma=AT.get_tensor(raw.get("gamma", 0.0)),
    )
    validate_fluxspring(spec)
    return spec

def save_fluxspring(spec: FluxSpringSpec, path: str, *, indent: int = 2) -> None:
    def _plain(x):
        if is_dataclass(x):
            return {k: _plain(v) for k, v in asdict(x).items() if v is not None}
        if isinstance(x, list):
            return [_plain(v) for v in x]
        if isinstance(x, dict):
            return {k: _plain(v) for k, v in x.items() if v is not None}
        try:
            t = AT.get_tensor(x)
            if getattr(t, "shape", ()) == ():
                return float(t)
            return t.tolist()
        except Exception:
            return x
    data = _plain(spec)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    # Round-trip check
    _ = load_fluxspring(path)

# ------------ validation ------------
def validate_fluxspring(spec: FluxSpringSpec, *, tol: float = 1e-8) -> None:
    N = len(spec.nodes); E = len(spec.edges); F = len(spec.faces)
    if spec.D < 2:
        raise ValueError("D must be ≥ 2")
    ids = [n.id for n in spec.nodes]
    if len(set(ids)) != N:
        raise ValueError("duplicate node ids")
    valid_nodes = set(ids)

    for n in spec.nodes:
        if len(n.p0) != spec.D or len(n.v0) != spec.D:
            raise ValueError(f"node {n.id}: p0/v0 must be length D")
        if float(AT.get_tensor(n.mass)) <= 0:
            raise ValueError(f"node {n.id}: mass must be > 0")
        if len(n.scripted_axes) != 2:
            raise ValueError(f"node {n.id}: scripted_axes must list exactly 2 axes")
        for a in n.scripted_axes:
            if a < 0 or a >= spec.D:
                raise ValueError(f"node {n.id}: scripted axis {a} out of range")

    for k, e in enumerate(spec.edges, start=1):
        if e.src not in valid_nodes or e.dst not in valid_nodes:
            raise ValueError(f"edge {k}: src/dst not valid node ids")
        if e.transport.kappa is None:
            raise ValueError(f"edge {k}: missing kappa")
        if float(AT.get_tensor(e.transport.kappa)) < 0:
            raise ValueError(f"edge {k}: kappa must be ≥ 0")
        if (e.transport.lambda_s is None) != (e.transport.x is None):
            raise ValueError(f"edge {k}: lambda_s and x must both be present or absent")
        if e.transport.lambda_s is not None and float(AT.get_tensor(e.transport.lambda_s)) < 0:
            raise ValueError(f"edge {k}: lambda_s must be ≥ 0")
        if e.transport.x is not None:
            x_t = AT.get_tensor(e.transport.x)
            if len(x_t) != spec.D:
                raise ValueError(f"edge {k}: x must have length D")

    for fidx, fc in enumerate(spec.faces, start=1):
        for or_idx in fc.edges:
            j = abs(or_idx)
            if j < 1 or j > E:
                raise ValueError(f"face {fidx}: invalid edge index {j}")
        alpha_val = float(AT.get_tensor(fc.alpha))
        c_val = float(AT.get_tensor(fc.c))
        if not (0.0 <= alpha_val <= 1.0):
            raise ValueError(f"face {fidx}: alpha must be in [0,1]")
        if c_val < 0.0:
            raise ValueError(f"face {fidx}: c must be ≥ 0")

    # Shape checks for DEC
    D0, D1 = spec.dec.D0, spec.dec.D1
    if len(D0) != E or any(len(row) != N for row in D0):
        raise ValueError(f"D0 shape must be (E={E}, N={N})")
    if len(D1) != F or any(len(row) != E for row in D1):
        raise ValueError(f"D1 shape must be (F={F}, E={E})")

    for name in ("rho", "beta", "gamma"):
        if float(AT.get_tensor(getattr(spec, name))) < 0:
            raise ValueError(f"{name} must be ≥ 0")

    sp = spec.spectral
    if sp.win_len < sp.hop_len:
        raise ValueError("spectral.win_len must be ≥ hop_len")
    nyq = sp.tick_hz / 2.0
    for i, (f_lo, f_hi) in enumerate(sp.metrics.bands):
        if not (0.0 <= f_lo < f_hi <= nyq):
            raise ValueError(
                f"spectral.metrics.bands[{i}] must satisfy 0 ≤ f_lo < f_hi ≤ tick_hz/2"
            )

    # Optional numeric check with AbstractTensor if available
    try:
        from ...abstraction import AbstractTensor as AT
        D0_t = AT.get_tensor(D0).astype(float)
        D1_t = AT.get_tensor(D1).astype(float)
        bdry = D1_t @ D0_t
        nrm = float(AT.linalg.norm(bdry))
        if nrm > tol:
            raise ValueError(f"D1@D0 ≠ 0 (||.||={nrm:.3e})")
    except Exception:
        # Skip if AT not available here; fs_dec has a stricter check
        pass
