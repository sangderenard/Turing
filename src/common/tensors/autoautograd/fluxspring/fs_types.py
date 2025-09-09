# -*- coding: utf-8 -*-
"""
FluxSpring graph types (JSON-faithful), independent of torch.
Ambient math elsewhere uses AbstractTensor.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

# ----- learning switches shared by node/edge controls -----
@dataclass
class LearnCtrl:
    alpha: bool = True
    w: bool = True
    b: bool = True

# ----- node control (data path) -----
@dataclass
class NodeCtrl:
    alpha: float = 0.0
    w: float = 1.0
    b: float = 0.0
    learn: LearnCtrl = field(default_factory=LearnCtrl)

# ----- edge Hooke params + learnability -----
@dataclass
class EdgeHookeLearn:
    k: bool = True
    l0: bool = True

@dataclass
class EdgeHooke:
    k: float = 1.0
    l0: float = 1.0
    learn: EdgeHookeLearn = field(default_factory=EdgeHookeLearn)

# ----- edge control (data path) -----
@dataclass
class EdgeCtrl:
    alpha: float = 0.0
    w: float = 1.0
    b: float = 0.0
    learn: LearnCtrl = field(default_factory=LearnCtrl)

# ----- node/edge/face specs -----
@dataclass
class NodeSpec:
    id: int
    p0: List[float]            # length D
    v0: List[float]            # length D
    mass: float
    ctrl: NodeCtrl
    scripted_axes: List[int]   # exactly 2 axes (Dirichlet/scripted)

@dataclass
class EdgeSpec:
    src: int
    dst: int
    hooke: EdgeHooke
    ctrl: EdgeCtrl

@dataclass
class FaceLearn:
    alpha: bool = True
    c: bool = True

@dataclass
class FaceSpec:
    edges: List[int]           # oriented: 1-based edge indices, negative flips orientation
    alpha: float               # activation wet/dry (mix)
    c: float                   # face weight
    learn: FaceLearn = field(default_factory=FaceLearn)

# ----- DEC + BC + regularizers -----
@dataclass
class DirichletCfg:
    window: Optional[int] = None    # moving-window size; if None, use EMA
    ema_beta: Optional[float] = 0.9 # EMA rate (ignored if window is set)
    gain: float = 3.0               # Dirichlet spring gain on scripted axes

@dataclass
class DECSpec:
    D0: List[List[float]]      # (E,N) edge-node incidence
    D1: List[List[float]]      # (F,E) face-edge incidence

@dataclass
class RegCfg:
    lambda_phi: float = 0.0
    mu_smooth: float = 0.0
    lambda_l0: float = 0.0
    lambda_b: float = 0.0
    lambda_c: float = 0.0
    lambda_w: float = 0.0

# ----- top-level FluxSpring spec -----
@dataclass
class FluxSpringSpec:
    version: str
    D: int
    nodes: List[NodeSpec]
    edges: List[EdgeSpec]
    faces: List[FaceSpec]
    dec: DECSpec
    dirichlet: Optional[DirichletCfg] = None
    regularizers: Optional[RegCfg] = None
