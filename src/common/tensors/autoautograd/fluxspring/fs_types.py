# -*- coding: utf-8 -*-
"""
FluxSpring graph types (JSON-faithful), independent of torch.
Ambient math elsewhere uses AbstractTensor.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from ...abstraction import AbstractTensor as AT

# ----- learning switches shared by node/edge controls -----
@dataclass
class LearnCtrl:
    alpha: bool = True
    w: bool = True
    b: bool = True

# ----- node control (data path) -----
@dataclass
class NodeCtrl:
    alpha: AT = field(default_factory=lambda: AT.tensor(0.0))
    w: AT = field(default_factory=lambda: AT.tensor(1.0))
    b: AT = field(default_factory=lambda: AT.tensor(0.0))
    learn: LearnCtrl = field(default_factory=LearnCtrl)

# ----- edge transport params + learnability -----
@dataclass
class EdgeTransportLearn:
    kappa: bool = True
    k: bool = True
    l0: bool = True
    lambda_s: bool = True
    x: bool = True

@dataclass
class EdgeTransport:
    kappa: AT = field(default_factory=lambda: AT.tensor(1.0))
    k: Optional[AT] = None
    l0: Optional[AT] = None
    lambda_s: Optional[AT] = None
    x: Optional[AT] = None
    learn: EdgeTransportLearn = field(default_factory=EdgeTransportLearn)

# ----- edge control (data path) -----
@dataclass
class EdgeCtrl:
    alpha: AT = field(default_factory=lambda: AT.tensor(0.0))
    w: AT = field(default_factory=lambda: AT.tensor(1.0))
    b: AT = field(default_factory=lambda: AT.tensor(0.0))
    learn: LearnCtrl = field(default_factory=LearnCtrl)

# ----- node/edge/face specs -----
@dataclass
class NodeSpec:
    id: int
    p0: AT                    # length D
    v0: AT                    # length D
    mass: AT
    ctrl: NodeCtrl
    scripted_axes: List[int]   # exactly 2 axes (Dirichlet/scripted)
    temperature: AT = field(default_factory=lambda: AT.tensor(0.0))  # placeholder for thermal models
    exclusive: bool = False  # True if node occupies exclusive geometry
    ring_size: Optional[int] = None
    ring: Optional[AT] = None
    ring_idx: int = 0

    def __post_init__(self) -> None:
        self.ensure_ring_buffer()

    def ensure_ring_buffer(self) -> None:
        """Allocate the ring buffer if a size is set."""
        if self.ring_size and self.ring_size > 0 and self.ring is None:
            D = int(AT.get_tensor(self.p0).shape[0])
            self.ring = AT.zeros((self.ring_size, D), dtype=float)
            self.ring_idx = 0

    def push_ring(self, val: AT) -> AT:
        """Insert ``val`` into the ring buffer and return the updated buffer.

        Uses :func:`AT.scatter_row` so the operation participates in the
        autograd tape instead of an in-place Python assignment.  The returned
        tensor remains connected to the computation graph.
        """
        if self.ring is None:
            raise RuntimeError("ring buffer not allocated")

        i = self.ring_idx % int(len(self.ring))
        self.ring = AT.scatter_row(self.ring.clone(), i, val, dim=0)
        self.ring_idx = i + 1
        return self.ring

@dataclass
class EdgeSpec:
    src: int
    dst: int
    transport: EdgeTransport
    ctrl: EdgeCtrl
    temperature: AT = field(default_factory=lambda: AT.tensor(0.0))  # placeholder for thermal models
    exclusive: bool = False  # True if edge occupies exclusive geometry
    ring_size: Optional[int] = None
    ring: Optional[AT] = None
    ring_idx: int = 0

    def __post_init__(self) -> None:
        self.ensure_ring_buffer()

    def ensure_ring_buffer(self) -> None:
        """Allocate the ring buffer if a size is set."""
        if self.ring_size and self.ring_size > 0 and self.ring is None:
            self.ring = AT.zeros(self.ring_size, dtype=float)
            self.ring_idx = 0

    def push_ring(self, val: AT) -> AT:
        """Insert ``val`` into the ring buffer and return the updated buffer.

        ``AT.scatter_row`` performs the update using differentiable tensor
        operations so gradients flow back to ``val`` when the ring is involved
        in later computations.
        """
        if self.ring is None:
            raise RuntimeError("ring buffer not allocated")

        i = self.ring_idx % int(len(self.ring))
        self.ring = AT.scatter_row(self.ring.clone(), i, val, dim=0)
        self.ring_idx = i + 1
        return self.ring

@dataclass
class FaceLearn:
    alpha: bool = True
    c: bool = True

@dataclass
class FaceSpec:
    edges: List[int]           # oriented: 1-based edge indices, negative flips orientation
    alpha: AT                  # activation wet/dry (mix)
    c: AT                      # face weight
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

# ----- spectral feature extraction -----
@dataclass
class SpectralMetrics:
    bands: List[List[float]] = field(default_factory=list)  # [ [f_lo, f_hi], ... ]
    centroid: bool = False
    flatness: bool = False
    coherence: bool = False


@dataclass
class SpectralCfg:
    enabled: bool = False
    tick_hz: float = 44100.0
    win_len: int = 1024
    hop_len: int = 256
    window: str = "hann"
    metrics: SpectralMetrics = field(default_factory=SpectralMetrics)

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
    spectral: SpectralCfg = field(default_factory=SpectralCfg)
    rho: AT = field(default_factory=lambda: AT.tensor(0.0))
    beta: AT = field(default_factory=lambda: AT.tensor(0.0))
    gamma: AT = field(default_factory=lambda: AT.tensor(0.0))

    def __post_init__(self) -> None:
        default_size = self.spectral.win_len if self.spectral.enabled else None
        for n in self.nodes:
            if n.ring_size is None:
                n.ring_size = default_size
            n.ensure_ring_buffer()
        for e in self.edges:
            if e.ring_size is None:
                e.ring_size = default_size
            e.ensure_ring_buffer()
