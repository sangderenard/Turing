# -*- coding: utf-8 -*-
"""
PyTorch bridge for FluxSpring:
- GraphDataNet: a data-network nn.Module with (alpha,w,b) on edges & nodes.
- Converters to sync parameters between FluxSpringSpec and the module.
- Optional: geometry tensor packs (D0, D1, k, l0, alpha_face, c) in torch.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

from .fs_types import FluxSpringSpec

# --------- tiny helpers ---------
def _act(name: str):
    s = (name or "tanh").lower()
    if s == "relu":    return torch.relu
    if s == "sigmoid": return torch.sigmoid
    return torch.tanh

# --------- Data-network Torch module ---------
class GraphDataNet(nn.Module):
    """
    Forward:
        For node j: y_j = NodeNonlin_j( sum_{e: dst(e)=j} EdgeNonlin_e( x_{src(e)} ) )
    Inputs: x (N,) or (B,N)
    Params (learnability respected by gradient masks):
        edge_alpha, edge_w, edge_b  : (E,)
        node_alpha, node_w, node_b  : (N,)
    """
    def __init__(self, spec: FluxSpringSpec, *, activation: str = "tanh",
                 device=None, dtype=torch.float32):
        super().__init__()
        self.N = len(spec.nodes)
        self.E = len(spec.edges)
        self.act = _act(activation)

        self.src_idx = torch.tensor([e.src for e in spec.edges], dtype=torch.long, device=device)
        self.dst_idx = torch.tensor([e.dst for e in spec.edges], dtype=torch.long, device=device)

        e_alpha = torch.tensor([e.ctrl.alpha for e in spec.edges], dtype=dtype, device=device)
        e_w     = torch.tensor([e.ctrl.w     for e in spec.edges], dtype=dtype, device=device)
        e_b     = torch.tensor([e.ctrl.b     for e in spec.edges], dtype=dtype, device=device)
        n_alpha = torch.tensor([n.ctrl.alpha for n in spec.nodes], dtype=dtype, device=device)
        n_w     = torch.tensor([n.ctrl.w     for n in spec.nodes], dtype=dtype, device=device)
        n_b     = torch.tensor([n.ctrl.b     for n in spec.nodes], dtype=dtype, device=device)

        self.edge_alpha = nn.Parameter(e_alpha)
        self.edge_w     = nn.Parameter(e_w)
        self.edge_b     = nn.Parameter(e_b)
        self.node_alpha = nn.Parameter(n_alpha)
        self.node_w     = nn.Parameter(n_w)
        self.node_b     = nn.Parameter(n_b)

        # mask grads by learnability flags
        self._edge_mask = {
            "alpha": torch.tensor([1.0 if e.ctrl.learn.alpha else 0.0 for e in spec.edges], dtype=dtype, device=device),
            "w":     torch.tensor([1.0 if e.ctrl.learn.w     else 0.0 for e in spec.edges], dtype=dtype, device=device),
            "b":     torch.tensor([1.0 if e.ctrl.learn.b     else 0.0 for e in spec.edges], dtype=dtype, device=device),
        }
        self._node_mask = {
            "alpha": torch.tensor([1.0 if n.ctrl.learn.alpha else 0.0 for n in spec.nodes], dtype=dtype, device=device),
            "w":     torch.tensor([1.0 if n.ctrl.learn.w     else 0.0 for n in spec.nodes], dtype=dtype, device=device),
            "b":     torch.tensor([1.0 if n.ctrl.learn.b     else 0.0 for n in spec.nodes], dtype=dtype, device=device),
        }
        self.edge_alpha.register_hook(lambda g: g * self._edge_mask["alpha"])
        self.edge_w.register_hook(    lambda g: g * self._edge_mask["w"])
        self.edge_b.register_hook(    lambda g: g * self._edge_mask["b"])
        self.node_alpha.register_hook(lambda g: g * self._node_mask["alpha"])
        self.node_w.register_hook(    lambda g: g * self._node_mask["w"])
        self.node_b.register_hook(    lambda g: g * self._node_mask["b"])

    def _edge_nonlin(self, x_src: torch.Tensor) -> torch.Tensor:
        lin = self.edge_w * x_src + self.edge_b
        return (1.0 - self.edge_alpha) * lin + self.edge_alpha * self.act(lin)

    def _node_nonlin(self, agg: torch.Tensor) -> torch.Tensor:
        lin = self.node_w * agg + self.node_b
        return (1.0 - self.node_alpha) * lin + self.node_alpha * self.act(lin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x_src = x.index_select(0, self.src_idx)   # (E,)
            m = self._edge_nonlin(x_src)              # (E,)
            y = torch.zeros(self.N, dtype=x.dtype, device=x.device)
            y.index_add_(0, self.dst_idx, m)
            return self._node_nonlin(y)               # (N,)
        elif x.dim() == 2:
            x_src = x.index_select(1, self.src_idx)   # (B,E)
            m = self._edge_nonlin(x_src)              # (B,E)
            y = torch.zeros(x.shape[0], self.N, dtype=x.dtype, device=x.device)
            y.index_add_(1, self.dst_idx, m)
            return self._node_nonlin(y)               # (B,N)
        else:
            raise ValueError("x must be (N,) or (B,N)")

# --------- Converters (spec ↔ torch module) ---------
def to_torch_model(spec: FluxSpringSpec, *, activation: str = "tanh",
                   device=None, dtype=torch.float32) -> GraphDataNet:
    return GraphDataNet(spec, activation=activation, device=device, dtype=dtype)

def copy_spec_to_module(spec: FluxSpringSpec, module: GraphDataNet) -> None:
    with torch.no_grad():
        module.edge_alpha.copy_(torch.tensor([e.ctrl.alpha for e in spec.edges], dtype=module.edge_alpha.dtype, device=module.edge_alpha.device))
        module.edge_w.copy_(    torch.tensor([e.ctrl.w     for e in spec.edges], dtype=module.edge_w.dtype,     device=module.edge_w.device))
        module.edge_b.copy_(    torch.tensor([e.ctrl.b     for e in spec.edges], dtype=module.edge_b.dtype,     device=module.edge_b.device))
        module.node_alpha.copy_(torch.tensor([n.ctrl.alpha for n in spec.nodes], dtype=module.node_alpha.dtype, device=module.node_alpha.device))
        module.node_w.copy_(    torch.tensor([n.ctrl.w     for n in spec.nodes], dtype=module.node_w.dtype,     device=module.node_w.device))
        module.node_b.copy_(    torch.tensor([n.ctrl.b     for n in spec.nodes], dtype=module.node_b.dtype,     device=module.node_b.device))

def copy_module_to_spec(module: GraphDataNet, spec: FluxSpringSpec) -> None:
    ea = module.edge_alpha.detach().cpu().tolist()
    ew = module.edge_w.detach().cpu().tolist()
    eb = module.edge_b.detach().cpu().tolist()
    na = module.node_alpha.detach().cpu().tolist()
    nw = module.node_w.detach().cpu().tolist()
    nb = module.node_b.detach().cpu().tolist()
    for e, a, w, b in zip(spec.edges, ea, ew, eb):
        e.ctrl.alpha = float(a); e.ctrl.w = float(w); e.ctrl.b = float(b)
    for n, a, w, b in zip(spec.nodes, na, nw, nb):
        n.ctrl.alpha = float(a); n.ctrl.w = float(w); n.ctrl.b = float(b)

# Optional geometry tensors for torch-side diagnostics or training
def geometry_tensors_to_torch(spec: FluxSpringSpec, *, dtype=torch.float32, device=None):
    D0 = torch.tensor(spec.dec.D0, dtype=dtype, device=device)
    D1 = torch.tensor(spec.dec.D1, dtype=dtype, device=device)
    k  = torch.tensor([e.hooke.k for e in spec.edges], dtype=dtype, device=device)
    l0 = torch.tensor([e.hooke.l0 for e in spec.edges], dtype=dtype, device=device)
    alpha_face = torch.tensor([f.alpha for f in spec.faces], dtype=dtype, device=device)
    c_face     = torch.tensor([f.c     for f in spec.faces], dtype=dtype, device=device)
    return D0, D1, k, l0, alpha_face, c_face

def geometry_tensors_from_torch(spec: FluxSpringSpec, D0: torch.Tensor, D1: torch.Tensor,
                                k: torch.Tensor, l0: torch.Tensor, alpha_face: torch.Tensor, c_face: torch.Tensor) -> None:
    spec.dec.D0 = D0.detach().cpu().tolist()
    spec.dec.D1 = D1.detach().cpu().tolist()
    kv = k.detach().cpu().tolist(); l0v = l0.detach().cpu().tolist()
    av = alpha_face.detach().cpu().tolist(); cv = c_face.detach().cpu().tolist()
    for e, kv_i, l0_i in zip(spec.edges, kv, l0v):
        e.hooke.k = float(kv_i); e.hooke.l0 = float(l0_i)
    for f, a_i, c_i in zip(spec.faces, av, cv):
        f.alpha = float(a_i); f.c = float(c_i)
