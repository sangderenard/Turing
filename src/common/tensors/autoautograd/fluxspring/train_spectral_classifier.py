# -*- coding: utf-8 -*-
"""
Train GraphDataNet (via FluxSpring spec) to classify the BAND CENTER
of a synthetic spectral-noise snapshot.

Pipeline
--------
1) Build FluxSpringSpec (AT-native) using make_classifier_spec(...)
2) Convert spec -> PyTorch GraphDataNet (data path only)
3) Generate synthetic spectra with a Gaussian band centered on one of C bins
4) Cross-entropy training on logits read from the 'output nodes'
5) Copy trained params back into the spec and save (round-trip I/O)
6) (Optional) Validate DEC: ||D1@D0||=0 using AbstractTensor

Run
---
  python -m examples.train_spectral_classifier \
        --M 128 --H 96 --C 8 --epochs 8 --device cpu

Notes
-----
- Torch is used only for the module + training. The graph & DEC routines remain AT.
- Outputs are the last C nodes by construction of `make_classifier_spec`.
"""
from __future__ import annotations
import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
from .fs_build_specs import make_classifier_spec, io_summary
from .fs_io import save_fluxspring, load_fluxspring, validate_fluxspring
from .fs_torch_bridge import to_torch_model, copy_module_to_spec, copy_spec_to_module
from .fs_dec import validate_boundary_of_boundary_AT
from .fs_types import FluxSpringSpec

# ---------- synthetic spectral dataset ----------
class SpectralNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, M: int, C: int, N: int, *, sigma_bins: float = 3.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        centers = torch.linspace(0, M - 1, steps=C)
        self.M, self.C, self.N = M, C, N
        self.centers = centers

        X = torch.empty(N, M)
        y = torch.empty(N, dtype=torch.long)

        k = torch.arange(M)  # reuse inside the loop
        for n in range(N):
            cls = torch.randint(low=0, high=C, size=(1,), generator=g).item()
            c = centers[cls]
            bump = torch.exp(-0.5 * ((k - c) / sigma_bins) ** 2)
            white = torch.randn(M, generator=g).abs()
            r = (0.2 + 0.8 * torch.rand((), generator=g)).item()  # U[0.2,1.0]
            slope = torch.pow(torch.linspace(1.0, 2.0, M), -r)

            s = (0.7 * bump + 0.3 * white) * slope
            x = torch.log1p(s)
            x = (x - x.mean()) / (x.std() + 1e-6)

            X[n] = x
            y[n] = cls

        self.X, self.y = X, y

    def __len__(self): return self.N
    def __getitem__(self, idx): return self.X[idx], self.y[idx]



# ---------- utilities for node indexing ----------
def split_indices(M: int, H: int, C: int):
    in_ids   = list(range(0, M))
    hid_ids  = list(range(M, M+H))
    out_ids  = list(range(M+H, M+H+C))
    return in_ids, hid_ids, out_ids

def pack_node_input(x_feat: torch.Tensor, N_total: int, in_ids):
    """
    x_feat: (M,) or (B,M) → scatter into a graph-wide input vector (N,) / (B,N)
    """
    if x_feat.dim() == 1:
        N = torch.zeros(N_total, dtype=x_feat.dtype, device=x_feat.device)
        N[in_ids] = x_feat
        return N
    else:
        B = x_feat.shape[0]
        out = torch.zeros(B, N_total, dtype=x_feat.dtype, device=x_feat.device)
        out[:, in_ids] = x_feat
        return out

# ---------- training ----------
def train_loop(
    model: nn.Module,
    spec: FluxSpringSpec,
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    *,
    in_ids, out_ids,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 3e-3,
    device: str = "cpu",
) -> Tuple[float, float]:
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    # --- per-epoch stats helpers ---
    def _tensor_stats(x: torch.Tensor):
        v = x.detach().float().view(-1)
        if v.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "l2": 0.0}
        return {
            "mean": float(v.mean().item()),
            "std":  float(v.std(unbiased=False).item()),
            "min":  float(v.min().item()),
            "max":  float(v.max().item()),
            "l2":   float(v.norm().item()),
        }

    class RunningStats:
        def __init__(self):
            self.n = 0
            self.sum = 0.0
            self.sumsq = 0.0
            self.min = float("inf")
            self.max = float("-inf")
        def update(self, g: torch.Tensor | None):
            if g is None:
                return
            v = g.detach().float().view(-1)
            if v.numel() == 0:
                return
            self.n += int(v.numel())
            s = float(v.sum().item())
            self.sum += s
            sq = float((v * v).sum().item())
            self.sumsq += sq
            mn = float(v.min().item()); mx = float(v.max().item())
            if mn < self.min: self.min = mn
            if mx > self.max: self.max = mx
        def summary(self):
            if self.n == 0:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "l2": 0.0}
            mean = self.sum / self.n
            var = max(0.0, self.sumsq / self.n - mean * mean)
            import math
            return {"mean": mean, "std": math.sqrt(var), "min": self.min, "max": self.max, "l2": math.sqrt(self.sumsq)}

    def eval_split(ds):
        model.eval()
        tot, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False):
                xb = xb.to(device); yb = yb.to(device)
                Xg = pack_node_input(xb, model.N, in_ids)
                logits_all = model(Xg)           # (B,N)
                logits = logits_all[:, out_ids]  # (B,C)
                loss = loss_fn(logits, yb)
                loss_sum += float(loss.item()) * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                tot += xb.size(0)
        return loss_sum / max(1, tot), correct / max(1, tot)

    for ep in range(1, epochs+1):
        model.train()
        # per-epoch gradient accumulators
        gstats = {
            "edge.alpha": RunningStats(),
            "edge.w":     RunningStats(),
            "edge.b":     RunningStats(),
            "node.alpha": RunningStats(),
            "node.w":     RunningStats(),
            "node.b":     RunningStats(),
        }

        dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)

            Xg = pack_node_input(xb, model.N, in_ids)
            logits_all = model(Xg)           # (B,N)
            logits = logits_all[:, out_ids]  # (B,C)
            loss = loss_fn(logits, yb)

            # tiny weight/bias L2 from spec.regularizers (data-path only)
            l2 = torch.tensor(0.0, device=device)
            if spec.regularizers:
                lam_w = spec.regularizers.lambda_w or 0.0
                lam_b = spec.regularizers.lambda_b or 0.0
                if lam_w > 0:
                    l2 = l2 + lam_w * (model.edge_w.pow(2).mean() + model.node_w.pow(2).mean())
                if lam_b > 0:
                    l2 = l2 + lam_b * (model.edge_b.pow(2).mean() + model.node_b.pow(2).mean())

            (loss + l2).backward()

            # accumulate grad stats this epoch
            gstats["edge.alpha"].update(model.edge_alpha.grad)
            gstats["edge.w"].update(model.edge_w.grad)
            gstats["edge.b"].update(model.edge_b.grad)
            gstats["node.alpha"].update(model.node_alpha.grad)
            gstats["node.w"].update(model.node_w.grad)
            gstats["node.b"].update(model.node_b.grad)

            opt.step()

        tr_loss, tr_acc = eval_split(train_ds)
        va_loss, va_acc = eval_split(val_ds)
        print(f"[ep {ep:02d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        # per-epoch parameter + grad summaries
        def fmt(d): return f"mean={d['mean']:.4e} std={d['std']:.4e} min={d['min']:.4e} max={d['max']:.4e} l2={d['l2']:.4e}"
        stats_lines = [
            ("edges.alpha", _tensor_stats(model.edge_alpha), gstats["edge.alpha"].summary()),
            ("edges.w    ", _tensor_stats(model.edge_w),     gstats["edge.w"].summary()),
            ("edges.b    ", _tensor_stats(model.edge_b),     gstats["edge.b"].summary()),
            ("nodes.alpha", _tensor_stats(model.node_alpha), gstats["node.alpha"].summary()),
            ("nodes.w    ", _tensor_stats(model.node_w),     gstats["node.w"].summary()),
            ("nodes.b    ", _tensor_stats(model.node_b),     gstats["node.b"].summary()),
        ]
        print(f"[ep {ep:02d}] param/grad stats:")
        for name, pstat, gstat in stats_lines:
            print(f"  {name}: param {fmt(pstat)} | grad {fmt(gstat)}")

    return va_loss, va_acc

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save", type=str, default="spectral_fluxspring.json")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sigma-bins", type=float, default=3.0)
    args = ap.parse_args()

    # 1) Build spec (AT-native)
    spec = make_classifier_spec("spectral-demo", M=args.M, H=args.H, C=args.C, D_geom=3)
    print(io_summary(spec))
    validate_fluxspring(spec)  # structural checks

    # 2) PyTorch module from spec (data network only)
    model = to_torch_model(spec, activation="tanh", device=args.device)
    copy_spec_to_module(spec, model)  # (no-op here but shows direction)

    # 3) Dataset
    tr = SpectralNoiseDataset(args.M, args.C, args.train, sigma_bins=args.sigma_bins, seed=args.seed)
    va = SpectralNoiseDataset(args.M, args.C, args.val, sigma_bins=args.sigma_bins, seed=args.seed + 1)
    in_ids, hid_ids, out_ids = split_indices(args.M, args.H, args.C)

    # 4) Train
    val_loss, val_acc = train_loop(
        model, spec, tr, va,
        in_ids=in_ids, out_ids=out_ids,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=args.device
    )
    print(f"[final] val loss {val_loss:.4f} acc {val_acc:.3f}")

    # 5) Copy trained params back into spec and save
    copy_module_to_spec(model, spec)
    save_fluxspring(spec, args.save)
    print(f"Saved trained FluxSpring spec → {args.save}")

    # 6) (Optional) DEC validator in AT
    try:
        _ = validate_boundary_of_boundary_AT(spec, tol=1e-12)
        print("DEC check: ||D1@D0|| = 0 within tolerance.")
    except Exception as e:
        print("DEC check failed:", e)

    # Sanity: reload & echo summary
    spec2 = load_fluxspring(args.save)
    print(io_summary(spec2))

if __name__ == "__main__":
    main()
