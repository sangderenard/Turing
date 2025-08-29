"""
runner_ready_demo.py (thin)
---------------------------

Thin demo that wires a model into the autograd auditing chain using
AutogradAuditSession. It demonstrates:

- Capture of the composed whole (forward+backward graphs)
- Optional strict-mode diagnostics prior to grad
- An eager training loop with per-step strict checks and early stopping
"""

from __future__ import annotations

from typing import Any, Dict
import argparse

from ....transmogrifier.ilpscheduler import ILPScheduler

from ..abstraction import AbstractTensor as AT
from .autograd_audit import AutogradAuditSession, AuditConfig
from .core import Linear, RectConv3d
from .activations import ReLU
from ..abstract_convolution.ndpca3conv import NDPCA3Conv3d


class TinyLinearConv3DNet:
    """Linear → reshape → Conv3D(3x3x3) → ReLU → Conv3D(1x1x1)."""

    def __init__(
        self,
        *,
        in_dim: int,
        vol_shape: tuple[int, int, int] = (4, 4, 4),
        Cin: int = 2,
        Cmid: int = 3,
        Cout: int = 2,
        like: AT | None = None,
    ) -> None:
        if like is None:
            like = AT.get_tensor()
        D, H, W = vol_shape
        self.in_dim = in_dim
        self.Cin, self.Cmid, self.Cout = Cin, Cmid, Cout
        self.vol_shape = vol_shape
        self.linear = Linear(in_dim, Cin * D * H * W, like=like, init="xavier")
        self.act0 = ReLU()
        self.conv1 = RectConv3d(Cin, Cmid, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), like=like, bias=True)
        self.act1 = ReLU()
        self.conv2 = RectConv3d(Cmid, Cout, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), like=like, bias=True)

    def parameters(self):
        params = []
        params.extend(self.linear.parameters())
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        return params

    def forward(self, X: AT) -> AT:
        N = X.shape[0]
        D, H, W = self.vol_shape
        Z = self.act0(self.linear.forward(X))
        V = Z.reshape(N, self.Cin, D, H, W)
        H1 = self.act1(self.conv1.forward(V))
        Yv = self.conv2.forward(H1)
        return Yv


class TinyLinearPCAConv3DNet:
    """Linear → reshape → NDPCA3Conv3d (metric-steered) → ReLU → 1x1 RectConv3d.

    Matches the tensor shapes of TinyLinearConv3DNet so we can compare fairly.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        vol_shape: tuple[int, int, int] = (4, 4, 4),
        Cin: int = 2,
        Cmid: int = 3,
        Cout: int = 2,
        like: AT | None = None,
    ) -> None:
        if like is None:
            like = AT.get_tensor()
        D, H, W = vol_shape
        self.in_dim = in_dim
        self.Cin, self.Cmid, self.Cout = Cin, Cmid, Cout
        self.vol_shape = vol_shape
        self.linear = Linear(in_dim, Cin * D * H * W, like=like, init="xavier")
        self.act0 = ReLU()
        # PCA conv in the middle; keep pointwise enabled to handle channel change
        self.pca = NDPCA3Conv3d(
            in_channels=Cin,
            out_channels=Cmid,
            like=like,
            grid_shape=vol_shape,
            pointwise=True,
            _label_prefix="RunnerDemo",
        )
        self.act1 = ReLU()
        self.conv2 = RectConv3d(Cmid, Cout, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), like=like, bias=True)

    def parameters(self):
        params = []
        params.extend(self.linear.parameters())
        params.extend(self.pca.parameters())
        params.extend(self.conv2.parameters())
        return params

    def forward(self, X: AT) -> AT:
        N = X.shape[0]
        D, H, W = self.vol_shape
        Z = self.act0(self.linear.forward(X))
        V = Z.reshape(N, self.Cin, D, H, W)
        # Identity metric per voxel for this demo
        I = AT.eye(3, batch_shape=(D, H, W))
        package = {"metric": {"g": I, "inv_g": I}}
        H1 = self.act1(self.pca.forward(V, package=package))
        Yv = self.conv2.forward(H1)
        return Yv


def print_schedules(title: str, session: AutogradAuditSession) -> None:
    if session.proc is None or session.proc.forward_graph is None:
        print(f"[{title}] No captured graph")
        return
    levels = {}
    for nid, data in session.proc.forward_graph.nodes(data=True):
        lvl = data.get("level")
        if lvl is not None:
            levels.setdefault(lvl, 0)
            levels[lvl] += 1
    print(f"\n[{title}] ASAP bands:")
    for lvl in sorted(levels):
        print(f"  L{lvl}: {levels[lvl]} nodes")


def main():
    parser = argparse.ArgumentParser(description="Runner-ready demo with selectable layers")
    parser.add_argument("--choice", type=str, default=None, help="Preset choice: linear_conv3d | linear_pcaconv3d")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    args = parser.parse_args()

    # Build a teacher/student with an interactive selection
    N, in_dim = 32, 8
    Cin, Cmid, Cout = 2, 3, 2
    vol_shape = (4, 4, 4)
    like = AT.get_tensor()
    X = AT.randn((N, in_dim), requires_grad=True)

    def _menu() -> str:
        print("Select model variant:")
        print("  1) Linear → Conv3D → Conv3D  [baseline]")
        print("  2) Linear → NDPCA3Conv3d → Conv3D(1x1)  [PCA conv]")
        s = input("> ").strip()
        return {"1": "linear_conv3d", "2": "linear_pcaconv3d"}.get(s, "linear_conv3d")

    choice = (args.choice or _menu()).lower()
    if choice not in {"linear_conv3d", "linear_pcaconv3d"}:
        print(f"Unknown choice '{choice}', defaulting to baseline")
        choice = "linear_conv3d"

    if choice == "linear_pcaconv3d":
        teacher = TinyLinearPCAConv3DNet(in_dim=in_dim, vol_shape=vol_shape, Cin=Cin, Cmid=Cmid, Cout=Cout, like=like)
        student = TinyLinearPCAConv3DNet(in_dim=in_dim, vol_shape=vol_shape, Cin=Cin, Cmid=Cmid, Cout=Cout, like=like)
    else:
        teacher = TinyLinearConv3DNet(in_dim=in_dim, vol_shape=vol_shape, Cin=Cin, Cmid=Cmid, Cout=Cout, like=like)
        student = TinyLinearConv3DNet(in_dim=in_dim, vol_shape=vol_shape, Cin=Cin, Cmid=Cmid, Cout=Cout, like=like)

    # Teacher target
    with AT.autograd.no_grad():
        Y_target = teacher.forward(X)

    # Capture composed whole (strict per AUTOGRAD_STRICT)
    session = AutogradAuditSession(student, X, Y_target, config=AuditConfig())
    session.capture()
    print(f"[capsule] Variant='{choice}' Runner-ready bundle: OK")
    print_schedules("capsule", session)

    # Train until epsilon using eager path with strict checks per step
    session.train(epochs=int(args.epochs), lr=float(args.lr), epsilon=float(args.epsilon), print_sched=False)


if __name__ == "__main__":
    main()
