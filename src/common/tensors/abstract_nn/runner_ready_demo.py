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

from ....transmogrifier.ilpscheduler import ILPScheduler

from ..abstraction import AbstractTensor as AT
from .autograd_audit import AutogradAuditSession, AuditConfig
from .core import Linear, RectConv3d
from .activations import ReLU


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
    # Build a teacher/student for: Linear → Conv3D → Conv3D
    N, in_dim = 32, 8
    Cin, Cmid, Cout = 2, 3, 2
    vol_shape = (4, 4, 4)
    like = AT.get_tensor()
    X = AT.randn((N, in_dim), requires_grad=True)

    teacher = TinyLinearConv3DNet(in_dim=in_dim, vol_shape=vol_shape, Cin=Cin, Cmid=Cmid, Cout=Cout, like=like)
    student = TinyLinearConv3DNet(in_dim=in_dim, vol_shape=vol_shape, Cin=Cin, Cmid=Cmid, Cout=Cout, like=like)

    # Teacher target
    with AT.autograd.no_grad():
        Y_target = teacher.forward(X)

    # Capture composed whole (strict per AUTOGRAD_STRICT)
    session = AutogradAuditSession(student, X, Y_target, config=AuditConfig())
    session.capture()
    print("[capsule] Runner-ready bundle: OK")
    print_schedules("capsule", session)

    # Train until epsilon using eager path with strict checks per step
    session.train(epochs=2000, lr=1e-2, epsilon=1e-5, print_sched=False)


if __name__ == "__main__":
    main()

