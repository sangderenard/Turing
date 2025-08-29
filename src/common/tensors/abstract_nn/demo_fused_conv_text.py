"""
demo_fused_conv_text.py
-----------------------

Simple interactive demo using the FusedProgram + IRGraphedModel wrapper.

Model:
- Input: 30-length byte-like vector (converted to AbstractTensor)
- Hidden: NDPCA3Conv3d with identity metric over a (D,H, W) = (3, 2, 5) grid
- Output: 30-length vector flattened back for byte decoding

Training:
- Uses IRGraphedModel interactive discriminator: decodes model output to UTF-8,
  prints it, and prompts for the expected output text per step.
- MSE loss between predicted 0..255 float vector and encoded expected text.

Run:
  python -m src.common.tensors.abstract_nn.demo_fused_conv_text

Then type expected outputs when prompted. Press Ctrl+C to exit.
"""

from __future__ import annotations

from typing import Any, Tuple
import argparse
import os

from .fused_program import IRGraphedModel
from .completion_training import CompletionTrainer
from ..abstraction import AbstractTensor as AT
from ..abstract_convolution.metric_steered_conv3d import MetricSteeredConv3DWrapper
from ..abstract_convolution.ndpca3transform import fit_metric_pca, PCANDTransform
from ..abstract_nn.core import Linear
from .activations import GELU, Sigmoid

class LinearStackModel:
    """Simple fallback: 30 → hidden → 30 via Linear layers only."""
    def __init__(self, like: AT, hidden_dim: int = 64):
        from ..abstract_nn.core import Linear, Model
        from .activations import GELU, Sigmoid
        self.model = Model(
            layers=[
                Linear(30, hidden_dim, like=like, init="xavier"),
                Linear(hidden_dim, 30, like=like, init="xavier"),
            ],
            activations=[GELU(), Sigmoid()]
        )
    def parameters(self):
        return self.model.parameters()
    def forward(self, x):
        return self.model.forward(x)


class ConvTextModel:
    """30 → metric‑aware conv over (D,H,W) → 30 using a real transform/grid.

    - Pre: Linear(30 → Cin·D·H·W), reshape to (B,Cin,D,H,W)
    - Metric‑steered conv: MetricSteeredConv3DWrapper with PCANDTransform
    - Post: Linear(Cout·D·H·W → 30)
    """

    def __init__(self, like: AT, grid_shape: Tuple[int, int, int] = (3, 2, 5), Cin: int = 1, Cout: int = 1) -> None:
        D, H, W = grid_shape
        self.grid_shape = grid_shape
        self.Cin, self.Cout = Cin, Cout

        # Define a simple intrinsic embedding φ(U,V,W) -> R^30 using polynomial + trig features.
        def _phi(U: AT, V: AT, W: AT) -> AT:
            pi = AT.get_tensor(3.141592653589793)
            feats = [
                U, V, W,
                U*V, V*W, W*U,
                U*U, V*V, W*W,
                (pi*U).sin(), (pi*U).cos(),
                (pi*V).sin(), (pi*V).cos(),
                (pi*W).sin(), (pi*W).cos(),
                U*V*W,
                (U+V+W), (U-V+W), (U+V-W),
                (U*U - V*V), (V*V - W*W), (W*W - U*U),
            ]
            # Pad/truncate to 30
            while len(feats) < 30:
                feats.append(AT.zeros_like(U))
            feats = feats[:30]
            return AT.stack(feats, dim=-1)

        # Fit PCA basis over the grid samples of φ to set up PCANDTransform
        U = AT.linspace(-1.0, 1.0, D).reshape(D, 1, 1) * AT.ones((1, H, W))
        V = AT.linspace(-1.0, 1.0, H).reshape(1, H, 1) * AT.ones((D, 1, W))
        Wg = AT.linspace(-1.0, 1.0, W).reshape(1, 1, W) * AT.ones((D, H, 1))
        u_nd = _phi(U, V, Wg).reshape(-1, 30)
        basis = fit_metric_pca(u_nd)
        self.transform = PCANDTransform(basis, phi_fn=_phi, d_visible=3)

        # Metric‑steered conv wrapper (builds the geometry package and runs NDPCA3Conv3d)
        self.wrapper = MetricSteeredConv3DWrapper(
            in_channels=Cin,
            out_channels=Cout,
            grid_shape=grid_shape,
            transform=self.transform,
            boundary_conditions=("dirichlet",) * 6,
            k=3,
            eig_from="g",
            pointwise=True,
        )

        # Pre/Post linear maps for 30 → conv → 30
        self.pre = Linear(30, Cin * D * H * W, like=like, init="xavier")
        self.post = Linear(Cout * D * H * W, 30, like=like, init="xavier")

        # Activations for spatial and output domains
        self.hidden_act = GELU()
        self.out_act = Sigmoid()

    def parameters(self):
        ps = []
        ps.extend(self.pre.parameters())
        ps.extend(self.wrapper.parameters())
        ps.extend(self.post.parameters())
        return ps

    def forward(self, x: Any) -> AT:
        t = AT.get_tensor(x).reshape(1, 30)
        D, H, W = self.grid_shape
        # Pre → (B,Cin,D,H,W)
        z = self.pre.forward(t)
        z = z.reshape(1, self.Cin, D, H, W)
        # Metric‑steered conv
        y = self.wrapper.forward(z)  # (1, Cout, D, H, W)
        y = self.hidden_act.forward(y)
        # Post → 30
        y = y.reshape(1, self.Cout * D * H * W)
        y = self.post.forward(y).reshape(30)
        y = self.out_act.forward(y) * 255.0
        return y
def run_eager_training_test(model, AT, args):
    print("--- Running pre-capture eager training test ---")
    from .optimizer import Adam
    from ..autograd import autograd
    eager_optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = lambda a, b: ((a - b) ** 2).mean()
    epsilon = 1e-4
    max_steps = 1000
    patience = 20
    patience_counter = patience
    last_loss = None
    best_loss = None
    inputs = AT.randn((args.input_len,), requires_grad=True)
    targets = AT.ones(args.input_len) * 65.0  # ASCII 'A'

    for i in range(max_steps):
        tape = autograd.tape
        tape._nodes.clear()
        tape.graph.clear()
        tape._op_index = 0
        tape._loss_tensor = None
        tape._loss_id = None
        params = model.parameters()
        keep = {id(p) for p in params}
        tape._tensor_refs = {tid: ref for tid, ref in tape._tensor_refs.items() if tid in keep}
        for p in params:
            tape.create_tensor_node(p)
        pred = model.forward(inputs)
        loss = loss_fn(pred, targets)
        try:
            loss_val = float(loss.item())
        except Exception:
            loss_val = float('nan')
        print(f"Eager step {i}, Loss: {loss_val}")

        # Early stopping: epsilon or patience
        if loss_val < epsilon:
            print(f"Eager training reached epsilon {epsilon} at step {i}")
            break
        if best_loss is None or loss_val < best_loss - 1e-8:
            best_loss = loss_val
            patience_counter = patience
            print(f"  [patience] New best loss {best_loss}. Patience reset to {patience}.")
        else:
            patience_counter -= 1
            print(f"  [patience] No new best. Patience left: {patience_counter}")
        last_loss = loss_val
        if patience_counter <= 0:
            print(f"Eager training stopped early due to patience at step {i}. Raising to trigger fallback.")
            raise RuntimeError("Patience exhausted in eager training.")

        try:
            loss.backward()
            params = model.parameters()
            grads = [p.grad for p in params]
            if any(g is None for g in grads):
                print("!!! Some gradients were None. Retrying backward with allow_unused=True and strict mode off.")
                for p_idx, p in enumerate(params):
                    if p.grad is None:
                        print(f"  - Param {p_idx} (label: {getattr(p, '_label', 'N/A')}) has no grad.")
                prev_strict = autograd.strict
                autograd.strict = False
                try:
                    for p in params:
                        if hasattr(p, 'zero_grad'):
                            p.zero_grad()
                        elif hasattr(p, '_grad'):
                            p._grad = None
                    autograd.grad(loss, params, allow_unused=True)
                    grads = [p.grad for p in params]
                    print("  (Suppressed strict mode for this step. Some gradients may be None.)")
                finally:
                    autograd.strict = prev_strict
            unused = [idx for idx, g in enumerate(grads) if g is None]
            if unused:
                print(f"  {len(unused)} parameter(s) unused; replacing missing gradients with zeros.")
                grads = [g if g is not None else AT.zeros_like(p) for p, g in zip(params, grads)]
            new_params = eager_optimizer.step(params, grads)
            for p, np_ in zip(params, new_params):
                AT.copyto(p, np_)
        except Exception as e:
            print(f"!!! Eager training step failed: {e} !!!")
            break
    else:
        print(f"Eager training finished after {max_steps} steps without reaching epsilon or patience.")

    print("--- Eager training test finished ---")

def main():
    parser = argparse.ArgumentParser(description="Fused Conv Text Demo")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive training mode")
    parser.add_argument("--filename", type=str, default=None, help="Path to corpus file for document training")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for document training")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (pairs per epoch) for document training")
    parser.add_argument("--input-len", type=int, default=30, help="Input slice length")
    parser.add_argument("--output-len", type=int, default=30, help="Output slice length")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    args = parser.parse_args()

    like = AT.get_tensor()


    # Try to initialize and ingest the complex model, fallback to linear if it fails (including deep failures)
    try:
        model = ConvTextModel(like=like, grid_shape=(3, 2, 5))
        x0 = AT.zeros(args.input_len)
        _ = model.forward(x0)
        try:
            run_eager_training_test(model, AT, args)
        except Exception as e:
            print(f"[WARNING] Eager training failed with ConvTextModel: {e}\nFalling back to LinearStackModel.")
            model = LinearStackModel(like=like)
            x0 = AT.zeros(args.input_len)
            _ = model.forward(x0)
            run_eager_training_test(model, AT, args)
    except Exception as e:
        print(f"[WARNING] Complex ConvTextModel failed to initialize or ingest: {e}\nFalling back to LinearStackModel.")
        model = LinearStackModel(like=like)
        x0 = AT.zeros(args.input_len)
        _ = model.forward(x0)
        run_eager_training_test(model, AT, args)

    gm = IRGraphedModel(model).config(
        interactive=args.interactive,
        recycle=True,
        queue_maxlen=512,
        max_sample_age=2048,
        lr=args.lr,
        epochs=args.epochs,
        # Conservative strict-mode whitelist for structural/non-differentiable labels
        #strict_whitelist_labels=[
        #    r'^grid\.',
        #    r'^laplace_nd\.grid\.',
        #    r'^laplace_nd\.meshgrid\.',
        #    r'^laplace_nd\.steps\.',
        #    r'^laplace_nd\.partials\.',
        #    r'^laplace_nd\.metric\.',
        #    r'^laplace_nd\.geometry\.',
        #    r'^laplace_nd\.DEC\.',
        #    r'^laplace_nd\.coo\.',
        #    r'^laplace_nd\.laplacian\.',
        #    r'^laplace_nd\.transform_spatial\.',
        #    r'^PCANDTransform\.',
        #    r'^fit_metric_pca\.',
        #    r'^SpectralConv3D\.(?!Wspec)',
        #],
    )

    # Build FusedProgram once; training loop will use the selected mode
    gm.capture(x0)
    trainer = CompletionTrainer(gm, lr=args.lr)

    if args.interactive:
        print("Interactive training: model outputs will be decoded to text.")
        print("Type the expected output (UTF-8) per step and press Enter. Ctrl+C to exit.")
        trainer.interactive_loop(initial_input=x0)
    else:
        # Default corpus filename if not provided: README.md at repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        default_file = os.path.join(repo_root, "README.md")
        corpus_path = args.filename or default_file
        try:
            with open(corpus_path, "rb") as f:
                data = f.read()
        except Exception as e:
            print(f"Failed to read corpus file '{corpus_path}': {e}")
            return
        print(f"Document training from: {corpus_path}")
        trainer.train_from_document(
            data,
            input_len=args.input_len,
            output_len=args.output_len,
            epochs=args.epochs,
            batch=args.batch,
        )
        # After document training, show a quick inference using the first input slice
        init_in = AT.get_tensor(list(data[:args.input_len])) if len(data) >= args.input_len else x0
        gm.capture(init_in)
        out = gm.infer()
        pred = out.get("pred", next(iter(out.values())))
        from .completion_training import decode_text
        print("\nModel completion (decoded):\n" + decode_text(pred))


if __name__ == "__main__":
    main()
