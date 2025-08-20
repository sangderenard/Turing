from __future__ import annotations
from ..abstraction import AbstractTensor
from .utils import as_list, from_list_like
from typing import Callable, List, Dict, Any


class HookedLoss:
    """
    Base class for loss functions with per-instance hooks.
    Hooks are lists of callables and are called at key points in forward/backward.
    """
    def __init__(self):
        self.hooks: Dict[str, List[Callable[..., None]]] = {}

    def register_hook(self, event: str, fn: Callable[..., None]):
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(fn)

    def run_hooks(self, event: str, **kwargs):
        for fn in self.hooks.get(event, []):
            fn(**kwargs)

class Loss(HookedLoss):
    def __init__(self):
        super().__init__()
    def __call__(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        return self.forward(pred, target)
    def forward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        raise NotImplementedError
    def backward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        raise NotImplementedError

class MSELoss(Loss):
    def forward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        self.run_hooks('before_forward', pred=pred, target=target)
        diff = pred - target
        out = (diff * diff).mean()
        self.run_hooks('after_forward', pred=pred, target=target, output=out)
        return out
    def backward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        self.run_hooks('before_backward', pred=pred, target=target)
        diff = pred - target
        if hasattr(pred, "numel"):
            N = pred.numel()
        elif hasattr(pred, "numel_"):
            N = pred.numel_()
        elif hasattr(pred, "shape"):
            N = 1
            for d in pred.shape:
                N *= d
        else:
            def _count(x):
                return sum(_count(v) for v in x) if isinstance(x, list) else 1
            N = _count(as_list(pred)) if hasattr(pred, "__len__") else 1
        grad = (2.0 / float(N)) * diff
        self.run_hooks('after_backward', pred=pred, target=target, grad=grad)
        return grad

class BCEWithLogitsLoss(Loss):
    def forward(self, logits: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        self.run_hooks('before_forward', pred=logits, target=target)
        z = logits
        y = target
        absz = (z * z).sqrt()
        out = (z.clamp_min(0.0) - z * y + ((absz * -1.0).exp() + 1.0).log()).mean()
        self.run_hooks('after_forward', pred=logits, target=target, output=out)
        return out

    def backward(self, logits: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        self.run_hooks('before_backward', pred=logits, target=target)
        z = logits
        y = target
        exp_neg = (z * -1.0).exp()
        ones = from_list_like([[1.0]] * exp_neg.shape[0], like=exp_neg)
        sig = ones / (ones + exp_neg)
        grad = sig - y
        if hasattr(grad, "numel"):
            N = grad.numel()
        elif hasattr(grad, "numel_"):
            N = grad.numel_()
        elif hasattr(grad, "shape"):
            N = 1
            for d in grad.shape:
                N *= d
        else:
            def _count(x):
                return sum(_count(v) for v in x) if isinstance(x, list) else 1
            N = _count(as_list(grad)) if hasattr(grad, "__len__") else 1
        grad_out = grad * (1.0 / float(N))
        self.run_hooks('after_backward', pred=logits, target=target, grad=grad_out)
        return grad_out

class CrossEntropyLoss(Loss):
    def forward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        self.run_hooks('before_forward', pred=pred, target=target)
        # log_probs: (batch, num_classes), target: (batch,) or (batch, 1)
        log_probs = pred.log_softmax(dim=-1)
        # If target is (batch, 1), squeeze to (batch,)
        if hasattr(target, 'shape') and len(target.shape) > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        # Gather the log-probabilities at the target indices
        # Assume target is integer class indices
        batch_indices = AbstractTensor.get_tensor(list(range(log_probs.shape[0])), faculty=None)
        nll = -log_probs[batch_indices, target]
        out = nll.mean()
        self.run_hooks('after_forward', pred=pred, target=target, output=out)
        return out

    def backward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        self.run_hooks('before_backward', pred=pred, target=target)
        # log_probs: (batch, num_classes), target: (batch,) or (batch, 1)
        log_probs = pred.log_softmax(dim=-1)
        probs = log_probs.exp()
        if hasattr(target, 'shape') and len(target.shape) > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        batch_size, num_classes = probs.shape
        # Create one-hot encoding for target
        one_hot = AbstractTensor.get_tensor(
            [[1 if j == int(target[i].item()) else 0 for j in range(num_classes)] for i in range(batch_size)],
            faculty=None
        )
        grad = (probs - one_hot) / float(batch_size)
        return grad
