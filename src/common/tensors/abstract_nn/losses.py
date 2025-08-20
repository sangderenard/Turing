from __future__ import annotations
from ..abstraction import AbstractTensor
from .utils import as_list, from_list_like

class Loss:
    def __call__(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        return self.forward(pred, target)
    def forward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor: ...
    def backward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor: ...

class MSELoss(Loss):
    def forward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        diff = pred - target
        return (diff * diff).mean()
    def backward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
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
        return (2.0 / float(N)) * diff

class BCEWithLogitsLoss(Loss):
    def forward(self, logits: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        z = logits
        y = target
        absz = (z * z).sqrt()
        return (z.clamp_min(0.0) - z * y + ((absz * -1.0).exp() + 1.0).log()).mean()

    def backward(self, logits: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
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
        return grad * (1.0 / float(N))

class CrossEntropyLoss(Loss):
    def forward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
        # log_probs: (batch, num_classes), target: (batch,) or (batch, 1)
        log_probs = pred.log_softmax(dim=-1)
        # If target is (batch, 1), squeeze to (batch,)
        if hasattr(target, 'shape') and len(target.shape) > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        # Gather the log-probabilities at the target indices
        # Assume target is integer class indices
        batch_indices = AbstractTensor.get_tensor(list(range(log_probs.shape[0])), faculty=None)
        nll = -log_probs[batch_indices, target]
        return nll.mean()

    def backward(self, pred: AbstractTensor, target: AbstractTensor) -> AbstractTensor:
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
