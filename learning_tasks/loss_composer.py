from __future__ import annotations

from typing import Callable, List, Tuple


class LossComposer:
    """Compose multiple loss components into a single scalar.

    Components are added via :meth:`add` with a slice selecting the predicted
    channels, a ``target_fn`` that derives the matching target tensor, and a
    ``loss_fn`` that produces a scalar loss given ``(pred, target, categories)``.
    """

    def __init__(self) -> None:
        self._components: List[Tuple[slice, Callable, Callable]] = []

    def add(
        self,
        pred_slice: slice,
        target_fn: Callable,
        loss_fn: Callable,
    ) -> None:
        """Register a loss component.

        Parameters
        ----------
        pred_slice:
            Slice applied along the channel dimension of the network output.
        target_fn:
            Function ``target_fn(target, categories)`` returning the tensor to
            compare against ``pred_slice``.
        loss_fn:
            Callable ``loss_fn(pred, target, categories)`` yielding a scalar
            loss value.
        """

        self._components.append((pred_slice, target_fn, loss_fn))

    def __call__(self, y, target, categories) -> object:
        total = 0
        for pred_slice, target_fn, loss_fn in self._components:
            pred_part = y[:, pred_slice]
            tgt_part = target_fn(target, categories)
            total = total + loss_fn(pred_part, tgt_part, categories)
        return total
