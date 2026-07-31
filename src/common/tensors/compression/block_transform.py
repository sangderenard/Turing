"""Block transforms composed only from AbstractTensor operations."""

from __future__ import annotations

from ..abstraction import AbstractTensor


def orthonormal_dct_basis(
    size: int,
    *,
    like: AbstractTensor,
) -> AbstractTensor:
    """Construct an orthonormal DCT-II basis on ``like``'s backend."""
    if size < 1:
        raise ValueError("DCT size must be positive")
    one = like.ensure_tensor(1.0)
    two = like.ensure_tensor(2.0)
    pi = (-one).acos()
    frequency = AbstractTensor.arange(size, cls=type(like)).unsqueeze(1)
    sample = AbstractTensor.arange(size, cls=type(like)).unsqueeze(0)
    angle = (pi / size) * frequency * (sample + 0.5)
    basis = angle.cos() * (two / size).sqrt()
    dc_scale = (frequency == 0).to_dtype("float32")
    scale = 1.0 + dc_scale * (one / two.sqrt() - 1.0)
    return basis * scale


def block_view_2d(
    values: AbstractTensor,
    *,
    block_height: int,
    block_width: int,
) -> AbstractTensor:
    """View ``(..., H, W)`` as ``(..., Hb, Wb, Bh, Bw)`` blocks."""
    if not isinstance(values, AbstractTensor):
        raise TypeError("values must be an AbstractTensor")
    if block_height < 1 or block_width < 1:
        raise ValueError("block dimensions must be positive")
    if values.ndims() < 2:
        raise ValueError("block_view_2d requires at least two dimensions")
    height, width = values.shape[-2:]
    if height % block_height or width % block_width:
        raise ValueError(
            "input dimensions must be divisible by block dimensions; "
            f"shape={tuple(values.shape)!r} "
            f"block=({block_height}, {block_width})"
        )

    prefix = values.shape[:-2]
    prefix_rank = len(prefix)
    reshaped = values.reshape(
        *(prefix + (
            height // block_height,
            block_height,
            width // block_width,
            block_width,
        ))
    )
    permutation = tuple(range(prefix_rank)) + (
        prefix_rank,
        prefix_rank + 2,
        prefix_rank + 1,
        prefix_rank + 3,
    )
    return reshaped.permute(permutation)


def dct_2d_blocks(
    blocks: AbstractTensor,
    *,
    basis: AbstractTensor | None = None,
) -> AbstractTensor:
    """Apply an orthonormal 2-D DCT to the final two dimensions."""
    if not isinstance(blocks, AbstractTensor):
        raise TypeError("blocks must be an AbstractTensor")
    if blocks.ndims() < 2 or blocks.shape[-1] != blocks.shape[-2]:
        raise ValueError("DCT blocks must be square on their final dimensions")
    transform = (
        basis
        if basis is not None
        else orthonormal_dct_basis(blocks.shape[-1], like=blocks)
    )
    if transform.shape != blocks.shape[-2:]:
        raise ValueError("DCT basis shape does not match block shape")
    return transform @ blocks @ transform.transpose(0, 1)


__all__ = ["block_view_2d", "dct_2d_blocks", "orthonormal_dct_basis"]
