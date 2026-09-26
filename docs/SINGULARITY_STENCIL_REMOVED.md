# Singularity stencil (removed)

Added in `b5cd9f6e` (PR #779) as a for-fun experiment; removed afterwards.
Recover it with `git show b5cd9f6e`.

## What it was

`src/common/singularity_stencil.py` classified a point by the winding number
of a planar field on a closed contour around it, never sampling the centre.
Tiers: min `|F|` on the ring, circle/box winding with a phase-step
reliability flag, and recursive box subdivision splitting a charge
(e.g. `2 -> 1 + 1`). Reference field `z**2 - a`; a demo plotted the
`W(r, a)` phase diagram, noise robustness, and the decomposition.

## Why it was removed

- It is the argument principle / residue detection, a known method, and its
  tests restated theorems (argument principle, Rouche, contour additivity)
  on one symmetric field rather than probing the method.
- The reliability flag passes aliased contours: `z**17` sampled at 16 points
  reported `W = 1, reliable = True`.
- It measures net charge, not zero count: a +1/-1 pair reads `W = 0`.
- The "centre removed" decomposition passed only because a hard-coded
  off-centre split cleared a 0.1 hole; at 0.13 it went unresolved.
- It was raw `torch` with its own conventions, parallel to the DEC.

## If it is ever needed

Per-cell winding is `d1` applied to the wrapped phase difference on each
edge: `DECSystem.d1` in
`src/common/tensors/abstract_convolution/dec_system.py`. Build it there, on
`AbstractTensor`, for a real caller (vortex cores, optical/cavity phase
singularities, field defects) rather than reviving this module.
