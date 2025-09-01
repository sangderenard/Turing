# MD5 Depth-Unrolled Learning Task

This module provides a synthetic task used by the Riemann demo to explore
learning across the MD5 hash's internal rounds.

Samples expose the state of the algorithm after each of the 64 MD5 steps.
For every sample the queue yields a tuple of `(inputs, targets, category)`:

* **Inputs** – bit‑plane representation of the 512‑bit message block
  concatenated with the previous state `(A,B,C,D)` for each supervised step.
* **Targets** – bit‑planes for the state after the supervised step.
* **Category** – dictionary containing the original message (hex encoded)
  and the supervision stride.

The helper `pump_queue` continuously generates random messages and pushes the
corresponding tensors to a queue for consumption by the training loop.
