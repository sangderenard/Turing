# Low Entropy Learning Task

This folder hosts the synthetic task used by the Riemann demo.  Samples are
constructed as follows:

* **Inputs** – random Gaussian fields with varying spectral profiles.
* **Targets** – low-entropy versions of a shared base field.
* **Category** – a dictionary with two keys:
  * `offset`: generation index controlling the low-entropy shift.
  * `spectrum`: index of the Gaussian spectrum used for the input.

The helper function `pump_queue` continuously generates tuples of
`(input, target, category)` and feeds them into a queue for the demo's training
loop.  The loop consumes as many items as available per iteration up to its
configured batch size and blocks when the queue is empty.
