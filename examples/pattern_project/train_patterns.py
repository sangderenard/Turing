"""Two ordinary training runs: an MLP and a CNN learn complex patterns.

Written like any user of abstract_nn would write it -- the packaged
``train_loop``, ``Adam``, and losses do all the work. The program reports
loss checkpoints and wall-clock timing for each phase, so a run demonstrates
both networks actually learning.

Patterns:
  * checkerboard -- a 2x2-cell parity pattern over [-1, 1]^2. Four separate
    decision regions; linearly inseparable everywhere.
  * stripes -- 8x8 single-channel textures, diagonal vs horizontal, with
    additive noise. Orientation is only decidable by spatial filtering,
    which is the convolutional layer's job.
"""

import random
import time

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_nn import (
    Adam,
    BCEWithLogitsLoss,
    Flatten,
    Linear,
    MaxPool2d,
    Model,
    MSELoss,
    RectConv2d,
    ReLU,
    Sigmoid,
    Tanh,
    set_seed,
    train_loop,
)
from src.common.tensors.abstract_nn.utils import from_list_like


def checkerboard_dataset(like, side=16):
    """Points on a grid over [-1, 1]^2; label = parity of the 2x2 cell."""

    points, labels = [], []
    for i in range(side):
        for j in range(side):
            x = -1.0 + (i + 0.5) * (2.0 / side)
            y = -1.0 + (j + 0.5) * (2.0 / side)
            cell = (int((x + 1.0) * 1.0) + int((y + 1.0) * 1.0)) % 2
            points.append([x, y])
            labels.append([float(cell)])
    X = from_list_like(points, like=like)
    Y = from_list_like(labels, like=like)
    return X, Y


def stripes_dataset(like, count=64, size=8, noise=0.15):
    """Single-channel textures: diagonal stripes (1) vs horizontal (0)."""

    images, labels = [], []
    for n in range(count):
        diagonal = n % 2 == 1
        rows = []
        for r in range(size):
            row = []
            for c in range(size):
                value = float(((r + c) if diagonal else r) % 2)
                row.append(value + random.gauss(0.0, noise))
            rows.append(row)
        images.append([rows])
        labels.append([1.0 if diagonal else 0.0])
    X = from_list_like(images, like=like)
    Y = from_list_like(labels, like=like)
    return X, Y


def checkpoints(losses, count=6):
    """A few evenly spaced (epoch, loss) points, first and last included."""

    if not losses:
        return []
    steps = max(1, (len(losses) - 1) // (count - 1)) if count > 1 else 1
    picked = list(range(0, len(losses), steps))
    if picked[-1] != len(losses) - 1:
        picked.append(len(losses) - 1)
    return [(index + 1, losses[index]) for index in picked]


def report(name, losses, seconds):
    print(f"\n== {name} ==")
    for epoch, loss in checkpoints(losses):
        print(f"  epoch {epoch:4d}  loss {loss:.6f}")
    first, last = losses[0], losses[-1]
    print(f"  loss {first:.6f} -> {last:.6f} "
          f"({(1.0 - last / first) * 100.0:.1f}% lower)")
    print(f"  {seconds:.1f}s total, {seconds / len(losses) * 1000.0:.1f} ms/epoch")


def compare(title, timings):
    """Side-by-side ms/epoch for one network across backends."""

    print(f"\n-- {title}: backend timing --")
    baseline = None
    for backend_name, seconds, epochs in timings:
        per_epoch = seconds / epochs * 1000.0
        if baseline is None:
            baseline = per_epoch
            print(f"  {backend_name:24s} {per_epoch:8.1f} ms/epoch")
        else:
            print(f"  {backend_name:24s} {per_epoch:8.1f} ms/epoch "
                  f"({per_epoch / baseline:.2f}x the first)")


def backend_label(like):
    return type(like).__name__


def train_mlp(like, epochs=800):
    X, Y = checkerboard_dataset(like)
    model = Model(
        layers=[
            Linear(2, 32, like=like, init="xavier"),
            Linear(32, 32, like=like, init="xavier"),
            Linear(32, 1, like=like, init="xavier"),
        ],
        activations=[Tanh(), Tanh(), Sigmoid()],
    )
    optimizer = Adam(model.parameters(), lr=1e-2)
    started = time.perf_counter()
    losses, _ = train_loop(
        model, MSELoss(), optimizer, X, Y, epochs=epochs, log_every=200,
    )
    seconds = time.perf_counter() - started
    report(f"MLP on checkerboard [{backend_label(like)}]", losses, seconds)
    return losses, seconds


def train_cnn(like, epochs=120):
    X, Y = stripes_dataset(like)
    model = Model(
        layers=[
            RectConv2d(1, 4, 3, padding=1, like=like),
            MaxPool2d(2, stride=2, like=like),
            Flatten(like=like),
            Linear(4 * 4 * 4, 16, like=like, init="xavier"),
            Linear(16, 1, like=like, init="xavier"),
        ],
        activations=[ReLU(), None, None, ReLU(), None],
    )
    optimizer = Adam(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    losses, _ = train_loop(
        model, BCEWithLogitsLoss(), optimizer, X, Y, epochs=epochs, log_every=40,
    )
    seconds = time.perf_counter() - started
    report(f"CNN on striped textures [{backend_label(like)}]", losses, seconds)
    return losses, seconds


def backends():
    """The pythonic numpy backend and the live nodus arena. The third racer
    -- the same training step AOT-compiled through repository SSA to LLVM --
    joins when the SSA->LLVM backend lands; it is a compiled program, not
    another eager backend, so it enters as its own timed entry."""

    from src.common.tensors.numpy_backend import NumPyTensorOperations
    like_numpy = AbstractTensor.get_tensor(cls=NumPyTensorOperations)
    pairs = [(backend_label(like_numpy), like_numpy)]
    try:
        from src.common.tensors.accelerator_backends.nodus_backend import (
            NodusTensorOperations,
        )
        like_nodus = AbstractTensor.get_tensor(cls=NodusTensorOperations)
        pairs.append((backend_label(like_nodus), like_nodus))
    except Exception as error:
        print("nodus backend unavailable:", error)
    return pairs


def main(argv):
    """Race the requested backends; with no arguments, race everything.

    Usage: python -m examples.pattern_project.train_patterns [numpy] [nodus]
    Each named backend runs both networks; the timing table compares them.
    """

    chosen = {argument.lower() for argument in argv} or None
    available = backends()
    if chosen is not None:
        keyed = {name.lower().replace("tensoroperations", ""): (name, like)
                 for name, like in available}
        unknown = chosen - set(keyed)
        if unknown:
            print("unknown backend(s):", ", ".join(sorted(unknown)),
                  "-- available:", ", ".join(sorted(keyed)))
            return 2
        available = [keyed[key] for key in sorted(chosen)]

    mlp_times, cnn_times = [], []
    lowered = True
    for name, like in available:
        print(f"\n#### backend: {name} ####")
        set_seed(0)
        mlp_losses, mlp_seconds = train_mlp(like)
        cnn_losses, cnn_seconds = train_cnn(like)
        mlp_times.append((name, mlp_seconds, len(mlp_losses)))
        cnn_times.append((name, cnn_seconds, len(cnn_losses)))
        lowered = lowered and (mlp_losses[-1] < mlp_losses[0]
                               and cnn_losses[-1] < cnn_losses[0])
    if len(mlp_times) > 1:
        compare("MLP", mlp_times)
        compare("CNN", cnn_times)
    print("\nall networks lowered their loss on every backend" if lowered
          else "\nWARNING: a network failed to lower its loss")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
