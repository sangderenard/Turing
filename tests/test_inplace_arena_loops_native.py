"""In-place array loops whose versions share one arena, checked natively.

Both programs computed wrong answers with no shortfall:

* ``store-after-inner-loop`` -- ``v`` is written inside an inner loop and
  again after it.  Placement moved the after-loop store into the inner loop
  because both versions resolve to arena ``v``; ownership of a region by a
  loop is now the composer's record (page ``loop_region_membership``).
* ``update-after-init-loop`` -- an init loop over ``v`` then a nested update
  loop.  The outer loop's carried initial (the post-init version) was not
  resolved to arena ``v`` and became a second, unconnected ``v`` formal.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_compiled_linalg import _run_native


_PROGRAMS = {
    "store-after-inner-loop": (
        "def f(v, w, n):\n"
        "    for i0 in range(n):\n"
        "        for j0 in range(n):\n"
        "            v[i0 * n + j0] = 0.0\n"
        "        v[i0 * n + i0] = 1.0\n"
        "    for i1 in range(n):\n"
        "        w[i1] = v[i1 * n + i1] * 2.0\n"
        "    return w\n",
        ("v", "w"),
    ),
    "update-after-init-loop": (
        "def f(a, v, n):\n"
        "    for i0 in range(n):\n"
        "        v[i0] = 1.0\n"
        "    for p0 in range(n):\n"
        "        c = a[p0] * 0.1\n"
        "        for k3 in range(n):\n"
        "            vk = v[k3]\n"
        "            v[k3] = vk + c\n"
        "    return a\n",
        ("a", "v"),
    ),
}


@pytest.mark.parametrize("label", sorted(_PROGRAMS))
def test_inplace_arena_loop_computes_the_authored_answer(label):
    source, arrays = _PROGRAMS[label]
    n = 3
    inputs = {
        "a": np.arange(1.0, n * n + 1.0),
        "v": np.full(n * n, 7.0),
        "w": np.zeros(n),
    }
    feeds = {name: inputs[name].copy() for name in arrays}
    produced = _run_native(
        source, "f", "arena_" + label.replace("-", "_"),
        feeds, {"n": n}, arrays, n * n,
    )
    namespace: dict = {}
    exec(compile(source, "<authored>", "exec"), namespace)
    expected = {name: inputs[name].copy() for name in arrays}
    namespace["f"](**expected, n=n)
    for name in arrays:
        assert np.allclose(produced[name], expected[name]), name
