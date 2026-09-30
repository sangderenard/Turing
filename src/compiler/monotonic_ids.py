"""Central issuer for compiler-created representational IDs.

Migration note (2026-09-30, docs/CONCORDANCE_SINGLE_API_DESIGN_2026-09-30.md
section 6.2): the sanctioned way to mint an id is a ``Novel`` post through
``IdentityBook.post`` in ``identity_concordance``, which draws the id from
``GLOBAL_MONOTONIC_IDS`` and writes the mint edge (transform, operands) in
the same clock tick, so the id is born joined to what it was made from.  A
direct ``GLOBAL_MONOTONIC_IDS.mint()`` call records nothing; the audit lists
every such id under ``unsourced-identity``.  Direct minting stays callable
only while the ~100 existing call sites migrate and is to be closed
afterwards: ``mint`` becomes unreachable except through the api.
"""

from __future__ import annotations

from threading import Lock

from .id_space import MINTED, compose


class MonotonicIdSource:
    """A single non-repeating source of compiler-created integer IDs.

    Every id this issues carries the ``MINTED`` flag, so a value the
    compiler invented says so in its own number.  The old arrangement
    encoded the same intent as a magnitude -- "start at a billion, anything
    at or above that was ours" -- which nothing enforced and which
    ``ssa_llvm_backend``'s history buffers silently violated by basing
    themselves at exactly the same billion.  A flag bit cannot be collided
    with by another group's counter.

    ``first_id`` remains the serial this source starts counting from, not
    the raw id; the flag is applied on the way out.
    """

    def __init__(self, first_id: int = 1_000_000_000) -> None:
        self._lock = Lock()
        self._next = int(first_id)

    def mint(self) -> int:
        with self._lock:
            value_id = self._next
            self._next += 1
            return compose(value_id, MINTED)

    def mint_block(self, count: int) -> tuple[int, ...]:
        return tuple(self.mint() for _ in range(int(count)))

    def peek(self) -> int:
        with self._lock:
            return self._next


GLOBAL_MONOTONIC_IDS = MonotonicIdSource()

