"""Where an absorbed module came from, stated on the module itself.

A translation with no recorded origin cannot be re-derived, re-verified, or
retired when its source changes, so :class:`Absorption` is a requirement of
this package rather than documentation.  It is deliberately plain data: it
records what was translated and how it was checked, and claims nothing about
quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Absorption:
    """The origin and verification status of one auto-ported module.

    Parameters
    ----------
    source_repository:
        The repository the original lives in, as a path relative to this
        tree's parent (``spectral-analyzer``), so a reader can find it.
    source_path:
        File within that repository.
    source_symbols:
        The function names translated, in the order they appear below.
    entrypoints:
        The names the compiler was asked to lower, which is what the emitted
        function names are derived from.
    verified_against:
        How agreement with the original was established.  A plain sentence
        naming the comparison, not a boolean -- "matches numpy.interp exactly
        including boundaries" is checkable, ``True`` is not.
    caveats:
        Anything a caller must know that the code does not say itself: a
        widened signature, a parameter the original did not have, a numerical
        domain the translation was only checked on.
    """

    source_repository: str
    source_path: str
    source_symbols: Tuple[str, ...]
    entrypoints: Tuple[str, ...]
    verified_against: str
    caveats: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        """One human-readable line, for logs and failure messages."""

        symbols = ", ".join(self.source_symbols)
        return (
            f"auto-ported {symbols} from "
            f"{self.source_repository}/{self.source_path}; "
            f"verified: {self.verified_against}"
        )


__all__ = ["Absorption"]
