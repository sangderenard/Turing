"""Loss-preserving operation records shared by ProcessGraph compiler passes.

``ProcessGraph`` nodes historically carried a loose collection of labels and
Python objects.  That remains useful for visualization, but compiler lowering
needs a stable payload whose fields survive AST, BitOps, SSA, and foreign graph
boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class SourceSpan:
    """A source location without retaining an AST object."""

    filename: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None


@dataclass(frozen=True)
class TensorSpec:
    """Backend-neutral tensor information known at graph construction time."""

    dtype: Optional[str] = None
    shape: Tuple[Optional[int], ...] = ()
    strides: Tuple[Optional[int], ...] = ()
    device: Optional[str] = None
    backend: Optional[str] = None


@dataclass(frozen=True)
class ProcessOp:
    """Serializable semantic payload for one ProcessGraph operation.

    ``op`` uses the AbstractTensor/canonical lower-case spelling.  Inputs are
    represented by graph edges; ``input_roles`` records their stable order and
    meaning.  ``attributes`` contains only operation parameters, never live
    tensor objects.
    """

    op: str
    input_roles: Tuple[str, ...] = ()
    output_roles: Tuple[str, ...] = ("result",)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    tensor: Optional[TensorSpec] = None
    constant: Any = None
    control: Mapping[str, Any] = field(default_factory=dict)
    source: Optional[SourceSpan] = None
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible dictionary where payload values permit."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProcessOp":
        """Reconstruct a payload produced by :meth:`to_dict`."""

        data = dict(payload)
        tensor = data.get("tensor")
        source = data.get("source")
        if tensor is not None and not isinstance(tensor, TensorSpec):
            tensor = TensorSpec(**tensor)
        if source is not None and not isinstance(source, SourceSpan):
            source = SourceSpan(**source)
        data["tensor"] = tensor
        data["source"] = source
        data["input_roles"] = tuple(data.get("input_roles", ()))
        data["output_roles"] = tuple(data.get("output_roles", ("result",)))
        return cls(**data)

