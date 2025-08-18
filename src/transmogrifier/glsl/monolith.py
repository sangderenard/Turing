"""Helpers for assembling GLSL compute shaders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ...common.dt_system.engine_api import ComputeShaderSpec


@dataclass
class MonolithicShader:
    """Bundle multiple :class:`ComputeShaderSpec` objects into a single source.

    The builder concatenates shader sources and merges buffer mappings while
    preserving the order provided. It performs no validation beyond ensuring
    buffer name uniqueness.
    """

    source: str
    buffers: Mapping[str, object]

    @classmethod
    def from_specs(
        cls, specs: Iterable[ComputeShaderSpec], *, version: str = "#version 450"
    ) -> "MonolithicShader":
        """Create a monolithic shader from engine specifications.

        Parameters
        ----------
        specs:
            Iterable of :class:`ComputeShaderSpec` objects describing shader
            stages.
        version:
            GLSL version directive to prepend. Defaults to ``"#version 450"``.
        """
        parts = [version]
        buffers: dict[str, object] = {}
        for spec in specs:
            parts.append(f"// {spec.name}")
            parts.append(spec.source)
            for name, buf in spec.buffers.items():
                if name in buffers:
                    raise ValueError(f"buffer '{name}' already bound")
                buffers[name] = buf
        return cls("\n".join(parts), buffers)
