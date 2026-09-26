"""Small native modules used by the compiler itself.

The registry in this module is deliberately about *compiler structure*, not
tensor execution.  A provider implements one backend-neutral semantic name and
must retain an equivalent Python fallback.  Native modules are installed
explicitly after they have been compiled; importing the compiler never invokes
a toolchain or makes native availability part of correctness.

The first module is lexicographical topological ordering.  Python normalizes a
graph's arbitrary node identities into the requested lexical order, while the
Fortran routine consumes only a compact integer edge table and returns an
ordering.  Consequently neither NetworkX objects nor tensor/backend concepts
cross the DLL ABI.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import networkx as nx


TOPOLOGICAL_ORDER = "compiler.graph.lexicographical_topological_order"


@dataclass
class CompilerAcceleratorProvider:
    """One implementation of a compiler-internal semantic operation."""

    semantic: str
    name: str
    run: Callable[..., Any]
    tier: str
    library_path: Path | None = None
    exports: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class CompilerAcceleratorRegistry:
    """Prefer registered native compiler modules while retaining an oracle."""

    def __init__(self) -> None:
        self._fallbacks: dict[str, CompilerAcceleratorProvider] = {}
        self._native: dict[str, CompilerAcceleratorProvider] = {}

    def register_fallback(
        self, semantic: str, run: Callable[..., Any], *, name: str
    ) -> CompilerAcceleratorProvider:
        provider = CompilerAcceleratorProvider(
            semantic=str(semantic), name=str(name), run=run, tier="python"
        )
        self._fallbacks[provider.semantic] = provider
        return provider

    def register_native(
        self,
        semantic: str,
        run: Callable[..., Any],
        *,
        name: str,
        library_path: str | Path,
        exports: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> CompilerAcceleratorProvider:
        semantic = str(semantic)
        if semantic not in self._fallbacks:
            raise KeyError(
                f"native compiler accelerator {semantic!r} has no correctness fallback"
            )
        path = Path(library_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        provider = CompilerAcceleratorProvider(
            semantic=semantic,
            name=str(name),
            run=run,
            tier="native",
            library_path=path,
            exports=tuple(map(str, exports)),
            metadata=dict(metadata or {}),
        )
        self._native[semantic] = provider
        return provider

    def unregister_native(self, semantic: str) -> None:
        self._native.pop(str(semantic), None)

    def resolve(self, semantic: str) -> CompilerAcceleratorProvider:
        semantic = str(semantic)
        try:
            return self._native.get(semantic) or self._fallbacks[semantic]
        except KeyError:
            raise KeyError(f"unknown compiler accelerator semantic {semantic!r}")

    def providers(self) -> tuple[CompilerAcceleratorProvider, ...]:
        semantics = sorted(set(self._fallbacks) | set(self._native))
        return tuple(self.resolve(semantic) for semantic in semantics)


def _python_lexicographical_topological_order(graph, *, key=None):
    return tuple(nx.lexicographical_topological_sort(graph, key=key))


compiler_accelerators = CompilerAcceleratorRegistry()
compiler_accelerators.register_fallback(
    TOPOLOGICAL_ORDER,
    _python_lexicographical_topological_order,
    name="python:networkx-lexicographical-topological-order",
)


def lexicographical_topological_order(
    graph,
    *,
    key=None,
    registry: CompilerAcceleratorRegistry = compiler_accelerators,
) -> tuple[Any, ...]:
    """Return the active provider's exact lexical topological ordering."""

    return tuple(registry.resolve(TOPOLOGICAL_ORDER).run(graph, key=key))


_TOPOLOGICAL_FORTRAN = """\
module turing_compiler_topology_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine turing_lexicographical_topological_order( &
      node_count, edge_count, sources, targets, output_order, status) &
      bind(C, name="turing_lexicographical_topological_order")
    integer(c_int32_t), value :: node_count
    integer(c_int32_t), value :: edge_count
    integer(c_int64_t), intent(in) :: sources(*)
    integer(c_int64_t), intent(in) :: targets(*)
    integer(c_int64_t), intent(out) :: output_order(*)
    integer(c_int32_t), intent(out) :: status
    integer(c_int32_t) :: indegree(max(1, node_count))
    logical :: emitted(max(1, node_count))
    integer(c_int32_t) :: edge_index
    integer(c_int32_t) :: position
    integer(c_int32_t) :: candidate
    integer(c_int32_t) :: selected
    integer(c_int64_t) :: source_index
    integer(c_int64_t) :: target_index

    status = 0_c_int32_t
    indegree = 0_c_int32_t
    emitted = .false.

    do edge_index = 1, edge_count
      source_index = sources(edge_index)
      target_index = targets(edge_index)
      if (source_index < 0_c_int64_t .or. source_index >= node_count .or. &
          target_index < 0_c_int64_t .or. target_index >= node_count) then
        status = 2_c_int32_t
        return
      end if
      indegree(int(target_index, c_int32_t) + 1) = &
          indegree(int(target_index, c_int32_t) + 1) + 1_c_int32_t
    end do

    do position = 1, node_count
      selected = 0_c_int32_t
      do candidate = 1, node_count
        if (.not. emitted(candidate) .and. indegree(candidate) == 0) then
          selected = candidate
          exit
        end if
      end do
      if (selected == 0_c_int32_t) then
        status = 1_c_int32_t
        return
      end if

      emitted(selected) = .true.
      output_order(position) = int(selected - 1, c_int64_t)
      do edge_index = 1, edge_count
        if (sources(edge_index) == int(selected - 1, c_int64_t)) then
          target_index = targets(edge_index)
          indegree(int(target_index, c_int32_t) + 1) = &
              indegree(int(target_index, c_int32_t) + 1) - 1_c_int32_t
        end if
      end do
    end do
  end subroutine turing_lexicographical_topological_order
end module turing_compiler_topology_fortran
"""


def _native_topological_runner(library: ctypes.CDLL):
    entry = library.turing_lexicographical_topological_order
    int64_pointer = ctypes.POINTER(ctypes.c_int64)
    entry.argtypes = (
        ctypes.c_int32,
        ctypes.c_int32,
        int64_pointer,
        int64_pointer,
        int64_pointer,
        ctypes.POINTER(ctypes.c_int32),
    )
    entry.restype = None

    def run(graph, *, key=None):
        if not graph.is_directed():
            raise nx.NetworkXError("topological ordering requires a directed graph")

        # NetworkX breaks equal lexical keys by graph insertion order.  Make
        # that total order explicit before crossing the integer-only ABI.
        nodes = tuple(graph.nodes)
        lexical_key = (lambda node: node) if key is None else key
        insertion = {node: index for index, node in enumerate(nodes)}
        normalized_nodes = tuple(sorted(
            nodes, key=lambda node: (lexical_key(node), insertion[node])
        ))
        index_of = {node: index for index, node in enumerate(normalized_nodes)}
        edges = tuple((index_of[source], index_of[target]) for source, target in graph.edges)

        edge_capacity = max(1, len(edges))
        node_capacity = max(1, len(nodes))
        sources = (ctypes.c_int64 * edge_capacity)(
            *(tuple(source for source, _target in edges) or (0,))
        )
        targets = (ctypes.c_int64 * edge_capacity)(
            *(tuple(target for _source, target in edges) or (0,))
        )
        output = (ctypes.c_int64 * node_capacity)()
        status = ctypes.c_int32()
        entry(
            len(nodes), len(edges), sources, targets, output, ctypes.byref(status)
        )
        if status.value == 1:
            raise nx.NetworkXUnfeasible(
                "Graph contains a cycle or graph changed during iteration"
            )
        if status.value != 0:
            raise nx.NetworkXError(
                f"native topological accelerator rejected edge table ({status.value})"
            )
        return tuple(normalized_nodes[int(output[index])] for index in range(len(nodes)))

    return run


def topology_fortran_module():
    """Describe the native topology module and its complete C calling ABI."""

    from .compiled_program_api import CompiledProgramAPI, EntryPoint, Parameter
    from .ssa_fortran_backend import FortranModule

    parameters = (
        Parameter("node_count", "extent", "int32", "int32_t", "c_int32", "value"),
        Parameter("edge_count", "extent", "int32", "int32_t", "c_int32", "value"),
        Parameter("sources", "input", "int64", "int64_t", "c_int64", "reference", extent="edge_count"),
        Parameter("targets", "input", "int64", "int64_t", "c_int64", "reference", extent="edge_count"),
        Parameter("output_order", "output", "int64", "int64_t", "c_int64", "reference", extent="node_count"),
        Parameter("status", "output", "int32", "int32_t", "c_int32", "reference"),
    )
    api = CompiledProgramAPI(
        module="turing_compiler_topology_fortran",
        language="fortran",
        entry="turing_lexicographical_topological_order",
        entry_points=(EntryPoint(
            "turing_lexicographical_topological_order",
            "turing_lexicographical_topological_order",
            "compiler-accelerator",
            parameters,
            note="Integer DAG ABI; Python owns arbitrary node identities and lexical keys.",
        ),),
        metadata={"semantic": TOPOLOGICAL_ORDER, "tensor_semantics": False},
    )
    return FortranModule(
        "turing_compiler_topology_fortran", _TOPOLOGICAL_FORTRAN, api=api
    )


def compile_and_register_topology_accelerator(
    directory: str | Path,
    *,
    registry: CompilerAcceleratorRegistry = compiler_accelerators,
    pipeline=None,
) -> CompilerAcceleratorProvider:
    """Compile, load, and register the first compiler-native module."""

    from .ssa_fortran_backend import compile_module

    module = topology_fortran_module()
    library_path = compile_module(module, directory=directory)
    library = ctypes.CDLL(str(library_path))
    provider = registry.register_native(
        TOPOLOGICAL_ORDER,
        _native_topological_runner(library),
        name="fortran:lexicographical-topological-order",
        library_path=library_path,
        exports=("turing_lexicographical_topological_order",),
        metadata={
            "abi": module.api.to_mapping(),
            # Keep the loaded library alive for as long as its callable is
            # registered; ctypes function pointers do not own the DLL.
            "library": library,
        },
    )
    if pipeline is not None:
        foundation = pipeline.register_foundation(
            "fortran:compiler-topology",
            library_path,
            accelerates=(TOPOLOGICAL_ORDER,),
        )
        foundation.loaded = True
        provider.metadata["pipeline_foundation"] = foundation
    return provider


__all__ = [
    "TOPOLOGICAL_ORDER",
    "CompilerAcceleratorProvider",
    "CompilerAcceleratorRegistry",
    "compiler_accelerators",
    "lexicographical_topological_order",
    "topology_fortran_module",
    "compile_and_register_topology_accelerator",
]
