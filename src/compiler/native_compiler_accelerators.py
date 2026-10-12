"""Small native modules used by the compiler itself.

The registry in this module is deliberately about *compiler structure*, not
tensor execution.  A provider implements one backend-neutral semantic name and
must retain an equivalent Python fallback.  Native modules are installed
explicitly after they have been compiled; importing the compiler never invokes
a toolchain or makes native availability part of correctness.

The first module is lexicographical topological ordering.  Python normalizes a
graph's arbitrary node identities into the requested lexical order, while the
authored kernel consumes only a compact integer edge table and returns an
ordering. The active build compiles graph mechanics through
``lower_ast_source_to_ssa``; the historical Fortran description below is not
used to build or register a provider. No graph objects cross the DLL ABI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import networkx as nx

from .graph_refinement_kernels import broadcast_shape_extents


TOPOLOGICAL_ORDER = "compiler.graph.lexicographical_topological_order"
DEPENDENCY_LEVELS = "compiler.graph.dependency_levels"
BROADCAST_SHAPES = "compiler.graph.broadcast_shapes"
DEAD_METADATA_ORDER = "compiler.graph.dead_metadata_order"
READ_ONLY_MODULE = "compiler.ssa.read_only_module"
METADATA_COPY = "compiler.graph.isolate_metadata_batch"


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
    METADATA_COPY, lambda values, *, copier: tuple(copier(value) for value in values),
    name="python:authored-metadata-copier",
)


def metadata_copy_batch(values, *, copier, registry=compiler_accelerators):
    """One shared identity domain for graph and node metadata; no item dispatch."""
    return registry.resolve(METADATA_COPY).run(values, copier=copier)


def metadata_copy_source():
    """Wrap the unchanged incumbent copier, never author a substitute."""
    import ast
    import inspect
    from .process_graph_fusion import extract_clean_process_subgraph
    extractor = ast.parse(inspect.getsource(extract_clean_process_subgraph)).body[0]
    copier = next(node for node in extractor.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'isolate_metadata')
    root = ast.parse('def metadata_copy_program(value, metadata_memo):\n'
                     '    return isolate_metadata(value)\n').body[0]
    root.body.insert(0, copier)
    imports = ast.ImportFrom(module='src.compiler.live_tiles',
                             names=[ast.alias(name='TiledSet')], level=0)
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[imports, root], type_ignores=[])))


def _native_metadata_copy_runner(artifact, abi):
    import numpy as np
    from .shell_external_references import ExternalReferenceValueCodec
    from .ssa_llvm_backend import prepare_artifact_execution
    artifact.entry()
    release = abi['release']
    release.entry()
    context, function, memo = abi['context'], abi['function'], abi['memo']

    def run(values, *, copier):
        # The synthetic tuple is only the batch root. The original recursive
        # program copies it with the same memo as every graph/node value.
        values = tuple(values)
        boundary = ExternalReferenceValueCodec().bind_native_objects(context, (values,))
        try:
            if boundary._opaque_container_handles:
                # The incumbent Python copier accepts builtin subclasses;
                # type evidence alone does not admit their native payload.
                # Decide BEFORE execution, never retry a native fault.
                return tuple(copier(value) for value in values)
            # A copy entry AND its source keepalive share the incumbent memo.
            # Tile-local reads can otherwise recycle an identity mid-copy.
            # Native temporary births add no keys; at most two per source.
            capacity = max(1, 2 * int(boundary.buffers[context.objects.length_address_id][0]))
            feeds = {
                abi['parameter']: np.array([boundary.roots[0]], dtype=np.int64),
                memo.column_value_ids[0]: np.zeros(capacity, dtype=np.int64),
                memo.column_value_ids[1]: np.zeros(capacity, dtype=np.int64),
                memo.length_address_id: np.zeros(1, dtype=np.int64),
                memo.capacity_value_id: np.array([capacity], dtype=np.int64),
                memo.status_address_id: np.zeros(1, dtype=np.int32),
                memo.live_flags_value_id: np.zeros(capacity, dtype=np.bool_),
            }
            invocation = boundary.bind_execution(artifact, function, feeds).run()
            if int(feeds[memo.status_address_id][0]) == 2:
                raise RuntimeError('native metadata copier exhausted its identity memo')
            result, = boundary.export((int(invocation.buffers[abi['output']].item()),))
            if type(result) is not tuple or len(result) != len(values):
                raise RuntimeError('native metadata copier returned an invalid batch')
            return result
        finally:
            prepare_artifact_execution(release, boundary.buffers).run()
            boundary.close()
    return run


def _metadata_copy_install_directory(directory=None):
    return (Path(directory) if directory is not None else
            Path(__file__).resolve().parents[2] / 'build/compiler_foundations/metadata_copy').resolve()


def _metadata_copy_install_key(store):
    import hashlib
    import platform
    import struct
    from .project_compilation_product import compiler_toolchain_fingerprint
    digest = hashlib.sha256(compiler_toolchain_fingerprint()['sha256'].encode())
    for filename in ('native_compiler_accelerators.py', 'native_object_boundary.py',
                     'ssa_object_context.py', 'ssa_object_operations.py'):
        digest.update(Path(__file__).with_name(filename).read_bytes())
    digest.update(metadata_copy_source().encode())
    digest.update(repr((platform.system(), platform.machine(), struct.calcsize('P'))).encode())
    return store.key('executable', METADATA_COPY, digest.hexdigest())


def load_metadata_copy_accelerator(directory=None, *, registry=compiler_accelerators):
    """Load matching compiler code only; missing/stale code stays serial."""
    import dataclasses
    import hashlib
    from .opportunistic_pipeline import ArtifactCheckpointStore
    directory = _metadata_copy_install_directory(directory)
    if not (directory / 'checkpoints').is_dir():
        return None
    store = ArtifactCheckpointStore(directory / 'checkpoints')
    checkpoint = store.load(_metadata_copy_install_key(store))
    if checkpoint is None:
        return None
    if checkpoint.kind != 'executable' or checkpoint.meta.get('semantic') != METADATA_COPY:
        raise ValueError('installed metadata copier has a different semantic ABI')
    abi = dict(checkpoint.meta['abi'])
    artifacts = []
    for product, digest in ((checkpoint.value, checkpoint.meta['library_sha256']),
                            (abi['release'], checkpoint.meta['release_sha256'])):
        library = (directory / product.library_path).resolve()
        if not library.is_relative_to(directory) or not library.is_file():
            raise ValueError('installed metadata copier library is absent or outside its installation')
        if hashlib.sha256(library.read_bytes()).hexdigest() != digest:
            raise ValueError('installed metadata copier library digest changed')
        artifacts.append(dataclasses.replace(product, library_path=library))
    artifact, abi['release'] = artifacts
    return registry.register_native(METADATA_COPY, _native_metadata_copy_runner(artifact, abi),
        name='turing:authored-metadata-copier', library_path=artifact.library_path,
        exports=(artifact.name,), metadata={**checkpoint.meta, 'artifact': artifact,
            'installation': str(directory), 'loaded_without_compilation': True})


def compile_and_register_metadata_copy_accelerator(directory=None, *, registry=compiler_accelerators, progress=None):
    """Build the whole ORIGINAL copier through our sanctioned source entry."""
    import dataclasses
    import hashlib
    from types import SimpleNamespace
    from .opportunistic_pipeline import Artifact, ArtifactCheckpointStore
    from .extraction_contract import ExtractionContract
    from .fortran_c_shell import lower_ast_source_to_ssa
    from .identity_concordance import concordance_report
    from .monotonic_ids import GLOBAL_MONOTONIC_IDS
    from .ssa_object_context import lower_object_context
    from .ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm
    from ..transmogrifier.ssa import SSAObjectContext, SSASequenceDescriptor, SSASequenceCapacityPolicy
    directory = _metadata_copy_install_directory(directory)
    incumbent = load_metadata_copy_accelerator(directory, registry=registry)
    if incumbent is not None:
        return incumbent
    store = ArtifactCheckpointStore(directory / 'checkpoints')
    key = _metadata_copy_install_key(store)
    mint = GLOBAL_MONOTONIC_IDS.mint
    owner = mint()
    def table(dtypes, keys):
        columns = tuple(mint() for _ in dtypes)
        return SSASequenceDescriptor(columns[0], columns, mint(), mint(),
            status_address_id=mint(), column_dtypes=dtypes, key_columns=keys,
            capacity_policy=SSASequenceCapacityPolicy.DYNAMIC, heap_owner_value_id=owner)
    kinds = tuple('builtins.' + name for name in ('dict', 'list', 'tuple', 'set', 'frozenset', 'object'))
    context = SSAObjectContext('compiler.metadata.context',
        table(('opaque_ref', 'int64', 'int64'), (0,)),
        table(('opaque_ref', 'opaque_ref', 'opaque_ref'), (0, 1)), mint(), kinds,
        type_vocabulary=(*kinds, *( 'builtins.' + name for name in ('bool','int','float','str','bytes','NoneType')),
                         'src.compiler.live_tiles.TiledSet'))
    repository = Path(__file__).resolve().parents[2]
    contract = ExtractionContract(repository / 'extraction_contracts/program_extraction.yaml').with_program_abi({
        'records': {}, 'bindings': [], 'values': [
            {'function': 'metadata_copy_program', 'parameter': 'value', 'storage': 'reference', 'python_type': 'builtins.object'},
            {'function': 'metadata_copy_program', 'parameter': 'metadata_memo', 'storage': 'keyed', 'dtype': 'opaque_ref',
             'mutable': True, 'key_encoding': 'integer_identity', 'python_type': 'builtins.dict'}]
    }).with_execution_file(repository / 'extraction_contracts/vehicle_full_native_execution.yaml')
    module, _, exports = lower_ast_source_to_ssa(metadata_copy_source(), 'metadata_copy_program',
        extraction_contract=contract, native_object_context=context, runtime_closure_only=True,
        progress=progress or (lambda _: None), progress_bars=False)
    report = concordance_report(module)
    if not report.splitlines()[0].endswith('0 finding(s)'):
        raise ValueError(report)
    function = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, function.name)
    release_module = lower_object_context(context, prefix='metadata_copy_release')
    release = emit_ssa_function_to_llvm(release_module, 'metadata_copy_release_release')
    if not artifact.complete or not release.complete:
        raise ValueError((artifact.shortfalls, release.shortfalls))
    product = directory / 'products' / key
    artifact = compile_artifact(artifact, directory=product / 'copier')
    release = compile_artifact(release, directory=product / 'release')
    def portable(value):
        return dataclasses.replace(value, emission=None, _entry=None, _library=None,
            _validation_error_reset=None, _validation_error_take=None,
            library_path=value.library_path.relative_to(directory))
    # Boundary ABI only, not a second saved graph or provenance registry.
    boundary_function = SimpleNamespace(name=function.name, args=tuple(function.args), metadata={
        name: function.metadata[name] for name in ('native_object_context', 'native_object_requires_member_storage')
        if name in function.metadata})
    memo = next(d for d in module.sequence_tables[function.name].sequences.values()
                if d.column_dtypes == ('int64', 'opaque_ref'))
    output = next(i.args[0].id for b in function.blocks.values() for i in b.instrs if i.op == 'Ret' and i.args)
    abi = dict(context=context, function=boundary_function, memo=memo,
        parameter=dict(function.metadata['parameter_names'])['value'], output=output, release=portable(release))
    metadata = dict(semantic=METADATA_COPY, abi=abi,
        library_sha256=hashlib.sha256(artifact.library_path.read_bytes()).hexdigest(),
        release_sha256=hashlib.sha256(release.library_path.read_bytes()).hexdigest())
    if store.save(key, Artifact('executable', portable(artifact), metadata)) is None:
        raise RuntimeError('could not save the compiled metadata copier')
    return load_metadata_copy_accelerator(directory, registry=registry)
compiler_accelerators.register_fallback(
    TOPOLOGICAL_ORDER,
    _python_lexicographical_topological_order,
    name="python:networkx-lexicographical-topological-order",
)


def _python_dependency_levels(graph, *, order, preference):
    from types import SimpleNamespace
    from ..transmogrifier.ilpscheduler import ILPScheduler
    return ILPScheduler(SimpleNamespace(G=graph)).compute_levels(preference, "dependency")


compiler_accelerators.register_fallback(
    DEPENDENCY_LEVELS, _python_dependency_levels, name="python:ILPScheduler-dependency-levels",
)


def _python_broadcast_shape(left, right):
    rank = max(len(left), len(right))
    left_aligned = (1,) * (rank - len(left)) + tuple(left)
    right_aligned = (1,) * (rank - len(right)) + tuple(right)
    output = [0] * rank
    if broadcast_shape_extents(rank, 0, left_aligned, right_aligned, output):
        return tuple(map(int, output))
    return None


def _python_broadcast_shapes(pairs):
    return tuple(_python_broadcast_shape(left, right) for left, right in pairs)


compiler_accelerators.register_fallback(
    BROADCAST_SHAPES, _python_broadcast_shapes, name="python:descriptor-broadcast-shapes",
)


def broadcast_shapes(pairs, *, registry=compiler_accelerators):
    """Transfer already admitted operand shapes, preserving input row order."""
    return registry.resolve(BROADCAST_SHAPES).run(pairs)


def broadcast_shape(left, right, *, registry=compiler_accelerators):
    """Descriptor single-query entry; no bulk packing on the eager route."""
    provider = registry.resolve(BROADCAST_SHAPES)
    if provider.run is _python_broadcast_shapes:
        return _python_broadcast_shape(left, right)
    return provider.run(((left, right),))[0]


def _dead_metadata_inputs(graph, eligible):
    """A view of the EXISTING exact graph, normalized only for integer ABI."""
    nodes = tuple(graph.nodes)
    positions = {node: position for position, node in enumerate(nodes)}
    offsets, parents = [0], []
    for node in nodes:
        # In-edges, not predecessor sets: preserve parallel-edge multiplicity.
        parents.extend(positions[parent] for parent, _child in graph.in_edges(node))
        offsets.append(len(parents))
    return nodes, offsets, parents, [int(node in eligible) for node in nodes]


def _python_dead_metadata_order(graph, *, eligible):
    from .graph_refinement_kernels import dead_metadata_order
    nodes, offsets, parents, flags = _dead_metadata_inputs(graph, eligible)
    count = len(nodes)
    capacity = max(1, count)
    output, status = [0] * capacity, [0]
    dead_metadata_order(count, len(parents), offsets, parents or [0], flags or [0],
                        [0] * capacity, [0] * capacity, output, status)
    return tuple(nodes[position] for position in output[:status[0]])


compiler_accelerators.register_fallback(
    DEAD_METADATA_ORDER, _python_dead_metadata_order, name="python:ordered-dead-metadata-worklist",
)


def _storage_effect_buffers(inputs, *, native=False):
    n = inputs['n']
    edge_capacity = max(1, n + inputs['instruction_count'] + inputs['use_count'])
    state_capacity = max(1, 2 * n)
    spans = {name: value for name, value in inputs.items() if not name.startswith('_')}
    sizes = dict(edge_sources=edge_capacity, edge_targets=edge_capacity, offsets=2 * n + 1,
                 cursor=state_capacity, targets=edge_capacity, unsafe=state_capacity,
                 queue=state_capacity, read_only=max(1, n), storage_types=max(1, n))
    if native:
        import numpy as np
        buffers = {name: np.asarray(value if isinstance(value, int) else value or [0], dtype=np.int64)
                   for name, value in spans.items()}
        # Native scratch is allocated directly. Do not first retain a second
        # complete collection of Python zero lists alongside these arrays.
        buffers.update({name: np.zeros(size, dtype=np.int64) for name, size in sizes.items()})
        return buffers
    spans.update({name: [0] * size for name, size in sizes.items()})
    return spans


def _storage_analysis_outputs(identities, inputs, buffers, include_storage_types):
    effects = {identity: bool(buffers['read_only'][position]) for position, identity in enumerate(identities)}
    if not include_storage_types:
        return effects
    # Tokens are invocation-local names for exact incumbent dtype objects,
    # not a new dtype lattice/vocabulary. 0/-1/'ptr' never infer a type.
    names = inputs['_dtype_names']
    storage = {identity: names[int(code)] for identity, code in zip(identities, buffers['storage_types'])
               if int(code) > 1}
    return effects, storage


def _python_read_only_module(functions, *, include_storage_types=False):
    from .ssa_call_input_adapters import _read_only_module_inputs
    from .graph_refinement_kernels import read_only_storage_effects
    identities, inputs = _read_only_module_inputs(functions)
    buffers = _storage_effect_buffers(inputs)
    read_only_storage_effects(**buffers)
    return _storage_analysis_outputs(identities, inputs, buffers, include_storage_types)


compiler_accelerators.register_fallback(
    READ_ONLY_MODULE, _python_read_only_module, name="python:whole-module-storage-effects",
)


def dependency_levels(graph, *, order, preference, registry=compiler_accelerators):
    return registry.resolve(DEPENDENCY_LEVELS).run(graph, order=order, preference=preference)


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


def _native_topological_runner(artifact, parameters):
    import numpy as np
    from .ssa_llvm_backend import prepare_artifact_execution

    # Load shared code on the coordinator. All storage below belongs to one
    # invocation; no native validation-error global or cached arena is shared.
    if "@turing_validation_error(" in artifact.llvm_ir:
        raise ValueError("topology kernel contains process-global validation state")
    artifact.entry()

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
        edge_count = graph.number_of_edges()
        node_capacity = max(1, len(nodes))
        offsets = np.empty(len(nodes) + 1, dtype=np.int64)
        targets = np.empty(max(1, edge_count), dtype=np.int64)
        offsets[0] = 0
        edge = 0
        for source_index, source in enumerate(normalized_nodes):
            for _source, target in graph.out_edges(source):
                if edge >= edge_count:
                    raise nx.NetworkXError("graph changed while its admitted edge table was built")
                targets[edge] = index_of[target]
                edge += 1
            offsets[source_index + 1] = edge
        if edge != edge_count:
            raise nx.NetworkXError("graph changed while its admitted edge table was built")
        output = np.zeros(node_capacity, dtype=np.int64)
        status = np.zeros(1, dtype=np.int64)
        values = {
            "n": np.array(len(nodes), dtype=np.int64),
            "edge_count": np.array(edge_count, dtype=np.int64),
            "offsets": offsets, "targets": targets,
            "indegree": np.zeros(node_capacity, dtype=np.int64),
            "heap": np.zeros(node_capacity, dtype=np.int64),
            "output": output, "status": status,
        }
        prepare_artifact_execution(artifact, {
            parameters[name]: value for name, value in values.items()
            if name in parameters
        }).run()
        if int(status[0]) < len(nodes):
            raise nx.NetworkXUnfeasible(
                "Graph contains a cycle or graph changed during iteration"
            )
        if int(status[0]) != len(nodes):
            raise nx.NetworkXError(
                f"native topological accelerator returned invalid count ({status[0]})"
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
    progress=None,
) -> CompilerAcceleratorProvider:
    """Self-host graph mechanics through the sanctioned source compiler."""

    return _compile_authored_graph_accelerator(
        directory, registry=registry, pipeline=pipeline, progress=progress,
        entrypoint="topological_order", module_name="compiler_graph_topology",
        names=("n", "edge_count", "offsets", "targets", "indegree", "heap", "output", "status"),
        scalars={"n", "edge_count"}, readonly={"offsets", "targets"},
        semantic=TOPOLOGICAL_ORDER, name="turing:lexicographical-topological-order",
        foundation_name="turing:compiler-topology", runner=_native_topological_runner,
    )


def _compile_authored_graph_accelerator(
    directory, *, registry, pipeline, progress, entrypoint, module_name, names, scalars,
    readonly, semantic, name, foundation_name, runner,
):

    from .extraction_contract import ExtractionContract
    from .fortran_c_shell import lower_ast_source_to_ssa
    from .identity_concordance import concordance_report
    from .ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

    repository = Path(__file__).resolve().parents[2]
    source = Path(__file__).with_name("graph_refinement_kernels.py").read_text()
    contract = ExtractionContract(repository / "extraction_contracts/program_extraction.yaml")
    contract = contract.with_program_abi({"records": {}, "bindings": [], "values": [
        {"function": entrypoint, "parameter": parameter,
         "storage": "scalar" if parameter in scalars else "span",
         "dtype": "int64", "rank": 0 if parameter in scalars else 1,
         "python_type": "builtins.int" if parameter in scalars
                        else "src.common.tensors.abstraction.AbstractTensor",
         "mutable": parameter not in readonly}
        for parameter in names
    ]}).with_execution_file(repository / "extraction_contracts/vehicle_full_native_execution.yaml")
    module, _, _ = lower_ast_source_to_ssa(
        source, entrypoint, name=module_name,
        extraction_contract=contract, runtime_closure_only=True, python_bindings={},
        progress=progress,
    )
    report = concordance_report(module)
    print(report, flush=True)
    if not report.splitlines()[0].endswith(", 0 finding(s)"):
        raise ValueError(f"self-hosted {entrypoint} kernel failed concordance")
    root = module.functions[f"{module_name}__{entrypoint}"]
    artifact = emit_ssa_function_to_llvm(module, root.name)
    if not artifact.complete:
        raise ValueError(f"self-hosted {entrypoint} kernel is incomplete: {artifact.shortfalls}")
    artifact = compile_artifact(artifact, directory=directory)
    parameters = dict(root.metadata["parameter_names"])
    library_path = artifact.library_path
    provider = registry.register_native(
        semantic,
        runner(artifact, parameters),
        name=name,
        library_path=library_path,
        exports=(artifact.name,),
        metadata={
            "abi": {"buffer_order": artifact.buffer_order,
                    "buffer_dtypes": artifact.buffer_dtypes,
                    "parameters": parameters,
                    "metadata": {"semantic": semantic, "tensor_semantics": False}},
            "source_compiler": "src.compiler.fortran_c_shell.lower_ast_source_to_ssa",
            "concordance": report,
            "artifact": artifact,
        },
    )
    if pipeline is not None:
        foundation = pipeline.register_foundation(
            foundation_name,
            library_path,
            accelerates=(semantic,),
        )
        foundation.loaded = True
        provider.metadata["pipeline_foundation"] = foundation
    return provider


def _native_dependency_levels_runner(artifact, parameters):
    import numpy as np
    from .ssa_llvm_backend import prepare_artifact_execution

    if "@turing_validation_error(" in artifact.llvm_ir:
        raise ValueError("worker kernel must not share process-global validation state")
    artifact.entry()  # Load once on the coordinator, never race the DLL loader.

    def run(graph, *, order, preference):
        if preference not in {"asap", "alap"}:
            raise ValueError("dependency levels require asap or alap")
        nodes = tuple(graph)
        if not nodes and preference == "alap":
            raise ValueError("max() arg is an empty sequence")  # ILPScheduler's exact empty-ALAP contract.
        index = {node: position for position, node in enumerate(nodes)}
        order = tuple(order)
        if len(order) != len(nodes) or set(order) != set(nodes):
            raise ValueError("dependency level order must contain every node exactly once")
        rank = {node: position for position, node in enumerate(order)}

        def csr(role):
            offsets = [0]
            values = []
            for node in nodes:
                for related, _label in graph.nodes[node].get(role, ()):
                    if related not in index or (rank[related] >= rank[node] if role == "parents"
                                               else rank[related] <= rank[node]):
                        raise ValueError(f"{role} relation contradicts the admitted DAG order")
                    values.append(index[related])
                offsets.append(len(values))
            return np.asarray(offsets, dtype=np.int64), np.asarray(values or [0], dtype=np.int64)

        parent_offsets, parents = csr("parents")
        child_offsets, children = csr("children")
        capacity = max(1, len(nodes))
        values = dict(n=np.array(len(nodes), dtype=np.int64),
                      preference=np.array(preference == "alap", dtype=np.int64),
                      order=np.asarray([index[node] for node in order] or [0], dtype=np.int64),
                      parent_offsets=parent_offsets, parents=parents,
                      child_offsets=child_offsets, children=children,
                      asap=np.zeros(capacity, dtype=np.int64), levels=np.zeros(capacity, dtype=np.int64))
        prepare_artifact_execution(artifact, {parameters[name]: value for name, value in values.items()}).run()
        return {node: int(values["levels"][position]) for position, node in enumerate(nodes)}
    return run


def compile_and_register_dependency_levels_accelerator(
    directory, *, registry=compiler_accelerators, pipeline=None, progress=None,
):
    return _compile_authored_graph_accelerator(
        directory, registry=registry, pipeline=pipeline, progress=progress,
        entrypoint="dependency_levels", module_name="compiler_graph_levels",
        names=("n", "preference", "order", "parent_offsets", "parents", "child_offsets", "children", "asap", "levels"),
        scalars={"n", "preference"}, readonly={"order", "parent_offsets", "parents", "child_offsets", "children"},
        semantic=DEPENDENCY_LEVELS, name="turing:dependency-levels",
        foundation_name="turing:compiler-dependency-levels", runner=_native_dependency_levels_runner,
    )


def _native_broadcast_shapes_runner(artifact, parameters):
    import numpy as np
    from .ssa_llvm_backend import prepare_artifact_execution

    if "@turing_validation_error(" in artifact.llvm_ir:
        raise ValueError("worker kernel must not share process-global validation state")
    artifact.entry()

    def run(pairs):
        pairs = tuple(pairs)
        offsets = [0]
        left_extents, right_extents = [], []
        for left, right in pairs:
            rank = max(len(left), len(right))
            left_extents.extend((1,) * (rank - len(left)))
            left_extents.extend(left)
            right_extents.extend((1,) * (rank - len(right)))
            right_extents.extend(right)
            offsets.append(offsets[-1] + rank)
        count = len(offsets) - 1
        if not count:
            return ()
        left_array = np.asarray(left_extents or [0])
        right_array = np.asarray(right_extents or [0])
        # This unit has an exact INTEGER ABI, not permission to coerce an
        # unsettled/symbolic extent or wrap an arbitrary-precision integer.
        # Inadmissible inputs stay with the SAME eager law before execution;
        # native execution faults are never caught or silently retried.
        for extents in (left_array, right_array):
            if extents.dtype.kind not in {"i", "u"} or (
                extents.dtype.kind == "u" and int(extents.max()) > np.iinfo(np.int64).max
            ):
                return _python_broadcast_shapes(pairs)
        # Per-invocation storage, including rank-zero dummy spans. Do not
        # retain feeds or scratch in the provider shared by compiler threads.
        values = dict(n=np.array(count, dtype=np.int64),
                      offsets=np.asarray(offsets, dtype=np.int64),
                      left=left_array.astype(np.int64, copy=False),
                      right=right_array.astype(np.int64, copy=False),
                      output=np.zeros(max(1, offsets[-1]), dtype=np.int64),
                      status=np.zeros(count, dtype=np.int64))
        prepare_artifact_execution(artifact, {
            parameters[name]: value for name, value in values.items()
        }).run()
        return tuple(
            tuple(values["output"][offsets[row]:offsets[row + 1]].tolist())
            if values["status"][row] else None
            for row in range(count)
        )
    return run


def compile_and_register_broadcast_shapes_accelerator(
    directory, *, registry=compiler_accelerators, pipeline=None, progress=None,
):
    """Compile the existing descriptor law, explicitly, through OUR compiler."""
    return _compile_authored_graph_accelerator(
        directory, registry=registry, pipeline=pipeline, progress=progress,
        entrypoint="broadcast_shapes", module_name="compiler_graph_shapes",
        names=("n", "offsets", "left", "right", "output", "status"),
        scalars={"n"}, readonly={"offsets", "left", "right"},
        semantic=BROADCAST_SHAPES, name="turing:descriptor-broadcast-shapes",
        foundation_name="turing:compiler-broadcast-shapes", runner=_native_broadcast_shapes_runner,
    )


def _native_dead_metadata_runner(artifact, parameters):
    import numpy as np
    from .ssa_llvm_backend import prepare_artifact_execution
    if "@turing_validation_error(" in artifact.llvm_ir:
        raise ValueError("worker kernel must not share process-global validation state")
    artifact.entry()

    def run(graph, *, eligible):
        nodes, offsets, parents, flags = _dead_metadata_inputs(graph, eligible)
        count = len(nodes)
        if not count:
            return ()
        capacity = max(1, count)
        values = dict(n=np.array(count, dtype=np.int64),
                      edge_count=np.array(len(parents), dtype=np.int64),
                      offsets=np.asarray(offsets, dtype=np.int64),
                      parents=np.asarray(parents or [0], dtype=np.int64),
                      eligible=np.asarray(flags, dtype=np.int64),
                      remaining=np.zeros(capacity, dtype=np.int64),
                      heap=np.zeros(capacity, dtype=np.int64),
                      output=np.zeros(capacity, dtype=np.int64),
                      status=np.zeros(1, dtype=np.int64))
        prepare_artifact_execution(artifact, {
            parameters[name]: value for name, value in values.items()
        }).run()  # ONE native call for the whole transitive worklist.
        emitted = int(values["status"][0])
        if not 0 <= emitted <= count:
            raise RuntimeError("native dead-metadata worklist returned an invalid count")
        positions = values["output"][:emitted].tolist()
        if len(set(positions)) != emitted or any(not 0 <= position < count for position in positions):
            raise RuntimeError("native dead-metadata worklist returned invalid node positions")
        return tuple(nodes[position] for position in positions)
    return run


def compile_and_register_dead_metadata_accelerator(
    directory, *, registry=compiler_accelerators, pipeline=None, progress=None,
):
    """Compile a whole graph-elimination program through OUR public entry."""
    return _compile_authored_graph_accelerator(
        directory, registry=registry, pipeline=pipeline, progress=progress,
        entrypoint="dead_metadata_order", module_name="compiler_graph_dead_metadata",
        names=("n", "edge_count", "offsets", "parents", "eligible", "remaining", "heap", "output", "status"),
        scalars={"n", "edge_count"}, readonly={"offsets", "parents", "eligible"},
        semantic=DEAD_METADATA_ORDER, name="turing:ordered-dead-metadata-worklist",
        foundation_name="turing:compiler-dead-metadata", runner=_native_dead_metadata_runner,
    )


def _native_read_only_module_runner(artifact, parameters):
    from .ssa_llvm_backend import prepare_artifact_execution
    from .ssa_call_input_adapters import _read_only_module_inputs
    if "@turing_validation_error(" in artifact.llvm_ir:
        raise ValueError("whole-module analysis must not share global validation state")
    artifact.entry()

    def run(functions, *, include_storage_types=False):
        identities, inputs = _read_only_module_inputs(functions)
        if not identities:
            return ({}, {}) if include_storage_types else {}
        buffers = _storage_effect_buffers(inputs, native=True)
        prepare_artifact_execution(artifact, {parameters[name]: value for name, value in buffers.items()}).run()
        return _storage_analysis_outputs(identities, inputs, buffers, include_storage_types)
    return run


def compile_and_register_storage_effects_accelerator(
    directory=None, *, registry=compiler_accelerators, pipeline=None, progress=None,
):
    """Compile the entire alias/call/write fixed point, not per-feed decisions."""
    import dataclasses
    import hashlib
    from .opportunistic_pipeline import Artifact, ArtifactCheckpointStore
    directory = _storage_effect_install_directory(directory)
    store = ArtifactCheckpointStore(directory / 'checkpoints')
    key = _storage_effect_install_key(store)
    if directory.is_dir():
        incumbent = load_storage_effects_accelerator(directory, registry=registry, pipeline=pipeline)
        if incumbent is not None:
            return incumbent
    provider = _compile_authored_graph_accelerator(
        directory / 'products' / key, registry=registry, pipeline=pipeline, progress=progress,
        entrypoint="read_only_storage_effects", module_name="compiler_storage_effects",
        names=("n", "instruction_count", "use_count", "blocked", "kinds", "results",
               "argument_offsets", "operands", "call_formals", "writes", "active", "memory_addresses",
               "memory_dtypes", "edge_sources", "edge_targets", "offsets", "cursor", "targets",
               "unsafe", "queue", "read_only", "storage_types"),
        scalars={"n", "instruction_count", "use_count"},
        readonly={"blocked", "kinds", "results", "argument_offsets", "operands", "call_formals", "writes",
                  "active", "memory_addresses", "memory_dtypes"},
        semantic=READ_ONLY_MODULE, name="turing:whole-module-storage-effects",
        foundation_name="turing:compiler-storage-effects", runner=_native_read_only_module_runner,
    )
    portable = dataclasses.replace(provider.metadata['artifact'], emission=None, _entry=None,
        _library=None, _validation_error_reset=None, _validation_error_take=None,
        library_path=provider.library_path.relative_to(directory))
    metadata = {name: value for name, value in provider.metadata.items()
                if name not in {'artifact', 'pipeline_foundation'}}
    metadata['library_sha256'] = hashlib.sha256(provider.library_path.read_bytes()).hexdigest()
    saved = store.save(key, Artifact('executable', portable, metadata))
    if saved is None:
        raise RuntimeError('could not save the compiled compiler storage-effect component')
    provider.metadata['installation'] = str(directory)
    return provider


def _storage_effect_install_directory(directory=None):
    return (Path(directory) if directory is not None else
            Path(__file__).resolve().parents[2] / 'build/compiler_foundations/storage_effects').resolve()


def _storage_effect_install_key(store):
    import hashlib
    import platform
    import struct
    from .project_compilation_product import compiler_toolchain_fingerprint
    digest = hashlib.sha256()
    digest.update(compiler_toolchain_fingerprint()['sha256'].encode())
    for filename in ('graph_refinement_kernels.py', 'native_compiler_accelerators.py', 'ssa_call_input_adapters.py'):
        digest.update(Path(__file__).with_name(filename).read_bytes())
    digest.update(repr((platform.system(), platform.machine(), struct.calcsize('P'))).encode())
    # Existing content-addressed artifact keys, not another cache/manifest system.
    return store.key('executable', READ_ONLY_MODULE, digest.hexdigest())


def load_storage_effects_accelerator(directory=None, *, registry=compiler_accelerators, pipeline=None):
    """Load a matching installed component; NEVER lower, emit or compile code.

    Missing/stale assets return None, retaining the ordinary serial compiler.
    A matching asset whose library bytes changed is rejected before DLL load.
    """
    import dataclasses
    import hashlib
    from .opportunistic_pipeline import ArtifactCheckpointStore
    directory = _storage_effect_install_directory(directory)
    if not (directory / 'checkpoints').is_dir():
        return None
    store = ArtifactCheckpointStore(directory / 'checkpoints')
    checkpoint = store.load(_storage_effect_install_key(store))
    if checkpoint is None:
        return None
    if checkpoint.kind != 'executable' or checkpoint.meta.get('abi', {}).get('metadata', {}).get('semantic') != READ_ONLY_MODULE:
        raise ValueError('installed storage-effect component has a different semantic ABI')
    library = (directory / checkpoint.value.library_path).resolve()
    if not library.is_relative_to(directory) or not library.is_file():
        raise ValueError('installed storage-effect library is absent or outside its installation')
    if hashlib.sha256(library.read_bytes()).hexdigest() != checkpoint.meta['library_sha256']:
        raise ValueError('installed storage-effect library digest changed')
    artifact = dataclasses.replace(checkpoint.value, library_path=library)
    provider = registry.register_native(READ_ONLY_MODULE,
        _native_read_only_module_runner(artifact, checkpoint.meta['abi']['parameters']),
        name='turing:whole-module-storage-effects', library_path=library, exports=(artifact.name,),
        metadata={**checkpoint.meta, 'artifact': artifact, 'installation': str(directory), 'loaded_without_compilation': True})
    if pipeline is not None:
        foundation = pipeline.register_foundation('turing:compiler-storage-effects', library, accelerates=(READ_ONLY_MODULE,))
        foundation.loaded = True
        provider.metadata['pipeline_foundation'] = foundation
    return provider


__all__ = [
    "METADATA_COPY",
    "metadata_copy_source",
    "metadata_copy_batch",
    "compile_and_register_metadata_copy_accelerator",
    "load_metadata_copy_accelerator",
    "TOPOLOGICAL_ORDER",
    "DEPENDENCY_LEVELS",
    "BROADCAST_SHAPES",
    "DEAD_METADATA_ORDER",
    "READ_ONLY_MODULE",
    "compile_and_register_storage_effects_accelerator",
    "load_storage_effects_accelerator",
    "compile_and_register_dead_metadata_accelerator",
    "broadcast_shape",
    "broadcast_shapes",
    "compile_and_register_broadcast_shapes_accelerator",
    "dependency_levels",
    "compile_and_register_dependency_levels_accelerator",
    "CompilerAcceleratorProvider",
    "CompilerAcceleratorRegistry",
    "compiler_accelerators",
    "lexicographical_topological_order",
    "topology_fortran_module",
    "compile_and_register_topology_accelerator",
]
