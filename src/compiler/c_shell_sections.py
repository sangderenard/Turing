"""Opt-in AOT C control around existing, explicitly selected SSA functions.

No AST pursuit or graph rewriting: the original SSA remains the semantic and
concordance owner. Native section calls reuse the C emitter's linked-piece ABI.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import subprocess

from .concordance_declarations import (C_SHELL_SECTION, C_SHELL_SECTION_DEMAND,
    C_SHELL_SECTION_BUILD, SSA_BLOCK_GEOMETRY, EMISSION_C)
from .identity_concordance import Derived, Mode


@dataclass(frozen=True)
class CShellControl:
    """Explicit permission for C control; compute functions are exact SSA names.

    Fortran requires a directory for its AOT object build. All other reachable
    functions retain the existing C materialization, including worker bodies.
    """
    compute_functions: tuple[str, ...]
    directory: Path | None = None

    def __post_init__(self):
        names = tuple(self.compute_functions)
        if not names or len(set(names)) != len(names) or not all(isinstance(n, str) for n in names):
            raise ValueError('C-shell control requires distinct explicit compute function names')
        object.__setattr__(self, 'compute_functions', names)


@dataclass(frozen=True)
class NativeSection:
    backend: str
    piece: object
    receipt: object
    snapshot: tuple


def _snapshot(function):
    return (tuple((v.id, v.dtype, tuple(v.shape or ())) for v in function.args),
            tuple((name, tuple(block.successors), tuple(
                (i.op, tuple((v.id, v.dtype, tuple(v.shape or ())) for v in i.args),
                 None if i.res is None else (i.res.id, i.res.dtype, tuple(i.res.shape or ())),
                 tuple(i.arg_roles), repr(i.attributes)) for i in block.instrs))
                  for name, block in function.blocks.items()))


def _piece_fact(piece):
    return tuple(sorted((k, repr(v)) for k, v in piece.items()))


def validate_section(module, entry, name, section):
    if _snapshot(module.functions[name]) != section.snapshot:
        raise ValueError('native C-shell section disagrees with accepted SSA')
    fact = module.metadata['identity_book'].page(C_SHELL_SECTION).cell(
        section.receipt.row, section.receipt.column)
    if fact != (True, section.backend, section.snapshot, _piece_fact(section.piece)):
        raise ValueError('native C-shell section disagrees with concordance')
    if section.receipt.row != (entry, name, section.backend):
        raise ValueError('native C-shell section belongs to another entry')


# First section ABI: scalar, leaf numerical work with no exceptional/effect
# operations. Extending this gate needs actual transport/failure contracts.
_SCALAR_LEAF = frozenset({'Const', 'const', 'Add', 'Sub', 'Mul', 'Neg', 'Fma',
                          'Abs', 'Min', 'Max', 'Ret', 'ret', 'Br', 'br'})


def _admit_leaf(module, name):
    function = module.functions[name]
    if function.metadata.get('llvm_piece') or not function.blocks:
        raise ValueError('compute section requires an ingested SSA body')
    for table_name, field in (('sequence_tables', 'sequences'), ('record_tables', 'records'),
                              ('reference_tables', None)):
        table = getattr(module, table_name, {}).get(name)
        if table and (getattr(table, field, {}) if field else table):
            raise ValueError('compute section requires aggregate/alias ABI transport')
    values = list(function.args)
    for block in function.blocks.values():
        for i in block.instrs:
            if i.op not in _SCALAR_LEAF:
                raise ValueError(f'compute section {name} contains {i.op}; retain it in C control or extend the section ABI')
            values.extend(i.args)
            if i.res is not None:
                values.append(i.res)
    if any(v.shape or (v.accounting or {}).get('program_abi_rank', 0) for v in values):
        raise ValueError('compute section requires array extent/layout transport')
    return function


def _fortran_piece(module, name, directory, demand):
    from .ssa_fortran_backend import emit_module, fortran_compiler
    from .fortran_c_shell import emit_fortran_packed_library_source
    function = module.functions[name]
    # No Fortran runtime/descriptor ABI is silently added to the C link.
    values = [*function.args, *(i.res for b in function.blocks.values() for i in b.instrs if i.res is not None)]
    if any(v.dtype != 'float64' for v in values):
        raise ValueError('initial Fortran C-shell compute sections require float64 scalars')
    compiler = fortran_compiler()
    if compiler is None:
        raise ValueError('Fortran C-shell section requires an AOT Fortran compiler')
    if directory is None:
        raise ValueError('Fortran C-shell sections require an explicit build directory')
    import shutil
    cc = shutil.which('gcc')
    if cc is None:
        raise ValueError('Fortran C-shell section requires a C compiler for its packed ABI')
    # The established mapping entry accepts the actual Function objects. A
    # scalar-only leaf needs no aggregate tables; do not manufacture a partial
    # IRModule whose unrelated tables still name excluded functions.
    artifact = emit_module({name: function}, name='turing_compute_section',
                           extra_roots=(name,), progress=lambda _:None)
    if not artifact.complete:
        raise ValueError(f'Fortran compute section refused: {artifact.shortfalls}')
    entry = artifact.api.entry_point(name)
    if any(p.role == 'extent' for p in entry.parameters):
        raise ValueError('Fortran C-shell section requires dynamic extent transport')
    ids = tuple(int(p.name.removeprefix('t')) for p in entry.parameters)
    native_symbol = f'{entry.symbol}__c_section'
    packed_source, packed_symbol = emit_fortran_packed_library_source(entry)
    packed_source += (f'\nextern void turing_validation_error(unsigned int);\n'
                      f'void {native_symbol}(void **buffers, int32_t *extents) {{\n'
                      f'  (void)extents; if (!{packed_symbol}(buffers, {len(ids)})) '
                      'turing_validation_error(1);\n}\n')
    root = Path(directory).resolve() / name
    root.mkdir(parents=True, exist_ok=True)
    fpath = artifact.write(root)
    cpath = root / 'packed.c'
    cpath.write_text(packed_source)
    objects = (root / 'compute.o', root / 'packed.o')
    build_cells = []
    commands = ([compiler, '-c', '-fPIC', '-O2', '-ffp-contract=off', str(fpath), '-o', str(objects[0])],
                [cc, '-c', '-fPIC', '-std=c11', str(cpath), '-o', str(objects[1])])
    book = module.metadata['identity_book']
    for source, output, command in zip((fpath, cpath), objects, commands, strict=True):
        row = (demand, source.suffix)
        payload = (source.read_text(), tuple(command), str(output))
        previous = book.latest_ref(C_SHELL_SECTION_BUILD, row)
        planned = book.post(C_SHELL_SECTION_BUILD, row, (*payload, None), stage=EMISSION_C,
                            provenance=Derived((demand, *((previous,) if previous else ()))),
                            mode=Mode.REVISE if previous else Mode.CONCORD)
        subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
        build_cells.append(book.post(C_SHELL_SECTION_BUILD, row,
            (*payload, hashlib.sha256(output.read_bytes()).hexdigest()), stage=EMISSION_C,
            provenance=Derived((planned,)), mode=Mode.REVISE))
    return dict(symbol=native_symbol, llvm_ir='', buffer_order=ids,
                extent_order=(), linked_objects=tuple((str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in objects),
                needs_validation_runtime=True), tuple(build_cells)


def emit_c_shell_sections(module, entry, backend, approval, *, entry_name=None):
    from .ssa_c_backend import emit_ssa_module_to_c, _module_call_closure
    from .ssa_record_return_state import function_scope_of
    if not isinstance(approval, CShellControl):
        raise TypeError('c_shell_control requires a CShellControl approval')
    reachable = _module_call_closure(module, entry)
    if entry in approval.compute_functions or not set(approval.compute_functions).issubset(reachable):
        raise ValueError('compute sections must be reachable non-entry functions')
    book = module.metadata.get('identity_book')
    if book is None:
        raise ValueError('C-shell sections require source/SSA concordance')
    sections = {}
    for name in approval.compute_functions:
        function = _admit_leaf(module, name)
        snapshot = _snapshot(function)
        causes = []
        for label in function.blocks:
            cell = book.latest_ref(SSA_BLOCK_GEOMETRY, (function_scope_of(function), name, label))
            if cell is None:
                raise ValueError('C-shell section requires ingested SSA geometry')
            block = function.blocks[label]
            actual = (tuple((str(i.op), tuple(int(v.id) for v in i.args),
                             None if i.res is None else int(i.res.id),
                             tuple(i.attributes.get('incoming_blocks') or ())) for i in block.instrs),
                      tuple(block.successors))
            if book.page(SSA_BLOCK_GEOMETRY).cell(cell.row, cell.column) != actual:
                raise ValueError('C-shell section geometry disagrees with concordance')
            causes.append(cell)
        demand = book.post(C_SHELL_SECTION_DEMAND, (entry, name, backend),
                           (True, snapshot, 'scalar_leaf'), stage=EMISSION_C,
                           provenance=Derived(tuple(causes)), mode=Mode.CONCORD)
        causes.append(demand)
        if backend == 'llvm':
            from .ssa_llvm_backend import emit_ssa_function_to_llvm
            native = emit_ssa_function_to_llvm(module, name, entry_name=name + '__c_section')
            if not native.complete or native.external_slots:
                raise ValueError(f'LLVM compute section refused: {native.shortfalls}')
            piece = dict(symbol=native.name, llvm_ir=native.llvm_ir,
                         buffer_order=tuple(native.buffer_order), extent_order=tuple(native.extent_order),
                         linked_objects=tuple(native.linked_objects), needs_validation_runtime=native.needs_validation_runtime)
            if native.emission is None or native.emission.module_text is None:
                raise ValueError('LLVM section requires its emitted module concordance')
            causes.append(native.emission.module_text)
        elif backend == 'fortran':
            piece, builds = _fortran_piece(module, name, approval.directory, demand)
            causes.extend(builds)
        else:
            raise ValueError('C-shell compute backend must be llvm or fortran')
        # Native emitters may legally refine their own accepted types. The
        # selection records the current SSA, never a stale pre-emission view.
        if _snapshot(function) != snapshot:
            raise ValueError('native emission changed section SSA; repeat selection from the revised graph')
        fact = (True, backend, snapshot, _piece_fact(piece))
        receipt = book.post(C_SHELL_SECTION, (entry, name, backend), fact, stage=EMISSION_C,
                            provenance=Derived(tuple(causes)), mode=Mode.CONCORD)
        sections[name] = NativeSection(backend, MappingProxyType(piece), receipt, snapshot)
    return emit_ssa_module_to_c(module, entry, entry_name=entry_name, native_sections=sections)
