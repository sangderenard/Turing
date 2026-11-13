"""Native pointer-cell snapshots; no allocation or ownership transfer claim."""
import numpy as np
import pytest
from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


def _snapshot_module(dtype='ptr', *, receipt=True, result_shape=()):
    source = SSAValue(1, 'ptr', shape=(1,), accounting={
        'heap_owner': True, 'compiler_frame_sequence_id': 11,
        'compiler_frame_member': 4})
    destination = SSAValue(2, 'ptr', shape=(1,), accounting={'heap_owner': True})
    snapshot = SSAValue(3, dtype, shape=result_shape)
    zero = SSAValue(4, dtype)
    empty = SSAValue(5, 'ptr', shape=(1,), accounting={'heap_owner': True,
        'compiler_frame_sequence_id': 12, 'compiler_frame_member': 4})
    function = Function('snapshot', [source, destination, empty], {'entry': BasicBlock('entry', [
        Instr('Load', [source], snapshot),
        Instr('Load', [empty], zero),
        Instr('Store', [zero, source], None),
        Instr('Store', [snapshot, destination], None),
        Instr('Ret', [], None),
    ])})
    body = function.blocks['entry']
    body.name = 'body'
    function.blocks = {'entry': BasicBlock('entry', [Instr('Br', [], None, attributes={'target': 'body'})], successors=['body']), 'body': body}
    module = IRModule(functions={'snapshot': function})
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book, Novel, Derived, Mode
    from src.compiler.concordance_declarations import SSA_VALUE, SSAValueFact, SSAValueOrigin, INGESTION, INGEST_SOURCE
    book, token = begin_identity_book()
    try:
        previous = None
        for v in (source, destination, empty, snapshot, zero):
            physical = ({'physical_dtype':dtype, 'storage':'scalar', 'rank':0}
                        if (v is snapshot or v is zero) and receipt else {})
            if v is snapshot and result_shape and physical:
                physical.update(storage='span', rank=len(result_shape))
            previous = book.post(SSA_VALUE, (function.name, v.id),
                SSAValueFact(v.dtype, v.shape, SSAValueOrigin.ADOPTED_GRAPH_ID, **physical),
                stage=INGESTION, mode=Mode.CONCORD,
                provenance=Novel(INGEST_SOURCE, ()) if previous is None else Derived((previous,)))
        from src.compiler.concordance_declarations import SSA_BLOCK, SSABlockKind
        for label, kind in [('entry',SSABlockKind.ENTRY),('body',SSABlockKind.FUNCTION_EXIT)]:
            book.post(SSA_BLOCK,(function.name,function.name,label),kind,stage=INGESTION,mode=Mode.CONCORD,provenance=Derived((previous,)))
        module.metadata['identity_book'] = book
    finally:
        end_identity_book(token)
    return module, function.name


@pytest.mark.parametrize('backend', ['llvm', 'c'])
@pytest.mark.parametrize('dtype', ['ptr', 'opaque_ref'])
def test_owner_cell_snapshot_survives_source_clear(dtype, backend, tmp_path):
    module, name = _snapshot_module(dtype)
    if backend == 'llvm':
        artifact = emit_ssa_function_to_llvm(module, name)
        assert artifact.complete, artifact.shortfalls
        native = compile_artifact(artifact, directory=tmp_path/dtype)
        prepare = lambda feeds: prepare_artifact_execution(native, feeds)
    else:
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact = emit_ssa_module_to_c(module, name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path/dtype)
        prepare = artifact.prepare_execution
    # Non-dereferenced sentinel bits only: never release these as allocations.
    left = np.array([0x12345678], dtype=np.uint64)
    right = np.array([0], dtype=np.uint64)
    prepare({1:left, 2:right, 5:np.zeros(1,dtype=np.uint64)}).run()
    assert left[0] == 0
    assert right[0] == 0x12345678



@pytest.mark.parametrize('backend', ['llvm', 'c'])
def test_pointer_head_move_retains_allocation_until_destination_release(backend, tmp_path):
    import ctypes
    from tests.test_ssa_owned_heap import make_allocation_module
    def native(module, name, lane=None):
        if (lane or backend) == 'llvm':
            artifact = emit_ssa_function_to_llvm(module, name)
            assert artifact.complete, artifact.shortfalls
            artifact = compile_artifact(artifact, directory=tmp_path/name)
            return lambda feeds: prepare_artifact_execution(artifact, feeds)
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact = emit_ssa_module_to_c(module, name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path/name)
        return artifact.prepare_execution
    allocation = make_allocation_module()
    # Established LLVM allocator fixture exposes a raw pointer return. The
    # C lane is exercised for the move, not its distinct host return ABI.
    allocate = native(allocation, 'allocate', 'llvm')
    release = native(allocation, 'release', 'llvm')
    module, name = _snapshot_module()
    move = native(module, name)
    source, destination, empty = (np.zeros(1,dtype=np.uint64) for _ in range(3))
    for n in (1,7,2,0,33):
        allocated = allocate({0:np.array([n],dtype=np.int64),1:source}).run()
        address = int(allocated.buffers[2].item())
        data = (ctypes.c_int64*n).from_address(address)
        for i in range(n): data[i] = 100+i
        original = int(source[0])
        move({1:source,2:destination,5:empty}).run()
        # Check ownership before any release or dereference of moved data.
        assert source[0] == 0 and destination[0] == original
        release({10:source}).run()
        assert list(data) == list(range(100,100+n))
        release({10:destination}).run()
        assert destination[0] == 0
        release({10:destination}).run()


@pytest.mark.parametrize('receipt, shape', [(False,()), (True,(1,))])
def test_unproven_or_ranked_pointer_load_retains_address_semantics(receipt, shape, tmp_path):
    module, name = _snapshot_module(receipt=receipt, result_shape=shape)
    artifact = emit_ssa_function_to_llvm(module, name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact,directory=tmp_path/'alias')
    left=np.array([0x12345678],dtype=np.uint64)
    right=np.zeros(1,dtype=np.uint64)
    prepare_artifact_execution(native,{1:left,2:right,5:np.zeros(1,dtype=np.uint64)}).run()
    assert left[0] == 0 and right[0] == 0
