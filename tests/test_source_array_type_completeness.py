"""Source typed arrays and checked row copy keep their declared layouts."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.ir_sequence_tables import lower_sequence_extend
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
from src.transmogrifier.ssa import SSASequenceDescriptor, IRModule

ROOT = Path(__file__).resolve().parents[1]

def require_fortran_toolchain(*, unsigned=False):
    from src.compiler.ssa_fortran_backend import fortran_compiler
    compiler = fortran_compiler()
    if compiler is None:
        pytest.skip('native Fortran compiler unavailable')
    if unsigned:
        from src.compiler.fortran_toolchain import native_fortran_flags, aggressive_fortran_flags
        try:
            native_fortran_flags(compiler, 'unsigned(c_int64_t) :: capability_probe',
                                 aggressive_fortran_flags(compiler))
        except ValueError as error:
            if str(error).startswith((
                'native unsigned Fortran requires a compiler supporting -funsigned;',
                'selected Fortran compiler/flags do not support exact native unsigned ISO C storage with -funsigned:',
            )):
                pytest.skip('selected Fortran toolchain lacks native unsigned ISO-C storage: ' + str(error))
            raise
    return compiler



@pytest.mark.parametrize('code,dtype', [('b','int8'),('B','uint8'),('h','int16'),('H','uint16'),
                                      ('i','int32'),('I','uint32'),('q','int64'),('Q','uint64'),
                                      ('f','float32'),('d','float64')])
def test_declared_array_row_is_one_type_at_constructor_and_callee(code, dtype):
    source = (ROOT/'tests/record_view_corpus/c632_guarded_record_length.py').read_text().replace('array("B")', f'array("{code}")')
    module, _, _ = lower_ast_source_to_ssa(source, 'root', name='c633_array_'+code,
        extraction_contract=ROOT/'extraction_contracts/program_extraction.yaml',
        progress=lambda _:None, progress_bars=False)
    fields = [(name, field) for name, table in module.record_tables.items()
              for record in table.records.values() for field in record.fields
              if field.storage_identity == 'Ledger.domains']
    assert fields
    for name, field in fields:
        descriptor = module.sequence_tables[name].by_id(field.sequence_id)
        assert descriptor.column_dtypes == (dtype,)
    report = concordance_report(module)
    assert '0 finding(s)' in report, report


def extend_module(target_dtype, source_dtype):
    destination = SSASequenceDescriptor(sequence_id=1, column_value_ids=(0,),
        length_address_id=1, capacity_value_id=2, column_dtypes=(target_dtype,))
    source = SSASequenceDescriptor(sequence_id=2, column_value_ids=(10,),
        length_address_id=11, capacity_value_id=12, column_dtypes=(source_dtype,))
    lowered = lower_sequence_extend(destination, source, function_name='root')
    assert lowered.complete, lowered.shortfalls
    # Test-authored fixed arenas carry real bounds, rather than asking an
    # emitter to invent capacity for a private canonical source list.
    for function in lowered.functions:
        for argument in function.args:
            if argument.id in {0,10}:
                argument.shape = (3,)
    return IRModule({function.name:function for function in lowered.functions})


def execution(module, target_dtype, source_dtype, values, lane, tmp_path):
    source_column = np.zeros(3, dtype=source_dtype)
    source_column[:len(values)] = values
    feeds = {0:np.zeros(3,dtype=target_dtype),1:np.array([0],dtype='int64'),
             2:np.array([3],dtype='int64'),10:source_column,
             11:np.array([len(values)],dtype='int64'),12:np.array([3],dtype='int64')}
    if lane == 'C':
        artifact = emit_ssa_to_c(module,'root')
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path/'native')
        return artifact.prepare_execution(feeds)
    if lane == 'Fortran':
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        require_fortran_toolchain(unsigned=str(target_dtype).startswith('uint'))
        artifact = emit_module(module, name='checked_extend', progress=lambda _:None)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(artifact, {}, tmp_path/'native',
                                               entrypoint='root', library=True)
        return FortranCoreNative(built.executable_path, artifact.api.entry_point('root'),
                                 artifact.api).prepare_execution(feeds)
    artifact = emit_ssa_function_to_llvm(module,'root')
    assert artifact.complete, artifact.shortfalls
    return prepare_artifact_execution(compile_artifact(artifact,directory=tmp_path),feeds)


@pytest.mark.parametrize('lane',['C','LLVM','Fortran'])
@pytest.mark.parametrize('dtype,values', [('uint8',[0,128,255]),('uint16',[0,32768,65535]),
    ('uint32',[0,2**31,2**32-1]),('uint64',[0,2**53+1,2**63-1]),
    ('int8',[-128,0,127]),('int16',[-32768,0,32767]),
    ('float32',[0,16777217,-16777217]),('float64',[0,2**53+1,-(2**53+1)])])
def test_extend_converts_loaded_rows_without_retyping_source_arena(lane,dtype,values,tmp_path):
    module=extend_module(dtype,'int64')
    buffers=execution(module,dtype,'int64',values,lane,tmp_path).run().buffers
    assert buffers[0].dtype == np.dtype(dtype)
    assert buffers[0].tolist() == np.asarray(values,dtype=dtype).tolist()
    assert buffers[10].dtype == np.dtype('int64')
    assert buffers[10].tolist() == values
    assert buffers[1].item() == 3


@pytest.mark.parametrize('lane',['C','LLVM','Fortran'])
@pytest.mark.parametrize('dtype,value',[('uint8',-1),('uint8',256),('int8',128),('int8',-129)])
def test_extend_rejects_out_of_range_integer_before_store(lane,dtype,value,tmp_path):
    if lane == 'Fortran':
        require_fortran_toolchain(unsigned=str(dtype).startswith('uint'))
        # Existing Fortran validation is ERROR STOP, so execute its rejection
        # in a child process. It must reach the built native call and stop
        # there; a compiler/setup failure must not count as validation.
        import subprocess
        import sys
        probe = f"""import runpy
from pathlib import Path
ns=runpy.run_path({str(Path(__file__).resolve())!r})
prepared=ns['execution'](ns['extend_module']({dtype!r},'int64'),{dtype!r},'int64',
                         [{value!r}],'Fortran',Path({str(tmp_path)!r}))
print('READY_NATIVE_VALIDATION_74',flush=True)
prepared.run()
print('UNEXPECTED_SUCCESS',flush=True)
"""
        child = subprocess.run([sys.executable,'-u','-c',probe], cwd=ROOT,
                               capture_output=True, text=True, timeout=30)
        assert 'READY_NATIVE_VALIDATION_74' in child.stdout, child.stdout + child.stderr
        assert child.returncode in {1,74}, child.stdout + child.stderr
        assert 'UNEXPECTED_SUCCESS' not in child.stdout
        return
    prepared=execution(extend_module(dtype,'int64'),dtype,'int64',[value],lane,tmp_path)
    with pytest.raises(RuntimeError,match='74'):
        prepared.run()
    assert prepared.buffers[1].item() == 0


def test_c634_source_extend_keeps_distinct_rows_and_clean_concordance():
    source = """from array import array
class Ledger:
    def __init__(self):
        self.tokens = array("Q")
    def add(self, token: int):
        cover = []
        cover.append(token)
        self.tokens.extend(cover)
        return len(self.tokens)
def root(token: int):
    ledger = Ledger()
    return ledger.add(token)
"""
    module, _, _ = lower_ast_source_to_ssa(source, 'root', name='c634_checked_extend',
        extraction_contract=ROOT/'extraction_contracts/program_extraction.yaml',
        progress=lambda _:None, progress_bars=False)
    report = concordance_report(module)
    assert '0 finding(s)' in report, report
    casts = [instruction for function in module.functions.values()
             for block in function.blocks.values() for instruction in block.instrs
             if instruction.op == 'Cast' and instruction.attributes.get('binding')
             == 'ssa_sequence_uint64_range']
    assert casts
    assert all(instruction.args[0].dtype == 'int64' and instruction.res.dtype == 'uint64'
               for instruction in casts)


@pytest.mark.parametrize('alias,canonical', [('u64','uint64'),('u32','uint32'),('u8','uint8'),
    ('i32','int32'),('long','int64'),('f32','float32'),('float','float32'),
    ('logical','bool'),('opaque_ref','opaque_ref')])
def test_program_abi_publishes_one_catalogue_type_for_aliases(alias, canonical):
    from src.compiler.extraction_contract import ProgramABIField
    scalar = ProgramABIField.from_mapping('audit.scalar', {'storage':'scalar','dtype':alias})
    assert scalar.dtype == canonical
    table = ProgramABIField.from_mapping('audit.table', {'storage':'table',
        'columns':[{'name':'value','dtype':alias}]})
    assert table.table_columns[0]['dtype'] == canonical
