from types import SimpleNamespace
from pathlib import Path
import hashlib
import pytest
from tools.add_arena_line_driver import patch_arena_line_driver,ANCHOR,CONTROL,PAINT,GUARD
from src.compiler.llvm_training_runtime import _compile_artifact_to_wasm


def test_patch_exact_reversal_and_hash_guard():
    data=('before\n'+ANCHOR+'\n'+GUARD+'\n'+CONTROL+'\n'+PAINT+'\nafter').encode()
    out,r=patch_arena_line_driver(data,expected_source_sha256=hashlib.sha256(data).hexdigest())
    assert r['source_recovered_exactly'] and not r['pose_writes']
    assert b'arenaLineControls(controls)' in out
    with pytest.raises(ValueError):patch_arena_line_driver(data,expected_source_sha256='bad')
    with pytest.raises(ValueError):patch_arena_line_driver(out,expected_source_sha256=hashlib.sha256(out).hexdigest())


def test_explicit_clang_preserves_ir_and_exports(tmp_path,monkeypatch):
    clang=tmp_path/'clang.exe';clang.write_bytes(b'fixture')
    monkeypatch.setenv('TURING_WASM_CLANG',str(clang))
    calls=[]
    def run(command,**kwargs):
        calls.append(command);Path(command[command.index('-o')+1]).write_bytes(b'\0asm')
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr('src.compiler.llvm_training_runtime.subprocess.run',run)
    a=SimpleNamespace(name='pilot',shortfalls=(),llvm_ir='source_filename = "same"\n')
    _compile_artifact_to_wasm(a,directory=tmp_path,what='test')
    assert calls[0][0]==str(clang)
    assert '--target=wasm32-unknown-unknown' in calls[0]
    assert '-Wl,--export=pilot' in calls[0]
    assert (tmp_path/'pilot.ll').read_text()==a.llvm_ir


def test_invalid_compiler_is_explicit_error(tmp_path,monkeypatch):
    monkeypatch.setenv('TURING_WASM_CLANG',str(tmp_path/'absent'))
    a=SimpleNamespace(name='pilot',shortfalls=(),llvm_ir='same')
    with pytest.raises(ValueError,match='TURING_WASM_CLANG'):_compile_artifact_to_wasm(a,directory=tmp_path,what='test')

def test_default_zig_driver_unchanged(tmp_path,monkeypatch):
    import sys
    monkeypatch.delenv('TURING_WASM_CLANG',raising=False)
    calls=[]
    def run(command,**kwargs):
        calls.append(command);Path(command[command.index('-o')+1]).write_bytes(b'\0asm')
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr('src.compiler.llvm_training_runtime.subprocess.run',run)
    a=SimpleNamespace(name='pilot',shortfalls=(),llvm_ir='same')
    _compile_artifact_to_wasm(a,directory=tmp_path,what='test')
    assert calls[0][:4]==[sys.executable,'-m','ziglang','cc']
    assert '--target=wasm32-freestanding' in calls[0]
