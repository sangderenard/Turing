"""Small public-source memory tests; authored Python is the numerical oracle."""
import dataclasses
import io
import sys

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.live_tiles import _ACTIVE_ARENA
from src.compiler.project_compilation_product import compiler_toolchain_fingerprint
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
from src.compiler.work_contract import PRESETS, active_contract, set_active_contract


def test_small_source_uses_paged_work_and_discards_unused_stdout(tmp_path, monkeypatch):
    from src.transmogrifier.graph.graph_express2 import ProcessGraph
    build = ProcessGraph.build_from_ast
    calls = []
    def observed_build(self, *args, **kwargs):
        assert not isinstance(sys.stdout, io.StringIO)
        calls.append(1)
        # The source closure's legacy diagnostics are suppressed, not buffered.
        print("discarded legacy graph diagnostic" * 1000)
        return build(self, *args, **kwargs)
    monkeypatch.setattr(ProcessGraph, "build_from_ast", observed_build)
    previous = active_contract()
    contract = dataclasses.replace(PRESETS["develop"], compress_cold_pages=True,
        spill_cold_pages=False, memory_budget_bytes=1, live_tile_rows=64,
        live_resident_tiles=4096, live_bucket_rows=64, live_decoded_budget_bytes=4 << 20)
    set_active_contract(contract)
    pinned = compiler_toolchain_fingerprint()
    source = "def root(x):\n    value = x\n"
    for index in range(16):
        source += f"    value = value + {(index % 7 + 1) / 8!r}\n"
    source += "    return value\n"
    try:
        module, outputs, exports = lower_ast_source_to_ssa(source, "root", name="bounded_source",
            extraction_contract=ExtractionContract("extraction_contracts/program_extraction.yaml"),
            runtime_closure_only=True, progress=lambda message: None)
    finally:
        set_active_contract(previous)
    assert _ACTIVE_ARENA.get() is None
    owner = object.__getattribute__(module, "__dict__")["_tile_arena"]
    stats = owner.flush()
    raw = sum(tile.raw_bytes for tile in owner._tiles if tile is not None)
    print(f"SOURCE_STORAGE raw_live_tiles={raw} decoded_limit={owner.decoded_budget_bytes} "
          f"disk={stats.disk_bytes} directory_resident={owner._tiles.resident_bytes()}")
    assert calls and stats.disk_tiles > 0 and stats.resident_encoded_bytes == 0
    assert stats.decoded_bytes <= owner.decoded_budget_bytes
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report
    artifact = compile_artifact(emit_ssa_function_to_llvm(module, exports[0]),
        directory=tmp_path / "native", optimization="O3")
    function = module.functions[exports[0]]
    parameter = dict(function.metadata["parameter_names"])["x"]
    namespace = {}
    exec(source, namespace)
    for value in (-2.5, 0.0, 3.25):
        result = prepare_artifact_execution(artifact, {parameter: value}).run()
        actual = result.buffers[outputs[function.name][0].id].reshape(-1)[0]
        assert actual == namespace["root"](value)
    current = compiler_toolchain_fingerprint()
    assert current["sha256"] == pinned["sha256"], "concurrent compiler changes confounded this source run"
