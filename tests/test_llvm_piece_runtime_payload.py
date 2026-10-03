"""Compiler payload retention is optional; the native piece ABI is exact."""

import pickle

import numpy as np
import pytest
import sympy as sp

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.native_law_kernels import (
    LLVMPiece, PieceBuildFact, batch_contract, piece_staleness, post_piece_book,
)
from src.compiler.native_package import piece_from_law
from src.compiler.symbolic_equation_compiler import compile_sympy_equations


@pytest.fixture(scope="module")
def compiled_piece(tmp_path_factory):
    x = sp.Symbol("x")
    equations = (
        sp.Eq(sp.Symbol("result"), x * x + 1, evaluate=False),
        sp.Eq(sp.Symbol("constant"), 7, evaluate=False),
    )
    law = compile_sympy_equations(equations, name="runtime_payload_law")
    return piece_from_law(law, "runtime_payload_law", 3,
                          directory=tmp_path_factory.mktemp("runtime_payload"))


def assert_runtime_payload(piece, full):
    assert piece is not full and piece.artifact is not full.artifact
    assert not piece.retain_compilation
    assert piece.module is None and piece.source is None and piece.outputs is None
    assert piece.artifact.emission is None
    assert piece.entry == full.entry
    assert piece.compiler == full.compiler
    assert piece_staleness(piece) == piece_staleness(full) == ()
    for name in ("name", "llvm_ir", "buffer_order", "buffer_shapes",
                 "buffer_dtypes", "extent_order", "shortfalls", "needs_text_sink",
                 "output_publications", "output_surfaces", "external_slots",
                 "training_steps_value_id", "learning_rate_value_id",
                 "optimizer_state_value_ids", "watched", "watch_shortfalls",
                 "library_path"):
        assert getattr(piece.artifact, name) == getattr(full.artifact, name)
    for name in ("argument_names", "argument_ids", "output_names", "output_ids",
                 "constant_outputs", "batch"):
        assert getattr(piece, name) == getattr(full, name)
    assert piece._execution is None and piece._bound is None and piece.in_place == ()


def test_runtime_copy_preserves_native_binding_and_original_book(compiled_piece):
    full = compiled_piece
    values = np.array([-2.0, 0.0, 3.5])
    # Copy an already executed artifact, including live ctypes handles.
    full(values)
    original_entry = full.artifact._entry
    original_emission = full.artifact.emission
    before = concordance_report(full.module)
    runtime = full.for_runtime()
    assert_runtime_payload(runtime, full)
    destination = np.zeros(3)
    runtime.instantiate({"x": values}, outputs={"result": destination})
    for _ in range(2):
        result = dict(zip(runtime.output_names, runtime(values)))
        assert result["result"] is destination
        np.testing.assert_array_equal(destination, values * values + 1)
        np.testing.assert_array_equal(result["constant"], np.full(3, 7.0))
        values += 0.5
    assert full.artifact._entry is original_entry
    assert full.artifact.emission is original_emission
    assert full.module is not None and full.source and full.outputs is not None
    assert full.retain_compilation
    assert concordance_report(full.module) == before
    print("full archive concordance, unchanged after runtime copy:\n" +
          "\n".join(before.splitlines()[:4]), flush=True)


def test_save_and_load_retention_is_explicit_and_archive_is_preserved(
        compiled_piece, tmp_path):
    full = compiled_piece
    archive = tmp_path / "full.piece"
    compact = tmp_path / "runtime.piece"
    full.save(archive)
    archive_bytes = archive.read_bytes()
    full.save(compact, retain_compilation=False)
    restored_full = LLVMPiece.load(archive)
    assert restored_full.retain_compilation and restored_full.module is not None
    assert restored_full.artifact.emission is not None
    for runtime in (LLVMPiece.load(compact),
                    LLVMPiece.load(archive, retain_compilation=False)):
        assert_runtime_payload(runtime, full)
        values = np.array([-2.0, 0.0, 3.5])
        result = dict(zip(runtime.output_names, runtime(values)))
        np.testing.assert_array_equal(result["result"], values * values + 1)
        np.testing.assert_array_equal(result["constant"], np.full(3, 7.0))
        assert not runtime.for_runtime(retain_compilation=True).retain_compilation
    assert archive.read_bytes() == archive_bytes
    assert compact.stat().st_size < archive.stat().st_size
    print(f"full={archive.stat().st_size}, runtime={compact.stat().st_size} bytes",
          flush=True)


def test_legacy_payload_and_build_receipt_defaults(compiled_piece, tmp_path):
    full = compiled_piece
    legacy = full.for_runtime(retain_compilation=True)
    del legacy.__dict__["retain_compilation"]
    archive = tmp_path / "legacy.piece"
    archive.write_bytes(pickle.dumps(legacy))
    restored = LLVMPiece.load(archive)
    assert restored.retain_compilation and restored.module is not None
    assert restored.artifact.emission is not None
    assert PieceBuildFact("digest", ()).retain_compilation
    runtime = restored.for_runtime()
    book = post_piece_book(tmp_path, "runtime_payload", 3, "law", built=runtime)
    facts = [value for value in book.pages["piece_build"].cells.values()
             if isinstance(value, PieceBuildFact)]
    assert len(facts) == 1 and not facts[0].retain_compilation


def test_runtime_only_piece_refuses_source_linking(compiled_piece):
    runtime = compiled_piece.for_runtime()
    with pytest.raises(ValueError, match="runtime-only.*source linking requires"):
        lower_ast_source_to_ssa(
            "def caller(x):\n    return law(x)\n", "caller",
            python_bindings={"law": runtime},
            extraction_contract=batch_contract("caller", ("x",), 3),
        )
