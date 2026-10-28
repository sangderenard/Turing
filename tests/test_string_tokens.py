"""Words lower to universal content tokens, compared as 64-bit identities.

A string constant interns to a content token (the same FNV-1a the container keys
and the runtime name primitive fold with), recorded in a central table; a
comparison of tokens lowers to a 64-bit identity test, not a float compare (the
token is held in the working type as reinterpreted bits). Verified in Node: a
runtime value equals a constant word iff their tokens match.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler import string_table as string_table_module
from src.compiler.string_table import (
    StringTable, StringTokenCollision, string_token,
)
from src.compiler.concordance_declarations import STRING_TABLE_ENTRY
from src.compiler.identity_concordance import (
    RowFieldKind, begin_identity_book, end_identity_book,
)
from src.compiler.ir_string_interning import intern_string_constants, STRING_TOKEN
from src.compiler.fused_program_wasm_backend import emit_wasm_module


def test_string_table_token_row_is_a_content_label(tmp_path):
    book, token = begin_identity_book()
    try:
        table = StringTable(root=tmp_path)
        value = table.intern("content-row")
        assert STRING_TABLE_ENTRY.row_fields[0].kind is RowFieldKind.LABEL
        assert book.page(STRING_TABLE_ENTRY).latest((value,)) == "content-row"
    finally:
        end_identity_book(token)


def test_string_table_refuses_a_second_word_for_one_token(tmp_path, monkeypatch):
    """TOK0: a hash collision is refused, never resolved by keeping the first
    word. The table persists across builds, so the second word may arrive
    under a fresh identity book that has no row to disagree with."""
    colliding = {"alpha", "beta"}
    real_token = string_table_module.string_token
    monkeypatch.setattr(
        string_table_module, "string_token",
        lambda text: 42 if text in colliding else real_token(text),
    )
    first_book, first = begin_identity_book()
    try:
        table = StringTable(root=tmp_path)
        assert table.intern("alpha") == 42
        assert table.intern("alpha") == 42
    finally:
        end_identity_book(first)
    _book, second = begin_identity_book()
    try:
        with pytest.raises(StringTokenCollision, match="recorded='alpha' proposed='beta'"):
            table.intern("beta")
    finally:
        end_identity_book(second)
    assert table.get(42) == "alpha"


def test_intern_fold_replaces_strings_and_tags_comparison(tmp_path):
    table = StringTable(root=tmp_path)
    program = FusedProgram(
        version=1, feeds={5},
        steps=[OpStep(0, "tensor_from_list", [], {"values": "node-wasm"}, 10),
               OpStep(1, "not_equal", [5, 10], {}, 11)],
        outputs={"r": 11}, meta={},
    )
    folded = intern_string_constants(program, table)
    tok_step = next(s for s in folded.steps if s.op_name == STRING_TOKEN)
    assert tok_step.attrs["token"] == string_token("node-wasm")
    cmp_step = next(s for s in folded.steps if s.op_name == "not_equal")
    assert cmp_step.attrs.get("string_compare") is True
    assert table.get(string_token("node-wasm")) == "node-wasm"


def test_constant_string_concatenation_folds_to_one_token(tmp_path):
    table = StringTable(root=tmp_path)
    program = FusedProgram(
        version=1, feeds=set(),
        steps=[OpStep(0, "tensor_from_list", [], {"values": "PE cycle: "}, 20),
               OpStep(1, "tensor_from_list", [], {"values": ".dll"}, 21),
               OpStep(2, "add", [20, 21], {}, 22)],
        outputs={"m": 22}, meta={},
    )
    folded = intern_string_constants(program, table)
    tokens = [s for s in folded.steps if s.op_name == STRING_TOKEN]
    assert len(tokens) == 1
    assert tokens[0].attrs["text"] == "PE cycle: .dll"
    assert tokens[0].result_id == 22


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_string_comparison_runs_as_identity_test(tmp_path):
    table = StringTable(root=tmp_path)
    program = FusedProgram(
        version=1, feeds={5},
        steps=[OpStep(0, "tensor_from_list", [], {"values": "node-wasm"}, 10),
               OpStep(1, "not_equal", [5, 10], {}, 11)],
        outputs={"r": 11}, meta={},
        extras={"capture_feed_origins": {5: {"binding_name": "x"}}},
    )
    module = emit_wasm_module(intern_string_constants(program, table),
                             name="strcmp", dtype="float64")
    assert module.complete, module.shortfall_report()
    wasm = tmp_path / "strcmp.wasm"
    wasm.write_bytes(module.binary)
    token = string_token("node-wasm")
    script = tmp_path / "run.mjs"
    # Pass the token as a BigInt literal (a JS Number would lose precision above
    # 2**53), write it into x's cell, and read the boolean result back.
    script.write_text(
        f"""
        import {{readFileSync}} from "node:fs";
        const {{run, memory}} = (await WebAssembly.instantiate(
          readFileSync(process.argv[2]), {{}})).instance.exports;
        const X = 64, OUT = 128;
        const i64 = new BigInt64Array(memory.buffer);
        function neq(xtok) {{ i64[X / 8] = xtok; run(1, X, OUT);
          return new Float64Array(memory.buffer)[OUT / 8]; }}
        console.log(JSON.stringify({{
          same: neq({token}n),   // x == 'node-wasm'  -> not_equal 0
          diff: neq(12345n),     // different word    -> not_equal 1
        }}));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(wasm)], capture_output=True, text=True, check=True,
    )
    import json
    assert json.loads(completed.stdout) == {"same": 0, "diff": 1}, completed.stdout
