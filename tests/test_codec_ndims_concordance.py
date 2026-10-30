"""The codec's tensor-rank accessor must carry its scalar class into planning."""

import inspect
import math
import textwrap
from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.compression.bitstream import tensor_octets_to_bytes
from src.compiler.concordance_declarations import SOURCE_VALUE_CLASS
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import begin_identity_book, end_identity_book


def test_codec_ndims_class_is_concorded_before_next_loop_value_refusal():
    # Program: tensor_octets_to_bytes. The tensor element dtype is unknown,
    # but ndims() is an int independent of that dtype. Its missing source
    # class used to stop the compile at node 7 before the loop binding was
    # considered. Grounding must post the class on that exact value ID.
    book, token = begin_identity_book()
    try:
        with pytest.raises(ValueError, match=r"region value 4 \(input 'value'"):
            lower_ast_source_to_ssa(
                textwrap.dedent(inspect.getsource(tensor_octets_to_bytes)),
                "tensor_octets_to_bytes", name="tensor_octets_to_bytes",
                source_module="src.common.tensors.compression.bitstream",
                python_package="src.common.tensors.compression",
                python_bindings={"AbstractTensor": AbstractTensor, "math": math},
                extraction_contract=(
                    Path(__file__).resolve().parents[1]
                    / "extraction_contracts" / "program_extraction.yaml"
                ),
                progress=lambda _message: None, progress_bars=False,
                identity_book=book,
            )
        page = book.page(SOURCE_VALUE_CLASS)
        rows = tuple(row for row in page.rows() if (
            row[1] == 7
            and row[0][0] == (
                "src.common.tensors.compression.bitstream",
                "tensor_octets_to_bytes",
            )
        ))
        assert len(rows) == 1
        assert page.latest(rows[0])[:2] == ("builtins.int", 1)
        receipt = book.latest_ref(SOURCE_VALUE_CLASS, rows[0])
        assert len(tuple(book.edges_into(receipt))) >= 2
    finally:
        end_identity_book(token)
