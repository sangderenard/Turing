"""An inserted private heap release revises its block from its booked effect."""

from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_BLOCK_GEOMETRY, TENSOR_HEAP_RELEASE_BINDING,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import IdentityBook


def test_private_release_geometry_cites_owner_decision(monkeypatch):
    source = """def root(lengths: AbstractTensor, bit_count: int) -> AbstractTensor:
    return AbstractTensor.arange(bit_count, cls=type(lengths)).to_dtype("float64").sum().item()
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "lengths", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "bit_count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    post = IdentityBook.post
    observed = []

    class ReleaseGeometry(Exception):
        pass

    def capture(book, page, row, fact, **options):
        result = post(book, page, row, fact, **options)
        if (page == SSA_BLOCK_GEOMETRY
                and row[1] == "root__root" and row[2] == "entry"
                and any(source.page == TENSOR_HEAP_RELEASE_BINDING
                        for source in options["provenance"].cells)):
            observed.append((book, result, row))
            raise ReleaseGeometry
        return result

    with monkeypatch.context() as patch:
        patch.setattr(IdentityBook, "post", capture)
        with pytest.raises(ReleaseGeometry):
            lower_ast_source_to_ssa(
                source, "root", extraction_contract=contract,
                python_bindings={"AbstractTensor": AbstractTensor},
                tensor_ssa_reference=c_backend_repository_ssa_reference(),
                progress=lambda _: None, progress_bars=False,
            )
    book, geometry, row = observed[-1]
    releases = tuple(
        cause for cause, _ in book.edges_into(geometry)
        if cause.page == TENSOR_HEAP_RELEASE_BINDING
    )
    assert len(releases) == 2
    instructions = book.page(SSA_BLOCK_GEOMETRY).cell(
        geometry.row, geometry.column,
    )[0]
    for release in releases:
        assert release.row[:2] == row[:2]
        assert book.page(TENSOR_HEAP_RELEASE_BINDING).cell(
            release.row, release.column,
        )[2] == row[2]
        assert any(instruction[0] == "Call"
                   and instruction[1] == (release.row[2],)
                   and instruction[2] is None
                   for instruction in instructions)
