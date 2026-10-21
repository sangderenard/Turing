"""Sequence lookup helpers and nested-sequence iterables post their physical
facts at the producer that first knows them.

Keyed rank-one Tensor lookup: the generated ``ssa_sequence_N_lookup`` helper's
formals (descriptor storage cells the caller minted, plus the helper's own
query formal) carried ``physical_dtype`` in accounting only (audit
``ssa-physical-unposted``).  ``list[list[float]]`` parameter: the iterable's
dtype was learned after its ``ssa_value`` row was posted (dtype None) and set
on the object only, so the settlement refused ("unresolved evidence cannot
erase resident type").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable,
    begin_identity_book,
    end_identity_book,
)

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / (
    "program_extraction.yaml"
)

_UNSOURCED = {"unsourced-fact", "unsourced-identity"}


@pytest.fixture(autouse=True)
def _isolated_book():
    _book, token = begin_identity_book()
    try:
        yield
    finally:
        end_identity_book(token)


def _unposted_or_unsourced(module):
    return [
        finding
        for finding in CorrelationTable.build(module).findings(module)
        if finding.kind in _UNSOURCED
        or finding.kind == "ssa-physical-unposted"
    ]


def test_keyed_tensor_lookup_helper_posts_physical_dtype():
    from tests.test_sequence_contract_concordance import (
        c_backend_repository_ssa_reference,
    )

    module = lower_ast_source_to_ssa(
        "from dataclasses import dataclass\n"
        "from typing import Mapping\n"
        "import torch\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Reading:\n"
        "    voltage_v: Mapping[str, torch.Tensor]\n"
        "\n"
        "def sample(reading: Reading, key: str):\n"
        "    selected = reading.voltage_v[key]\n"
        "    return selected.real.sum()\n",
        "sample",
        name="keyed_tensor_rows",
        extraction_contract=ExtractionContract(CONTRACT),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )[0]
    assert any(
        name.endswith("ssa_sequence_2_lookup") for name in module.functions
    )
    assert _unposted_or_unsourced(module) == []


def test_nested_list_parameter_lowers_with_posted_iterable_dtype():
    module = lower_ast_source_to_ssa(
        "def f(rows: list[list[float]]) -> float:\n"
        " total = 0.0\n"
        " for r in rows:\n"
        "  for x in r:\n"
        "   total = total + x\n"
        " return total\n",
        "f",
        extraction_contract=ExtractionContract(CONTRACT),
        progress=lambda _message: None, progress_bars=False,
    )[0]
    assert _unposted_or_unsourced(module) == []
