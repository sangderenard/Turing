from __future__ import annotations

from src.compiler.influence_field import DYNAMIC, InfluenceContract, InfluenceField
from src.compiler.spectral_graph_analysis import analyze_region_spectrum
from tools.spectral_graph_report import _region_record


def _ring_with_tail() -> InfluenceField:
    field = InfluenceField(InfluenceContract(enabled=True, categories=(DYNAMIC,)))
    field.add_sources([(0, DYNAMIC, 0, "n0", "")])
    for index in range(6):
        field.add_edge(index, (index + 1) % 6,
                       role="loop-back" if index == 5 else "control-next")
    field.add_edge(5, "exit", role="control-next")
    return field


def test_public_single_region_analysis_can_avoid_the_whole_graph_dense_path():
    field = _ring_with_tail()
    region = analyze_region_spectrum(field, tuple(range(6)))
    assert region.method == "circulant_fft"
    assert region.is_loop_body is False


def test_report_analyses_a_non_circulant_region_with_dense_eigh():
    field = _ring_with_tail()
    result = _region_record(
        "whole graph", field, field.node_keys(), loop=None, circulant_tol=1e-9,
    )
    assert result["status"] == "analysed"
    assert result["method"] == "dense_eigh"


def test_report_keeps_an_oversized_circulant_region_on_the_fft_path():
    field = _ring_with_tail()
    result = _region_record(
        "loop", field, tuple(range(6)), loop=None, circulant_tol=1e-9,
    )
    assert result["status"] == "analysed"
    assert result["method"] == "circulant_fft"
