from src.compiler.influence_field import DYNAMIC, InfluenceContract, InfluenceField
from src.compiler.spectral_trace_dye import (
    analyse_trace_dye, compare_emission_sequences, emissions_from_shell_payload,
)


def _field():
    field = InfluenceField(InfluenceContract(enabled=True, categories=(DYNAMIC,)))
    field.add_sources([(("f", "entry", 1), DYNAMIC, 0, "x", "")])
    field.add_edge(("f", "entry", 1), ("f", "body", 2), role="data")
    field.propagate()
    return field


def test_telemetry_trace_keeps_real_timestamps_and_durations():
    emissions = emissions_from_shell_payload({"records": [
        {"kind": "trace", "sequence": 2, "at_ns": 30,
         "detail": {"region": 7, "nanoseconds": 11}},
        {"kind": "trace", "sequence": 1, "at_ns": 10,
         "detail": {"region": 7, "nanoseconds": 13}},
    ]})
    assert [(item.at_ns, item.duration_ns) for item in emissions] == [(10, 13), (30, 11)]


def test_native_trace_ring_uses_measured_durations_to_place_events():
    emissions = emissions_from_shell_payload({"trace": {"launches": [
        {"seq": 3, "region": 4, "shell_ns": 7},
        {"seq": 4, "region": 4, "shell_ns": 11},
    ]}})
    assert [(item.at_ns, item.duration_ns) for item in emissions] == [(0, 7), (7, 11)]


def test_trace_only_report_keeps_real_timing_without_guessing_ssa_targets():
    report = analyse_trace_dye({"trace": {"launches": [
        {"seq": 0, "region": 4, "shell_ns": 7},
        {"seq": 1, "region": 4, "shell_ns": 11},
    ]}})
    target = report["targets"][0]
    assert target["total_duration_ns"] == 18
    assert target["field_keys"] == [] and target["paths"] == []
    assert target["resolution"].startswith("unresolved")


def test_trace_analysis_resolves_target_dye_paths_and_cadence_phase():
    report = analyse_trace_dye(
        {"records": [
            {"kind": "trace", "sequence": 1, "at_ns": 0, "detail": {"region": 0, "nanoseconds": 5}},
            {"kind": "trace", "sequence": 2, "at_ns": 1_000_000_000, "detail": {"region": 0, "nanoseconds": 7}},
        ]},
        {"sites": [{"site": 0}], "levels": {"ssa": {"0": [2]}}}, _field(),
    )
    target = report["targets"][0]
    assert target["field_keys"] == ["('f', 'body', 2)"]
    assert target["frequency_hz"] == 1.0
    assert target["timings"][1]["phase"] == 0.0
    assert target["paths"] and target["paths"][0]["edges"][0]["role"] == "data"


def test_authored_target_selection_keeps_only_its_manifest_producing_site():
    payload = {"records": [
        {"kind": "trace", "sequence": 1, "at_ns": 0, "detail": {"region": 0}},
        {"kind": "trace", "sequence": 2, "at_ns": 1, "detail": {"region": 1}},
    ]}
    manifest = {
        "sites": [{"site": 0}, {"site": 1}],
        "levels": {"ssa": {"0": [2], "1": [9]}},
        "names": {"loss": [9]},
    }
    report = analyse_trace_dye(payload, manifest, _field(), target_names=("loss", "missing"))
    assert [target["site"] for target in report["targets"]] == [1]
    assert report["unmatched_target_names"] == ["missing"]


def test_paired_trace_names_the_first_control_split_before_any_value_claim():
    reference = {"trace": {"launches": [{"seq": 0, "region": 1, "shell_ns": 4}]}}
    observed = {"trace": {"launches": [{"seq": 0, "region": 2, "shell_ns": 4}]}}
    result = compare_emission_sequences(reference, observed)
    assert result["equal"] is False and result["kind"] == "control" and result["index"] == 0
