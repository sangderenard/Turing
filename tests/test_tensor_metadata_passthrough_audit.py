"""A folded metadata output retains its exact result and call proofs."""

from src.compiler.identity_concordance import CorrelationTable
from tests import test_dynamic_arange_metadata as arange_case


def test_folded_metadata_result_and_call_reject_wrong_live_rebinding(
    monkeypatch, tmp_path,
):
    captured = []
    lower = arange_case.lower_ast_source_to_ssa

    def capture(*args, **kwargs):
        result = lower(*args, **kwargs)
        captured.append(result[0])
        return result

    monkeypatch.setattr(arange_case, "lower_ast_source_to_ssa", capture)
    arange_case.test_dynamic_arange_cast_and_sum_keep_runtime_count_across_regions(
        tmp_path,
    )
    module = captured[0]
    assert not CorrelationTable._tensor_metadata_result_binding_findings(module)
    assert not CorrelationTable._tensor_metadata_call_binding_findings(module)

    root = next(function for function in module.functions.values()
                if function.name.endswith("__root"))
    calls = tuple(instruction for block in root.blocks.values()
                  for instruction in block.instrs if instruction.op == "Call")
    cast = next(instruction for instruction in calls
                if instruction.attributes.get("callee", "").endswith(
                    "__planned_region_1"
                ))
    bindings = cast.attributes["aggregate_output_passthrough_bindings"]
    altered = list(bindings)
    first = list(altered[0])
    first[3] = altered[1][3]
    altered[0] = tuple(first)
    cast.attributes["aggregate_output_passthrough_bindings"] = tuple(altered)
    assert CorrelationTable._tensor_metadata_result_binding_findings(module)
    cast.attributes["aggregate_output_passthrough_bindings"] = bindings
    assert not CorrelationTable._tensor_metadata_result_binding_findings(module)

    consumer = next(instruction for instruction in calls
                    if instruction.attributes.get("callee", "").endswith(
                        "__planned_region_2"
                    ))
    actuals = consumer.args
    swapped = list(actuals)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    consumer.args = tuple(swapped)
    assert CorrelationTable._tensor_metadata_call_binding_findings(module)
    consumer.args = actuals
    assert not CorrelationTable._tensor_metadata_call_binding_findings(module)
