"""A runtime Mul's display shape cannot certify a fixed consumer extent."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book
from src.compiler.ssa_record_return_state import posted_ssa_value_fact
import src.compiler.tensor_ssa_lowering as tensor_ssa_lowering
from src.compiler.concordance_declarations import (
    TENSOR_HEAP_OWNER_CALL_BINDING, TENSOR_REDUCTION_DOMAIN,
)


def test_runtime_product_and_static_weights_keep_distinct_call_shapes(monkeypatch):
    class AtMetadataBoundary(Exception):
        pass

    observed = []

    def inspect_before_metadata(module, *, finalize_private_owners=True):
        regions = tuple(
            fn for fn in module.functions.values()
            if any(
                ins.op == "Call"
                and ins.attributes.get("callee") == "reduce_dim_double"
                for block in fn.blocks.values() for ins in block.instrs
            )
        )
        assert len(regions) == 1
        consumer = regions[0]
        source = consumer.args[0]
        table = module.tensor_tables[consumer.name]
        dynamic_input = table.by_id(int(source.id))
        assert dynamic_input is not None
        assert dynamic_input.metadata_state != "static"
        assert dynamic_input.byte_size is None
        if dynamic_input.metadata_state == "dynamic":
            assert all(
                getattr(dynamic_input, field) is not None
                for field in (
                    "shape_value_id", "rank_value_id",
                    "element_count_value_id",
                )
            )
        weight_region = next(
            fn for fn in module.functions.values()
            if any(
                (arg.accounting or {}).get("ssa_storage_view", {}).get(
                    "operation"
                ) == "unsqueeze"
                for arg in fn.args
            )
        )
        weight_view = next(
            arg for arg in weight_region.args
            if (arg.accounting or {}).get("ssa_storage_view", {}).get(
                "operation"
            ) == "unsqueeze"
        )
        assert module.tensor_tables[weight_region.name].by_id(
            int(weight_view.id)
        ).metadata_state == "static"
        assert tuple(source.shape) == (1, 8)  # display shape, not full extent
        book = identity_book(module)
        assert not any(
            row[0] == consumer.name and row[1] == int(source.id)
            for row in book.page("ssa_call_shape").rows()
        )
        observed.append(True)
        raise AtMetadataBoundary

    monkeypatch.setattr(
        tensor_ssa_lowering,
        "publish_computed_region_result_metadata",
        inspect_before_metadata,
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "bits", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    return (bits.reshape(-1, 8) * weights.unsqueeze(0)).sum(dim=1)
"""
    try:
        lower_ast_source_to_ssa(
            source, "root", name="dynamic_mixed_call_shape_probe",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtMetadataBoundary:
        pass
    assert observed == [True]


def test_static_unsqueeze_view_keeps_rank_two_when_carrier_is_rank_one(monkeypatch):
    class AtEnrichmentBoundary(Exception):
        pass

    observed = []
    original = tensor_ssa_lowering.propagate_repository_ssa_call_metadata

    def inspect_after_enrichment(module):
        result = original(module)
        regions = tuple(
            fn for fn in module.functions.values()
            if any(ins.op == "Mul" for block in fn.blocks.values()
                   for ins in block.instrs)
            and any(
                (arg.accounting or {}).get("ssa_storage_view", {}).get(
                    "operation"
                ) == "unsqueeze"
                for arg in fn.args
            )
        )
        if not regions:
            return result
        assert len(regions) == 1
        region = regions[0]
        view = next(arg for arg in region.args if
                    (arg.accounting or {}).get("ssa_storage_view", {}).get(
                        "operation"
                    ) == "unsqueeze")
        carrier_id = view.accounting["ssa_storage_view"]["storage_value_id"]
        owner = module.functions[region.metadata["source_region_integral"]["owner"]]
        book = identity_book(module)
        view_fact = posted_ssa_value_fact(region, int(view.id), book=book)
        carrier_fact = posted_ssa_value_fact(owner, int(carrier_id), book=book)
        assert (view_fact.storage, view_fact.rank, view_fact.shape) == (
            "span", 2, (1, 8),
        )
        assert (carrier_fact.storage, carrier_fact.rank) == ("span", 1)
        observed.append(True)
        raise AtEnrichmentBoundary

    monkeypatch.setattr(
        tensor_ssa_lowering,
        "propagate_repository_ssa_call_metadata",
        inspect_after_enrichment,
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "bits", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    return (bits.reshape(-1, 8) * weights.unsqueeze(0)).sum(dim=1)
"""
    try:
        lower_ast_source_to_ssa(
            source, "root", name="static_view_rank_probe",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtEnrichmentBoundary:
        pass
    assert observed == [True]


def test_dynamic_axis_sum_projects_result_count_from_source_shape(monkeypatch):
    class AtReductionBoundary(Exception):
        pass

    observed = []
    original = tensor_ssa_lowering.publish_computed_region_result_metadata

    def inspect(module, *, finalize_private_owners=True):
        book = identity_book(module)
        rows = tuple(
            row for row in book.page(TENSOR_REDUCTION_DOMAIN).rows()
            if book.page(TENSOR_REDUCTION_DOMAIN).latest(row)[0]
            == "axis-runtime"
        )
        if not rows:
            return original(
                module, finalize_private_owners=finalize_private_owners,
            )
        assert len(rows) == 1
        function_name, result_id = rows[0]
        fact = book.page(TENSOR_REDUCTION_DOMAIN).latest(rows[0])
        assert fact[:5] == ("axis-runtime", fact[1], "sum", 1, False)
        function = module.functions[function_name]
        table = module.tensor_tables[function_name]
        source = table.by_id(fact[1])
        result = table.by_id(result_id)
        assert source.metadata_state == result.metadata_state == "dynamic"
        assert (source.shape_value_id, source.rank_value_id,
                source.element_count_value_id) == fact[5:8]
        assert (result.shape_value_id, result.rank_value_id,
                result.element_count_value_id) == fact[8:11]
        kernel_calls = tuple(
            ins for block in function.blocks.values() for ins in block.instrs
            if ins.op == "Call"
            and ins.attributes.get("callee") == "reduce_dim_double"
        )
        assert len(kernel_calls) == 1
        assert tuple(int(value.id) for value in kernel_calls[0].args[2:4]) \
            == fact[5:7]
        owner_rows = tuple(
            row for row in book.page(TENSOR_HEAP_OWNER_CALL_BINDING).rows()
            if row[-2:] == (function_name, result_id)
        )
        assert len(owner_rows) == 1
        owner_row = owner_rows[0]
        owner_fact = book.page(TENSOR_HEAP_OWNER_CALL_BINDING).latest(owner_row)
        caller = module.functions[owner_row[1]]
        region_call = next(
            ins for block in caller.blocks.values() for ins in block.instrs
            if ins.op == "Call" and ins.res is not None
            and int(ins.res.id) == owner_row[2]
            and ins.attributes.get("callee") == function_name
        )
        assert int(region_call.args[-1].id) == owner_fact[1]
        assert int(function.args[-1].id) == owner_fact[2]
        assert region_call.attributes["feed_ids"][-1] == owner_fact[1]
        assert region_call.attributes["feed_dtypes"][-1] == "ptr"
        assert len(region_call.attributes["feed_ids"]) == len(region_call.args)
        assert len(region_call.attributes["feed_shapes"]) == len(region_call.args)
        assert len(region_call.attributes["feed_dtypes"]) == len(region_call.args)
        assert not any(
            ins.op == "extent" and ins.args
            and int(ins.args[0].id) == result_id
            for block in function.blocks.values() for ins in block.instrs
        )
        observed.append(True)
        raise AtReductionBoundary

    monkeypatch.setattr(
        tensor_ssa_lowering,
        "publish_computed_region_result_metadata", inspect,
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "bits", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    shaped = bits.reshape(-1, 8).to_dtype('float64')
    return (shaped * weights.unsqueeze(0)).sum(dim=1).to_dtype('int64').sum().item()
"""
    try:
        lower_ast_source_to_ssa(
            source, "root", name="runtime_axis_sum_geometry_probe",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtReductionBoundary:
        pass
    assert observed == [True]
