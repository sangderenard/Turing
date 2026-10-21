import pytest
from pathlib import Path
import ast

from src.transmogrifier.ssa import (
    SSAValueUnionAlternative,
    SSAValueUnionDescriptor,
    SSAValueUnionStorage,
    SSAValueUnionTable,
    value_union_frame_members,
)


def test_program_abi_record_aliases_are_one_native_type_identity():
    from src.compiler.fortran_c_shell import (
        _canonical_program_abi_type_identity,
        _program_abi_record_for_type_identity,
    )

    program_abi = {"records": {
        "Metrics": {
            "identity": "src.common.dt_system.dt_scaler.Metrics",
            "fields": {},
        },
        "CompilerSSAExtent": {
            "identity": "builtins.int",
            "fields": {
                "value": {"storage": "scalar", "dtype": "int64"},
            },
        },
    }}

    assert _canonical_program_abi_type_identity(
        "Metrics", program_abi,
    ) == "Metrics"
    assert _canonical_program_abi_type_identity(
        "src.common.dt_system.dt_scaler.Metrics", program_abi,
    ) == "Metrics"
    assert _canonical_program_abi_type_identity(
        "int", program_abi,
    ) == "builtins.int"
    assert _canonical_program_abi_type_identity(
        "CompilerSSAExtent", program_abi,
    ) == "CompilerSSAExtent"
    assert _program_abi_record_for_type_identity(
        "builtins.int", program_abi,
    ) is None
    assert _program_abi_record_for_type_identity(
        "CompilerSSAExtent", program_abi,
    ) == "CompilerSSAExtent"
    assert _program_abi_record_for_type_identity(
        "src.common.dt_system.dt_scaler.Metrics", program_abi,
    ) == "Metrics"


def test_optional_fixed_tuple_annotation_retains_exact_member_contracts():
    from src.compiler.fortran_c_shell import (
        _authored_annotation_field_receipt,
        _authored_annotation_python_types,
        _authored_fixed_tuple_member_annotations,
    )

    annotation = ast.parse(
        "tuple[object, int] | None", mode="eval",
    ).body

    members = _authored_fixed_tuple_member_annotations(annotation)

    assert len(members) == 2
    assert _authored_annotation_field_receipt(members[0]) == {
        "storage": "reference",
        "dtype": "opaque_ref",
        "rank": 0,
        "mutable": False,
    }
    assert _authored_annotation_python_types(members[1]) == (
        "builtins.int",
    )
    assert _authored_annotation_field_receipt(members[1]) == {
        "storage": "scalar",
        "dtype": "int64",
        "rank": 0,
        "mutable": False,
    }


def test_optional_sequence_of_bare_collections_uses_reference_elements():
    from src.compiler.fortran_c_shell import (
        _authored_annotation_field_receipt,
        _authored_annotation_python_types,
    )

    annotation = ast.parse("list[dict] | None", mode="eval").body

    assert _authored_annotation_python_types(annotation) == (
        "builtins.NoneType",
        "builtins.list",
    )
    assert _authored_annotation_field_receipt(annotation) == {
        "storage": "span",
        "dtype": "opaque_ref",
        "rank": 1,
        "mutable": True,
        "aggregate_kind": "list",
        "optional": True,
    }


def test_value_union_table_keeps_per_arm_physical_storage():
    descriptor = SSAValueUnionDescriptor(
        union_id=40,
        identity="builtins.int | builtins.str",
        tag_value_id=41,
        alternatives=(
            SSAValueUnionAlternative(
                "builtins.int", SSAValueUnionStorage.SCALAR,
                dtype="int64", value_ids=(42,),
            ),
            SSAValueUnionAlternative(
                "builtins.str", SSAValueUnionStorage.SPAN,
                dtype="int64", shape=(-1,), value_ids=(43,),
            ),
        ),
    )
    table = SSAValueUnionTable(owner="row_admits")
    from src.compiler.concordance_declarations import SCOPE_REGISTRY

    owner_cell = table.book.latest_ref(
        SCOPE_REGISTRY, tuple(table.owner),
    )
    assert owner_cell is not None
    assert table.register(descriptor, sources=(owner_cell,)) is descriptor
    assert table.by_id(40) == descriptor
    assert table.member_claims(41) == ((40, "tag", ("tag",)),)
    assert table.member_claims(42) == (
        (40, "builtins.int", ("value", 0)),
    )


def test_value_union_refuses_unrepresented_payload_arm():
    with pytest.raises(ValueError, match="requires payload value ids"):
        SSAValueUnionAlternative(
            "builtins.str", SSAValueUnionStorage.SPAN, dtype="int64",
        )


def test_value_union_none_arm_has_no_payload():
    arm = SSAValueUnionAlternative(
        "builtins.NoneType", SSAValueUnionStorage.NONE,
    )
    assert arm.value_ids == ()
    with pytest.raises(ValueError, match="cannot own payload storage"):
        SSAValueUnionAlternative(
            "builtins.NoneType", SSAValueUnionStorage.NONE,
            dtype="int64", value_ids=(9,),
        )


@pytest.mark.parametrize("descriptor_id", (200, 201))
@pytest.mark.parametrize("through_wrapper", (False, True))
def test_linked_union_result_claims_provisional_call_tag(
    descriptor_id, through_wrapper,
):
    from src.compiler.fortran_c_shell import (
        _propagate_native_type_tag_results,
    )
    from src.transmogrifier.ssa import (
        BasicBlock,
        Function,
        Instr,
        SSAValue,
    )

    callee_payload = SSAValue(201, dtype="opaque_ref")
    callee_tag = SSAValue(202, dtype="int64")
    callee_unions = SSAValueUnionTable(owner="callee")
    from src.compiler.concordance_declarations import SCOPE_REGISTRY

    callee_owner_cell = callee_unions.book.latest_ref(
        SCOPE_REGISTRY, tuple(callee_unions.owner),
    )
    callee_unions.register(SSAValueUnionDescriptor(
        union_id=descriptor_id,
        identity="builtins.NoneType | builtins.tuple",
        tag_value_id=202,
        alternatives=(
            SSAValueUnionAlternative(
                "builtins.NoneType", SSAValueUnionStorage.NONE,
            ),
            SSAValueUnionAlternative(
                "builtins.tuple", SSAValueUnionStorage.REFERENCE,
                dtype="opaque_ref", value_ids=(201,),
            ),
        ),
    ), sources=(callee_owner_cell,))
    callee = Function(
        "callee", [],
        {"entry": BasicBlock("entry", [
            Instr("Ret", [callee_payload, callee_tag], None),
        ])},
        metadata={
            "native_value_union_return_layouts": ((
                200,
                (
                    (("alternative", "builtins.tuple", 0), 0),
                    (("tag",), 1),
                ),
            ),),
            "native_value_type_abi": {
                200: {
                    "alternatives": (
                        "builtins.NoneType", "builtins.tuple",
                    ),
                    "tags": (
                        ("builtins.NoneType", 0),
                        ("builtins.tuple", 1),
                    ),
                },
            },
            "value_union_table": callee_unions,
        },
    )

    union_fact = {
        "alternatives": ("builtins.NoneType", "builtins.tuple"),
        "tags": (("builtins.NoneType", 0), ("builtins.tuple", 1)),
    }
    functions = {"callee": callee}
    called_symbol = "callee"
    called_outputs = (201, 202)
    call_link_rank = {"callee": 0, "caller": 1}
    if through_wrapper:
        wrapper_payload = SSAValue(301, dtype="opaque_ref")
        wrapper_tag = SSAValue(302, dtype="int64")
        wrapper_aggregate = SSAValue(350, dtype="ssa.aggregate")
        wrapper = Function(
            "wrapper", [],
            {"entry": BasicBlock("entry", [
                Instr("Call", [], wrapper_aggregate, attributes={
                    "callee": "callee",
                    "callee_output_ids": (201, 202),
                    "physical_result_bindings": ((201, 301), (202, 302)),
                }),
                Instr("Load", [wrapper_aggregate], wrapper_payload),
                Instr("Load", [wrapper_aggregate], wrapper_tag),
                Instr("Ret", [wrapper_payload, wrapper_tag], None),
            ])},
            metadata={
                "native_value_union_return_layouts": ((
                    300,
                    (
                        (("alternative", "builtins.tuple", 0), 0),
                        (("tag",), 1),
                    ),
                ),),
                "native_value_type_abi": {300: union_fact},
                "value_union_table": SSAValueUnionTable(owner="wrapper"),
            },
        )
        functions["wrapper"] = wrapper
        called_symbol = "wrapper"
        called_outputs = (301, 302)
        call_link_rank = {"callee": 0, "wrapper": 1, "caller": 2}

    provisional = SSAValue(101, dtype="int64", accounting={
        "native_type_tag_pending_for_value_id": 31,
        "native_type_tag_pending_call_result": True,
    })
    payload = SSAValue(31, dtype="opaque_ref", accounting={
        "native_type_tag_value_id": 101,
    })
    physical_tag = SSAValue(102, dtype="int64")
    aggregate = SSAValue(150, dtype="ssa.aggregate")
    merged_tag = SSAValue(160, dtype="int64")
    linked_call = Instr("Call", [], aggregate, attributes={
        "callee": called_symbol,
        "callee_output_ids": called_outputs,
        "physical_result_bindings": (
            (called_outputs[0], 31), (called_outputs[1], 102),
        ),
    })
    consumer = Instr("Phi", [provisional, physical_tag], merged_tag)
    caller = Function(
        "caller", [provisional],
        {"entry": BasicBlock("entry", [
            linked_call,
            Instr("Load", [aggregate], payload),
            Instr("Load", [aggregate], physical_tag),
            consumer,
            Instr("Ret", [payload], None),
        ])},
        metadata={
            "native_value_type_tags": ((31, 101),),
            "value_union_table": SSAValueUnionTable(owner="caller"),
        },
    )

    functions["caller"] = caller
    _propagate_native_type_tag_results(
        # Deliberately present callers before callees: the sanctioned link
        # rank, not dictionary insertion order, governs propagation.
        dict(reversed(tuple(functions.items()))),
        lambda *_args, **_kwargs: 999,
        call_link_rank,
    )

    assert consumer.args[0] is physical_tag
    assert provisional not in caller.args
    assert caller.metadata["native_value_type_tags"] == ((31, 102),)
    assert payload.accounting["native_type_tag_value_id"] == 102
    descriptor = caller.metadata["value_union_table"].by_id(31)
    assert descriptor is not None
    assert descriptor.tag_value_id == 102


def test_conditional_none_arm_merges_distinct_exact_source_proofs():
    from src.compiler.concordance_declarations import (
        INGEST_SOURCE,
        INGESTION,
        SOURCE_VALUE_CLASS,
        ValueTypeABIAlternative,
    )
    from src.compiler.fortran_c_shell import (
        _merge_conditional_value_type_abi_arm,
    )
    from src.compiler.identity_concordance import (
        Mode,
        Novel,
        begin_identity_book,
        end_identity_book,
    )

    book, token = begin_identity_book()
    try:
        call_cell = book.post(
            SOURCE_VALUE_CLASS, ("conditional-none", 31),
            ("builtins.NoneType",), stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
        literal_cell = book.post(
            SOURCE_VALUE_CLASS, ("conditional-none", 32),
            ("builtins.NoneType",), stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
        call_return = ValueTypeABIAlternative(
            "builtins.NoneType", "none", "", (), 0, (), (call_cell,),
        )
        literal_else = ValueTypeABIAlternative(
            "builtins.NoneType", "none", "", (), 0, (), (literal_cell,),
        )

        merged = _merge_conditional_value_type_abi_arm(
            call_return, literal_else, value_id=63,
        )

        assert merged.source_cells == (call_cell, literal_cell)
        assert (
            merged.identity, merged.storage, merged.dtype,
            merged.shape, merged.rank, merged.storage_identity,
        ) == (
            "builtins.NoneType", "none", "", (), 0, (),
        )
        with pytest.raises(
            ValueError, match="conditional union arm changes its physical ABI"
        ):
            _merge_conditional_value_type_abi_arm(
                ValueTypeABIAlternative(
                    "builtins.int", "scalar", "int64", (), 0, (),
                    (call_cell,),
                ),
                ValueTypeABIAlternative(
                    "builtins.int", "scalar", "float64", (), 0, (),
                    (literal_cell,),
                ),
                value_id=63,
            )
    finally:
        end_identity_book(token)


def test_closed_class_union_uses_one_shared_native_reference_payload():
    descriptor = SSAValueUnionDescriptor(
        union_id=50,
        identity="example.Left | example.Right",
        tag_value_id=51,
        alternatives=tuple(
            SSAValueUnionAlternative(
                identity, SSAValueUnionStorage.REFERENCE,
                dtype="opaque_ref", value_ids=(50,),
            )
            for identity in ("example.Left", "example.Right")
        ),
        layout="shared",
    )
    assert descriptor.to_mapping()["layout"] == "shared"
    assert tuple(
        (member.alternative, member.role, member.value_id)
        for member in value_union_frame_members(descriptor)
    ) == (
        ("", ("tag",), 51),
        ("", ("shared", "value", 0), 50),
    )


def test_program_abi_closed_class_union_lowers_to_payload_and_tag():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report

    repository = Path(__file__).resolve().parents[1]
    contract = ExtractionContract(
        repository / "extraction_contracts" / "program_extraction.yaml"
    ).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [{
            "function": "accepts_mapping",
            "parameter": "value",
            "storage": "reference",
            "type_expectations": ["builtins.dict", "builtins.list"],
        }],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        "from typing import Any\n"
        "def accepts_mapping(value: Any) -> bool:\n"
        "    if isinstance(value, dict):\n"
        "        return value == value\n"
        "    return False\n",
        "accepts_mapping",
        extraction_contract=contract,
        progress=lambda _message: None,
        progress_bars=False,
    )

    function = module.functions[exports[0]]
    descriptor, = module.value_union_tables[function.name].unions.values()
    assert descriptor.layout == "shared"
    payload_id, = {
        value_id
        for arm in descriptor.alternatives
        for value_id in arm.value_ids
    }
    assert {arm.value_ids for arm in descriptor.alternatives} == {
        (payload_id,),
    }
    assert [argument.dtype for argument in function.args] == [
        "opaque_ref", "int64",
    ]
    assert concordance_report(module).splitlines()[0].endswith(
        "0 finding(s)"
    )


def test_authored_class_optional_parameter_keeps_declared_union_abi():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    repository = Path(__file__).resolve().parents[1]
    contract = ExtractionContract(
        repository / "extraction_contracts" / "program_extraction.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        "class Registry:\n"
        "    pass\n"
        "def accepts(registry: Registry | None) -> bool:\n"
        "    return registry is None\n",
        "accepts",
        extraction_contract=contract,
        progress=lambda _message: None,
        progress_bars=False,
    )

    function = module.functions[exports[0]]
    fact = function.metadata["native_value_type_abi"][0]
    assert fact.alternatives[1] == "builtins.NoneType"
    assert fact.alternatives[0].endswith(".Registry")
    assert tuple(
        (arm.identity.rsplit(".", 1)[-1], arm.storage, arm.dtype)
        for arm in fact.physical_alternatives
    ) == (
        ("Registry", "reference", "opaque_ref"),
        ("NoneType", "none", ""),
    )
    assert any(
        source.page.name == "parameter_annotation"
        for arm in fact.physical_alternatives
        for source in arm.source_cells
    )
    assert any(
        source.page.name == "class_declaration"
        for arm in fact.physical_alternatives
        for source in arm.source_cells
    )


@pytest.mark.parametrize("annotation", ["Any", "object"])
def test_open_type_predicate_declares_every_source_class_at_section_api(
    annotation,
):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report

    repository = Path(__file__).resolve().parents[1]
    contract = ExtractionContract(
        repository / "extraction_contracts" / "program_extraction.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        "from typing import Any\n"
        "class Marker:\n"
        "    pass\n"
        f"def classify(value: {annotation}) -> bool:\n"
        "    if isinstance(value, int):\n"
        "        return value == value\n"
        "    return False\n",
        "classify",
        extraction_contract=contract,
        progress=lambda _message: None,
        progress_bars=False,
    )

    function = module.functions[exports[0]]
    descriptor, = module.value_union_tables[function.name].unions.values()
    alternatives = {arm.name for arm in descriptor.alternatives}
    assert descriptor.layout == "shared"
    assert {"Marker", "builtins.int", "builtins.str"} <= alternatives
    assert [argument.dtype for argument in function.args] == [
        "opaque_ref", "int64",
    ]
    assert concordance_report(module).splitlines()[0].endswith(
        "0 finding(s)"
    )


def test_exact_local_receiver_class_does_not_invent_a_physical_type_abi():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    repository = Path(__file__).resolve().parents[1]
    contract = ExtractionContract(
        repository / "extraction_contracts" / "program_extraction.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        "class Descriptor:\n"
        "    def to_mapping(self, value: int) -> int:\n"
        "        return value + 1\n",
        "Descriptor.to_mapping",
        extraction_contract=contract,
        progress=lambda _message: None,
        progress_bars=False,
    )

    function = module.functions[exports[0]]
    assert [argument.dtype for argument in function.args] == ["int64"]
    table = module.value_union_tables.get(function.name)
    assert table is None or not table.unions


def test_mapping_value_column_uses_writer_union_abi():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    repository = Path(__file__).resolve().parents[1]
    contract = ExtractionContract(
        repository / "extraction_contracts" / "program_extraction.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        "from typing import Any\n"
        "def mapping(integer: int, flag: bool) -> dict[str, Any]:\n"
        "    return {'integer': integer, 'flag': flag}\n",
        "mapping",
        extraction_contract=contract,
        progress=lambda _message: None,
        progress_bars=False,
    )

    function = module.functions[exports[0]]
    stores = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation")
        == "table_store"
    ]
    assert len(stores) == 2
    assert all(
        instruction.attributes.get("native_type_tag_sequence")
        for instruction in stores
    )
