from src.compiler.binary_ingestion import (
    X86EncodingFields, eligible_reverse_selections, plan_reverse_selection,
    write_reverse_selection,
)
from src.compiler.machine_reference_vocabulary import X86InstructionToken
from src.compiler.machine_code_lifting import raise_binary_region_to_ssa
from src.transmogrifier.ssa_registry import Handler
from src.transmogrifier.ssa import Instr, SSAValue


PADDQ_FACTS = {
    "two-independent-lanes",
    "modulo-2^64",
    "no-cross-lane-carry",
    "xmm-destination-available",
}


def test_reverse_multi_lane_selection_is_disabled_by_default():
    assert eligible_reverse_selections(
        (Handler.VectorAddModulo.value,), proven_facts=PADDQ_FACTS,
    ) == ()


def test_scalar_add_immediate_selection_requires_exact_integer_flags():
    facts = {
        "register-or-memory-destination", "signed-immediate-8",
        "width-64", "modulo-2^64", "all-add-flags-exact",
    }
    selection = eligible_reverse_selections(
        (Handler.Add.value,), proven_facts=facts, allow_multi_lane=True,
    )

    assert len(selection) == 1
    assert selection[0].target_token == int(X86InstructionToken.ADD_R64_IMM8)


def test_reverse_multi_lane_selection_requires_every_state_fact():
    assert eligible_reverse_selections(
        (Handler.VectorAddModulo.value,),
        proven_facts=PADDQ_FACTS - {"no-cross-lane-carry"},
        allow_multi_lane=True,
    ) == ()


def test_reverse_multi_lane_selection_returns_exact_pe_encoding_contract():
    (selection,) = eligible_reverse_selections(
        (Handler.VectorAddModulo.value,), proven_facts=PADDQ_FACTS,
        allow_multi_lane=True,
    )
    assert selection.target_token == int(X86InstructionToken.PADDQ_XMM_XMMM128)
    assert selection.lane_count == 2
    assert selection.lane_width == 64
    assert {"rflags", "mxcsr"} <= selection.preserved_state


def test_scalar_add_is_not_collapsed_by_spelling():
    assert eligible_reverse_selections(
        (Handler.Add.value,), proven_facts=PADDQ_FACTS,
        allow_multi_lane=True,
    ) == ()


def test_reverse_packed_subtraction_requires_no_cross_lane_borrow():
    facts = {
        "two-independent-lanes",
        "modulo-2^64",
        "xmm-destination-available",
    }
    assert eligible_reverse_selections(
        (Handler.VectorSubtractModulo.value,), proven_facts=facts,
        allow_multi_lane=True,
    ) == ()
    (selection,) = eligible_reverse_selections(
        (Handler.VectorSubtractModulo.value,),
        proven_facts=facts | {"no-cross-lane-borrow"},
        allow_multi_lane=True,
    )
    assert selection.target_token == int(X86InstructionToken.PSUBQ_XMM_XMMM128)


def test_rep_movsq_reverse_selection_requires_ordered_architectural_registers():
    facts = {
        "source-register-rsi", "destination-register-rdi",
        "count-register-rcx", "direction-flag-df",
        "ordered-overlap-semantics", "qword-elements",
    }
    assert eligible_reverse_selections(
        (Handler.StridedMemoryCopy.value,),
        proven_facts=facts - {"ordered-overlap-semantics"},
        allow_multi_lane=True,
    ) == ()
    (selection,) = eligible_reverse_selections(
        (Handler.StridedMemoryCopy.value,), proven_facts=facts,
        allow_multi_lane=True,
    )
    assert selection.target_token == int(X86InstructionToken.REP_MOVSQ)


def test_atomic_xadd_reverse_selection_requires_observed_source_and_ordering():
    tokens = (
        Handler.AtomicExchangeAddObserved.value,
        Handler.AtomicExchangeAddMemory.value,
    )
    facts = {
        "memory-destination", "register-source", "width-32",
        "sequentially-consistent", "locked",
        "source-receives-observed", "all-add-flags-exact",
    }
    assert eligible_reverse_selections(
        tokens, proven_facts=facts - {"source-receives-observed"},
        allow_multi_lane=True,
    ) == ()
    (selection,) = eligible_reverse_selections(
        tokens, proven_facts=facts, allow_multi_lane=True,
    )
    assert selection.target_token == int(X86InstructionToken.XADD_RM32_R32)


def test_btc_reverse_selection_does_not_match_unproved_generic_xor_chain():
    tokens = (
        Handler.Shr.value, Handler.And.value,
        Handler.Shl.value, Handler.Xor.value,
    )
    facts = {
        "width-32", "immediate-bit-index", "destination-bit-complement",
        "cf-is-prior-bit", "other-flags-preserved",
    }
    assert eligible_reverse_selections(
        tokens, proven_facts=facts - {"cf-is-prior-bit"},
        allow_multi_lane=True,
    ) == ()
    (selection,) = eligible_reverse_selections(
        tokens, proven_facts=facts, allow_multi_lane=True,
    )
    assert selection.target_token == int(X86InstructionToken.BTC_RM32_IMM8)


def test_reverse_plan_retains_exact_bytes_only_with_consistent_decode_provenance():
    facts = {
        "source-register-rsi", "destination-register-rdi",
        "count-register-rcx", "direction-flag-df",
        "ordered-overlap-semantics", "qword-elements",
    }
    instruction = Instr(
        Handler.StridedMemoryCopy.value, [], SSAValue(9, dtype="memory"),
        attributes={
            "machine_address": 0x1000,
            "machine_token": int(X86InstructionToken.REP_MOVSQ),
            "machine_bytes": "f348a5",
        },
    )
    plan = plan_reverse_selection(
        (instruction,), proven_facts=facts, allow_multi_lane=True,
    )

    assert plan is not None
    assert plan.mode == "exact-retention"
    assert plan.encoded == b"\xf3\x48\xa5"
    assert plan.machine_address == 0x1000
    assert len(plan.witness) == 64


def test_reverse_plan_for_transformed_ssa_is_template_only():
    facts = {
        "source-register-rsi", "destination-register-rdi",
        "count-register-rcx", "direction-flag-df",
        "ordered-overlap-semantics", "qword-elements",
    }
    instruction = Instr(
        Handler.StridedMemoryCopy.value, [], SSAValue(9, dtype="memory"),
    )
    plan = plan_reverse_selection(
        (instruction,), proven_facts=facts, allow_multi_lane=True,
    )

    assert plan is not None
    assert plan.mode == "template-selection"
    assert plan.encoded is None
    assert write_reverse_selection(
        plan, X86EncodingFields(rex=0x48, legacy_prefixes=(0xF3,)),
    ) == b"\xf3\x48\xa5"


def test_reverse_plan_accepts_complete_real_rep_movsq_lowering_group():
    facts = {
        "source-register-rsi", "destination-register-rdi",
        "count-register-rcx", "direction-flag-df",
        "ordered-overlap-semantics", "qword-elements",
    }
    lifting = raise_binary_region_to_ssa(
        b"\xf3\x48\xa5\xc3", maximum_file_size=4, size=4,
        base_address=0x1000, name="rep_movsq_reverse",
        full_vocabulary_report=True, cfg_decode=True,
    )
    group = tuple(
        instruction for block in lifting.function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("machine_address") == 0x1000
    )
    plan = plan_reverse_selection(
        group, proven_facts=facts, allow_multi_lane=True,
    )

    assert len(group) > 1
    assert plan is not None
    assert plan.mode == "exact-retention"
    assert plan.encoded == b"\xf3\x48\xa5"
    assert write_reverse_selection(plan) == b"\xf3\x48\xa5"
