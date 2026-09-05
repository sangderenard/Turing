from src.compiler.machine_code_lifting import raise_binary_region_to_ssa
from src.compiler.pe_recompilation import (
    PERecompilationLedger, build_pe_recompilation_ledger,
    recompile_pe_in_place,
)
from src.compiler.binary_ingestion import parse_pe_image
from src.compiler.machine_reference_vocabulary import X86InstructionToken
from src.compiler.x86_tensor_read_head import X86AllocatedInstruction
from tests.test_machine_state_buffer import _minimal_amd64_pe_return
from src.transmogrifier.ssa import IRModule


def _module(encoded: bytes):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="subject",
        full_vocabulary_report=True, cfg_decode=True,
    )
    assert lifting.failed_vocabulary == ()
    return IRModule({"subject": lifting.function})


def test_unchanged_ssa_groups_form_complete_exact_retention_ledger():
    ledger = build_pe_recompilation_ledger(
        _module(b"\x83\xc0\x01\xc3"),
    )

    assert ledger.complete
    assert ledger.unresolved == ()
    assert tuple(item.encoded for item in ledger.occurrences) == (
        b"\x83\xc0\x01", b"\xc3",
    )
    assert all(item.disposition == "exact-retention" for item in ledger.occurrences)


def test_changed_ssa_group_cannot_claim_original_bytes():
    module = _module(b"\x83\xc0\x01\xc3")
    function = module.functions["subject"]
    instruction = next(
        item for block in function.blocks.values() for item in block.instrs
        if item.attributes.get("machine_address") == 0x1000 and item.res is not None
    )
    instruction.attributes["tampered"] = True

    ledger = build_pe_recompilation_ledger(module)

    changed = next(item for item in ledger.occurrences if item.machine_address == 0x1000)
    assert changed.disposition == "unresolved"
    assert changed.encoded is None
    assert not ledger.complete


def test_complete_ledger_recompiles_and_reparses_pe_in_place():
    encoded = _minimal_amd64_pe_return()
    image, _statistics = parse_pe_image(
        encoded, maximum_file_size=len(encoded),
    )
    function_rva = image.runtime_functions[0].begin_rva
    file_offset = image.file_offset_for_rva(function_rva)
    assert file_offset is not None
    module = _module(encoded[file_offset:file_offset + 1])
    # The helper used a fixed 0x1000 base; align provenance with this PE.
    function = module.functions["subject"]
    delta = int(image.image_base) + function_rva - 0x1000
    for block in function.blocks.values():
        for instruction in block.instrs:
            address = instruction.attributes.get("machine_address")
            if address is not None:
                instruction.attributes["machine_address"] = int(address) + delta
    # Restamp because address provenance is deliberately part of the witness.
    from src.compiler.machine_code_lifting import _stamp_machine_group_fingerprints
    _stamp_machine_group_fingerprints(function)
    ledger = build_pe_recompilation_ledger(module)

    recompiled = recompile_pe_in_place(image, ledger)

    assert recompiled.encoded == encoded
    assert recompiled.image.entrypoint_rva == image.entrypoint_rva
    assert len(recompiled.rewritten_ranges) == 1


def test_identical_duplicate_occurrences_are_retained_but_written_once():
    encoded = _minimal_amd64_pe_return()
    image, _statistics = parse_pe_image(encoded, maximum_file_size=len(encoded))
    function_rva = image.runtime_functions[0].begin_rva
    file_offset = image.file_offset_for_rva(function_rva)
    module = _module(encoded[file_offset:file_offset + 1])
    function = module.functions["subject"]
    delta = int(image.image_base) + function_rva - 0x1000
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.attributes.get("machine_address") is not None:
                instruction.attributes["machine_address"] += delta
    from src.compiler.machine_code_lifting import _stamp_machine_group_fingerprints
    _stamp_machine_group_fingerprints(function)
    single = build_pe_recompilation_ledger(module)
    duplicated = PERecompilationLedger(single.occurrences + single.occurrences)

    result = recompile_pe_in_place(image, duplicated)

    assert len(duplicated.occurrences) == 2
    assert len(result.rewritten_ranges) == 1
    assert result.encoded == encoded


def test_changed_group_is_rewritten_by_bidirectional_head_after_proof():
    module = _module(b"\xf3\x48\xa5\xc3")
    function = module.functions["subject"]
    group = tuple(
        item for block in function.blocks.values() for item in block.instrs
        if item.attributes.get("machine_address") == 0x1000
    )
    group[0].attributes["allocation_revision"] = 1
    facts = {
        "source-register-rsi", "destination-register-rdi",
        "count-register-rcx", "direction-flag-df",
        "ordered-overlap-semantics", "qword-elements",
    }

    ledger = build_pe_recompilation_ledger(
        module,
        proven_facts_by_address={0x1000: facts},
        allocated_instructions_by_address={
            0x1000: X86AllocatedInstruction(
                int(X86InstructionToken.REP_MOVSQ), (), 0x1000,
            ),
        },
        allow_new_selection=True,
    )

    selected = next(item for item in ledger.occurrences if item.machine_address == 0x1000)
    assert selected.target_token == int(X86InstructionToken.REP_MOVSQ)
    assert selected.encoded == b"\xf3\x48\xa5"
    assert selected.disposition == "allocated-selection"


def test_scoped_ledger_emits_only_selected_repository_ssa_block():
    module = _module(b"\x74\x03\x83\xc0\x01\xc3")
    function = module.functions["subject"]
    blocks = {
        name: tuple(addresses)
        for name, addresses in function.metadata["machine_block_addresses"]
        if addresses
    }
    selected_name = next(iter(blocks))

    ledger = build_pe_recompilation_ledger(
        module, selected_blocks=(("subject", selected_name),),
    )

    assert ledger.complete
    assert tuple(item.machine_address for item in ledger.occurrences) == blocks[selected_name]


def test_scoped_ledger_rejects_machine_address_membership_loss():
    module = _module(b"\x83\xc0\x01\xc3")
    function = module.functions["subject"]
    block_name, addresses = next(
        (name, addresses)
        for name, addresses in function.metadata["machine_block_addresses"]
        if addresses
    )
    removed_address = int(addresses[0])
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.attributes.get("machine_address") == removed_address:
                instruction.attributes.pop("machine_address")

    try:
        build_pe_recompilation_ledger(
            module, selected_blocks=(("subject", block_name),),
        )
    except ValueError as error:
        assert "machine-address membership changed" in str(error)
    else:
        raise AssertionError("scoped emission accepted a deleted machine group")
