import json
from types import SimpleNamespace

from src.compiler.machine_block_recompiler import JOURNAL_STRIDE
from src.compiler.machine_block_web_bundle import build_machine_block_web_bundle


class _Artifact:
    entry_address = 0x140001000
    block_digest = "block-digest"
    binary = b"\0asm\x01\0\0\0"
    wat = '(module (memory (export "memory") 1) (func (export "run")))'
    witnesses = (SimpleNamespace(
        operation_index=0,
        address=0x140001000,
        semantic="NO_OPERATION",
        semantic_id=19,
        encoded=b"\x90",
        encoded_digest="9e076ceaf246b600c7a9b067c11c5d808e25c8ecb3f3e7e90d1b6c6bb42ad72b",
        journal_offset=0,
        possible_next_addresses=(0x140001001,),
        expected_stack_effect=(0, 0, 0),
    ),)
    shortfalls = (SimpleNamespace(
        operation_index=1, address=0x140001001,
        semantic="RETURN", reason="outer return stays in lifecycle tier",
    ),)
    continuation_address = 0x140001001
    possible_continuations = (0x140001001,)
    guest_memory_base = 0
    guest_memory_size = 0
    state_abi = {"schema": "turing.machine-block-state.v2"}
    specialization_guard = {}
    complete = False
    covered_operation_count = 1

    @staticmethod
    def pack_state(_state):
        return b"state"

    @staticmethod
    def pack_guest_memory(_state):
        return b""


def test_machine_block_web_bundle_retains_wasm_state_and_instruction_provenance():
    bundle = build_machine_block_web_bundle(
        _Artifact(), object(), subject_sha256="ab" * 32,
    )

    assert bundle.assets["machine/recompiled-entry/block.wasm"].startswith(b"\0asm")
    assert bundle.assets["machine/recompiled-entry/initial-state.bin"] == b"state"
    assert bundle.descriptor["journal_bytes"] == JOURNAL_STRIDE
    assert bundle.descriptor["expected_first_witness"] == {
        "address": 0x140001000,
        "semantic_id": 19,
        "digest_prefix": "9e076ceaf246b600",
    }
    plan = json.loads(bundle.assets[
        "machine/recompiled-entry/dispatch-plan.json"
    ])
    assert plan["subject_sha256"] == "ab" * 32
    assert plan["witnesses"][0]["encoded_hex"] == "90"
    assert plan["witnesses"][0]["possible_next_addresses"] == [0x140001001]
    assert plan["shortfalls"][0]["semantic"] == "RETURN"


def test_machine_block_web_bundle_rejects_escaping_asset_root():
    try:
        build_machine_block_web_bundle(
            _Artifact(), object(), subject_sha256="ab" * 32, root="../escape",
        )
    except ValueError as error:
        assert "escapes" in str(error)
    else:
        raise AssertionError("escaping browser-block root must fail closed")
