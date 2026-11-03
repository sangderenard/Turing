"""A dynamic byte sequence retains its proved int64 storage across tensor views."""

import inspect
import textwrap
from pathlib import Path

from src.common.tensors.compression import octet_huffman
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable


def test_public_octets_same_dtype_uses_proved_dynamic_sequence_view():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "_octets", "parameter": "data", "storage": "span",
            "dtype": "int64", "rank": 1, "python_type": "builtins.bytes",
        }],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        textwrap.dedent(inspect.getsource(octet_huffman._octets)),
        "_octets", python_bindings=vars(octet_huffman),
        source_module=octet_huffman.__name__,
        name="dynamic_byte_view", extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    assert not any(
        instruction.op == "Call"
        and str(instruction.attributes.get("callee", "")).startswith("cast_")
        for function in module.functions.values()
        if function.name.startswith("dynamic_byte_view___octets")
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    root = module.functions["dynamic_byte_view___octets"]
    assert root.metadata["value_aliases"][8] == 0
