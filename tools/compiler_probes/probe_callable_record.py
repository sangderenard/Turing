from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

source = r'''
from dataclasses import dataclass
from typing import Callable

@dataclass
class Law:
    advance: Callable[[float], float]

def add_one(value):
    return value + 1.0

def root(value):
    law = Law(add_one)
    return law.advance(value)
'''

module, outputs, exports = lower_ast_source_to_ssa(
    source,
    "root",
    name="callable_record",
    extraction_contract=ExtractionContract(
        Path(__file__).resolve().parents[2]
        / "extraction_contracts" / "program_extraction.yaml"
    ),
    progress=lambda message: print(f"[lower] {message}", flush=True),
)
print(f"exports={exports!r}", flush=True)
print(f"functions={tuple(module.functions)!r}", flush=True)
for name, table in module.sequence_tables.items():
    print("sequence_table=" + repr((name, {
        key: value.to_mapping() for key, value in table.sequences.items()
    })), flush=True)
for name, table in module.record_tables.items():
    print("record_table=" + repr((name, {
        key: value.to_mapping() for key, value in table.records.items()
    })), flush=True)
for name, function in module.functions.items():
    print("function_args=" + repr((name, [
        (value.id, value.dtype, value.shape, value.accounting)
        for value in function.args
    ])), flush=True)
    for block_name, block in function.blocks.items():
        for instruction in block.instrs:
            print("instruction=" + repr((
                name, block_name, instruction.op,
                None if instruction.res is None else (
                    instruction.res.id, instruction.res.dtype,
                    instruction.res.shape, instruction.res.accounting,
                ),
                [(value.id, value.dtype, value.shape, value.accounting)
                 for value in instruction.args],
                instruction.attributes,
            )), flush=True)
