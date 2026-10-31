"""Probe ordinary authored str operations through the public SSA compiler."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.concordance_declarations import STRING_TABLE_ENTRY  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402


SOURCE = '''def root(text: str):
    return len(text + "cd")
'''


def main():
    print("authored_source=str parameter + literal concat + len", flush=True)
    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="authored_string_orchard_probe",
        extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts" / "program_extraction.yaml"
        ),
        progress=lambda message: print(message, flush=True)
        if message.startswith("stage ") else None,
        progress_bars=False,
    )
    book = module.metadata["identity_book"]
    rows = tuple(book.page(STRING_TABLE_ENTRY).rows())
    print(f"ssa_functions={len(module.functions)} exports={exports!r}", flush=True)
    print(f"string_table_rows={len(rows)} token_orchard={bool(module.metadata.get('token_orchard'))}", flush=True)


if __name__ == "__main__":
    main()
