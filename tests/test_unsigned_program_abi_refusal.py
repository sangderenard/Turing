"""An authored unsigned span must not silently become floating storage."""

from pathlib import Path

import pytest
import yaml

from src.compiler.extraction_contract import ExtractionContractError
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "unsigned_span_corpus" / "c621_unsigned_remainder.py"
BASE_CONTRACT = ROOT / "extraction_contracts" / "program_extraction.yaml"


@pytest.mark.parametrize(
    "dtype, storage, shape",
    (("uint64", "span", None), ("u64", "span", (3,)),
     ("uint64", "scalar", ())),
)
def test_c621_unsigned_program_abi_refuses_before_native_emission(
    tmp_path, dtype, storage, shape,
):
    raw = yaml.safe_load(BASE_CONTRACT.read_text(encoding="utf-8"))
    values = []
    for parameter in ("x", "n"):
        field = {
            "function": "f", "parameter": parameter,
            "storage": storage if parameter == "x" else "scalar",
            "dtype": dtype,
            "python_type": (
                "src.common.tensors.abstraction.AbstractTensor"
                if parameter == "x" and storage == "span" else "builtins.int"
            ),
        }
        if parameter == "x" and storage == "span":
            field["rank"] = 1
            if shape is not None:
                field["shape"] = list(shape)
        values.append(field)
    raw["program_abi"] = {"records": {}, "bindings": [], "values": values}
    contract = tmp_path / "unsigned-program-abi.yaml"
    contract.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(
        ExtractionContractError,
        match=r"program_abi\.values\[0\]\.dtype .*no exact native C/LLVM/Fortran unsigned lane",
    ):
        lower_ast_source_to_ssa(
            SOURCE.read_text(encoding="utf-8"), "f", name="c621_unsigned_remainder",
            extraction_contract=str(contract), progress=lambda _: None,
        )
