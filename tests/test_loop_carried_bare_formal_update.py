from pathlib import Path

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

CONTRACT = str(
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


def test_loop_carried_phi_updated_from_a_callee_bare_formal_lowers():
    """``h, l = step(h, l, i)`` where ``step`` returns ``low ^ byte, high``.

    The second result is the callee's formal, so the passthrough fold rebinds
    it to the actual (the other carried Phi) as a projection occurrence of that
    id.  The loop-scope rule read the occurrence as a private resident.
    """

    lower_ast_source_to_ssa(
        "def step(high: int, low: int, byte: int):\n"
        "    return low ^ byte, high\n"
        "\n"
        "def root(n: int) -> int:\n"
        "    h = 1\n"
        "    l = 2\n"
        "    for i in range(n):\n"
        "        h, l = step(h, l, i)\n"
        "    return h + l\n",
        "root", name="x", extraction_contract=CONTRACT,
        progress=lambda _: None,
    )
