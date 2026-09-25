"""Make the unary tables reachable by the spelling the SSA actually emits."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LLVM = ROOT / "src/compiler/ssa_llvm_backend.py"
C = ROOT / "src/compiler/ssa_c_backend.py"

OLD_LOOKUP = "    template = _BINARY.get(operation) or _UNARY.get(operation)"
NEW_LOOKUP = '''    # Looked up by the SPELLING THE SSA EMITS, which is not the spelling
    # these tables are keyed by: the reducer plants ``floor`` where the
    # table says ``Floor``, and an exact match therefore reports a
    # capability that is present as missing. That is the same failure the
    # C lane's casefolded tables were built to end -- "a capability that
    # exists but is spelled differently is indistinguishable from a
    # missing one" -- and it kept range reduction, and with it every
    # full-domain transcendental, from compiling at all.
    template = _BINARY.get(operation) or _UNARY.get(operation)
    if template is None:
        folded = str(operation).casefold()
        template = _BINARY_FOLDED.get(folded) or _UNARY_FOLDED.get(folded)'''

FOLDED_TABLES = '''

#: The same tables keyed by casefolded name, for operations whose emitted
#: spelling differs in case from the canonical one.
_BINARY_FOLDED: dict[str, str] = {
    key.casefold(): value for key, value in _BINARY.items()
}
_UNARY_FOLDED: dict[str, str] = {
    key.casefold(): value for key, value in _UNARY.items()
}
'''

OLD_C_UNARY = '_UNARY = {"Abs": "fabs", "Sqrt": "sqrt", "Neg": None}'
NEW_C_UNARY = '''_UNARY = {
    "Abs": "fabs", "Sqrt": "sqrt", "Neg": None,
    # Range reduction is floor and nothing else, so a lane without it
    # cannot compile a transcendental outside its core's own interval.
    # C99 has all three in <math.h>.
    "Floor": "floor", "Ceil": "ceil", "Round": "nearbyint",
}'''


def main() -> None:
    llvm = LLVM.read_text(encoding="utf-8")
    if "_BINARY_FOLDED" in llvm:
        raise SystemExit("llvm already folded; refusing to re-apply")
    if llvm.count(OLD_LOOKUP) != 1:
        raise SystemExit(f"llvm lookup anchor appears {llvm.count(OLD_LOOKUP)} times")
    llvm = llvm.replace(OLD_LOOKUP, NEW_LOOKUP, 1)
    anchor = "def scalar_likeness("
    if llvm.count(anchor) != 1:
        raise SystemExit("llvm table anchor not unique")
    llvm = llvm.replace(anchor, FOLDED_TABLES.strip("\n") + "\n\n\n" + anchor, 1)
    LLVM.write_text(llvm, encoding="utf-8")

    text = C.read_text(encoding="utf-8")
    if text.count(OLD_C_UNARY) != 1:
        raise SystemExit("c unary anchor not unique")
    C.write_text(text.replace(OLD_C_UNARY, NEW_C_UNARY, 1), encoding="utf-8")
    print("floor reachable in both lanes")


if __name__ == "__main__":
    main()
