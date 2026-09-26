"""Stop the backward carry from widening boolean-derived masks."""
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "src/compiler/ir_identities.py"

OLD = '''                        producer = definitions.get(int(argument.id))
                        if (
                            producer is None
                            or str(producer.op)
                            not in PRECISION_CLOSED_OPERATIONS
                        ):
                            continue'''

NEW = '''                        producer = definitions.get(int(argument.id))
                        if (
                            producer is None
                            or str(producer.op)
                            not in PRECISION_CLOSED_OPERATIONS
                        ):
                            continue
                        # A value built from a COMPARISON is a mask: it is
                        # zero or one, exactly representable in one limb,
                        # and there is nothing for a second limb to hold.
                        # Widening it anyway is not merely wasted work --
                        # it puts an error-free transformation on a
                        # boolean, and the LLVM lane then tracks the same
                        # register as both i1 and double and emits a
                        # two_product residual over a predicate. Measured,
                        # a quadrant blend that the C lane computed
                        # exactly came back off by half on LLVM. The
                        # forward pass never reached these because a mask
                        # has no limbed operand; only this backward half
                        # can, so the guard belongs here.
                        if _is_predicate_derived(
                            producer, definitions, predicates
                        ):
                            continue'''

HELPER = '''

def _is_predicate_derived(instruction, definitions, predicates,
                          depth: int = 4) -> bool:
    """Whether this value descends from a comparison within a few steps.

    Shallow on purpose: a mask is formed close to the comparison that
    made it, and a deep search would start refusing genuine arithmetic
    that merely happens to sit downstream of a branch condition.
    """

    if depth <= 0:
        return False
    for argument in getattr(instruction, "args", ()):
        identifier = int(argument.id)
        if identifier in predicates:
            return True
        producer = definitions.get(identifier)
        if producer is not None and _is_predicate_derived(
            producer, definitions, predicates, depth - 1
        ):
            return True
    return False

'''

OLD_DEFINITIONS = '''            definitions = {
                int(instruction.res.id): instruction
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            }'''

NEW_DEFINITIONS = OLD_DEFINITIONS + '''
            predicates = {
                int(instruction.res.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
                and str(instruction.op) in _PREDICATE_NAMES
            }'''

NAMES = '''

#: Operations whose result is a truth value rather than a quantity.
_PREDICATE_NAMES = frozenset({
    "Lt", "Le", "Gt", "Ge", "Eq", "Ne",
    "lt", "le", "gt", "ge", "eq", "ne",
})

'''


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if "_is_predicate_derived" in text:
        raise SystemExit("guard already present; refusing to re-apply")
    for anchor in (OLD, OLD_DEFINITIONS):
        if text.count(anchor) != 1:
            raise SystemExit(f"anchor appears {text.count(anchor)} times")
    text = text.replace(OLD_DEFINITIONS, NEW_DEFINITIONS, 1)
    text = text.replace(OLD, NEW, 1)
    marker = "\ndef carry_precision_through_ssa("
    if text.count(marker) != 1:
        raise SystemExit("carry anchor not unique")
    text = text.replace(marker, NAMES + HELPER + marker, 1)
    PATH.write_text(text, encoding="utf-8")
    print("mask guard installed")


if __name__ == "__main__":
    main()
