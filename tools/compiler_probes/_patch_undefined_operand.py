"""Report an operand that nothing in the function defines.

A loop's carried update stored into a slot other than the one its latch Phi
names produces a program that is well formed, that LLVM accepts, whose every
block runs and whose publishing copy executes -- and that publishes whatever
an uninitialized alloca happened to hold.  With a zero-filled buffer it reads
as a plausible answer; it took a NaN fill and four runtime probes to see it at
all.

The check that notices it is cheap: an operand that is neither a formal nor
the result of any instruction in the function is a value nothing writes.  That
is always a defect, it is decidable from the IR alone, and it needs no
dominance or CFG reasoning.  Phi operands are included deliberately -- the
latch incoming of a carried value is exactly where this hides.
"""

from pathlib import Path

ANCHOR = """        found.extend(self._sequence_descriptor_findings(module))"""

CALL = """            found.extend(
                self._undefined_operand_findings(str(name), function)
            )
        found.extend(self._sequence_descriptor_findings(module))"""

METHOD_ANCHOR = """    @staticmethod
    def _callable_identity_findings(module: Any) -> list[Finding]:"""

METHOD = '''    @staticmethod
    def _undefined_operand_findings(
        name: str, function: Any,
    ) -> list[Finding]:
        """An operand no instruction defines and no formal supplies.

        Reading such a value yields whatever its storage happened to hold, so
        the program computes an answer from uninitialized memory instead of
        failing.  The latch incoming of a loop-carried Phi is where this hides:
        the body computes the update and stores it somewhere other than the
        slot the Phi names, and every later stage -- emission, the LLVM
        verifier, execution -- accepts the result.
        """

        found: list[Finding] = []
        formals = {int(value.id) for value in getattr(function, "args", ())}
        defined: set[int] = set()
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is not None:
                    defined.add(int(instruction.res.id))
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                for position, argument in enumerate(instruction.args):
                    argument_id = getattr(argument, "id", None)
                    if argument_id is None:
                        continue
                    argument_id = int(argument_id)
                    if argument_id in formals or argument_id in defined:
                        continue
                    found.append(Finding(
                        "operand-never-written", name, argument_id,
                        f"{block_name}#{index} {instruction.op} operand "
                        f"{position} is neither a formal nor defined by any "
                        "instruction",
                    ))
        return found

    @staticmethod
    def _callable_identity_findings(module: Any) -> list[Finding]:'''

path = Path(__file__).resolve().parents[2] / "src/compiler/identity_concordance.py"
text = path.read_text(encoding="utf-8")
assert text.count(ANCHOR) == 1, text.count(ANCHOR)
text = text.replace(ANCHOR, CALL)
assert text.count(METHOD_ANCHOR) == 1, text.count(METHOD_ANCHOR)
text = text.replace(METHOD_ANCHOR, METHOD)
path.write_text(text, encoding="utf-8")
print("undefined-operand finding added")
