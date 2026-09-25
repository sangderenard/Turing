"""Report a use, inside a loop, of the value the loop carries FROM BEFORE it.

`operand-never-written` found nothing here because nothing is undefined: the
body reads value 12, and value 12 is a perfectly good instruction result --
it is just the value `x` held before the loop started.  Once a value has a
carried Phi, every use inside the loop must name the Phi.  A use that names
the pre-loop definition instead is a read of a stale generation: the program
is well formed, LLVM accepts it, every block runs, and the arithmetic is
simply wrong.

The loop is derived from DECLARED branch targets (`target`, `true_target`,
`false_target`) and the Phi's declared `incoming_blocks`, not from block
names, so the check holds wherever the shape occurs.
"""

from pathlib import Path

CALL_ANCHOR = """            found.extend(
                self._undefined_operand_findings(str(name), function)
            )"""

CALL = """            found.extend(
                self._undefined_operand_findings(str(name), function)
            )
            found.extend(
                self._stale_carried_read_findings(str(name), function)
            )"""

METHOD_ANCHOR = """    @staticmethod
    def _undefined_operand_findings("""

METHOD = '''    @staticmethod
    def _stale_carried_read_findings(
        name: str, function: Any,
    ) -> list[Finding]:
        """A loop body reading the value its carried Phi superseded.

        The Phi names two generations of one value: what it held on entry and
        what the latch produced.  Inside the loop only the Phi speaks for it.
        An instruction that still names the entry generation is reading the
        value as it was before the first iteration, every iteration -- the
        accumulation silently does not accumulate.
        """

        successors: dict[str, set[str]] = {}
        for block_name, block in function.blocks.items():
            targets: set[str] = set()
            for instruction in block.instrs:
                for key in ("target", "true_target", "false_target"):
                    declared = instruction.attributes.get(key)
                    if declared is not None:
                        targets.add(str(declared))
            successors[str(block_name)] = targets

        def reaches(source: str, goal: str) -> set[str]:
            """Blocks on some path from `source` to `goal`, inclusive."""

            forward: set[str] = set()
            frontier = [source]
            while frontier:
                current = frontier.pop()
                if current in forward:
                    continue
                forward.add(current)
                frontier.extend(successors.get(current, ()))
            backward: set[str] = set()
            frontier = [goal]
            while frontier:
                current = frontier.pop()
                if current in backward:
                    continue
                backward.add(current)
                for candidate, onward in successors.items():
                    if current in onward:
                        frontier.append(candidate)
            return forward & backward

        found: list[Finding] = []
        for header_name, header in function.blocks.items():
            for phi in header.instrs:
                if str(phi.op).lower() != "phi":
                    continue
                if phi.attributes.get("binding") != "loop_carried":
                    continue
                incoming = phi.attributes.get("incoming_blocks") or ()
                if len(incoming) != len(phi.args):
                    continue
                latches = [
                    str(origin) for origin in incoming
                    if str(header_name) in reaches(str(header_name), str(origin))
                ]
                if not latches:
                    continue
                body = set()
                for latch in latches:
                    body |= reaches(str(header_name), latch)
                stale = {
                    int(argument.id)
                    for origin, argument in zip(incoming, phi.args)
                    if str(origin) not in latches
                    and getattr(argument, "id", None) is not None
                }
                if not stale:
                    continue
                for block_name in sorted(body):
                    block = function.blocks.get(block_name)
                    if block is None:
                        continue
                    for index, instruction in enumerate(block.instrs):
                        if str(instruction.op).lower() == "phi":
                            continue
                        for position, argument in enumerate(instruction.args):
                            argument_id = getattr(argument, "id", None)
                            if argument_id is None:
                                continue
                            if int(argument_id) not in stale:
                                continue
                            found.append(Finding(
                                "stale-carried-read", name, int(argument_id),
                                f"{block_name}#{index} {instruction.op} operand "
                                f"{position} names the pre-loop value carried by "
                                f"{header_name} Phi {int(phi.res.id)}; inside the "
                                "loop only the Phi speaks for it",
                            ))
        return found

    @staticmethod
    def _undefined_operand_findings('''

path = Path(__file__).resolve().parents[2] / "src/compiler/identity_concordance.py"
text = path.read_text(encoding="utf-8")
assert text.count(CALL_ANCHOR) == 1, text.count(CALL_ANCHOR)
text = text.replace(CALL_ANCHOR, CALL)
assert text.count(METHOD_ANCHOR) == 1, text.count(METHOD_ANCHOR)
text = text.replace(METHOD_ANCHOR, METHOD)
path.write_text(text, encoding="utf-8")
print("stale-carried-read finding added")
