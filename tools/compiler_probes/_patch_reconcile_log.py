"""Have the loop-result reconciliation record its decisions.

The concordance reports `use-not-dominated` for a value while this pass,
walking the same instructions in the same blocks, substitutes nothing.  Two
passes looking at one function and disagreeing is a fact worth recording
rather than inferring, so every declined substitution now says which test it
failed, on a page keyed by the value identity the finding already names.
"""

from pathlib import Path

ANCHOR = """            resolved_args = list(instruction.args)
            for argument_index, argument in enumerate(instruction.args):"""

LOGGED = '''            resolved_args = list(instruction.args)
            for argument_index, argument in enumerate(instruction.args):
                def _note(outcome: str, detail: Any = None) -> None:
                    """Record this argument's reconciliation decision."""

                    try:
                        from .identity_concordance import (
                            current_identity_book,
                        )

                        page = current_identity_book().page(
                            "loop_result_reconciliation"
                        )
                        row = (
                            str(function.name), int(argument.id),
                        )
                        page.set(row, len(page.history(row)), (
                            outcome,
                            f"{block_name}#{instruction_index}",
                            str(instruction.op),
                            detail,
                        ))
                    except Exception:
                        pass
'''

DECISIONS = (
    (
        """                if replacement is None or replacement is argument:
                    continue""",
        """                if replacement is None:
                    _note("no-candidate", tuple(
                        definition_sites.get(int(argument.id), ())
                    )[:3])
                    continue
                if replacement is argument:
                    continue""",
    ),
    (
        """                if dominates(
                    argument, block_name, instruction_index, incoming_block
                ):
                    continue""",
        """                if dominates(
                    argument, block_name, instruction_index, incoming_block
                ):
                    continue
                _note("argument-not-dominating", tuple(
                    definition_sites.get(int(argument.id), ())
                )[:3])""",
    ),
    (
        """                if not dominates(
                    replacement, block_name, instruction_index, incoming_block
                ):
                    continue""",
        """                if not dominates(
                    replacement, block_name, instruction_index, incoming_block
                ):
                    _note("candidate-not-dominating", (
                        int(replacement.id),
                        tuple(definition_sites.get(int(replacement.id), ()))[:3],
                    ))
                    continue""",
    ),
    (
        """                resolved_args[argument_index] = replacement
                receipts.append((""",
        """                _note("replaced", int(replacement.id))
                resolved_args[argument_index] = replacement
                receipts.append((""",
    ),
)

path = Path(__file__).resolve().parents[2] / "src/compiler/precompile_to_ssa.py"
text = path.read_text(encoding="utf-8")
assert text.count(ANCHOR) == 1, text.count(ANCHOR)
text = text.replace(ANCHOR, LOGGED)
for anchor, replacement in DECISIONS:
    assert text.count(anchor) == 1, (text.count(anchor), anchor[:50])
    text = text.replace(anchor, replacement)
path.write_text(text, encoding="utf-8")
print("reconciliation decisions are recorded")
