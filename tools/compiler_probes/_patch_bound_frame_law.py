"""Restore the frame reapplication, bounded to what a bare id can name.

The pass is not wrong.  Its job is real: aggregate legalization can rewrite
positional projection occurrences after the linked call-table fixed point, and
without reapplying the frame receipt the emitted call and its provenance table
end on different residents.  Deleting it wholesale fixes this program and
leaves every other one exposed to the desync it was written for.

What is wrong is its SCOPE OF AUTHORITY.  The receipt stores a bare caller id
and resolves it against ``function_values(caller)`` -- one answer per id for
the whole function.  That is authoritative over a value with one generation,
and says nothing about a value with several: inside a loop the single answer
is the one from before the first iteration.  The measured consequence was an
operand that ``emit_plan_callsite`` had already resolved correctly at its
position (the callsite page records ``graph 12 -> <carried phi> mapped``)
being overwritten back to 12 and receipted as ``post_aggregate_frame_reconciled``.

So the receipt keeps its authority everywhere it has any, and abstains where
a bare id cannot name the generation.  This is not compensating for the law
with a better resident -- there is no better resident to pick, because the
identity the receipt names is genuinely ambiguous at that position.  It is
the law declining to speak about something it cannot say.

Outside a loop, and for any id no loop rebinds, nothing changes.
"""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SHELL = ROOT / "src/compiler/fortran_c_shell.py"

Q = chr(34) * 3

original = subprocess.run(
    ["git", "show", "HEAD:src/compiler/fortran_c_shell.py"],
    cwd=ROOT, capture_output=True, text=True, check=True,
).stdout

START = "    # Aggregate legalization can rewrite positional projection occurrences"
END = "    # A source return expression may be owned by a path-correlated numerical"

law = original[original.index(START):original.index(END)]

# 1. The receipt abstains on an operand whose identity has generations.
BOUND_HEADER = (
    "    # Aggregate legalization can rewrite positional projection occurrences\n"
    "    # after the linked call-table fixed point.  Reapply the final exact frame\n"
    "    # receipt once so emitted calls and their provenance table end on the same\n"
    "    # resident identity.  Equal-priority duplicates keep the receipt's first\n"
    "    # incumbent; no new identity or schema relation is inferred here.\n"
    "    #\n"
    "    # The receipt stores a BARE caller id and resolves it against a value\n"
    "    # table with one answer per id for the whole function.  That is\n"
    "    # authoritative over a value with one generation and says nothing about\n"
    "    # a value a loop rebinds, where the single answer is the one from before\n"
    "    # the first iteration.  Such an operand was already resolved at its\n"
    "    # position by the callsite, so the receipt abstains rather than\n"
    "    # overwrite a more specific fact with a less specific one.\n"
)
law = law.replace(
    "    # Aggregate legalization can rewrite positional projection occurrences\n"
    "    # after the linked call-table fixed point.  Reapply the final exact frame\n"
    "    # receipt once so emitted calls and their provenance table end on the same\n"
    "    # resident identity.  Equal-priority duplicates keep the receipt's first\n"
    "    # incumbent; no new identity or schema relation is inferred here.\n",
    BOUND_HEADER,
)

law = law.replace(
    "        caller_values = function_values(caller)\n"
    "        for record in records:",
    "        caller_values = function_values(caller)\n"
    "        generational_ids = _loop_rebound_value_ids(caller)\n"
    "        for record in records:",
)

law = law.replace(
    "                caller_id = frame.get(callee_id)\n"
    "                resident = (\n",
    "                caller_id = frame.get(callee_id)\n"
    "                if caller_id is not None and int(caller_id) in generational_ids:\n"
    "                    abstained_positions.append((\n"
    "                        position, callee_id, int(caller_id),\n"
    "                    ))\n"
    "                    continue\n"
    "                resident = (\n",
)

law = law.replace(
    "            changed_positions = []\n",
    "            changed_positions = []\n"
    "            abstained_positions = []\n",
)

law = law.replace(
    "            if changed_positions:\n",
    "            if abstained_positions:\n"
    "                linked_call.attributes[\n"
    "                    \"post_aggregate_frame_abstained\"\n"
    "                ] = tuple(abstained_positions)\n"
    "            if changed_positions:\n",
)

# 2. The helper that says which ids have generations.
HELPER_ANCHOR = "def _concordant_function_aliases("

HELPER = (
    "def _loop_rebound_value_ids(function: Any) -> frozenset:\n"
    "    " + Q + "Caller ids a loop rebinds, so a bare id cannot name them.\n"
    "\n"
    "    Empty when the function declares no loop scope, which is the common\n"
    "    case and costs one page lookup.\n"
    "    " + Q + "\n"
    "\n"
    "    try:\n"
    "        from .identity_concordance import (\n"
    "            current_identity_book, loop_scope_declarations,\n"
    "        )\n"
    "\n"
    "        return frozenset(\n"
    "            int(rebind[\"outer\"])\n"
    "            for declaration in loop_scope_declarations(\n"
    "                current_identity_book(), str(function.name),\n"
    "            )\n"
    "            for rebind in declaration[\"rebinds\"]\n"
    "        )\n"
    "    except Exception:\n"
    "        return frozenset()\n"
    "\n"
    "\n"
    "def _concordant_function_aliases("
)

text = SHELL.read_text(encoding="utf-8")

# Replace the note left where the law used to be.
note_start = text.index(
    "    # The post-aggregate frame reconciliation used to run here."
)
note_end = text.index(END, note_start)
text = text[:note_start] + law + text[note_end:]

assert text.count(HELPER_ANCHOR) == 1
text = text.replace(HELPER_ANCHOR, HELPER)

SHELL.write_text(text, encoding="utf-8")
print("frame reapplication restored, bounded to ids without generations")
