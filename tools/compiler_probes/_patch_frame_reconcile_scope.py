"""Resolve a frame receipt's caller id at the position it is used.

The post-aggregate frame reconciliation reapplies each call's exact frame
receipt so the emitted call and its provenance table end on one resident
identity.  The receipt stores a bare caller id, and the resident is looked up
in ``function_values(caller)`` -- one answer per id for the whole function.

Inside a loop that answer is the wrong generation.  ``_row(x, hot_i)`` had
already been resolved correctly at its position: the callsite page records
``graph 12 -> 2305843010213693998 mapped``.  This pass then overwrote operand
0 with ``caller_values[12]``, the pre-loop clone, and receipted the overwrite
as ``post_aggregate_frame_reconciled: ((0, 0, 12),)``.  Nothing was rewritten
in the structural sense, which is why no rebinding receipt existed and the
identity-collapse rule stayed silent: the operand was not changed from one
value to another, it was rebuilt from a record that never carried a scope.

The receipt is right about WHICH value -- it is the same variable.  It cannot
be right about which generation, because a bare id cannot say.  So resolve it
through the loop scope at the block the call sits in before taking the
resident.  Outside a loop, and for any id the loop does not rebind, this
returns the same id and the pass behaves exactly as before.

R-B, at the line that motivated it: a binding resolved at a position carries
the scope it resolved in; a later stage re-deriving it from a bare id has
thrown the scope away.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONCORDANCE = ROOT / "src/compiler/identity_concordance.py"
SHELL = ROOT / "src/compiler/fortran_c_shell.py"

Q = chr(34) * 3

# A resolver that reads the declarations once per function rather than per
# operand -- the rules run inside a fixed point, so the scan matters.
RESOLVER_ANCHOR = "def refuse_cross_generation_identities("

RESOLVER = (
    "def scope_resolver(book: Any, name: Any, function: Any):\n"
    "    " + Q + "A positional name resolver for one function.\n"
    "\n"
    "    Returns ``resolve(block, value_id)`` giving what that id means at\n"
    "    that position: inside a loop the carried generation, everywhere else\n"
    "    the id unchanged.  The declarations are read once, because callers\n"
    "    are inside fixed points and a per-operand page scan is not free.\n"
    "    " + Q + "\n"
    "\n"
    "    scopes = []\n"
    "    for declaration in loop_scope_declarations(book, name):\n"
    "        boundary = declaration[\"boundary\"]\n"
    "        if not boundary:\n"
    "            continue\n"
    "        header, latch, _exit = boundary\n"
    "        inside = _blocks_between(function, header, latch)\n"
    "        if not inside:\n"
    "            continue\n"
    "        carried = {\n"
    "            int(rebind[\"outer\"]): int(rebind[\"carried\"])\n"
    "            for rebind in declaration[\"rebinds\"]\n"
    "        }\n"
    "        if carried:\n"
    "            scopes.append((inside, carried))\n"
    "\n"
    "    def resolve(block: Any, value_id: int) -> int:\n"
    "        for inside, carried in scopes:\n"
    "            if str(block) not in inside:\n"
    "                continue\n"
    "            replacement = carried.get(int(value_id))\n"
    "            if replacement is not None:\n"
    "                return int(replacement)\n"
    "        return int(value_id)\n"
    "\n"
    "    return resolve\n"
    "\n"
    "\n"
    "def note_scope_resolution(\n"
    "    function: Any, block: Any, value_id: int, resolved: int, *, stage: str,\n"
    ") -> None:\n"
    "    " + Q + "Record a binding that meant something else at its position." + Q + "\n"
    "\n"
    "    page = current_identity_book().page(\"scope_resolution\")\n"
    "    row = (authored_function_name(function), int(value_id))\n"
    "    page.set(row, len(page.history(row)), (\n"
    "        str(stage), str(block), int(resolved),\n"
    "    ))\n"
    "\n"
    "\n"
    "def refuse_cross_generation_identities("
)

# ------------------------------------------------------------------- the pass

FIND_ANCHOR = (
    "            linked_call = next((\n"
    "                instruction\n"
    "                for block in caller.blocks.values()\n"
    "                for instruction in block.instrs\n"
    "                if instruction.op in {\"Call\", \"call\"}\n"
    "                and instruction.attributes.get(\"source_linked\")\n"
    "                and int(instruction.attributes.get(\n"
    "                    \"plan_callsite_id\", -1\n"
    "                )) == int(record.callsite_id)\n"
    "                and str(instruction.attributes.get(\"callee\") or \"\")\n"
    "                == str(record.callee_symbol or \"\")\n"
    "            ), None)\n"
    "            if linked_call is None:\n"
    "                continue"
)

FIND = (
    "            linked_call = None\n"
    "            linked_block = None\n"
    "            for _block_name, _block in caller.blocks.items():\n"
    "                for instruction in _block.instrs:\n"
    "                    if (\n"
    "                        instruction.op in {\"Call\", \"call\"}\n"
    "                        and instruction.attributes.get(\"source_linked\")\n"
    "                        and int(instruction.attributes.get(\n"
    "                            \"plan_callsite_id\", -1\n"
    "                        )) == int(record.callsite_id)\n"
    "                        and str(instruction.attributes.get(\"callee\") or \"\")\n"
    "                        == str(record.callee_symbol or \"\")\n"
    "                    ):\n"
    "                        linked_call = instruction\n"
    "                        linked_block = str(_block_name)\n"
    "                        break\n"
    "                if linked_call is not None:\n"
    "                    break\n"
    "            if linked_call is None:\n"
    "                continue"
)

APPLY_ANCHOR = (
    "                caller_id = frame.get(callee_id)\n"
    "                resident = (\n"
    "                    None if caller_id is None\n"
    "                    else caller_values.get(int(caller_id))\n"
    "                )\n"
    "                if (\n"
    "                    resident is None\n"
    "                    or position >= len(linked_call.args)\n"
    "                    or int(linked_call.args[position].id) == int(caller_id)\n"
    "                ):\n"
    "                    continue\n"
    "                linked_call.args[position] = resident\n"
    "                changed_positions.append((\n"
    "                    position, callee_id, int(caller_id),\n"
    "                ))"
)

APPLY = (
    "                caller_id = frame.get(callee_id)\n"
    "                # The receipt is right about WHICH value -- it is the\n"
    "                # same variable -- but a bare id cannot say which\n"
    "                # generation, and inside a loop the function-wide answer\n"
    "                # is the one from before the first iteration.  Resolve it\n"
    "                # at the block this call sits in; outside a loop, and for\n"
    "                # any id no loop rebinds, this is the identity.\n"
    "                resolved_id = caller_id\n"
    "                if caller_id is not None and resolve_in_scope is not None:\n"
    "                    resolved_id = resolve_in_scope(\n"
    "                        linked_block, int(caller_id),\n"
    "                    )\n"
    "                    if int(resolved_id) != int(caller_id):\n"
    "                        _note_scope_resolution(\n"
    "                            caller.name, linked_block, int(caller_id),\n"
    "                            int(resolved_id),\n"
    "                        )\n"
    "                resident = (\n"
    "                    None if resolved_id is None\n"
    "                    else caller_values.get(int(resolved_id))\n"
    "                )\n"
    "                if (\n"
    "                    resident is None\n"
    "                    or position >= len(linked_call.args)\n"
    "                    or int(linked_call.args[position].id) == int(resolved_id)\n"
    "                ):\n"
    "                    continue\n"
    "                linked_call.args[position] = resident\n"
    "                changed_positions.append((\n"
    "                    position, callee_id, int(resolved_id),\n"
    "                ))"
)

# The resolver is built once per caller, beside the value table it corrects.
BUILD_ANCHOR = (
    "        caller_values = function_values(caller)\n"
    "        for record in records:"
)

BUILD = (
    "        caller_values = function_values(caller)\n"
    "        resolve_in_scope = _scope_resolver_for(caller)\n"
    "        for record in records:"
)

HELPER_ANCHOR = "def _concordant_function_aliases("

HELPER = (
    "def _scope_resolver_for(function: Any):\n"
    "    " + Q + "A positional name resolver for one function, or None.\n"
    "\n"
    "    None when the function declares no loop scope, so a caller pays\n"
    "    nothing for the common case.\n"
    "    " + Q + "\n"
    "\n"
    "    try:\n"
    "        from .identity_concordance import (\n"
    "            current_identity_book, loop_scope_declarations, scope_resolver,\n"
    "        )\n"
    "\n"
    "        book = current_identity_book()\n"
    "        if not loop_scope_declarations(book, str(function.name)):\n"
    "            return None\n"
    "        return scope_resolver(book, str(function.name), function)\n"
    "    except Exception:\n"
    "        return None\n"
    "\n"
    "\n"
    "def _note_scope_resolution(\n"
    "    function: Any, block: Any, value_id: int, resolved: int,\n"
    ") -> None:\n"
    "    " + Q + "Record one binding that meant another generation here." + Q + "\n"
    "\n"
    "    try:\n"
    "        from .identity_concordance import note_scope_resolution\n"
    "\n"
    "        note_scope_resolution(\n"
    "            function, block, value_id, resolved,\n"
    "            stage=\"post_aggregate_frame_reconciliation\",\n"
    "        )\n"
    "    except Exception:\n"
    "        pass\n"
    "\n"
    "\n"
    "def _concordant_function_aliases("
)


def patch(path, pairs):
    text = path.read_text(encoding="utf-8")
    for anchor, replacement, expected in pairs:
        count = text.count(anchor)
        assert count == expected, (path.name, count, expected, anchor[:70])
        text = text.replace(anchor, replacement)
    path.write_text(text, encoding="utf-8")


patch(CONCORDANCE, [(RESOLVER_ANCHOR, RESOLVER, 1)])
patch(SHELL, [
    (HELPER_ANCHOR, HELPER, 1),
    (BUILD_ANCHOR, BUILD, 1),
    (FIND_ANCHOR, FIND, 1),
    (APPLY_ANCHOR, APPLY, 1),
])

print("frame receipts resolve at the position they are applied")
