"""Drop the rules that found nothing; fix the pass that caused it.

Kept: the loop scope declaration, and the one rule that located the defect --
an in-scope use of the outer generation.

Dropped:
  operand-never-written           0 findings; nothing here is undefined
  stale-carried-read              redundant with the declared rule
  loop-scope-latch-renamed        reports a legitimate construction: the
  loop-scope-inner-outside        latch names the region's unpacked result,
                                  which IS the new value.  The slot is
                                  abandoned, not misused.

Fixed: the post-aggregate frame reconciliation overwrote an operand that had
already been resolved correctly at its position.  The receipt stores a bare
caller id and the resident comes from ``function_values(caller)`` -- one
answer per id for the whole function -- so inside a loop it is the generation
from before the first iteration.  The receipt is right about WHICH value; a
bare id cannot say which generation.  Resolve it at the block the call sits
in.  Outside a loop, and for any id no loop rebinds, this is the identity.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONCORDANCE = ROOT / "src/compiler/identity_concordance.py"
SHELL = ROOT / "src/compiler/fortran_c_shell.py"

Q = chr(34) * 3

text = CONCORDANCE.read_text(encoding="utf-8")


def cut(source: str, start_marker: str, end_marker: str) -> str:
    """Remove [start_marker, end_marker), keeping end_marker."""

    assert source.count(start_marker) == 1, start_marker[:60]
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[:start] + source[end:]


def swap(source: str, old: str, new: str, expected: int = 1) -> str:
    assert source.count(old) == expected, (source.count(old), old[:60])
    return source.replace(old, new)


# 1. findings(): one call, no fallback.
text = swap(
    text,
    "            found.extend(\n"
    "                self._undefined_operand_findings(str(name), function)\n"
    "            )\n"
    "            declared = self._loop_scope_findings(module, str(name), function)\n"
    "            found.extend(declared)\n"
    "            # The inferred check recovers the scope from branch topology; it\n"
    "            # covers loops built by a path that does not declare one.  Where a\n"
    "            # declaration exists it is authoritative, so do not report twice.\n"
    "            if not loop_scope_declarations(identity_book(module), str(name)):\n"
    "                found.extend(\n"
    "                    self._stale_carried_read_findings(str(name), function)\n"
    "                )\n",
    "            found.extend(\n"
    "                self._loop_scope_findings(module, str(name), function)\n"
    "            )\n",
)

# 2. Rules 2 and 3 report a legitimate construction.
text = cut(
    text,
    "\n                # Rule 2 -- the backedge must carry the declared inner",
    "\n        return found\n\n    @staticmethod\n"
    "    def _stale_carried_read_findings(",
)

# 3. The two checks that found nothing.
text = cut(
    text,
    "    @staticmethod\n    def _stale_carried_read_findings(",
    "    @staticmethod\n    def _callable_identity_findings(",
)

# 4. Their helpers.
text = cut(text, "def _carried_phi(", "class IdentityPage:")
text = swap(
    text,
    "        definitions: dict[int, str] = {}\n"
    "        for block_name, block in function.blocks.items():\n"
    "            for instruction in block.instrs:\n"
    "                if instruction.res is not None:\n"
    "                    definitions[int(instruction.res.id)] = str(block_name)\n"
    "\n",
    "",
)

# 5. The resolver the fix needs.
text = swap(
    text,
    "def record_proven_shape(",
    "def scope_resolver(book: Any, name: Any, function: Any):\n"
    "    " + Q + "A positional name resolver for one function, or None.\n"
    "\n"
    "    Returns ``resolve(block, value_id)`` giving what that id means at\n"
    "    that position: inside a loop the carried generation, everywhere else\n"
    "    the id unchanged.  None when the function declares no loop scope, so\n"
    "    a caller pays nothing for the common case.  The declarations are read\n"
    "    once, because callers sit inside fixed points.\n"
    "    " + Q + "\n"
    "\n"
    "    scopes = []\n"
    "    for declaration in loop_scope_declarations(book, name):\n"
    "        boundary = declaration[\"boundary\"]\n"
    "        if not boundary:\n"
    "            continue\n"
    "        header, latch, _exit = boundary\n"
    "        inside = _blocks_between(function, header, latch)\n"
    "        carried = {\n"
    "            int(rebind[\"outer\"]): int(rebind[\"carried\"])\n"
    "            for rebind in declaration[\"rebinds\"]\n"
    "        }\n"
    "        if inside and carried:\n"
    "            scopes.append((inside, carried))\n"
    "    if not scopes:\n"
    "        return None\n"
    "\n"
    "    def resolve(block: Any, value_id: int) -> int:\n"
    "        for inside, carried in scopes:\n"
    "            if str(block) in inside:\n"
    "                replacement = carried.get(int(value_id))\n"
    "                if replacement is not None:\n"
    "                    return int(replacement)\n"
    "        return int(value_id)\n"
    "\n"
    "    return resolve\n"
    "\n"
    "\n"
    "def record_proven_shape(",
)

CONCORDANCE.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------- fix

shell = SHELL.read_text(encoding="utf-8")

shell = swap(
    shell,
    "def _concordant_function_aliases(",
    "def _scope_resolver_for(function: Any):\n"
    "    " + Q + "A positional name resolver for one function, or None." + Q + "\n"
    "\n"
    "    try:\n"
    "        from .identity_concordance import (\n"
    "            current_identity_book, scope_resolver,\n"
    "        )\n"
    "\n"
    "        return scope_resolver(\n"
    "            current_identity_book(), str(function.name), function,\n"
    "        )\n"
    "    except Exception:\n"
    "        return None\n"
    "\n"
    "\n"
    "def _concordant_function_aliases(",
)

shell = swap(
    shell,
    "        caller_values = function_values(caller)\n"
    "        for record in records:",
    "        caller_values = function_values(caller)\n"
    "        resolve_in_scope = _scope_resolver_for(caller)\n"
    "        for record in records:",
)

shell = swap(
    shell,
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
    "                continue",
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
    "                continue",
)

shell = swap(
    shell,
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
    "                ))",
    "                caller_id = frame.get(callee_id)\n"
    "                # The receipt is right about WHICH value -- it is the same\n"
    "                # variable -- but a bare id cannot say which generation,\n"
    "                # and ``caller_values`` has one answer per id for the whole\n"
    "                # function.  Inside a loop that answer is the value from\n"
    "                # before the first iteration, so a call whose operand was\n"
    "                # already resolved correctly at its position gets it\n"
    "                # overwritten.  Resolve at the block the call sits in.\n"
    "                if caller_id is not None and resolve_in_scope is not None:\n"
    "                    caller_id = resolve_in_scope(\n"
    "                        linked_block, int(caller_id),\n"
    "                    )\n"
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
    "                ))",
)

SHELL.write_text(shell, encoding="utf-8")

print("inert rules removed; frame receipts resolve at their position")
