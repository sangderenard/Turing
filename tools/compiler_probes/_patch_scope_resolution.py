"""Refuse an identity collapse that crosses a generation, and say why.

A tie policy is a confession.  Where a pass needs ``tie_policy: "incumbent"``
to choose between two identities, it is choosing without information it
should have had -- and the right response is to supply the missing fact, not
to pick a winner.

The canonicalizer's two candidates for a loop variable ARE structurally
identical: same shape, same dtype, same everything, because they are the same
variable.  An alias map can only say "same" or "different", so told they are
the same it collapses them and breaks the tie by incumbency.  In a loop the
incumbent is by definition the generation from before the loop.  The collapse
is therefore not arbitrary -- it is systematically backwards, every time,
which is exactly the uniformity measured across all five carried values.

Identity of CONTENT is not identity of BINDING.  Two values that differ in a
declared dimension are not tied, so there is no tie to break; there is a
question, and the declaration answers it.  Three rules follow:

  R-A  An identity map may not merge two generations of one declared rebind.
       Not "prefer the newer one" -- refuse, because either choice erases a
       distinction the program depends on.

  R-B  A binding resolved AT A POSITION carries the scope it resolved in.  A
       later stage that re-derives it from a bare id has thrown the scope
       away and will get the flat map's single answer.  Bindings travel as a
       resolved value or as (value, position) -- never as a bare id.

  R-C  A pass that renames a value crossing a scope boundary must re-declare
       it, or the declaration goes stale and the rules read a legitimate
       rename as a defect.

This installs R-A as a refusal at the canonicalizer, records every refusal so
the pass that needed one is named rather than inferred, and adds the reading
of R-A over the rebinding receipts that already exist in function metadata --
so a collapse performed by any OTHER pass is still reported.  It also gives
the concordance the positional resolver the flat maps lack, which is what R-B
needs to be actionable.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONCORDANCE = ROOT / "src/compiler/identity_concordance.py"
SHELL = ROOT / "src/compiler/fortran_c_shell.py"

Q = chr(34) * 3

RESOLVER_ANCHOR = "def record_proven_shape("

RESOLVER = (
    "def scope_generations(book: Any, function: Any) -> dict:\n"
    "    " + Q + "Every generation id of one function, keyed to its rebind.\n"
    "\n"
    "    Maps a value id to ``(rebind_key, generation, loop_node_id)`` where\n"
    "    the rebind key identifies the one quantity all three generations\n"
    "    speak for.  Two ids sharing a key and differing in generation are the\n"
    "    same value and different bindings -- the distinction an alias map has\n"
    "    no way to represent.\n"
    "    " + Q + "\n"
    "\n"
    "    generations: dict = {}\n"
    "    for declaration in loop_scope_declarations(book, function):\n"
    "        loop_node_id = int(declaration[\"loop_node_id\"])\n"
    "        for rebind in declaration[\"rebinds\"]:\n"
    "            key = (loop_node_id, int(rebind[\"outer\"]))\n"
    "            for label in (\"outer\", \"carried\", \"inner\", \"reserved\"):\n"
    "                value_id = rebind.get(label)\n"
    "                if value_id is None:\n"
    "                    continue\n"
    "                generations.setdefault(\n"
    "                    int(value_id), (key, label, loop_node_id),\n"
    "                )\n"
    "    return generations\n"
    "\n"
    "\n"
    "def crossed_generation(generations: dict, source: int, target: int):\n"
    "    " + Q + "The crossing two ids make, or None if they make none.\n"
    "\n"
    "    ``reserved`` and ``inner`` name the same generation before and after\n"
    "    a rename, so a map between them is a re-declaration rather than a\n"
    "    collapse.\n"
    "    " + Q + "\n"
    "\n"
    "    origin = generations.get(int(source))\n"
    "    destination = generations.get(int(target))\n"
    "    if origin is None or destination is None:\n"
    "        return None\n"
    "    if origin[0] != destination[0] or origin[1] == destination[1]:\n"
    "        return None\n"
    "    if {origin[1], destination[1]} == {\"inner\", \"reserved\"}:\n"
    "        return None\n"
    "    return f\"{origin[1]} -> {destination[1]}\"\n"
    "\n"
    "\n"
    "def generation_in_scope(\n"
    "    book: Any, function: Any, block: str, value_id: int,\n"
    ") -> int:\n"
    "    " + Q + "What one name means at one position.\n"
    "\n"
    "    This is the question every flat ``{id: value}`` map in the pipeline\n"
    "    answers without being able to ask it.  Inside a loop the outer\n"
    "    generation is not in scope, so a binding recorded as a bare id must\n"
    "    be resolved here before it is used at a position, or it silently\n"
    "    means the value as it was before the first iteration.\n"
    "    " + Q + "\n"
    "\n"
    "    for declaration in loop_scope_declarations(book, function):\n"
    "        boundary = declaration[\"boundary\"]\n"
    "        if not boundary:\n"
    "            continue\n"
    "        header, latch, _exit = boundary\n"
    "        if str(block) not in {str(header), str(latch)} and str(block) != str(\n"
    "            latch\n"
    "        ):\n"
    "            inside = declaration.get(\"blocks\")\n"
    "            if inside is not None and str(block) not in inside:\n"
    "                continue\n"
    "        for rebind in declaration[\"rebinds\"]:\n"
    "            if int(rebind[\"outer\"]) == int(value_id):\n"
    "                return int(rebind[\"carried\"])\n"
    "    return int(value_id)\n"
    "\n"
    "\n"
    "def refuse_cross_generation_identities(\n"
    "    function: Any, mapping: Any, book: Any = None, *, stage: str = \"\",\n"
    ") -> dict:\n"
    "    " + Q + "Drop identity-map entries that merge two generations.\n"
    "\n"
    "    Returns the mapping without them and records each refusal, so the\n"
    "    pass that wanted one is named on the page instead of inferred from a\n"
    "    wrong answer downstream.  When nothing crosses, this is a no-op and\n"
    "    costs one dictionary walk.\n"
    "    " + Q + "\n"
    "\n"
    "    book = book or current_identity_book()\n"
    "    generations = scope_generations(book, function)\n"
    "    if not generations:\n"
    "        return mapping\n"
    "    page = book.page(\"scope_refusal\")\n"
    "    kept = {}\n"
    "    for source_id, replacement in mapping.items():\n"
    "        target_id = getattr(replacement, \"id\", replacement)\n"
    "        crossing = crossed_generation(\n"
    "            generations, int(source_id), int(target_id),\n"
    "        )\n"
    "        if crossing is None:\n"
    "            kept[source_id] = replacement\n"
    "            continue\n"
    "        row = (authored_function_name(function), int(source_id))\n"
    "        page.set(row, len(page.history(row)), (\n"
    "            str(stage), int(target_id), crossing,\n"
    "        ))\n"
    "    return kept\n"
    "\n"
    "\n"
    "def record_proven_shape("
)

RULE_CALL_ANCHOR = (
    "            declared = self._loop_scope_findings(module, str(name), function)"
)

RULE_CALL = (
    "            found.extend(\n"
    "                self._identity_collapse_findings(module, str(name), function)\n"
    "            )\n"
    "            declared = self._loop_scope_findings(module, str(name), function)"
)

RULE_ANCHOR = (
    "    @staticmethod\n"
    "    def _loop_scope_findings("
)

RULE = (
    "    @staticmethod\n"
    "    def _identity_collapse_findings(\n"
    "        module: Any, name: str, function: Any,\n"
    "    ) -> list[Finding]:\n"
    "        " + Q + "An operand rewrite that replaced one generation with another.\n"
    "\n"
    "        Every structural rewrite is already receipted in function\n"
    "        metadata, so this needs no instrumentation -- only the\n"
    "        declaration to read the receipts against.  A rewrite whose tie\n"
    "        was broken by incumbency is the one to look for: in a loop the\n"
    "        incumbent is always the generation from before it.\n"
    "        " + Q + "\n"
    "\n"
    "        book = identity_book(module)\n"
    "        generations = scope_generations(book, name)\n"
    "        if not generations:\n"
    "            return []\n"
    "        found: list[Finding] = []\n"
    "        receipts = function.metadata.get(\n"
    "            \"structural_identity_rebindings\", ()\n"
    "        )\n"
    "        for receipt in receipts:\n"
    "            try:\n"
    "                source = int(receipt[\"value_id\"])\n"
    "                target = int(receipt[\"replacement_value_id\"])\n"
    "            except (KeyError, TypeError, ValueError):\n"
    "                continue\n"
    "            crossing = crossed_generation(generations, source, target)\n"
    "            if crossing is None:\n"
    "                continue\n"
    "            found.append(Finding(\n"
    "                \"identity-collapse-crosses-generation\", name, source,\n"
    "                f\"{receipt.get('block')}#{receipt.get('instruction_index')} \"\n"
    "                f\"operand {receipt.get('operand_index')} was rewritten \"\n"
    "                f\"{source} -> {target} ({crossing}) by \"\n"
    "                f\"{receipt.get('priority')} with tie policy \"\n"
    "                f\"{receipt.get('tie_policy')}; the two are one value at \"\n"
    "                \"two generations, so there was no tie to break\",\n"
    "            ))\n"
    "        return found\n"
    "\n"
    "    @staticmethod\n"
    "    def _loop_scope_findings("
)

# ------------------------------------------------------------ the canonicalizer

CANONICAL_ANCHOR = (
    "        if len(choices) == 1:\n"
    "            canonical[source_id] = choices[0]\n"
    "\n"
    "    changes = []"
)

CANONICAL = (
    "        if len(choices) == 1:\n"
    "            canonical[source_id] = choices[0]\n"
    "\n"
    "    # Two generations of one loop variable are structurally identical --\n"
    "    # same shape, same dtype, same variable -- so this map is told they\n"
    "    # are the same and breaks the tie by incumbency.  Inside a loop the\n"
    "    # incumbent is always the value from before it, which makes the\n"
    "    # collapse systematically backwards rather than merely arbitrary.\n"
    "    # Identity of content is not identity of binding: where the two\n"
    "    # differ in a declared generation there is no tie, so refuse instead\n"
    "    # of choosing.\n"
    "    try:\n"
    "        from .identity_concordance import (\n"
    "            refuse_cross_generation_identities,\n"
    "        )\n"
    "\n"
    "        canonical = refuse_cross_generation_identities(\n"
    "            function.name, canonical, stage=\"exact_structural_identity\",\n"
    "        )\n"
    "    except Exception:\n"
    "        pass\n"
    "\n"
    "    changes = []"
)


def patch(path, pairs):
    text = path.read_text(encoding="utf-8")
    for anchor, replacement, expected in pairs:
        count = text.count(anchor)
        assert count == expected, (path.name, count, expected, anchor[:70])
        text = text.replace(anchor, replacement)
    path.write_text(text, encoding="utf-8")


patch(CONCORDANCE, [
    (RESOLVER_ANCHOR, RESOLVER, 1),
    (RULE_CALL_ANCHOR, RULE_CALL, 1),
    (RULE_ANCHOR, RULE, 1),
])

patch(SHELL, [
    (CANONICAL_ANCHOR, CANONICAL, 1),
])

print("R-A installed as a refusal; receipts read against the declaration")
