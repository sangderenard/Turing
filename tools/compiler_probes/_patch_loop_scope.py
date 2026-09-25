"""Declare a loop as an evolving scope, and rule on it in the concordance.

Every construction in this pipeline encodes "x evolves across iterations" in
a medium a transformation is free to change: the SLOT encodes it in mutation
order (outlining moved the write into a region and the aggregate return
renamed it); the PHI encodes it in edge order (specialization captured its
operand before that edge existed).  Neither is a fixpoint under the
transformations the pipeline performs, so neither is more right than the
other -- they fail symmetrically, at the same boundary, because neither is a
DECLARATION.  The builder's own comment says as much: the latch operand is
"the LAST publish of a carried id" and "that dependency is invisible to
feed/output signatures", which is why the only defense on hand is to forbid
reordering wholesale.  It stops reordering.  It does not stop outlining.

The concordance already owns the right mechanism -- ``bind_alias`` is a
scoped, history-preserving alias table whose rebinding appends a column.  Two
things kept it from covering loops: every caller passes the FUNCTION as the
scope, so there is no granularity below a function; and ``resolve_alias``
chases to a terminal fixpoint and raises on cycles, which is precisely what a
loop rebinding is.  So this adds a scope one level finer, where the column is
the GENERATION rather than a round, and cycles are the subject rather than an
error.

One name, three generations:

    OUTER    the value before the loop            (12)
    CARRIED  the header Phi that merges them      (...998)
    INNER    the slot the body redefines          (63)

and three rules that follow from the declaration alone.  Rule 1 catches the
six stale reads; rule 2 catches the aggregate return minting a new name for
the inner slot; rule 3 catches an inner name whose only definition sits in
the preheader.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONCORDANCE = ROOT / "src/compiler/identity_concordance.py"
BUILDER = ROOT / "src/compiler/precompile_to_ssa.py"

Q = chr(34) * 3

# ---------------------------------------------------------------- declaration

DECL_ANCHOR = "def record_proven_shape("

DECL = (
    "OUTER, CARRIED, INNER = 0, 1, 2\n"
    "\n"
    "_GENERATION_NAMES = {OUTER: \"outer\", CARRIED: \"carried\", INNER: \"inner\"}\n"
    "\n"
    "\n"
    "def declare_loop_scope(\n"
    "    function: Any, loop_node_id: Any, header: str, latch: str,\n"
    "    exit_block: str, rebinds: Any,\n"
    ") -> None:\n"
    "    " + Q + "Declare one loop as a scope whose bindings evolve per iteration.\n"
    "\n"
    "    A nested scope binds a name once for its whole extent.  A loop body\n"
    "    binds it differently on every entry, so the page's COLUMN is the\n"
    "    generation -- outer, carried, inner -- rather than a round.  The\n"
    "    boundary is recorded with the rebinds because a transformation that\n"
    "    moves code across it must be able to ask where it is, instead of\n"
    "    recovering it from block names or branch topology that the\n"
    "    transformation itself may have rewritten.\n"
    "    " + Q + "\n"
    "\n"
    "    page = current_identity_book().page(\"loop_scope\")\n"
    "    scope = (authored_function_name(function), int(loop_node_id))\n"
    "    page.set(\n"
    "        (*scope, \"boundary\"), 0,\n"
    "        (str(header), str(latch), str(exit_block)),\n"
    "    )\n"
    "    for outer_id, carried_id, inner_id, graph_outer, graph_inner in rebinds:\n"
    "        row = (*scope, int(outer_id))\n"
    "        page.set(row, OUTER, int(outer_id))\n"
    "        page.set(row, CARRIED, int(carried_id))\n"
    "        page.set(row, INNER, int(inner_id))\n"
    "        page.set(row, INNER + 1, (\"graph\", int(graph_outer), int(graph_inner)))\n"
    "\n"
    "\n"
    "def loop_scope_declarations(book: Any, function: Any) -> list[dict]:\n"
    "    " + Q + "Every loop scope declared for one authored function." + Q + "\n"
    "\n"
    "    page = book.page(\"loop_scope\")\n"
    "    wanted = authored_function_name(function)\n"
    "    scopes: dict[int, dict] = {}\n"
    "    for row in page.rows():\n"
    "        if not (isinstance(row, tuple) and len(row) == 3):\n"
    "            continue\n"
    "        name, loop_node_id, key = row\n"
    "        if name != wanted:\n"
    "            continue\n"
    "        record = scopes.setdefault(\n"
    "            int(loop_node_id),\n"
    "            {\n"
    "                \"loop_node_id\": int(loop_node_id),\n"
    "                \"boundary\": None,\n"
    "                \"rebinds\": [],\n"
    "            },\n"
    "        )\n"
    "        if key == \"boundary\":\n"
    "            record[\"boundary\"] = page.latest(row)\n"
    "            continue\n"
    "        generations = dict(page.history(row))\n"
    "        if (\n"
    "            OUTER in generations\n"
    "            and CARRIED in generations\n"
    "            and INNER in generations\n"
    "        ):\n"
    "            record[\"rebinds\"].append({\n"
    "                \"outer\": int(generations[OUTER]),\n"
    "                \"carried\": int(generations[CARRIED]),\n"
    "                \"inner\": int(generations[INNER]),\n"
    "            })\n"
    "    return [record for record in scopes.values() if record[\"boundary\"]]\n"
    "\n"
    "\n"
    "def record_proven_shape("
)

# ---------------------------------------------------------------------- rules

RULE_CALL_ANCHOR = (
    "            found.extend(\n"
    "                self._stale_carried_read_findings(str(name), function)\n"
    "            )"
)

RULE_CALL = (
    "            found.extend(\n"
    "                self._loop_scope_findings(module, str(name), function)\n"
    "            )\n"
    "            found.extend(\n"
    "                self._stale_carried_read_findings(str(name), function)\n"
    "            )"
)

RULE_ANCHOR = (
    "    @staticmethod\n"
    "    def _stale_carried_read_findings("
)

RULE = (
    "    @staticmethod\n"
    "    def _loop_scope_findings(\n"
    "        module: Any, name: str, function: Any,\n"
    "    ) -> list[Finding]:\n"
    "        " + Q + "Rule on a loop against the scope it declared.\n"
    "\n"
    "        These are the checks that no amount of shape, dominance or type\n"
    "        agreement can supply, because every name involved has the same\n"
    "        shape, the same dtype, and a definition that dominates every use.\n"
    "        The only thing separating them is which generation they speak\n"
    "        for, and that is knowable only from the declaration.\n"
    "        " + Q + "\n"
    "\n"
    "        book = identity_book(module)\n"
    "        declarations = loop_scope_declarations(book, name)\n"
    "        if not declarations:\n"
    "            return []\n"
    "\n"
    "        definitions: dict[int, str] = {}\n"
    "        for block_name, block in function.blocks.items():\n"
    "            for instruction in block.instrs:\n"
    "                if instruction.res is not None:\n"
    "                    definitions[int(instruction.res.id)] = str(block_name)\n"
    "\n"
    "        found: list[Finding] = []\n"
    "        for declaration in declarations:\n"
    "            header, latch, exit_block = declaration[\"boundary\"]\n"
    "            inside = _blocks_between(function, header, latch)\n"
    "            if not inside:\n"
    "                continue\n"
    "            for rebind in declaration[\"rebinds\"]:\n"
    "                outer = rebind[\"outer\"]\n"
    "                carried = rebind[\"carried\"]\n"
    "                inner = rebind[\"inner\"]\n"
    "\n"
    "                # Rule 1 -- inside the scope the outer generation is not\n"
    "                # in scope.  A use of it reads the value as it was before\n"
    "                # the first iteration, on every iteration.\n"
    "                for block_name in sorted(inside):\n"
    "                    block = function.blocks.get(block_name)\n"
    "                    if block is None:\n"
    "                        continue\n"
    "                    for index, instruction in enumerate(block.instrs):\n"
    "                        if str(instruction.op).lower() == \"phi\":\n"
    "                            continue\n"
    "                        for position, argument in enumerate(instruction.args):\n"
    "                            if getattr(argument, \"id\", None) is None:\n"
    "                                continue\n"
    "                            if int(argument.id) != outer:\n"
    "                                continue\n"
    "                            found.append(Finding(\n"
    "                                \"loop-scope-outer-read\", name, outer,\n"
    "                                f\"{block_name}#{index} {instruction.op} \"\n"
    "                                f\"operand {position} names the outer \"\n"
    "                                f\"generation of a value the loop rebinds; \"\n"
    "                                f\"in scope it is {carried} (carried) or \"\n"
    "                                f\"{inner} (inner)\",\n"
    "                            ))\n"
    "\n"
    "                # Rule 2 -- the backedge must carry the declared inner\n"
    "                # name.  An outlined body whose result returns through an\n"
    "                # aggregate arrives under a name the parent minted after\n"
    "                # the crossing; the value is right, the identity is lost.\n"
    "                phi = _carried_phi(function, header, carried)\n"
    "                if phi is not None and len(phi.args) == 2:\n"
    "                    incoming = phi.attributes.get(\"incoming_blocks\") or ()\n"
    "                    for origin, argument in zip(incoming, phi.args):\n"
    "                        if str(origin) != str(latch):\n"
    "                            continue\n"
    "                        if getattr(argument, \"id\", None) is None:\n"
    "                            continue\n"
    "                        if int(argument.id) == inner:\n"
    "                            continue\n"
    "                        found.append(Finding(\n"
    "                            \"loop-scope-latch-renamed\", name, inner,\n"
    "                            f\"{header} Phi {carried} takes \"\n"
    "                            f\"{int(argument.id)} from {latch}, but the \"\n"
    "                            f\"loop declared its inner generation as \"\n"
    "                            f\"{inner}; a transformation renamed the value \"\n"
    "                            \"crossing the boundary without re-declaring \"\n"
    "                            \"it\",\n"
    "                        ))\n"
    "\n"
    "                # Rule 3 -- the inner generation must be defined inside\n"
    "                # the scope.  A seed in the preheader is storage\n"
    "                # initialization and is correct; a seed that is its ONLY\n"
    "                # definition means the body never wrote the slot.\n"
    "                where = definitions.get(inner)\n"
    "                if where is not None and where not in inside:\n"
    "                    found.append(Finding(\n"
    "                        \"loop-scope-inner-outside\", name, inner,\n"
    "                        f\"the inner generation is defined in {where}, \"\n"
    "                        f\"outside the scope ({header}..{latch}); nothing \"\n"
    "                        \"in the body redefines it\",\n"
    "                    ))\n"
    "        return found\n"
    "\n"
    "    @staticmethod\n"
    "    def _stale_carried_read_findings("
)

HELPERS_ANCHOR = "class IdentityPage:"

HELPERS = (
    "def _blocks_between(function: Any, header: str, latch: str) -> set[str]:\n"
    "    " + Q + "Blocks on some path from the declared header to the declared latch."
    + Q + "\n"
    "\n"
    "    successors: dict[str, set[str]] = {}\n"
    "    for block_name, block in function.blocks.items():\n"
    "        targets: set[str] = set()\n"
    "        for instruction in block.instrs:\n"
    "            for key in (\"target\", \"true_target\", \"false_target\"):\n"
    "                declared = instruction.attributes.get(key)\n"
    "                if declared is not None:\n"
    "                    targets.add(str(declared))\n"
    "        successors[str(block_name)] = targets\n"
    "    forward: set[str] = set()\n"
    "    frontier = [str(header)]\n"
    "    while frontier:\n"
    "        current = frontier.pop()\n"
    "        if current in forward:\n"
    "            continue\n"
    "        forward.add(current)\n"
    "        frontier.extend(successors.get(current, ()))\n"
    "    backward: set[str] = set()\n"
    "    frontier = [str(latch)]\n"
    "    while frontier:\n"
    "        current = frontier.pop()\n"
    "        if current in backward:\n"
    "            continue\n"
    "        backward.add(current)\n"
    "        for candidate, onward in successors.items():\n"
    "            if current in onward:\n"
    "                frontier.append(candidate)\n"
    "    return forward & backward\n"
    "\n"
    "\n"
    "def _carried_phi(function: Any, header: str, carried_id: int) -> Any:\n"
    "    " + Q + "The header Phi that speaks for one declared carried generation."
    + Q + "\n"
    "\n"
    "    block = function.blocks.get(str(header))\n"
    "    for instruction in (block.instrs if block is not None else ()):\n"
    "        if str(instruction.op).lower() != \"phi\":\n"
    "            continue\n"
    "        if instruction.res is None:\n"
    "            continue\n"
    "        if int(instruction.res.id) == int(carried_id):\n"
    "            return instruction\n"
    "    return None\n"
    "\n"
    "\n"
    "class IdentityPage:"
)

# ------------------------------------------------------------------- builders

PHI_ATTRS = (
    "                    \"incoming_blocks\": (preheader.name, latch.name),\n"
    "                    \"binding\": \"loop_carried\",\n"
    "                    \"initial_value_id\": initial_id,\n"
    "                    \"updated_value_id\": updated_id,\n"
    "                    \"recursion_region_id\": recursion_region_id,"
)

PHI_ATTRS_DECLARED = (
    PHI_ATTRS + "\n"
    "                    # The loop this generation belongs to, so a consumer\n"
    "                    # can reach the scope declaration without recovering\n"
    "                    # it from branch topology a transformation may have\n"
    "                    # rewritten.\n"
    "                    \"source_loop_node_id\": loop.source_loop_node_id,"
)

DECLARE_ANCHOR = (
    "                bound_initial_ids.add(initial_id)\n"
    "        break_bound_initials = {"
)

DECLARE_CALL = (
    "                bound_initial_ids.add(initial_id)\n"
    "        # Declare the loop as an evolving scope.  The rebind table is\n"
    "        # exactly what ``carried`` already holds; what was missing is any\n"
    "        # statement that these three names are one quantity at three\n"
    "        # generations, which is why three of them could disagree in\n"
    "        # silence.\n"
    "        try:\n"
    "            from .identity_concordance import declare_loop_scope\n"
    "\n"
    "            declare_loop_scope(\n"
    "                self.function_name, loop.source_loop_node_id,\n"
    "                header.name, latch.name, exit_block.name,\n"
    "                tuple(\n"
    "                    (\n"
    "                        int(_initial.id), int(_current.id),\n"
    "                        int(_updated.id), int(_initial_id),\n"
    "                        int(_updated_id),\n"
    "                    )\n"
    "                    for _updated_id, _initial_id, _initial, _updated,\n"
    "                    _current in carried\n"
    "                ),\n"
    "            )\n"
    "        except Exception:\n"
    "            pass\n"
    "        break_bound_initials = {"
)


def patch(path, pairs):
    text = path.read_text(encoding="utf-8")
    for anchor, replacement, expected in pairs:
        count = text.count(anchor)
        assert count == expected, (path.name, count, expected, anchor[:60])
        text = text.replace(anchor, replacement)
    path.write_text(text, encoding="utf-8")


patch(CONCORDANCE, [
    (DECL_ANCHOR, DECL, 1),
    (RULE_CALL_ANCHOR, RULE_CALL, 1),
    (RULE_ANCHOR, RULE, 1),
    (HELPERS_ANCHOR, HELPERS, 1),
])

patch(BUILDER, [
    (PHI_ATTRS, PHI_ATTRS_DECLARED, 2),
    (DECLARE_ANCHOR, DECLARE_CALL, 2),
])

print("loop scope declared; three rules added to the concordance")
