"""Make the pass that renames a carried generation re-declare it.

The builder deliberately repoints a carried Phi's latch operand away from the
slot reserved before the body was lowered, because a nested loop -- or, as it
turns out, any outlined body whose result returns through an aggregate --
publishes the update under a different name.  That repoint is correct.  What
was missing is that it never told anyone, so the declaration went stale and
the scope rules reported a rename as though it were a defect.

Re-declaration is the third obligation the scope design asks of a pass: a
transformation may rename a value crossing the boundary, but it must say so.
The page keeps every generation it was ever told, at its own column, so the
history of a carried value's identity survives the rename instead of being
overwritten by it.

Also records what each scheduled callsite resolved its arguments to.  The
docstring on ``emit_plan_callsite`` says arguments resolve through
``external_value`` at the callsite's position, which inside a loop body is
exactly what maps them to the current iteration -- yet six calls hold the
outer generation.  Either they do not take this path, or the binding was
replaced before they asked.  The page distinguishes those two without
inference.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONCORDANCE = ROOT / "src/compiler/identity_concordance.py"
BUILDER = ROOT / "src/compiler/precompile_to_ssa.py"

Q = chr(34) * 3

# ------------------------------------------------------------- re-declaration

REDECLARE_ANCHOR = "def loop_scope_declarations("

REDECLARE = (
    "def redeclare_loop_inner(\n"
    "    function: Any, loop_node_id: Any, carried_id: int, inner_id: int,\n"
    ") -> None:\n"
    "    " + Q + "Record that a pass renamed a loop's inner generation.\n"
    "\n"
    "    Keyed by the CARRIED generation because that is the one name a pass\n"
    "    repointing a backedge always has in hand.  The new generation is\n"
    "    appended at its own column, so the page keeps every identity the\n"
    "    value has been given rather than the last one to win a race.\n"
    "    " + Q + "\n"
    "\n"
    "    page = current_identity_book().page(\"loop_scope\")\n"
    "    scope = (authored_function_name(function), int(loop_node_id))\n"
    "    for row in page.rows():\n"
    "        if not (isinstance(row, tuple) and len(row) == 3):\n"
    "            continue\n"
    "        if row[:2] != scope or row[2] == \"boundary\":\n"
    "            continue\n"
    "        generations = dict(page.history(row))\n"
    "        recorded = generations.get(CARRIED)\n"
    "        if recorded is None or int(recorded) != int(carried_id):\n"
    "            continue\n"
    "        column = max(max(generations) + 1, INNER + 2)\n"
    "        page.set(row, column, (\"inner\", int(inner_id)))\n"
    "        return\n"
    "\n"
    "\n"
    "def loop_scope_declarations("
)

READER_ANCHOR = (
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
    "            })"
)

READER = (
    "        generations = dict(page.history(row))\n"
    "        if (\n"
    "            OUTER in generations\n"
    "            and CARRIED in generations\n"
    "            and INNER in generations\n"
    "        ):\n"
    "            # A later column re-declares the inner generation after a\n"
    "            # pass renamed it; the deepest one is current, and the ones\n"
    "            # before it remain on the page as the value's history.\n"
    "            inner = int(generations[INNER])\n"
    "            for column in sorted(generations):\n"
    "                fact = generations[column]\n"
    "                if (\n"
    "                    isinstance(fact, tuple)\n"
    "                    and len(fact) == 2\n"
    "                    and fact[0] == \"inner\"\n"
    "                ):\n"
    "                    inner = int(fact[1])\n"
    "            record[\"rebinds\"].append({\n"
    "                \"outer\": int(generations[OUTER]),\n"
    "                \"carried\": int(generations[CARRIED]),\n"
    "                \"inner\": inner,\n"
    "                \"reserved\": int(generations[INNER]),\n"
    "            })"
)

# ------------------------------------------------------------------- builders

REPOINT_ANCHOR = (
    "            published = self.external_values.get(updated_id, reserved)\n"
    "            carried_updates[updated_id] = published\n"
    "            if published is not reserved:\n"
    "                carried_phis[updated_id].args[1] = published"
)

REPOINT = (
    "            published = self.external_values.get(updated_id, reserved)\n"
    "            carried_updates[updated_id] = published\n"
    "            if published is not reserved:\n"
    "                carried_phis[updated_id].args[1] = published\n"
    "                _redeclare_loop_inner(\n"
    "                    self, loop, carried_phis[updated_id], published,\n"
    "                )"
)

BACKEDGE_STORAGE_ANCHOR = (
    "                    current = carried_phis[updated_id].args[0]\n"
    "                    carried_phis[updated_id].args[1] = current\n"
    "                    carried_updates[updated_id] = current\n"
    "                    self.external_values[updated_id] = current\n"
    "                    current.accounting[\"ssa_storage_identity_backedge\"] = True"
)

BACKEDGE_STORAGE = (
    "                    current = carried_phis[updated_id].args[0]\n"
    "                    carried_phis[updated_id].args[1] = current\n"
    "                    carried_updates[updated_id] = current\n"
    "                    self.external_values[updated_id] = current\n"
    "                    current.accounting[\"ssa_storage_identity_backedge\"] = True\n"
    "                    _redeclare_loop_inner(\n"
    "                        self, loop, carried_phis[updated_id], current,\n"
    "                    )"
)

BACKEDGE_IDENTITY_ANCHOR = (
    "                    carried_phis[updated_id].args[1] = current\n"
    "                    carried_updates[updated_id] = current\n"
    "                    self.external_values[updated_id] = current\n"
    "                    current.accounting[\"ssa_identity_backedge\"] = True"
)

BACKEDGE_IDENTITY = (
    "                    carried_phis[updated_id].args[1] = current\n"
    "                    carried_updates[updated_id] = current\n"
    "                    self.external_values[updated_id] = current\n"
    "                    current.accounting[\"ssa_identity_backedge\"] = True\n"
    "                    _redeclare_loop_inner(\n"
    "                        self, loop, carried_phis[updated_id], current,\n"
    "                    )"
)

HELPER_ANCHOR = "    def emit_plan_callsite(self, callsite_id: int, *, location: str) -> None:"

HELPER = (
    "    def _note_callsite_arguments(\n"
    "        self, callsite_id: int, argument_ids: Any, arguments: Any,\n"
    "    ) -> None:\n"
    "        " + Q + "Record what one scheduled callsite resolved its arguments to.\n"
    "\n"
    "        A graph id that resolves to itself inside a loop body means the\n"
    "        carried machinery did not map it to the current iteration -- which\n"
    "        is either because the binding was replaced before the call asked,\n"
    "        or because the call never asked.\n"
    "        " + Q + "\n"
    "\n"
    "        try:\n"
    "            from .identity_concordance import current_identity_book\n"
    "\n"
    "            page = current_identity_book().page(\"callsite_argument\")\n"
    "            for position, (graph_id, value) in enumerate(\n"
    "                zip(argument_ids, arguments)\n"
    "            ):\n"
    "                row = (str(self.function_name), int(callsite_id), position)\n"
    "                page.set(row, len(page.history(row)), (\n"
    "                    int(graph_id), int(value.id),\n"
    "                    bool(self.loop_targets),\n"
    "                ))\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "    def emit_plan_callsite(self, callsite_id: int, *, location: str) -> None:"
)

TRACE_ANCHOR = (
    "        argument_ids, result_ids = bindings\n"
    "        arguments = [\n"
    "            self.external_value(int(value_id)) for value_id in argument_ids\n"
    "        ]"
)

TRACE = TRACE_ANCHOR + "\n        self._note_callsite_arguments(callsite_id, argument_ids, arguments)"

MODULE_HELPER_ANCHOR = "class _ControlSSABuilder"

MODULE_HELPER = (
    "def _redeclare_loop_inner(builder: Any, loop: Any, phi: Any, value: Any) -> None:\n"
    "    " + Q + "Tell the concordance a carried generation was renamed.\n"
    "\n"
    "    A pass may repoint a backedge -- a nested loop publishes under its own\n"
    "    name, and an outlined body's result returns through an aggregate that\n"
    "    is unpacked into a fresh one -- but the declaration has to follow, or\n"
    "    the scope rules read a legitimate rename as a defect and a real one as\n"
    "    nothing at all.\n"
    "    " + Q + "\n"
    "\n"
    "    try:\n"
    "        from .identity_concordance import redeclare_loop_inner\n"
    "\n"
    "        redeclare_loop_inner(\n"
    "            builder.function_name, loop.source_loop_node_id,\n"
    "            int(phi.res.id), int(value.id),\n"
    "        )\n"
    "    except Exception:\n"
    "        pass\n"
    "\n"
    "\n"
    "class _ControlSSABuilder"
)


def patch(path, pairs):
    text = path.read_text(encoding="utf-8")
    for anchor, replacement, expected in pairs:
        count = text.count(anchor)
        assert count == expected, (path.name, count, expected, anchor[:70])
        text = text.replace(anchor, replacement)
    path.write_text(text, encoding="utf-8")


# The concordance half applied on an earlier run of this script.
if "def redeclare_loop_inner(" not in CONCORDANCE.read_text(encoding="utf-8"):
    patch(CONCORDANCE, [
        (REDECLARE_ANCHOR, REDECLARE, 1),
        (READER_ANCHOR, READER, 1),
    ])

patch(BUILDER, [
    (MODULE_HELPER_ANCHOR, MODULE_HELPER, 1),
    (REPOINT_ANCHOR, REPOINT, 2),
    (BACKEDGE_STORAGE_ANCHOR, BACKEDGE_STORAGE, 1),
    (BACKEDGE_IDENTITY_ANCHOR, BACKEDGE_IDENTITY, 1),
    (HELPER_ANCHOR, HELPER, 1),
    (TRACE_ANCHOR, TRACE, 1),
])

print("re-declaration obligation added; callsite argument resolution recorded")
