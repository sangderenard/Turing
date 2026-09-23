"""Master correlation table over one lowered SSA module.

A side product of compilation: neither source, nor IR, nor artifact.  The
lowering keeps many partial records of what a value *is* -- ``parameter_names``,
``storage_formals``, ``program_abi_*`` accounting, sequence descriptors,
record tables, ``value_aliases``, ``callee_input_ids`` -- each written by a
different pass and each correlated through its own key (a name, a signature
position, a source coordinate, an id).  Every fault traced on 2026-09-17 was
two of those records disagreeing about one value while nothing compared them.

This module collects every such claim into one table, one row per
``(function, value_id)``, and reports the places where the claims disagree.
It writes nothing back.  The table is the thing a future single identity
authority would own; until then it is the audit that says where the
authority is missing.

Findings (each is one concrete disagreement, with the two claims):

``multiple-definition``
    one value id defined by more than one instruction (in/out ABI fields
    that are deliberately redefined by every producer are exempt).
``unaccounted-formal``
    a formal no record names: not a parameter, not ABI storage, not leased
    frame storage, not a closure/member formal.
``conflicting-storage-claims``
    a formal claimed as two different ABI fields, or as an ABI field and
    as leased frame storage at once.
``descriptor-member-unknown``
    a sequence descriptor names a member id the function neither receives
    nor defines.
``descriptor-member-shared``
    one storage id plays a role in two descriptors of one function.
``helper-operand-outside-descriptor``
    a keyed lookup helper call reads storage no descriptor of the function
    names (the rewrite bound the call but not the descriptor, or vice
    versa).
``duplicate-storage-across-call``
    a callee formal identified as an ABI field or a sequence member is fed
    freshly leased caller storage although the caller already owns a value
    with that same identity.
``use-not-dominated``
    an instruction reads a value whose definition does not dominate the
    read (Phi operands exempt; they arrive over predecessor edges).
``alias-target-missing``
    a recorded value alias points at a value the function never defines.
``alias-not-concorded``
    a durable function-local alias receipt is absent from, or disagrees with,
    the shared planning identity page.
``source-field-identity-disagreement``
    repeated source-stage reads assign different class identities to one
    authored object field.
``callable-identity-disagreement``
    one exact source value is assigned different function-table addresses as
    it moves from first-class function syntax through a callable record field.
``source-parameter-identity-disagreement``
    one discovered static parameter identity changes between source stages.
"""

from __future__ import annotations

import contextvars
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .id_space import group_by_prefix, label as id_label


@dataclass(frozen=True)
class Claim:
    """One record's statement about a value's identity."""

    kind: str
    key: str
    source: str


@dataclass
class ValueRow:
    function: str
    value_id: int
    dtype: str | None = None
    is_formal: bool = False
    claims: list[Claim] = field(default_factory=list)
    definitions: list[tuple[str, int, str]] = field(default_factory=list)
    uses: list[tuple[str, int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Finding:
    kind: str
    function: str
    value_id: int | None
    detail: str


_INOUT_REDEFINED = ("program_abi_mutable", "program_abi_field_written")


class CorrelationTable:
    """Every identity claim the module makes, keyed by (function, value id)."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, int], ValueRow] = {}
        self.sequence_roles: dict[tuple[str, int], list[tuple[int, str]]] = (
            defaultdict(list)
        )

    # ------------------------------------------------------------------ build
    def row(self, function: str, value_id: int, dtype: Any = None) -> ValueRow:
        key = (str(function), int(value_id))
        row = self.rows.get(key)
        if row is None:
            row = ValueRow(str(function), int(value_id), dtype=None)
            self.rows[key] = row
        if dtype is not None and row.dtype is None:
            row.dtype = str(dtype)
        return row

    def claim(self, function: str, value_id: int, kind: str, key: Any,
              source: str) -> None:
        self.row(function, value_id).claims.append(
            Claim(str(kind), str(key), str(source))
        )

    @classmethod
    def build(cls, module: Any) -> "CorrelationTable":
        table = cls()
        for name, function in module.functions.items():
            table._build_function(module, str(name), function)
        return table

    def _build_function(self, module: Any, name: str, function: Any) -> None:
        metadata = dict(getattr(function, "metadata", {}) or {})
        for formal in function.args:
            row = self.row(name, int(formal.id), formal.dtype)
            row.is_formal = True
            accounting = dict(formal.accounting or {})
            abi_field = accounting.get("program_abi_field")
            abi_parameter = accounting.get("program_abi_parameter")
            if abi_field is not None:
                self.claim(
                    name, formal.id, "abi-field",
                    f"{accounting.get('program_abi_record') or abi_parameter}"
                    f".{abi_field}",
                    "accounting.program_abi_field",
                )
            elif abi_parameter is not None:
                self.claim(
                    name, formal.id, "abi-parameter", abi_parameter,
                    "accounting.program_abi_parameter",
                )
            if accounting.get("program_abi_keyed_owner") is not None:
                self.claim(
                    name, formal.id, "keyed-part",
                    f"{accounting['program_abi_keyed_owner']}."
                    f"{accounting.get('program_abi_keyed_part')}",
                    "accounting.program_abi_keyed_owner",
                )
            if accounting.get("linked_call_frame_storage"):
                self.claim(
                    name, formal.id, "frame-storage",
                    f"{accounting['linked_call_frame_storage']}"
                    f"#{accounting.get('propagated_formal_id', '?')}",
                    "accounting.linked_call_frame_storage",
                )
            if accounting.get("compiler_frame_storage"):
                self.claim(
                    name, formal.id, "compiler-frame-storage",
                    accounting["compiler_frame_storage"],
                    "accounting.compiler_frame_storage",
                )
            if accounting.get("projected_row_source_id") is not None:
                self.claim(
                    name, formal.id, "projected-row",
                    f"{accounting['projected_row_source_id']}"
                    f"[{accounting.get('projected_row_column')}]",
                    "accounting.projected_row_source_id",
                )
        for label, value_id in metadata.get("parameter_names", ()) or ():
            self.claim(name, value_id, "parameter", label,
                       "metadata.parameter_names")
        for entry in metadata.get("storage_formals", ()) or ():
            self.claim(name, entry["value_id"], "storage-formal",
                       entry.get("kind", "storage"), "metadata.storage_formals")
        for entry in metadata.get("closure_formals", ()) or ():
            self.claim(name, entry["value_id"], "closure-formal",
                       entry.get("name", "?"), "metadata.closure_formals")
        for entry in metadata.get("parameter_member_formals", ()) or ():
            self.claim(
                name, entry["value_id"], "member-formal",
                f"{entry.get('parameter')}{list(entry.get('path', ()))}",
                "metadata.parameter_member_formals",
            )
        aliases = metadata.get("value_aliases", ()) or ()
        alias_pairs = (
            aliases.items() if isinstance(aliases, Mapping) else aliases
        )
        for pair in alias_pairs:
            try:
                alias, target = pair
            except (TypeError, ValueError):
                continue
            self.claim(name, alias, "alias-of", int(target),
                       "metadata.value_aliases")
        for pair in metadata.get("output_identity_aliases", ()) or ():
            try:
                alias, target = pair
            except (TypeError, ValueError):
                continue
            self.claim(name, alias, "alias-of", int(target),
                       "metadata.output_identity_aliases")
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is not None:
                    row = self.row(name, int(instruction.res.id),
                                   instruction.res.dtype)
                    row.definitions.append((block_name, index, instruction.op))
                for argument in instruction.args:
                    argument_id = getattr(argument, "id", None)
                    if argument_id is None:
                        continue
                    self.row(name, int(argument_id)).uses.append(
                        (block_name, index, instruction.op)
                    )
        sequence_table = (getattr(module, "sequence_tables", {}) or {}).get(name)
        if sequence_table is not None:
            for descriptor in sequence_table.sequences.values():
                roles = [
                    (int(descriptor.sequence_id), "handle"),
                    *(
                        (int(column), f"column{position}")
                        for position, column in enumerate(
                            descriptor.column_value_ids
                        )
                    ),
                    (int(descriptor.length_address_id), "length"),
                    (int(descriptor.capacity_value_id), "capacity"),
                ]
                if descriptor.status_address_id is not None:
                    roles.append((int(descriptor.status_address_id), "status"))
                if descriptor.live_flags_value_id is not None:
                    roles.append(
                        (int(descriptor.live_flags_value_id), "live_flags")
                    )
                for value_id, role in roles:
                    self.claim(
                        name, value_id, "sequence-member",
                        f"seq{int(descriptor.sequence_id)}.{role}",
                        "sequence_table",
                    )
                    self.sequence_roles[(name, value_id)].append(
                        (int(descriptor.sequence_id), role)
                    )
        record_table = (getattr(module, "record_tables", {}) or {}).get(name)
        if record_table is not None:
            for record in record_table.records.values():
                for record_field in record.fields:
                    for value_id in record_field.value_ids:
                        self.claim(
                            name, value_id, "record-field",
                            f"{record.identity}#{record.record_id}"
                            f".{record_field.name}",
                            "record_table",
                        )

    # --------------------------------------------------------------- findings
    def findings(self, module: Any) -> list[Finding]:
        found: list[Finding] = []
        for name, function in module.functions.items():
            found.extend(self._function_findings(module, str(name), function))
            found.extend(
                self._undefined_operand_findings(str(name), function)
            )
            declared = self._loop_scope_findings(module, str(name), function)
            found.extend(declared)
            # The inferred check recovers the scope from branch topology; it
            # covers loops built by a path that does not declare one.  Where a
            # declaration exists it is authoritative, so do not report twice.
            if not loop_scope_declarations(identity_book(module), str(name)):
                found.extend(
                    self._stale_carried_read_findings(str(name), function)
                )
        found.extend(self._sequence_descriptor_findings(module))
        found.extend(self._binding_kind_findings(module))
        found.extend(self._source_field_identity_findings(module))
        found.extend(self._source_parameter_identity_findings(module))
        found.extend(self._callable_identity_findings(module))
        return found

    @staticmethod
    def _loop_scope_findings(
        module: Any, name: str, function: Any,
    ) -> list[Finding]:
        """Rule on a loop against the scope it declared.

        These are the checks that no amount of shape, dominance or type
        agreement can supply, because every name involved has the same
        shape, the same dtype, and a definition that dominates every use.
        The only thing separating them is which generation they speak
        for, and that is knowable only from the declaration.
        """

        book = identity_book(module)
        declarations = loop_scope_declarations(book, name)
        if not declarations:
            return []

        definitions: dict[int, str] = {}
        for block_name, block in function.blocks.items():
            for instruction in block.instrs:
                if instruction.res is not None:
                    definitions[int(instruction.res.id)] = str(block_name)

        found: list[Finding] = []
        for declaration in declarations:
            header, latch, exit_block = declaration["boundary"]
            inside = _blocks_between(function, header, latch)
            if not inside:
                continue
            for rebind in declaration["rebinds"]:
                outer = rebind["outer"]
                carried = rebind["carried"]
                inner = rebind["inner"]

                # Rule 1 -- inside the scope the outer generation is not
                # in scope.  A use of it reads the value as it was before
                # the first iteration, on every iteration.
                for block_name in sorted(inside):
                    block = function.blocks.get(block_name)
                    if block is None:
                        continue
                    for index, instruction in enumerate(block.instrs):
                        if str(instruction.op).lower() == "phi":
                            continue
                        for position, argument in enumerate(instruction.args):
                            if getattr(argument, "id", None) is None:
                                continue
                            if int(argument.id) != outer:
                                continue
                            found.append(Finding(
                                "loop-scope-outer-read", name, outer,
                                f"{block_name}#{index} {instruction.op} "
                                f"operand {position} names the outer "
                                f"generation of a value the loop rebinds; "
                                f"in scope it is {carried} (carried) or "
                                f"{inner} (inner)",
                            ))

                # Rule 2 -- the backedge must carry the declared inner
                # name.  An outlined body whose result returns through an
                # aggregate arrives under a name the parent minted after
                # the crossing; the value is right, the identity is lost.
                phi = _carried_phi(function, header, carried)
                if phi is not None and len(phi.args) == 2:
                    incoming = phi.attributes.get("incoming_blocks") or ()
                    for origin, argument in zip(incoming, phi.args):
                        if str(origin) != str(latch):
                            continue
                        if getattr(argument, "id", None) is None:
                            continue
                        if int(argument.id) == inner:
                            continue
                        found.append(Finding(
                            "loop-scope-latch-renamed", name, inner,
                            f"{header} Phi {carried} takes "
                            f"{int(argument.id)} from {latch}, but the "
                            f"loop declared its inner generation as "
                            f"{inner}; a transformation renamed the value "
                            "crossing the boundary without re-declaring "
                            "it",
                        ))

                # Rule 3 -- the inner generation must be defined inside
                # the scope.  A seed in the preheader is storage
                # initialization and is correct; a seed that is its ONLY
                # definition means the body never wrote the slot.
                where = definitions.get(inner)
                if where is not None and where not in inside:
                    found.append(Finding(
                        "loop-scope-inner-outside", name, inner,
                        f"the inner generation is defined in {where}, "
                        f"outside the scope ({header}..{latch}); nothing "
                        "in the body redefines it",
                    ))
        return found

    @staticmethod
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
    def _callable_identity_findings(module: Any) -> list[Finding]:
        """Report a first-class callable whose exact address changed."""

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "callable_identity_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            distinct = tuple(dict.fromkeys(
                int(fact) for _column, fact in history
            ))
            if len(distinct) <= 1:
                continue
            function, value_id = (
                row if isinstance(row, tuple) and len(row) == 2
                else (str(row), None)
            )
            found.append(Finding(
                "callable-identity-disagreement",
                str(function),
                None if value_id is None else int(value_id),
                "first-class callable changed function-table address across "
                f"source stages: {distinct!r}",
            ))
        return found

    @staticmethod
    def _source_field_identity_findings(module: Any) -> list[Finding]:
        """Report a field whose source-class identity changed between reads."""

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_field_identity_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            distinct = tuple(dict.fromkeys(repr(fact) for _column, fact in history))
            if len(distinct) <= 1:
                continue
            owner, field = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, "?")
            )
            found.append(Finding(
                "source-field-identity-disagreement",
                str(owner),
                None,
                f"field {field!r} changed identity across source stages: "
                f"{distinct!r}",
            ))
        return found

    @staticmethod
    def _source_parameter_identity_findings(module: Any) -> list[Finding]:
        """Report a static parameter whose discovered identity changed."""

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_parameter_identity_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in history
            ))
            if len(distinct) <= 1:
                continue
            scope, parameter = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, "?")
            )
            found.append(Finding(
                "source-parameter-identity-disagreement",
                str(scope),
                None,
                f"parameter {parameter!r} changed identity across source "
                f"stages: {distinct!r}",
            ))
        return found

    @staticmethod
    def _binding_kind_findings(module: Any) -> list[Finding]:
        """One caller slot bound under two different kinds across callsites.

        A frame binding's kind is not a label on the slot; it selects which
        machinery may materialize it.  ``caller_storage`` can restore a slot
        a structural cleanup removed, ``caller_alias`` and ``caller_value``
        cannot.  So when two callsites name the SAME caller id for the same
        callee formal but disagree on the kind, they do not merely describe
        it differently -- one of them can supply the argument and the other
        reports ``missing_<kind>`` and refuses the call.

        Neither callsite can see this on its own: each consults its own
        private maps in its own elif order and records a locally consistent
        answer.  The disagreement only exists across the pair, which is the
        whole reason the decisions are written to a shared page.
        """
        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = (getattr(book, "pages", {}) or {}).get("argument_binding")
        if page is None:
            return []
        found: list[Finding] = []
        for row in page.rows():
            kinds_by_source: dict[Any, dict[str, list[int]]] = {}
            for column, fact in page.history(row):
                if not (isinstance(fact, tuple) and len(fact) == 2):
                    continue
                kind, source = fact
                if not isinstance(source, int):
                    continue
                kinds_by_source.setdefault(int(source), {}).setdefault(
                    str(kind), []
                ).append(int(column))
            for source, by_kind in sorted(kinds_by_source.items()):
                if len(by_kind) < 2:
                    continue
                function_name, value_id = (
                    (row[0], row[1]) if isinstance(row, tuple) and len(row) >= 2
                    else (str(row), -1)
                )
                resolution = book.page(
                    "argument_binding_resolution"
                ).latest((str(function_name), int(value_id), int(source)))
                if (
                    isinstance(resolution, tuple)
                    and resolution
                    and str(resolution[0]) in by_kind
                ):
                    # The raw history remains intact, but the shared
                    # concordance has selected the one kind capable of
                    # materializing this exact source slot.
                    continue
                found.append(Finding(
                    "binding-kind-disagreement",
                    str(function_name),
                    int(value_id),
                    f"caller id {id_label(int(source))} is bound as "
                    + "; ".join(
                        f"{kind!r} at callsite(s) {sorted(columns)}"
                        for kind, columns in sorted(by_kind.items())
                    )
                    + " -- one slot, and only some of those kinds can "
                      "materialize it",
                ))
        return found

    @staticmethod
    def _sequence_descriptor_findings(module: Any) -> list[Finding]:
        """Descriptors whose own two records contradict each other.

        A descriptor states which of its columns are keys AND what each
        column holds.  When it says column 0 is a key and also says that
        column is float64, those are two of its own claims disagreeing --
        exactly what this table exists to catch, and catchable without
        knowing which pass wrote it.

        It matters because a key is not merely imprecise as a float: a
        string key lowers to an fnv1a-**64** token and float64 carries only
        53 bits exactly, so a token above 2**53 is silently rounded and then
        never matches its own lookup.  The entry goes missing rather than
        failing loudly, which is the worst available outcome.
        """
        integral = {
            "int", "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64", "bool",
        }
        found: list[Finding] = []
        for function_name, table in (
            getattr(module, "sequence_tables", {}) or {}
        ).items():
            for sequence_id, descriptor in sorted(
                getattr(table, "sequences", {}).items()
            ):
                dtypes = tuple(getattr(descriptor, "column_dtypes", ()) or ())
                for column in getattr(descriptor, "key_columns", ()) or ():
                    if int(column) >= len(dtypes):
                        continue
                    dtype = str(dtypes[int(column)])
                    if dtype in integral:
                        continue
                    found.append(Finding(
                        "key-column-not-integral",
                        str(function_name),
                        int(sequence_id),
                        f"column {int(column)} is declared a key but holds "
                        f"{dtype!r}; a key column is an index and cannot be "
                        f"a float (dtypes={dtypes})",
                    ))
        return found

    def _identity_claims(self, row: ValueRow) -> list[Claim]:
        return [
            claim for claim in row.claims
            if claim.kind in {
                "abi-field", "abi-parameter", "keyed-part", "parameter",
                "storage-formal", "closure-formal", "member-formal",
                "frame-storage", "compiler-frame-storage", "projected-row",
            }
        ]

    @staticmethod
    def _is_authored(function: Any) -> bool:
        """Source-level function (not a planned region or generated helper).

        Regions and sequence/tensor helpers receive their formals by the
        planner's own slot convention and never carry ``parameter_names``;
        judging them by the authored-function records would only report
        the convention itself.
        """

        metadata = dict(getattr(function, "metadata", {}) or {})
        return bool(
            metadata.get("parameter_names")
            or metadata.get("authored_parameters")
        ) and not metadata.get("source_region_integral")

    def _function_findings(self, module: Any, name: str,
                           function: Any) -> list[Finding]:
        found: list[Finding] = []
        metadata = dict(getattr(function, "metadata", {}) or {})
        authored = self._is_authored(function)
        formals = {int(formal.id): formal for formal in function.args}
        rows = {
            value_id: row for (function_name, value_id), row in self.rows.items()
            if function_name == name
        }
        # multiple-definition
        for value_id, row in rows.items():
            if len(row.definitions) <= 1:
                continue
            formal = formals.get(value_id)
            accounting = dict(getattr(formal, "accounting", {}) or {})
            if formal is not None and all(
                accounting.get(key) for key in _INOUT_REDEFINED
            ):
                continue
            found.append(Finding(
                "multiple-definition", name, value_id,
                f"defined {len(row.definitions)} times: "
                f"{row.definitions[:4]!r}",
            ))
        # formal accountability and conflicts (authored functions only)
        for value_id, formal in formals.items():
            row = rows[value_id]
            identity = self._identity_claims(row)
            if not identity and not authored:
                continue
            if not identity:
                found.append(Finding(
                    "unaccounted-formal", name, value_id,
                    f"dtype={formal.dtype} accounting_keys="
                    f"{sorted(dict(formal.accounting or {}))[:6]!r}",
                ))
                continue
            abi_fields = {c.key for c in identity if c.kind == "abi-field"}
            frame = [c for c in identity if c.kind == "frame-storage"]
            accounting = dict(formal.accounting or {})
            if accounting.get("returned_record_storage") is not None:
                # Storage leased for a returned record's field carries that
                # field's identity by design; leased + field is the
                # convention here, not a conflict.
                frame = []
            if len(abi_fields) > 1 or (abi_fields and frame):
                found.append(Finding(
                    "conflicting-storage-claims", name, value_id,
                    f"claims={[(c.kind, c.key) for c in identity]!r}",
                ))
        # sequence descriptor members (authored functions only: a generated
        # helper's descriptor names the caller's cells it is handed)
        for (function_name, value_id), roles in self.sequence_roles.items():
            if function_name != name or not authored:
                continue
            row = rows.get(value_id)
            if row is None or (not row.is_formal and not row.definitions):
                found.append(Finding(
                    "descriptor-member-unknown", name, value_id,
                    f"roles={roles!r}: not a formal, never defined",
                ))
            distinct = {(sequence_id, role.rstrip("0123456789"))
                        for sequence_id, role in roles}
            sequences = {sequence_id for sequence_id, _role in distinct}
            if len(sequences) > 1:
                found.append(Finding(
                    "descriptor-member-shared", name, value_id,
                    f"roles={roles!r}",
                ))
        # keyed helper operands
        described = {
            value_id for (function_name, value_id) in self.sequence_roles
            if function_name == name
        }
        keyed_parts = {
            value_id for value_id, row in rows.items()
            if any(c.kind == "keyed-part" for c in row.claims)
        }
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                attributes = instruction.attributes or {}
                if attributes.get("keyed_lookup_owner") is None:
                    continue
                for position, argument in enumerate(instruction.args[:4]):
                    argument_id = int(argument.id)
                    if argument_id in described or argument_id in keyed_parts:
                        continue
                    found.append(Finding(
                        "helper-operand-outside-descriptor", name, argument_id,
                        f"{block_name}#{index} {attributes.get('callee')} "
                        f"operand {position} (owner "
                        f"{attributes.get('keyed_lookup_owner')!r})",
                    ))
        # duplicate storage across a call
        callee_identity: dict[str, dict[int, str]] = {}
        for callee_name, callee in module.functions.items():
            callee_identity[str(callee_name)] = {
                int(formal.id): key
                for formal in callee.args
                for key in (self._abi_or_member_key(str(callee_name), formal),)
                if key is not None
            }
        caller_by_key: dict[str, list[int]] = defaultdict(list)
        for value_id, formal in formals.items():
            key = self._abi_or_member_key(name, formal)
            if key is not None:
                caller_by_key[key].append(value_id)
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.op not in {"Call", "call"}:
                    continue
                attributes = instruction.attributes or {}
                callee_name = str(attributes.get("callee") or "")
                declared = attributes.get("callee_input_ids")
                if callee_name not in callee_identity or declared is None:
                    continue
                for callee_id, argument in zip(declared, instruction.args):
                    key = callee_identity[callee_name].get(int(callee_id))
                    if key is None:
                        continue
                    argument_id = int(argument.id)
                    accounting = dict(getattr(argument, "accounting", {}) or {})
                    if not accounting.get("linked_call_frame_storage"):
                        continue
                    owned = [
                        owner for owner in caller_by_key.get(key, ())
                        if owner != argument_id
                    ]
                    if owned:
                        found.append(Finding(
                            "duplicate-storage-across-call", name, argument_id,
                            f"{block_name}#{index} -> {callee_name} formal "
                            f"{callee_id} ({key}) fed leased storage "
                            f"{argument_id} while caller owns {owned!r}",
                        ))
        # dominance
        found.extend(self._dominance_findings(name, function, formals))
        # alias targets
        for value_id, row in rows.items():
            for claim in row.claims:
                if claim.kind != "alias-of":
                    continue
                target = rows.get(int(claim.key))
                if target is None or (
                    not target.is_formal and not target.definitions
                ):
                    found.append(Finding(
                        "alias-target-missing", name, value_id,
                        f"alias of {claim.key} which is never defined",
                    ))
        # A private alias snapshot is allowed only as a durable copy of the
        # shared authority. This catches the exact class of failure where a
        # late pass proves an identity in ``metadata.value_aliases`` but the
        # next pass reads only ``planning_value_concordance`` (or vice versa).
        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "planning_value_concordance"
            )
        )
        if book is not None:
            durable_aliases: list[tuple[str, int, int]] = []
            local_aliases = metadata.get("value_aliases", ()) or ()
            local_pairs = (
                local_aliases.items()
                if isinstance(local_aliases, Mapping) else local_aliases
            )
            durable_aliases.extend(
                ("metadata.value_aliases", int(alias), int(target))
                for alias, target in local_pairs
            )
            durable_aliases.extend(
                (
                    "metadata.output_identity_aliases",
                    int(alias), int(target),
                )
                for alias, target in (
                    metadata.get("output_identity_aliases", ()) or ()
                )
            )
            for source, alias, target in durable_aliases:
                concorded = (
                    None if page is None else page.latest((name, alias))
                )
                if concorded is not None and int(concorded) == target:
                    continue
                found.append(Finding(
                    "alias-not-concorded", name, alias,
                    f"{source} says {target}, planning_value_concordance "
                    f"says {concorded!r}",
                ))
        return found

    def _abi_or_member_key(self, function: str, formal: Any) -> str | None:
        accounting = dict(getattr(formal, "accounting", {}) or {})
        abi_field = accounting.get("program_abi_field")
        if abi_field is not None:
            # Identity is per record INSTANCE: the parameter that carries
            # the record, or the call whose result record this storage
            # holds.  Two instances of one record type in one function
            # (``metrics`` and ``coerced = coerce_metrics(metrics)``)
            # legitimately own the same field name each.
            instance = accounting.get("returned_record_storage")
            if instance is not None:
                instance = f"{instance}@{accounting.get('callsite_id')}"
            else:
                instance = accounting.get("program_abi_parameter") or accounting.get("program_abi_record")
            return f"{instance}.{abi_field}"
        roles = self.sequence_roles.get((function, int(formal.id)))
        if roles:
            sequence_id, role = roles[0]
            return f"seq{sequence_id}.{role}"
        return None

    @staticmethod
    def _dominators(function: Any) -> dict[str, set[str]]:
        blocks = list(function.blocks)
        if not blocks:
            return {}
        entry = blocks[0]
        predecessors: dict[str, set[str]] = {b: set() for b in blocks}
        for block_name, block in function.blocks.items():
            for successor in block.successors:
                if successor in predecessors:
                    predecessors[successor].add(block_name)
        reachable = {entry}
        stack = [entry]
        while stack:
            current = stack.pop()
            for successor in function.blocks[current].successors:
                if successor in predecessors and successor not in reachable:
                    reachable.add(successor)
                    stack.append(successor)
        dominators = {b: set(reachable) for b in reachable}
        dominators[entry] = {entry}
        changed = True
        while changed:
            changed = False
            for block_name in reachable:
                if block_name == entry:
                    continue
                incoming = [
                    dominators[p] for p in predecessors[block_name]
                    if p in reachable
                ]
                updated = ({block_name} | set.intersection(*incoming)
                           if incoming else {block_name})
                if updated != dominators[block_name]:
                    dominators[block_name] = updated
                    changed = True
        return dominators

    def _dominance_findings(self, name: str, function: Any,
                            formals: Mapping[int, Any]) -> list[Finding]:
        found: list[Finding] = []
        dominators = self._dominators(function)
        if not dominators:
            return found
        definition_sites: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is not None:
                    definition_sites[int(instruction.res.id)].append(
                        (block_name, index)
                    )
        for block_name, block in function.blocks.items():
            if block_name not in dominators:
                continue
            for index, instruction in enumerate(block.instrs):
                if instruction.op in {"Phi", "phi"}:
                    continue
                for argument in instruction.args:
                    argument_id = getattr(argument, "id", None)
                    if argument_id is None or int(argument_id) in formals:
                        continue
                    sites = definition_sites.get(int(argument_id))
                    if not sites:
                        continue
                    if any(
                        (site_block == block_name and site_index <= index)
                        or (site_block != block_name
                            and site_block in dominators[block_name])
                        for site_block, site_index in sites
                    ):
                        # ``site_index == index``: a region call names its
                        # own result among its operands (the out-pointer
                        # convention), which is a definition, not a read.
                        continue
                    found.append(Finding(
                        "use-not-dominated", name, int(argument_id),
                        f"{block_name}#{index} {instruction.op} reads a "
                        f"value defined at {sites[:3]!r}",
                    ))
        return found


def concordance_report(module: Any, *, limit: int = 12) -> str:
    """Build the table and render its findings as text."""

    table = CorrelationTable.build(module)
    findings = table.findings(module)
    by_kind: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        by_kind[finding.kind].append(finding)
    census = group_by_prefix(value_id for _function, value_id in table.rows)
    lines = [
        f"identity concordance: {len(table.rows)} rows across "
        f"{len(module.functions)} functions, {len(findings)} finding(s)"
    ]
    # Every row gathered under its own id group, before any finding: a
    # count per prefix says at a glance which spaces this module actually
    # uses, and a group that should be empty (``history`` ids among a
    # function's own values, say) shows up as a number rather than having
    # to be hunted for.
    if len(census) > 1 or (census and census[0].label != "legacy"):
        lines.append(
            "  id groups: "
            + ", ".join(
                f"{group.label}={len(group.value_ids)}" for group in census
            )
        )
    for kind in sorted(by_kind):
        entries = by_kind[kind]
        lines.append(f"  [{kind}] x{len(entries)}")
        for finding in entries[:limit]:
            named = (
                "?" if finding.value_id is None
                else id_label(finding.value_id)
            )
            lines.append(
                f"     {finding.function} value {named}: "
                f"{finding.detail}"
            )
        if len(entries) > limit:
            lines.append(f"     ... {len(entries) - limit} more")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Identity pages: row x column x page, for facts a finished-module table
# cannot see.
#
# ``CorrelationTable`` above audits one finished module -- it has no notion
# of time, so it can only ever compare a value against itself, never against
# what it USED to be.  Two 2026-09-17/18 defects were exactly that: a
# correct fact computed once but never carried to the one consumer that
# needed it (an alias map read in two places, not the third that mattered),
# and a value's shape flipping A -> B -> A within a single fixed-point round
# while every step honestly reported "changed" -- a round-boundary snapshot
# necessarily reads that as no change at all, because it is none, net.
#
# A page is one pipeline stage's table.  A row is one identity -- whatever a
# page decides makes two facts "about the same thing" (a value id, an
# (function, value id) pair, ...).  A column is one round -- whatever
# "round" means on that page (a fixed-point iteration, a phase index).  A
# cell is the fact that identity held at that round.  Reading one row across
# its own columns finds an in-stage oscillation.  Reading one row's key
# across two different pages finds a cross-stage disagreement -- the shape
# of every fault above and, going forward, the general instrument for both.
# --------------------------------------------------------------------------


def _blocks_between(function: Any, header: str, latch: str) -> set[str]:
    """Blocks on some path from the declared header to the declared latch."""

    successors: dict[str, set[str]] = {}
    for block_name, block in function.blocks.items():
        targets: set[str] = set()
        for instruction in block.instrs:
            for key in ("target", "true_target", "false_target"):
                declared = instruction.attributes.get(key)
                if declared is not None:
                    targets.add(str(declared))
        successors[str(block_name)] = targets
    forward: set[str] = set()
    frontier = [str(header)]
    while frontier:
        current = frontier.pop()
        if current in forward:
            continue
        forward.add(current)
        frontier.extend(successors.get(current, ()))
    backward: set[str] = set()
    frontier = [str(latch)]
    while frontier:
        current = frontier.pop()
        if current in backward:
            continue
        backward.add(current)
        for candidate, onward in successors.items():
            if current in onward:
                frontier.append(candidate)
    return forward & backward


def _carried_phi(function: Any, header: str, carried_id: int) -> Any:
    """The header Phi that speaks for one declared carried generation."""

    block = function.blocks.get(str(header))
    for instruction in (block.instrs if block is not None else ()):
        if str(instruction.op).lower() != "phi":
            continue
        if instruction.res is None:
            continue
        if int(instruction.res.id) == int(carried_id):
            return instruction
    return None


@dataclass
class IdentityPage:
    """One pipeline stage's row (identity) x column (round) table of facts."""

    name: str
    cells: dict[tuple[Any, int], Any] = field(default_factory=dict)
    columns: list[int] = field(default_factory=list)

    def set(self, row: Any, column: int, fact: Any) -> None:
        if column not in self.columns:
            self.columns.append(column)
        self.cells[(row, column)] = fact

    def latest(self, row: Any, default: Any = None) -> Any:
        """Return the most recently recorded fact for ``row``."""
        entries = self.history(row)
        return entries[-1][1] if entries else default

    def bind_alias(self, scope: Any, alias: int, resident: int) -> None:
        """Concord one planning value occurrence with its resident identity.

        Rebinding a row appends a new column so the page remains both the live
        planning authority and the history of every decision it supplied.
        """
        row = (scope, int(alias))
        entries = self.history(row)
        column = entries[-1][0] + 1 if entries else 0
        self.set(row, column, int(resident))

    def alias_bindings(self, scope: Any) -> dict[int, int]:
        """Materialize the latest alias facts owned by one planning scope."""
        return {
            int(row[1]): int(self.latest(row))
            for row in self.rows()
            if (
                isinstance(row, tuple)
                and len(row) == 2
                and row[0] == scope
            )
        }

    def resolve_alias(self, scope: Any, value_id: int) -> int:
        """Resolve one value through this page's current planning facts."""
        current = int(value_id)
        path: list[int] = []
        while True:
            target = self.latest((scope, current))
            if target is None or int(target) == current:
                return current
            if current in path:
                raise ValueError(
                    f"cyclic planning identity concordance for {scope!r}: "
                    f"{tuple((*path, current))}"
                )
            path.append(current)
            current = int(target)

    def rows(self) -> tuple[Any, ...]:
        return tuple(dict.fromkeys(row for row, _ in self.cells))

    def history(self, row: Any) -> tuple[tuple[int, Any], ...]:
        """This row's fact at every column it was recorded on, in order."""
        return tuple(
            (column, self.cells[(row, column)])
            for column in self.columns
            if (row, column) in self.cells
        )

    def spans(self, row: Any) -> tuple[tuple[int, int, Any], ...]:
        """This row's history collapsed to contiguous (start, end, fact) runs."""
        entries = self.history(row)
        if not entries:
            return ()
        runs: list[tuple[int, int, Any]] = []
        start_column, current_fact = entries[0]
        end_column = start_column
        for column, fact in entries[1:]:
            if fact != current_fact:
                runs.append((start_column, end_column, current_fact))
                start_column, current_fact = column, fact
            end_column = column
        runs.append((start_column, end_column, current_fact))
        return tuple(runs)

    def oscillating_rows(
        self, key: Any = None, *, old_key: Any = None,
    ) -> dict[Any, tuple[tuple[int, int, Any], ...]]:
        """Rows that left a value and later came back to it -- a round-trip,
        never a settle, and the exact shape a round-boundary-only snapshot
        cannot see (the net effect across the trip is zero).

        ``key`` extracts the comparable "resulting" value from a fact
        (default: the fact itself).  For a fact that bundles its own
        transition -- ``(old, new, ...)``, as a mutation-log style page does
        -- pass ``key=lambda fact: fact[1]``.

        That alone still misses the most common real case: fact A moves a
        value from its UNRECORDED starting point to X, fact B moves it from
        X back to that same starting point.  The visited-value sequence is
        genuinely [start, X, start] -- a real round-trip -- but only X and
        start-as-B's-new ever get compared unless the implicit start is
        counted too.  Pass ``old_key`` (extracting the "before" side of the
        SAME fact shape, e.g. ``lambda fact: fact[0]``) to prepend that
        first recorded starting value to the sequence before checking.
        """
        project = key or (lambda fact: fact)
        found = {}
        for row in self.rows():
            runs = self.spans(row)
            facts = [project(fact) for _, _, fact in runs]
            if old_key is not None and runs:
                facts = [old_key(runs[0][2]), *facts]
            if len(set(facts)) < len(facts):
                found[row] = runs
        return found


class IdentityBook:
    """Every stage's page, so one identity's claim can be read across all
    of them -- the comparison none of them makes on its own."""

    def __init__(self) -> None:
        self.pages: dict[str, IdentityPage] = {}

    def page(self, name: str) -> IdentityPage:
        return self.pages.setdefault(name, IdentityPage(name))

    def latest_by_page(self, row: Any) -> dict[str, Any]:
        """The final fact recorded for `row` on each page that ever saw it."""
        result = {}
        for name, page in self.pages.items():
            runs = page.spans(row)
            if runs:
                result[name] = runs[-1][2]
        return result

    def disagreements(self, row: Any) -> dict[str, Any] | None:
        """The per-page facts for `row`, if more than one distinct fact
        exists among them -- else None (the pages agree, or only one saw it)."""
        latest = self.latest_by_page(row)
        if len({repr(fact) for fact in latest.values()}) > 1:
            return latest
        return None


@dataclass(frozen=True)
class SequenceContract:
    """The physical row contract owned by one resident sequence identity."""

    policy: str
    column_count: int
    writable: bool


def _canonical_sequence_row_dtype(dtype: Any) -> str:
    spelling = "unknown" if dtype is None else str(dtype)
    return "unknown" if spelling in {"", "None", "unknown"} else spelling


def committed_sequence_row_dtypes(
    scope: Any,
    sequence_id: int,
    *,
    page: IdentityPage | None = None,
) -> tuple[str, ...] | None:
    """Read the row dtype contract for one resident sequence identity."""

    if page is None:
        page = current_identity_book().page(
            "sequence_row_dtype_concordance"
        )
    fact = page.latest((scope, int(sequence_id)))
    if fact is None:
        return None
    return tuple(map(str, fact[0]))


def concord_sequence_row_dtypes(
    scope: Any,
    claims: Mapping[int, Iterable[Any]],
    *,
    source: str,
    page: IdentityPage | None = None,
) -> tuple[str, ...]:
    """Resolve one row layout across sequence identities proven equivalent.

    ``unknown`` is absence of a claim, not a competing dtype.  A replace or
    carried-state edge proves its two arenas have the same physical row, so a
    known column on either side refines the other.  Two different known
    dtypes are a real disagreement and compilation stops here.
    """

    if page is None:
        page = current_identity_book().page(
            "sequence_row_dtype_concordance"
        )
    normalized: dict[int, tuple[str, ...]] = {}
    for sequence_id, raw_dtypes in claims.items():
        sid = int(sequence_id)
        proposed = tuple(
            _canonical_sequence_row_dtype(dtype) for dtype in raw_dtypes
        )
        incumbent = committed_sequence_row_dtypes(
            scope, sid, page=page
        )
        if incumbent is not None and len(incumbent) != len(proposed):
            raise ValueError(
                "sequence row dtype concordance width disagreement for "
                f"{scope!r} value {sid}: recorded={incumbent!r}, "
                f"{source} says {proposed!r}"
            )
        normalized[sid] = proposed if incumbent is None else tuple(
            recorded if proposed_dtype == "unknown" else proposed_dtype
            if recorded == "unknown" else recorded
            for recorded, proposed_dtype in zip(incumbent, proposed)
        )
        if incumbent is not None:
            for recorded, proposed_dtype in zip(incumbent, proposed):
                if (
                    recorded != "unknown"
                    and proposed_dtype != "unknown"
                    and recorded != proposed_dtype
                ):
                    raise ValueError(
                        "sequence row dtype concordance disagreement for "
                        f"{scope!r} value {sid}: recorded={incumbent!r}, "
                        f"{source} says {proposed!r}"
                    )
    widths = {len(dtypes) for dtypes in normalized.values()}
    if len(widths) > 1:
        raise ValueError(
            "sequence row dtype concordance cannot equate different row "
            f"widths for {scope!r} at {source}: {normalized!r}"
        )
    width = next(iter(widths), 0)
    resolved: list[str] = []
    for column in range(width):
        known = {
            dtypes[column] for dtypes in normalized.values()
            if dtypes[column] != "unknown"
        }
        if len(known) > 1:
            raise ValueError(
                "sequence row dtype concordance disagreement for "
                f"{scope!r} column {column} at {source}: {normalized!r}"
            )
        resolved.append(next(iter(known), "unknown"))
    result = tuple(resolved)
    for sequence_id in normalized:
        row = (scope, int(sequence_id))
        history = page.history(row)
        column = history[-1][0] + 1 if history else 0
        page.set(row, column, (result, str(source)))
    return result


def committed_sequence_contract(
    scope: Any,
    sequence_id: int,
    *,
    page: IdentityPage | None = None,
) -> SequenceContract | None:
    """Read the sequence contract already committed for this exact identity."""

    if page is None:
        page = current_identity_book().page("sequence_contract_concordance")
    fact = page.latest((scope, int(sequence_id)))
    if fact is None:
        return None
    return SequenceContract(
        policy=str(fact[0]),
        column_count=int(fact[1]),
        writable=bool(fact[2]),
    )


def commit_sequence_contract(
    scope: Any,
    sequence_id: int,
    policy: str,
    column_count: int,
    writable: bool,
    *,
    source: str,
    page: IdentityPage | None = None,
) -> SequenceContract:
    """Commit or verify one resident sequence's physical row contract.

    The first source-stage fact owns policy and row width. Later compiler
    stages may repeat that contract and may prove the same storage writable,
    but they may not silently replace its policy or width. Every accepted
    statement is retained in the page history together with its source stage.
    """

    if page is None:
        page = current_identity_book().page("sequence_contract_concordance")
    row = (scope, int(sequence_id))
    proposed = SequenceContract(
        policy=str(policy),
        column_count=int(column_count),
        writable=bool(writable),
    )
    if proposed.column_count < 1:
        raise ValueError(
            f"sequence contract for {scope!r} value {sequence_id} declares "
            f"invalid column count {proposed.column_count} at {source}"
        )
    incumbent = committed_sequence_contract(scope, sequence_id, page=page)
    if incumbent is not None and (
        incumbent.policy != proposed.policy
        or incumbent.column_count != proposed.column_count
    ):
        prior = page.latest(row)
        raise ValueError(
            f"sequence contract concordance disagreement for {scope!r} "
            f"value {sequence_id}: {prior[3]} committed "
            f"{incumbent.policy}/{incumbent.column_count}, {source} says "
            f"{proposed.policy}/{proposed.column_count}"
        )
    resolved = SequenceContract(
        policy=proposed.policy,
        column_count=proposed.column_count,
        writable=bool(proposed.writable or (
            incumbent.writable if incumbent is not None else False
        )),
    )
    history = page.history(row)
    column = history[-1][0] + 1 if history else 0
    page.set(row, column, (
        resolved.policy,
        resolved.column_count,
        resolved.writable,
        str(source),
    ))
    return resolved


# One book per top-level compile, reachable from anywhere in the call stack
# without a `module` argument -- a contextvar rather than module.metadata,
# because the module a mid-pipeline pass is building is not always the same
# object the top-level entry point will eventually return, and a crash deep
# inside one stage (exactly tonight's fault) must not lose everything
# recorded before it.  `lower_ast_source_to_ssa` opens one with
# `begin_identity_book()` and dumps it in a `finally` with
# `end_identity_book()`, success or failure, at the one small, honest cost
# of the book itself: it is only ever appended to as a side effect of work
# the pass was already doing.
_ACTIVE_IDENTITY_BOOK: contextvars.ContextVar[IdentityBook | None] = (
    contextvars.ContextVar("identity_concordance_active_book", default=None)
)


def begin_identity_book() -> tuple[IdentityBook, contextvars.Token]:
    """Start a fresh book for one compile and make it current.

    Returns the book and a reset token.  A nested compile (one
    ``lower_ast_source_to_ssa`` invoked while another is already on the
    stack) must not clobber the outer compile's book to ``None`` when it
    finishes -- that would silently orphan everything the outer compile
    recorded before the nested one started.  The token lets ``end_identity_
    book`` restore exactly the PREVIOUS value instead of blanking it.
    """
    book = IdentityBook()
    token = _ACTIVE_IDENTITY_BOOK.set(book)
    return book, token


def current_identity_book() -> IdentityBook:
    """The active compile's book, creating a detached one if none is open
    (so a page write is never a hard error just because nothing called
    ``begin_identity_book`` -- it simply has nowhere to be dumped later)."""
    book = _ACTIVE_IDENTITY_BOOK.get()
    if book is None:
        book = IdentityBook()
        _ACTIVE_IDENTITY_BOOK.set(book)
    return book


def concordant_alias_bindings(
    scope: Any,
    *ledgers: Mapping[int, int] | Iterable[tuple[int, int]],
    page: IdentityPage | None = None,
) -> dict[int, int]:
    """Return one checked alias ledger for a planning scope.

    ``planning_value_concordance`` is the shared identity authority.  Some
    finished SSA functions also retain a local ``value_aliases`` snapshot or
    an ``output_identity_aliases`` receipt because backends need the facts
    after the active compile context has closed.  A consumer must not choose
    one of those records and silently ignore the others: combine them here,
    and refuse any disagreement about the same alias.

    The page is read last only to make the authority explicit; agreement is
    required, so insertion order never decides an identity.
    """

    if page is None:
        page = current_identity_book().page("planning_value_concordance")
    sources: list[tuple[str, Iterable[tuple[int, int]]]] = []
    for index, ledger in enumerate(ledgers):
        pairs = ledger.items() if isinstance(ledger, Mapping) else ledger
        sources.append((f"ledger[{index}]", pairs))
    sources.append((page.name, page.alias_bindings(scope).items()))

    result: dict[int, int] = {}
    owners: dict[int, str] = {}
    for source, pairs in sources:
        for alias, resident in pairs:
            alias = int(alias)
            resident = int(resident)
            incumbent = result.get(alias)
            if incumbent is not None and incumbent != resident:
                raise ValueError(
                    f"identity concordance disagreement for {scope!r} "
                    f"value {alias}: {owners[alias]} says {incumbent}, "
                    f"{source} says {resident}"
                )
            result[alias] = resident
            owners.setdefault(alias, source)
    return result


def resolved_concordant_alias_bindings(
    scope: Any,
    *ledgers: Mapping[int, int] | Iterable[tuple[int, int]],
    page: IdentityPage | None = None,
) -> dict[int, int]:
    """Merge exact alias receipts and resolve every transitive chain.

    Each input ledger may own a different segment of one identity path. A
    consumer must not stop at the boundary between pages: ``a -> b`` on the
    planning page and ``b -> resident`` on the control page are one proven
    identity. Cycles remain an error and name the complete path.
    """

    sources: list[tuple[str, Iterable[tuple[int, int]]]] = []
    for index, ledger in enumerate(ledgers):
        pairs = ledger.items() if isinstance(ledger, Mapping) else ledger
        sources.append((f"ledger[{index}]", pairs))
    if page is None:
        page = current_identity_book().page("planning_value_concordance")
    sources.append((page.name, page.alias_bindings(scope).items()))

    edges: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for source, pairs in sources:
        for alias, target in pairs:
            edge = (int(target), str(source))
            if edge not in edges[int(alias)]:
                edges[int(alias)].append(edge)

    memo: dict[int, int] = {}

    def resolve(value_id: int, path: tuple[int, ...] = ()) -> int:
        value_id = int(value_id)
        if value_id in memo:
            return memo[value_id]
        if value_id in path:
            raise ValueError(
                f"cyclic planning identity concordance for {scope!r}: "
                f"{(*path, value_id)}"
            )
        targets = tuple(
            target for target, _source in edges.get(value_id, ())
            if int(target) != value_id
        )
        if not targets:
            memo[value_id] = value_id
            return value_id
        roots = {
            resolve(target, (*path, value_id)) for target in targets
        }
        if len(roots) != 1:
            claims = tuple(edges.get(value_id, ()))
            raise ValueError(
                f"identity concordance disagreement for {scope!r} value "
                f"{value_id}: claims={claims!r}, terminal residents="
                f"{tuple(sorted(roots))!r}"
            )
        root = next(iter(roots))
        memo[value_id] = root
        return root

    return {alias: resolve(alias) for alias in edges}


def end_identity_book(
    token: contextvars.Token | None = None,
) -> IdentityBook | None:
    """Detach and return the book that was active (or None if none was
    open), restoring whatever was active before it via `token` when given
    (see `begin_identity_book`) instead of unconditionally clearing to None.
    """
    book = _ACTIVE_IDENTITY_BOOK.get()
    if token is not None:
        _ACTIVE_IDENTITY_BOOK.reset(token)
    else:
        _ACTIVE_IDENTITY_BOOK.set(None)
    return book


def identity_book(module: Any) -> IdentityBook:
    """The current compile's book, also cached on ``module.metadata`` when
    available -- so code with a module in hand (an already-finished
    ``IRModule``, inspected after the fact) and code with only the ambient
    compile context (a pass mid-construction, or an exception handler with
    no module at all) read the exact same instance."""
    metadata = getattr(module, "metadata", None)
    # The compile closes its book in a ``finally`` before returning, so after
    # that point ``current_identity_book()`` mints a fresh EMPTY one.  A
    # caller holding the finished module would then read an empty book and
    # conclude the compile recorded nothing -- which is the opposite of what
    # this function promises.  The module's own attached book wins whenever
    # there is one.
    attached = None if metadata is None else metadata.get("identity_book")
    if attached is not None:
        return attached
    book = current_identity_book()
    if metadata is not None:
        metadata["identity_book"] = book
    return book


def authored_function_name(name: Any) -> str:
    """The authored name behind a lowered symbol.

    Stores are keyed by the function as written -- ``solve`` -- while a
    lowered symbol carries the module prefix, the callsite specialization
    hash and the region index.  Without stripping those, rows never line up
    and every value appears to have exactly one source.
    """

    text = str(name)
    for separator in ("__specialized_", "__planned_region"):
        if separator in text:
            text = text.split(separator)[0]
    # Split at the FIRST separator: the artifact prefix is at the front, and
    # an authored name that itself begins with an underscore makes the last
    # separator fall inside ``___``, which would eat that underscore.
    return text.split("__", 1)[-1] if "__" in text else text


OUTER, CARRIED, INNER = 0, 1, 2

_GENERATION_NAMES = {OUTER: "outer", CARRIED: "carried", INNER: "inner"}


def declare_loop_scope(
    function: Any, loop_node_id: Any, header: str, latch: str,
    exit_block: str, rebinds: Any,
) -> None:
    """Declare one loop as a scope whose bindings evolve per iteration.

    A nested scope binds a name once for its whole extent.  A loop body
    binds it differently on every entry, so the page's COLUMN is the
    generation -- outer, carried, inner -- rather than a round.  The
    boundary is recorded with the rebinds because a transformation that
    moves code across it must be able to ask where it is, instead of
    recovering it from block names or branch topology that the
    transformation itself may have rewritten.
    """

    page = current_identity_book().page("loop_scope")
    scope = (authored_function_name(function), int(loop_node_id))
    page.set(
        (*scope, "boundary"), 0,
        (str(header), str(latch), str(exit_block)),
    )
    for outer_id, carried_id, inner_id, graph_outer, graph_inner in rebinds:
        row = (*scope, int(outer_id))
        page.set(row, OUTER, int(outer_id))
        page.set(row, CARRIED, int(carried_id))
        page.set(row, INNER, int(inner_id))
        page.set(row, INNER + 1, ("graph", int(graph_outer), int(graph_inner)))


def rebind_loop_scope_inner(
    function: Any,
    loop_node_id: Any,
    declared_inner: int,
    resident_inner: int,
    reason: Any,
) -> None:
    """Register a transformation of the value crossing a loop backedge.

    The scope declaration preserves the authored outer/carried/inner
    generations.  Outlining or aggregate projection can subsequently mint a
    resident SSA value for that same inner generation.  Record that transition
    separately so the original declaration remains historical evidence while
    every later consumer sees the resident identity.
    """

    page = current_identity_book().page("loop_scope_inner_transition")
    row = (
        authored_function_name(function), int(loop_node_id),
        int(declared_inner),
    )
    history = page.history(row)
    column = history[-1][0] + 1 if history else 0
    page.set(row, column, (int(resident_inner), str(reason)))


def loop_scope_declarations(book: Any, function: Any) -> list[dict]:
    """Every loop scope declared for one authored function."""

    page = book.page("loop_scope")
    wanted = authored_function_name(function)
    scopes: dict[int, dict] = {}
    for row in page.rows():
        if not (isinstance(row, tuple) and len(row) == 3):
            continue
        name, loop_node_id, key = row
        if name != wanted:
            continue
        record = scopes.setdefault(
            int(loop_node_id),
            {
                "loop_node_id": int(loop_node_id),
                "boundary": None,
                "rebinds": [],
            },
        )
        if key == "boundary":
            record["boundary"] = page.latest(row)
            continue
        generations = dict(page.history(row))
        if (
            OUTER in generations
            and CARRIED in generations
            and INNER in generations
        ):
            declared_inner = int(generations[INNER])
            transition = book.page("loop_scope_inner_transition").latest((
                wanted, int(loop_node_id), declared_inner,
            ))
            resident_inner = (
                int(transition[0])
                if isinstance(transition, tuple) and transition
                else declared_inner
            )
            record["rebinds"].append({
                "outer": int(generations[OUTER]),
                "carried": int(generations[CARRIED]),
                "inner": resident_inner,
                "declared_inner": declared_inner,
            })
    return [record for record in scopes.values() if record["boundary"]]


def concord_loop_scope_latch_residents(module: Any) -> tuple[dict, ...]:
    """Publish the final resident identity on every transformed backedge.

    Aggregate legalization and linked-call projection happen after control
    lowering declared the authored inner generation.  At the completed-module
    seam the loop Phi is the exact authority on what now crosses the latch.
    Reconcile that resident into the concordance so the declaration tracks the
    transformation instead of leaving a locally recorded but globally unknown
    rename.
    """

    book = identity_book(module)
    receipts: list[dict] = []
    for function_name, function in getattr(module, "functions", {}).items():
        for declaration in loop_scope_declarations(book, function_name):
            header, latch, _exit = declaration["boundary"]
            loop_node_id = int(declaration["loop_node_id"])
            for rebind in declaration["rebinds"]:
                carried = int(rebind["carried"])
                declared_inner = int(rebind.get(
                    "declared_inner", rebind["inner"]
                ))
                phi = _carried_phi(function, str(header), carried)
                if phi is None:
                    continue
                incoming = tuple(phi.attributes.get("incoming_blocks") or ())
                resident = next((
                    int(argument.id)
                    for predecessor, argument in zip(incoming, phi.args)
                    if str(predecessor) == str(latch)
                    and getattr(argument, "id", None) is not None
                ), None)
                if resident is None or resident == int(rebind["inner"]):
                    continue
                rebind_loop_scope_inner(
                    function_name,
                    loop_node_id,
                    declared_inner,
                    resident,
                    "completed_module_latch_projection",
                )
                receipt = {
                    "function": str(function_name),
                    "loop_node_id": loop_node_id,
                    "carried": carried,
                    "declared_inner": declared_inner,
                    "resident_inner": resident,
                    "reason": "completed_module_latch_projection",
                }
                receipts.append(receipt)
                argument = next((
                    argument
                    for predecessor, argument in zip(incoming, phi.args)
                    if str(predecessor) == str(latch)
                    and int(argument.id) == resident
                ), None)
                if argument is not None:
                    argument.accounting = {
                        **dict(argument.accounting or {}),
                        "loop_scope_inner_transition": (
                            declared_inner, resident,
                            "completed_module_latch_projection",
                        ),
                    }
    if receipts:
        metadata = getattr(module, "metadata", None)
        if metadata is not None:
            metadata["loop_scope_inner_reconciliations"] = tuple(receipts)
    return tuple(receipts)


def concord_compiler_frame_formals(module: Any) -> tuple[dict, ...]:
    """Account for hidden formals proved to be compiler-owned at every call.

    A structural value such as a tensor dtype can survive specialization as a
    scalar formal even when it is not an authored parameter.  If every exact
    incoming call position supplies linked frame storage, the concordance can
    classify that formal without guessing from its numerical dtype or use.
    """

    functions = getattr(module, "functions", {}) or {}
    incoming: dict[tuple[str, int], list[tuple[str, Any]]] = defaultdict(list)
    for caller_name, caller in functions.items():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Call", "call"}:
                    continue
                callee_name = str(instruction.attributes.get("callee") or "")
                callee = functions.get(callee_name)
                if callee is None or len(instruction.args) != len(callee.args):
                    continue
                for formal, actual in zip(callee.args, instruction.args):
                    incoming[(callee_name, int(formal.id))].append((
                        str(caller_name), actual,
                    ))

    receipts: list[dict] = []
    book = identity_book(module)
    for function_name, function in functions.items():
        metadata = function.metadata
        named = {
            int(value_id)
            for _name, value_id in metadata.get("parameter_names", ())
        }
        recorded = {
            int(item["value_id"])
            for key in ("storage_formals", "closure_formals", "member_formals")
            for item in (metadata.get(key, ()) or ())
            if isinstance(item, Mapping) and item.get("value_id") is not None
        }
        for formal in function.args:
            formal_id = int(formal.id)
            accounting = dict(formal.accounting or {})
            if (
                formal_id in named
                or formal_id in recorded
                or any(accounting.get(key) not in {None, ""} for key in (
                    "program_abi_storage", "program_abi_parameter",
                    "program_abi_field", "linked_call_frame_storage",
                    "compiler_frame_storage", "returned_record_storage",
                ))
            ):
                continue
            sources = incoming.get((str(function_name), formal_id), ())
            if not sources or not all(
                (actual.accounting or {}).get("linked_call_frame_storage")
                or (actual.accounting or {}).get("compiler_frame_storage")
                for _caller, actual in sources
            ):
                continue
            source_receipts = tuple(
                (caller, int(actual.id)) for caller, actual in sources
            )
            formal.accounting = {
                **accounting,
                "compiler_frame_storage": str(function_name),
                "compiler_frame_sources": source_receipts,
            }
            storage_entry = {
                "value_id": formal_id,
                "dtype": str(formal.dtype or "unknown"),
                "shape": tuple(formal.shape or ()),
                "kind": "compiler_frame_storage",
                "sources": source_receipts,
            }
            prior = tuple(metadata.get("storage_formals", ()) or ())
            metadata["storage_formals"] = (
                *prior,
                *( () if storage_entry in prior else (storage_entry,) ),
            )
            receipt = {
                "function": str(function_name),
                "formal_id": formal_id,
                "sources": source_receipts,
                "kind": "compiler_frame_storage",
            }
            receipts.append(receipt)
            page = book.page("formal_storage_resolution")
            row = (str(function_name), formal_id)
            history = page.history(row)
            column = history[-1][0] + 1 if history else 0
            page.set(row, column, (
                "compiler_frame_storage", source_receipts,
            ))
    if receipts:
        getattr(module, "metadata", {})[
            "compiler_frame_formal_reconciliations"
        ] = tuple(receipts)
    return tuple(receipts)


def materializing_binding_kind(
    book: Any, callee_symbol: Any, formal_id: int, source_id: int, kind: Any,
) -> str:
    """The kind that can materialize one caller slot, across every callsite.

    A frame binding's kind is not a label on the slot; it selects which
    machinery may supply the argument.  ``caller_storage`` can restore a slot
    a structural cleanup removed, ``caller_alias`` and ``caller_value``
    cannot.  Each callsite decides the kind from its own private map in its
    own order, so two callsites can name the same slot under different kinds
    and the one that cannot materialize it reports ``missing_<kind>`` and
    refuses the call -- after which the callee's output is never written and
    whatever reads it gets uninitialized memory, with no shortfall anywhere.

    The decisions are already written to the shared page precisely so this is
    answerable.  If any callsite proved the slot is caller storage, that is
    what it is, and every callsite naming it gets the kind that works.
    """

    page = book.page("argument_binding")
    row = (str(callee_symbol), int(formal_id), "binding")
    for _column, fact in page.history(row):
        if not (isinstance(fact, tuple) and len(fact) == 2):
            continue
        recorded_kind, recorded_source = fact
        if not isinstance(recorded_source, int):
            continue
        if int(recorded_source) != int(source_id):
            continue
        if str(recorded_kind) == "caller_storage":
            resolution_page = book.page("argument_binding_resolution")
            resolution_row = (
                str(callee_symbol), int(formal_id), int(source_id),
            )
            history = resolution_page.history(resolution_row)
            column = history[-1][0] + 1 if history else 0
            resolution_page.set(resolution_row, column, (
                "caller_storage", str(kind),
                "materializing_binding_kind",
            ))
            return "caller_storage"
    return str(kind)


def record_proven_shape(
    function: Any, value_id: int, extents: Any, dtype: Any,
    level: int | None = 0,
) -> None:
    """Record extents proven for one value identity at one causal level.

    Only EXTENTS are recorded.  An empty shape is both a rank-0 scalar and
    what a query returns when recovery stops, so storing it would let an
    unknown win a race against a real shape.  A deeper level supersedes a
    shallower one because it was derived from more of the program; the same
    answer arriving deeper is recorded at its own level so the page shows how
    far it has been confirmed.
    """

    extents = tuple(int(extent) for extent in (extents or ()))
    if not extents:
        return
    page = current_identity_book().page("proven_shape")
    row = (authored_function_name(function), int(value_id))
    recorded = page.history(row)
    fact = ("proven", extents, str(dtype or "float64"))
    target_level = (
        max((int(column) for column, _fact in recorded), default=0)
        if level is None else int(level)
    )
    if not recorded:
        page.set(row, target_level, fact)
        return
    deepest = max(recorded, key=lambda entry: int(entry[0]))
    if isinstance(deepest[1], tuple) and deepest[1] and (
        deepest[1][0] == "invalidated"
    ):
        # An upstream identity changed after this proof was derived.  The next
        # descriptor query is a new proof generation, not a disagreement with
        # the invalidated fact.  Keep both events in causal order.
        page.set(row, int(deepest[0]) + 1, fact)
        return
    if (
        tuple(deepest[1][1]) == extents
        or target_level > int(deepest[0])
    ):
        page.set(row, target_level, fact)
        return
    page.set(
        row, target_level,
        ("conflicting", extents, str(dtype or "")),
    )


def invalidate_proven_shape(
    function: Any, value_id: int, source_id: int, reason: Any,
) -> None:
    """Withdraw a derived shape after one of its exact dependencies changes."""

    page = current_identity_book().page("proven_shape")
    row = (authored_function_name(function), int(value_id))
    recorded = page.history(row)
    column = max(
        (int(existing) for existing, _fact in recorded), default=-1,
    ) + 1
    page.set(
        row, column,
        ("invalidated", int(source_id), str(reason)),
    )


def proven_shape_of(function: Any, value_id: int) -> tuple[int, ...] | None:
    """The extents proven for this identity, deepest first, or None.

    This is the question every store was answering separately.  A row that
    two derivations contradict at the same causal level answers nothing --
    concurrent and genuinely in conflict is not a fact.
    """

    page = current_identity_book().page("proven_shape")
    row = (authored_function_name(function), int(value_id))
    recorded = page.history(row)
    if not recorded:
        return None
    deepest = max(recorded, key=lambda entry: int(entry[0]))[1]
    if not isinstance(deepest, tuple) or deepest[0] != "proven":
        return None
    return tuple(int(extent) for extent in deepest[1])


def shape_store_report(book: Any, stores: Any = None) -> str:
    """Where the stores of one shape disagree, as a report.

    Kept because it turned a day of inference into three named rows: a value
    whose stores differ is a row, not a hunt.
    """

    names = tuple(stores or ("node", "linked", "ssa"))
    pages = {name: book.page(f"shape.{name}") for name in names}
    pages["proven"] = book.page("proven_shape")
    rows: set = set()
    for page in pages.values():
        rows.update(page.rows())

    def extents(name: str, row: Any):
        page = pages[name]
        if row not in set(page.rows()):
            return None
        recorded = page.history(row)
        if not recorded:
            return None
        fact = max(recorded, key=lambda entry: int(entry[0]))[1]
        if name == "proven":
            return tuple(fact[1]) if fact[0] == "proven" else None
        return tuple(fact)

    lines = []
    disagreeing = []
    for row in sorted(rows, key=str):
        present = {}
        for name in pages:
            value = extents(name, row)
            if value is not None:
                present[name] = value
        if len({tuple(value) for value in present.values()}) > 1:
            disagreeing.append((row, present))
    lines.append(
        f"shape stores: {len(disagreeing)} disagreeing of {len(rows)} "
        "value identit(ies)"
    )
    for row, present in disagreeing[:10]:
        lines.append(f"  {render_row(row)} {present}")
    for name, page in pages.items():
        lines.append(f"  store {name}: {len(page.rows())} row(s)")
    return "\n".join(lines)


def row_value_id(row: Any) -> int | None:
    """The SSA value id a page row is about, when it names one.

    Pages key rows differently -- ``(function, value id, field)`` for a
    mutation page, a bare id for a reference count -- so presentation takes
    the first integer it finds and says nothing when there is none, rather
    than guessing a position that happens to work for one page's shape.
    """
    if isinstance(row, int):
        return int(row)
    if isinstance(row, tuple):
        for item in row:
            if isinstance(item, int):
                return int(item)
    return None


def render_identity_book(book: IdentityBook) -> str:
    """Every page, every row, its full span history -- the dense log."""
    lines = [f"identity book: {len(book.pages)} page(s)"]
    for page_name in sorted(book.pages):
        page = book.pages[page_name]
        rows = page.rows()
        lines.append(f"[{page_name}] {len(rows)} row(s), {len(page.cells)} cell(s)")
        # Rows gathered under the id group they belong to, so one page's
        # entries read as the few spaces they actually span rather than as
        # one undifferentiated list.  A row whose key names no id keeps its
        # place under "unkeyed" instead of being dropped or invented into
        # a group.
        by_group: dict[str, list[Any]] = {}
        for row in rows:
            value_id = row_value_id(row)
            group = (
                "unkeyed" if value_id is None
                else (group_by_prefix([value_id])[0].label)
            )
            by_group.setdefault(group, []).append(row)
        for group in sorted(by_group):
            group_rows = by_group[group]
            if len(by_group) > 1:
                lines.append(f"  ({group}) {len(group_rows)} row(s)")
            for row in group_rows:
                spans = page.spans(row)
                trail = " -> ".join(
                    f"{start}..{end}={fact}" for start, end, fact in spans
                )
                lines.append(f"  {render_row(row)}: {trail}")
    return "\n".join(lines)


def render_row(row: Any) -> str:
    """A page row with its ids named rather than spelled out in full.

    A flagged id is a nineteen-digit number; printing it raw makes the log
    unsearchable by the serial a reader actually has in hand (from a
    traceback, say) and unreadable at a glance.  Each integer in the row is
    rendered through :func:`id_space.label`, which leaves an unflagged id
    exactly as it was -- so nothing about legacy output changes -- and
    turns a flagged one into ``minted#1000013548``.
    """
    if isinstance(row, int):
        return id_label(row)
    if isinstance(row, tuple):
        return (
            "(" + ", ".join(
                id_label(item) if isinstance(item, int) and not isinstance(item, bool)
                else repr(item)
                for item in row
            ) + ")"
        )
    return repr(row)
