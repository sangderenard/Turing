"""Retire exact positional call occurrences at the signature-pruning seam."""
from .concordance_declarations import (
    FORMAL_ACTUAL_OCCURRENCE as OCC, FORMAL_OCCURRENCE_PRUNED,
    PRUNED_CALLEE_FORMAL, SSA_VALUE, FRAME_BINDING,
)


def _scope(function):
    return str(function.metadata.get("tensor_shape_concordance_scope") or function.name)


def _fact(book, cell):
    return book.page(cell.page).cell(cell.row, cell.column)


def capture_pruned_occurrences(functions, callee, keep_indices, book):
    page = book.pages.get(OCC.name)
    if page is None:
        return ()
    captured = []
    for caller in functions.values():
        accepted = dict(caller.metadata.get("accepted_ssa_component") or {}).get("identity_book")
        if accepted is not None and accepted is not book:
            continue
        for label, block in caller.blocks.items():
            for index, call in enumerate(block.instrs):
                if call.op not in {"Call", "call"} or call.attributes.get("callee") != callee.name:
                    continue
                refs = tuple(book.latest_ref(OCC, (callee.name, int(formal.id), caller.name,
                    str(label), index, position, _scope(caller)))
                    for position, formal in enumerate(callee.args))
                if not any(ref is not None for ref in refs):
                    continue
                if (None in refs or len(call.args) != len(callee.args)
                        or any(_fact(book, ref) != int(actual.id) for ref, actual in zip(refs, call.args))):
                    raise ValueError("signature pruning lacks its complete exact prior occurrence census")
                captured.append((tuple(refs), tuple(map(int, keep_indices))))
    return tuple(captured)


def retire_pruned_occurrences(functions, captures, book):
    from .identity_concordance import Derived, Mode, Unresolved, publish_formal_actual_occurrences
    if not captures:
        return
    publish_formal_actual_occurrences(functions, book=book)
    for refs, keep in captures:
        removed = tuple(ref for i, ref in enumerate(refs) if i not in keep)
        prunes = tuple(book.latest_ref(PRUNED_CALLEE_FORMAL, ref.row[:2]) for ref in removed)
        if None in prunes:
            raise ValueError("occurrence retirement lacks exact pruned formal receipts")
        for old_position, old in enumerate(refs):
            if old_position in keep and keep.index(old_position) == old_position:
                continue
            replacements = ()
            if old_position in keep:
                row = (*old.row[:5], keep.index(old_position), old.row[6])
                new = book.latest_ref(OCC, row)
                if new is None or _fact(book, new) != _fact(book, old):
                    raise ValueError("signature pruning lacks exact replacement occurrence")
                replacements = (new,)
            causes = tuple(dict.fromkeys((old, *refs, *prunes, *replacements)))
            book.post(OCC, old.row, Unresolved(FORMAL_OCCURRENCE_PRUNED, read=causes),
                stage=FRAME_BINDING, provenance=Derived(causes), mode=Mode.REVISE)


def _current_occurrence(book, ref, seen):
    current = book.latest_ref(OCC, ref.row)
    if current == ref:
        return ref, ()
    proof = occurrence_retirement_sources(book, current, _seen=seen)
    if proof is None:
        raise ValueError("unproven later occurrence transition")
    fact = _fact(book, current)
    if any(r.page == PRUNED_CALLEE_FORMAL and r.row == ref.row[:2] for r in fact.read):
        return None, proof
    replacements = [r for r in fact.read if r.page == OCC and r.row[:2] == ref.row[:2]
                    and r.row[2:5] == ref.row[2:5] and r.row[6] == ref.row[6]
                    and r.row[5] != ref.row[5]]
    if len(replacements) != 1:
        raise ValueError("ambiguous later occurrence replacement")
    terminal, later = _current_occurrence(book, replacements[0], seen | {current})
    return terminal, (*proof, *later)


def occurrence_retirement_sources(book, cell, functions=None, _seen=frozenset()):
    """Return exact retirement evidence, never accept a bare tombstone."""
    from .identity_concordance import Unresolved
    if cell in _seen:
        return None
    try:
        fact = _fact(book, cell)
        if (cell.page != OCC or book.latest_ref(OCC, cell.row) != cell
                or not isinstance(fact, Unresolved) or fact.reason != FORMAL_OCCURRENCE_PRUNED):
            return None
        causes = tuple(parent for parent, _ in book.edges_into(cell))
        if set(causes) != set(fact.read):
            return None
        history = book.page(OCC).history(cell.row)
        previous = [book_ref for book_ref in causes if book_ref.page == OCC and book_ref.row == cell.row]
        if (len(previous) != 1 or len(history) < 2 or previous[0].column != history[-2][0]
                or not isinstance(_fact(book, previous[0]), int)):
            return None
        old = previous[0]
        removed = [ref for ref in causes if ref.page == PRUNED_CALLEE_FORMAL]
        if not removed:
            return None
        boundary = min(book.stamp_of(ref) for ref in removed)
        originals = [ref for ref in causes if ref.page == OCC
                     and ref.row[0] == cell.row[0] and ref.row[2:5] == cell.row[2:5]
                     and ref.row[6] == cell.row[6] and isinstance(_fact(book, ref), int)
                     and book.stamp_of(ref) < boundary]
        # Reconstruct the whole live occurrence census immediately before the
        # pruning receipts, not just whichever original rows the claimant cites.
        # Earlier retired generations stay historical and do not become live.
        historical = []
        from .identity_concordance import Ref
        for row in book.page(OCC).rows():
            if row[0] != cell.row[0] or row[2:5] != cell.row[2:5] or row[6] != cell.row[6]:
                continue
            candidates = [Ref(OCC, row, column) for column, _ in book.page(OCC).history(row)
                          if book.stamp_of(Ref(OCC, row, column)) < boundary]
            if candidates:
                prior = max(candidates, key=book.stamp_of)
                if isinstance(_fact(book, prior), int):
                    historical.append(prior)
        if set(originals) != set(historical):
            return None
        originals.sort(key=lambda ref: int(ref.row[5]))
        if (old not in originals or [int(ref.row[5]) for ref in originals] != list(range(len(originals)))
                or len({ref.row[1] for ref in originals}) != len(originals)):
            return None
        removed_occurrences = []
        for prune in removed:
            if (book.latest_ref(PRUNED_CALLEE_FORMAL, prune.row) != prune
                    or _fact(book, prune) != "unused" or prune.row[0] != cell.row[0]):
                return None
            own = [r for r, _ in book.edges_into(prune) if r.page == SSA_VALUE
                   and r.row == (cell.row[0], prune.row[1])]
            if len(own) != 1:
                return None
            matches = [r for r in originals if r.page == OCC and r.row[:2] == prune.row
                       and r.row[2:5] == cell.row[2:5] and r.row[6] == cell.row[6]
                       and isinstance(_fact(book, r), int)]
            if len(matches) != 1:
                return None
            removed_occurrences.append(matches[0])
        positions = [int(r.row[5]) for r in removed_occurrences]
        if len(set(positions)) != len(positions):
            return None
        removed_formal = cell.row[:2] in {r.row for r in removed}
        new_position = int(cell.row[5]) - sum(p < int(cell.row[5]) for p in positions)
        expected_row = (*cell.row[:5], new_position, cell.row[6])
        remaining = set(causes) - {*originals, *removed}
        proof = list(causes)
        if removed_formal:
            if remaining:
                return None
        else:
            if new_position == cell.row[5] or len(remaining) != 1:
                return None
            replacement, = remaining
            if (replacement.page != OCC or replacement.row != expected_row
                    or _fact(book, replacement) != _fact(book, old)):
                return None
            current = book.latest_ref(OCC, expected_row)
            if current != replacement:
                nested = occurrence_retirement_sources(book, current, functions, _seen | {cell})
                if nested is None:
                    return None
                proof.extend(nested)
            elif functions is not None:
                caller = functions.get(str(cell.row[2])); callee = functions.get(str(cell.row[0]))
                if caller is None or callee is None or _scope(caller) != cell.row[6]:
                    return None
                call = caller.blocks[str(cell.row[3])].instrs[int(cell.row[4])]
                if (call.op not in {"Call", "call"} or call.attributes.get("callee") != callee.name
                        or len(call.args) != len(callee.args)
                        or int(callee.args[new_position].id) != int(cell.row[1])
                        or int(call.args[new_position].id) != _fact(book, old)):
                    return None
        if functions is not None:
            callee = functions.get(str(cell.row[0])); caller = functions.get(str(cell.row[2]))
            if (callee is None or caller is None or _scope(caller) != cell.row[6]
                    or any(int(arg.id) in {int(p.row[1]) for p in removed} for arg in callee.args)):
                return None
            call = caller.blocks[str(cell.row[3])].instrs[int(cell.row[4])]
            kept = []
            for ref in originals:
                if ref in removed_occurrences:
                    continue
                terminal, later = _current_occurrence(book, replacement if ref == old else ref, _seen | {cell})
                proof.extend(later)
                if terminal is not None:
                    kept.append(terminal)
            kept.sort(key=lambda ref: int(ref.row[5]))
            if ([int(ref.row[5]) for ref in kept] != list(range(len(kept)))
                    or call.op not in {"Call", "call"} or call.attributes.get("callee") != callee.name
                    or len(call.args) != len(callee.args) or len(kept) != len(callee.args)
                    or any((int(formal.id), int(actual.id)) != (int(ref.row[1]), _fact(book, ref))
                           for formal, actual, ref in zip(callee.args, call.args, kept))):
                return None
        return tuple(dict.fromkeys((cell, *proof)))
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        return None


def live_occurrence_rows(book, rows, functions=None):
    from .identity_concordance import Unresolved
    live = []
    for row in rows:
        cell = book.latest_ref(OCC, row)
        if isinstance(_fact(book, cell), Unresolved):
            if occurrence_retirement_sources(book, cell, functions) is None:
                raise ValueError("invalid positional occurrence retirement")
        else:
            live.append(row)
    return tuple(live)
