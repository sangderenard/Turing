"""A staged decision tree for "why doesn't this translate/compile/run right".

The pipeline is Python -> ProcessGraph -> repository SSA -> backend
(Fortran/LLVM) -> runtime buffers. A defect at any stage shows up at the
END as a wrong number or a compiler error, and the expensive mistake is
guessing which stage owns it. This walks the stages in order and reports
the FIRST one that is provably inconsistent, so the search starts where
the evidence is instead of where the symptom is.

Each check answers one question and says what it does NOT prove, because
a check that silently degrades (returning a default when it cannot
observe something) is worse than no check -- that exact failure mode
produced a day of false diagnosis here.

    python tools/diagnose_translation.py                # all stages
    python tools/diagnose_translation.py --ids 116,47   # focus on values
    python tools/diagnose_translation.py \
        --repository-ssa build/product/units/call/repository-ssa.pkl \
        --entry call                                    # meta-compilation
    python tools/diagnose_translation.py \
        --compilation-unit build/product/units/call     # live/failed unit
    python tools/diagnose_translation.py \
        --compilation-product build/product             # aggregate frontier

DECISION TREE
=============

STAGE 0  Isolated meta-compilation worker
    Does a catalogue unit have a live worker, a durable failure, or published
    repository SSA? An absent artifact is not malformed SSA.
    RESOURCE/FRONTEND FAIL -> remain above SSA and use the recorded phase.
    PUBLISHED -> continue through the ordinary stages below.

STAGE 1  Lowering shortfalls
    Does the SSA even claim to be complete?
    FAIL -> read the shortfall; nothing downstream is meaningful.

STAGE 2  SSA well-formedness
    2a Duplicate producers: is one id produced by two DIFFERENT SSAValue
       objects? (Same object twice is in-place reuse and is FINE.)
    2b Phi arity: does every phi's incoming-block count match its
       operand count?
    2c Dangling operands: is any operand neither a formal, nor produced,
       nor a known constant?
    2e Query dominance: does a generator-backed scalar query execute before
       every condition or region that consumes its result?
    FAIL -> the defect is at/above precompile_to_ssa; the backend is
    faithfully rendering a broken graph.

STAGE 3  In-place aliasing safety  (in-place is good; unsafe is not)
    For every region call: which out-params share a pointer with a feed?
    3a An out-param that aliases a feed is IN-PLACE -- expected, fast.
    3b TWO DIFFERENT outputs sharing ONE pointer is a real defect: the
       second write destroys the first.
    3c An out-param aliasing a feed that the callee reads AFTER writing
       that output is order-dependent and must be verified.
    FAIL -> the fusion decision is wrong, not the arithmetic.

STAGE 4  ABI observability
    Which ids are actually in artifact.buffer_order?
    An id that is NOT there cannot be read; it is not zero. Any
    conclusion drawn from reading one is void.

STAGE 5  Influence (dye) -- what actually reaches a suspect value
    `influence_field.field_from_ssa` IS wired to lowered SSA (verified on
    this program: ~8k transports over the advance function). It answers
    "what reaches this, from where, and how much survived", which is the
    question static reading of IR text cannot answer cheaply.
    Read it as:
      - dominant BAKED  -> mostly compile-time-constant influence; per the
        module's own semantics such a node is constant-foldable, so a
        runtime-varying value that reads dominantly baked is suspicious.
      - dominant DYNAMIC -> genuinely runtime-fed.
      - RECURRENT       -> arrived through a loop-carried edge; state.
    Two values that SHOULD be independent but show identical weights are
    sharing an influence path -- though note the def-use view treats all
    outputs of one region call as one node, so equal weights for two
    outputs of the SAME call is expected, not evidence.

STAGE 6  Runtime
    Run and compare against independently computed truth.

WHAT THIS DELIBERATELY DOES NOT DO
    It does not mutate the program to observe it. Adding a source-level
    probe (`state.x = accumulator + 0.0`) shifts value ids and can rebind
    the very thing being measured -- that has already produced a wrong
    conclusion here. Prefer reading an EXISTING public id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_id_spec(spec: str) -> tuple[int, ...]:
    """Parse an id selection: ``116``, ``116,47``, ``100-120``, mixed.

    Accepts commas and/or whitespace as separators and ``a-b`` (inclusive)
    or ``a..b`` as ranges, so a caller can sweep a numbering neighbourhood
    without typing every id. Order is preserved and duplicates collapse,
    because these ids are printed as report sections and a repeated
    section reads as a different value rather than the same one twice.
    """
    found: list[int] = []
    for token in str(spec).replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        body = token.replace("..", "-")
        if "-" in body.lstrip("-"):
            low_text, _, high_text = body.lstrip("-").partition("-")
            try:
                low, high = int(low_text), int(high_text)
            except ValueError:
                raise SystemExit(f"could not read id range {token!r}")
            if low > high:
                low, high = high, low
            if high - low > 4096:
                raise SystemExit(
                    f"id range {token!r} spans {high - low} ids; narrow it"
                )
            found.extend(range(low, high + 1))
            continue
        try:
            found.append(int(body))
        except ValueError:
            raise SystemExit(f"could not read id {token!r}")
    return tuple(dict.fromkeys(found))


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"         {msg}")


def load_compilation_unit(path: Path) -> dict[str, Any]:
    """Read one isolated compiler worker's durable diagnostic surface.

    A meta-compilation attempt may stop before repository SSA exists. Its
    ``compile-progress.json`` and ``failure.json`` are then the authoritative
    frontier; pretending the absent SSA is merely incomplete would route the
    investigation one stage too far downstream.
    """

    supplied = path.resolve()
    root = supplied.parent if supplied.is_file() else supplied
    if not root.is_dir():
        raise SystemExit(f"compilation unit does not exist: {root}")

    def read(name: str) -> dict[str, Any] | None:
        candidate = root / name
        if not candidate.is_file():
            return None
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise SystemExit(f"could not read {candidate}: {error}") from error
        if not isinstance(value, dict):
            raise SystemExit(f"{candidate} is not a JSON object")
        return value

    receipt = read("unit.json")
    failure = read("failure.json")
    progress = read("compile-progress.json")
    repository = root / "repository-ssa.pkl"
    qualified_name = next((
        str(record["qualified_name"])
        for record in (receipt, failure, progress)
        if record and record.get("qualified_name")
    ), None)
    if qualified_name is None and receipt is not None:
        qualified_names = tuple(receipt.get("qualified_names") or ())
        if len(qualified_names) == 1:
            qualified_name = str(qualified_names[0])
    if receipt is not None and str(receipt.get("status") or "") == "source-only":
        state = "source-only"
    elif receipt is not None and repository.is_file():
        state = "published"
    elif failure is not None:
        state = "resource-failure" if str(
            failure.get("error_type") or ""
        ) == "ResourceLimitExceeded" else "compile-failure"
    elif progress is not None:
        process_id = progress.get("process_id")
        running = False
        if process_id is not None:
            try:
                from src.compiler.project_compilation_product import (
                    process_memory_bytes,
                )
                running = process_memory_bytes(int(process_id)) is not None
            except (ImportError, OSError, TypeError, ValueError):
                running = False
        state = "running" if running else "interrupted"
    else:
        state = "empty"
    return {
        "root": root,
        "state": state,
        "qualified_name": qualified_name,
        "receipt": receipt,
        "failure": failure,
        "progress": progress,
        "repository": repository if repository.is_file() else None,
    }


def planned_dependency_repair_order(
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Resolve one partial unit's unlinked dependency closure, leaves first.

    Function-table integers are local correlation data, so the user-facing
    route is always expressed with the plan's authored qualified names.  The
    indices are retained only as stable positions in this exact hashed plan.
    """

    plan_value = receipt.get("process_graph_unit_plan")
    if not plan_value:
        return ()
    plan_path = Path(str(plan_value)).resolve()
    if not plan_path.is_file():
        raise ValueError(f"unit plan is unavailable: {plan_path}")
    expected_hash = str(receipt.get("process_graph_unit_plan_sha256") or "")
    if expected_hash:
        actual_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                "unit plan changed after compilation: "
                f"expected {expected_hash}, found {actual_hash}"
            )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "turing.compilation-unit-plan.v1":
        raise ValueError("unsupported ProcessGraph compilation-unit plan")
    units = tuple(map(dict, plan.get("units") or ()))
    selected_index = int(receipt["unit_index"])
    if selected_index < 0 or selected_index >= len(units):
        raise ValueError(f"unit index {selected_index} is outside its plan")

    linked = set()
    for record in receipt.get("linked_verified_units") or ():
        linked.add(int(
            record.get("unit_index") if isinstance(record, dict) else record
        ))

    ordered: list[int] = []
    visited: set[int] = set()

    def visit(index: int) -> None:
        if index in visited or index in linked:
            return
        if index < 0 or index >= len(units):
            raise ValueError(f"dependency unit {index} is outside its plan")
        visited.add(index)
        for dependency in map(
            int, units[index].get("dependency_units") or (),
        ):
            visit(dependency)
        ordered.append(index)

    for dependency in map(
        int, units[selected_index].get("dependency_units") or (),
    ):
        visit(dependency)
    return tuple({
        "unit_index": index,
        "qualified_names": tuple(map(
            str, units[index].get("qualified_names") or ("?",),
        )),
        "dependency_units": tuple(map(
            int, units[index].get("dependency_units") or (),
        )),
    } for index in ordered)


def stage_0_compilation_unit(snapshot: dict[str, Any]) -> bool:
    """Route a meta-compilation attempt before repository-SSA diagnostics."""

    print("STAGE 0  isolated meta-compilation worker")
    state = str(snapshot["state"])
    failure = snapshot.get("failure") or {}
    progress = snapshot.get("progress") or {}
    current = progress.get("current") or failure.get("stage") or {}
    if state == "source-only":
        receipt = snapshot.get("receipt") or {}
        _ok("unit has no numeric regions; authored source remains authoritative")
        for accounting in receipt.get("repository_ssa_accounting") or ():
            for shortfall in accounting.get("shortfalls") or ():
                _info(
                    f"{accounting.get('qualified_name', '?')}: "
                    f"{shortfall.get('kind', 'source-only')} -> "
                    f"{shortfall.get('action', 'retain-authored-source')}"
                )
        _info("There is no repository SSA or native installation to diagnose.")
        return True
    if state == "published":
        receipt = snapshot.get("receipt") or {}
        pinned_toolchain = receipt.get("compiler_toolchain")
        if isinstance(pinned_toolchain, dict) and pinned_toolchain.get("sha256"):
            from src.compiler.project_compilation_product import (
                changed_compiler_toolchain_files,
            )

            changed = changed_compiler_toolchain_files(pinned_toolchain)
            if changed:
                _bad(
                    "unit was compiled from a stale compiler-toolchain plan; "
                    "regenerate the frozen ProcessGraph before trusting this receipt"
                )
                _info(
                    "changed toolchain files: "
                    + ", ".join(changed[:8])
                    + (f" (+{len(changed) - 8} more)" if len(changed) > 8 else "")
                )
            else:
                _ok("frozen plan matches the current compiler toolchain")
        else:
            _info(
                "legacy plan has no compiler-toolchain fingerprint; source/graph "
                "hashes are pinned, but compiler staleness cannot be ruled out"
            )
        if receipt.get("status") == "partial" or not bool(
            receipt.get("repository_ssa_complete", True)
        ):
            _bad("unit published partial repository SSA; it is not installable")
            for accounting in receipt.get("repository_ssa_accounting") or ():
                for shortfall in accounting.get("shortfalls") or ():
                    _info(
                        f"{accounting.get('qualified_name', '?')}: "
                        f"{shortfall.get('kind', 'unspecified shortfall')}"
                    )
            _info(
                "Continue through structural stages to localize the defect; "
                "do not emit or install this unit."
            )
            try:
                repair_order = planned_dependency_repair_order(receipt)
            except (OSError, TypeError, ValueError) as error:
                _info(f"cannot resolve the pinned dependency plan: {error}")
            else:
                if repair_order:
                    _info("Meta-compilation repair order (dependencies first):")
                    for dependency in repair_order:
                        role = (
                            "leaf" if not dependency["dependency_units"]
                            else "after " + ",".join(map(
                                str, dependency["dependency_units"],
                            ))
                        )
                        _info(
                            f"  unit {dependency['unit_index']}: "
                            f"{', '.join(dependency['qualified_names'])} "
                            f"[{role}]"
                        )
                    _info(
                        "Verify those units in that order, then retry this "
                        "unit so the scheduler can link them. Reassess this "
                        "unit's boundary and conditional shortfalls only "
                        "after that dependency pass."
                    )
                elif (receipt.get("unit") or {}).get("dependency_units"):
                    _info(
                        "All declared dependencies were linked; repair this "
                        "unit's own Stage 1 boundary/control shortfalls."
                    )
                else:
                    _info(
                        "This is a dependency leaf; repair its own Stage 1 "
                        "boundary/control shortfalls before retrying callers."
                    )
        else:
            _ok("unit published repository SSA; continue with structural stages")
        return True
    if current:
        elapsed = current.get("elapsed_seconds")
        elapsed_text = (
            "unknown time" if elapsed is None else f"{float(elapsed):.3f}s"
        )
        phase = str(current.get("phase") or "unspecified-phase")
        _info(
            f"last durable phase {phase} at {elapsed_text}: "
            f"{current.get('message', '?')}"
        )
        resident = current.get("resident_bytes")
        if resident is not None:
            _info(f"worker resident memory then: {int(resident) / 1024 ** 3:.3f} GiB")
    if state == "resource-failure":
        _bad(
            f"worker stopped at a resource boundary: "
            f"{failure.get('error', 'unspecified limit')}"
        )
        phase = str(current.get("phase") or "")
        if phase == "deployment-instantiation":
            _info(
                "No repository SSA was published. Divide the selected "
                "authored/SCC activation closure before retrying; backend "
                "and SSA checks do not apply at deployment instantiation."
            )
        elif phase == "region-precompile":
            _info(
                "No repository SSA was published. Resume from the owned "
                "region/subdivision receipts, or introduce a smaller "
                "deterministic region integral at this shell."
            )
        else:
            _info(
                "No repository SSA was published. Retry with a deliberate "
                "larger bound, or partition at the last durable phase if the "
                "same phase repeatedly makes no progress; downstream "
                "SSA/backend checks do not apply yet."
            )
        return False
    if state == "compile-failure":
        _bad(
            f"worker raised {failure.get('error_type', 'an exception')}: "
            f"{failure.get('error', 'no message')}"
        )
        if failure.get("frontier_kind") == "compilation-subdivision-required":
            for boundary in failure.get("subdivision_boundaries") or ():
                _info(
                    "subdivide at "
                    f"{boundary.get('kind', 'source boundary')} "
                    f"node={boundary.get('loop_node_id', '?')} "
                    f"regions={tuple(boundary.get('region_indices') or ())} "
                    f"because={tuple(boundary.get('blockers') or ())}"
                )
            _info(
                "The enclosing authored function remains source fallback; "
                "enqueue the named boundary as the next deterministic "
                "compilation integral."
            )
            return False
        _info(
            "This is a frontend/planning/lowering failure before publication; "
            "fix the named phase before inspecting a backend."
        )
        return False
    if state == "running":
        _ok(
            "worker process "
            f"{progress.get('process_id')} is still running; no terminal "
            "receipt exists yet"
        )
        _info("There is no SSA verdict until the worker publishes or stops.")
        return True
    if state == "interrupted":
        _bad("worker has progress telemetry but its process is no longer running")
        _info(
            "The attempt was interrupted without a terminal receipt. Preserve "
            "the last durable phase when deciding whether to retry or divide."
        )
        return False
    _bad("unit directory has no receipt, failure, or progress telemetry")
    return False


def load_compilation_product(path: Path) -> dict[str, Any]:
    """Read a sealed or live isolated-unit catalogue without rebuilding it."""

    supplied = path.resolve()
    root = supplied.parent if supplied.is_file() else supplied
    if not root.is_dir():
        raise SystemExit(f"compilation product does not exist: {root}")

    def read(name: str) -> dict[str, Any] | None:
        candidate = root / name
        if not candidate.is_file():
            return None
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise SystemExit(f"could not read {candidate}: {error}") from error
        if not isinstance(value, dict):
            raise SystemExit(f"{candidate} is not a JSON object")
        return value

    manifest = read("manifest.json")
    progress = read("progress.json")
    unit = read("unit.json")
    source = manifest or progress or unit
    if source is None:
        raise SystemExit(
            "compilation product has no manifest.json, progress.json, or "
            f"unit.json: {root}"
        )
    schema = str(source.get("schema") or "")
    subdivision_product = (
        "subdivision" in schema
        or "integrals" in source
        or any("integral_index" in item for item in source.get("completed") or ())
    )
    if unit is not None and manifest is None and progress is None:
        records = (unit,)
    elif "units" in source:
        records = tuple(source.get("units") or ())
    elif "integrals" in source:
        records = tuple(source.get("integrals") or ())
    else:
        records = tuple(source.get("completed") or ())
    return {
        "root": root,
        "sealed": manifest is not None or unit is not None,
        "kind": "subdivision" if subdivision_product else "unit-catalogue",
        "schema": schema,
        "records": records,
        "running": tuple(() if progress is None else progress.get("running") or ()),
        "pending": tuple(() if progress is None else progress.get("pending") or ()),
        "subdivision_integral_count": int(
            source.get("subdivision_integral_count") or 0
        ),
    }


def _failure_signature(record: dict[str, Any]) -> str:
    error_type = str(record.get("error_type") or "failure")
    error = " ".join(str(record.get("error") or "no message").split())
    if error_type == "ResourceLimitExceeded":
        resource = "memory" if "memory" in error.casefold() else (
            "time" if "elapsed time" in error.casefold() else "other"
        )
        return f"{error_type}:{resource}"
    # Compiler failures consistently put their stable category before the
    # first colon and volatile ids/details after it. Preserve that authored
    # category instead of regex-normalizing numbers and accidentally merging
    # semantically different failures.
    category = error.partition(":")[0]
    return f"{error_type}:{category[:180]}"


def _record_names(record: dict[str, Any]) -> tuple[str, ...]:
    """Read authored identities from either catalogue receipt generation."""

    if record.get("qualified_name"):
        return (str(record["qualified_name"]),)
    names = tuple(map(str, record.get("qualified_names") or ()))
    if names:
        return names
    return tuple(map(
        str, (record.get("unit") or {}).get("qualified_names") or ("?",),
    ))


def _identity_token_summary(tokens: Any) -> str:
    """Render one deterministic context chain without falling back to its id."""

    chain = tuple(map(str, tokens or ()))
    fields: dict[str, str] = {}
    for index, token in enumerate(chain[:-1]):
        if not token.startswith("field:node."):
            continue
        value = chain[index + 1]
        if value.startswith("value:"):
            fields[token.removeprefix("field:node.")] = value.removeprefix(
                "value:"
            )
    line = next((item.removeprefix("line:") for item in chain if item.startswith("line:")), "?")
    column = next((item.removeprefix("column:") for item in chain if item.startswith("column:")), "?")
    line = line.lstrip("0") or "0"
    column = column.lstrip("0") or "0"
    operation = fields.get("op") or fields.get("label") or fields.get("type") or "value"
    version = next((item for item in reversed(chain) if item.startswith("version:")), "version:?")
    # Boundary formals are deliberately assigned a sentinel source position.
    # Showing that sentinel as a line number obscures the stable authored name
    # that is already present in the same identity chain.
    if operation == "input" and fields.get("label"):
        return f"input[{fields['label']}]/{version}"
    ast_start = next((
        index + 1 for index, token in enumerate(chain)
        if token == "field:node.ast"
    ), None)
    if ast_start is not None:
        ast_end = next((
            index for index in range(ast_start, len(chain))
            if chain[index].startswith("field:")
        ), len(chain))
        ast_tokens = chain[ast_start:ast_end]
        if "value:Attribute" in ast_tokens or "value:Subscript" in ast_tokens:
            quoted = []
            for index, token in enumerate(ast_tokens[:-1]):
                if token == "value:'" and index + 1 < len(ast_tokens):
                    candidate = ast_tokens[index + 1]
                    if candidate.startswith("value:"):
                        value = candidate.removeprefix("value:")
                        if (
                            re.fullmatch(r"[A-Za-z_]\w*", value)
                            and value not in quoted
                        ):
                            quoted.append(value)
            if "value:Subscript" in ast_tokens and len(quoted) >= 2:
                base = ".".join(quoted[:-1])
                return f"{base}[{quoted[-1]}]/{version}"
            if "value:Attribute" in ast_tokens and quoted:
                return f"{'.'.join(quoted)}/{version}"
    return f"{operation}@{line}:{column}/{version}"


def stage_0_compilation_product(snapshot: dict[str, Any]) -> bool:
    """Summarize the causal frontier across one bounded catalogue pass."""

    noun = "integral" if snapshot.get("kind") == "subdivision" else "unit"
    print(f"STAGE 0  meta-compilation {noun} product")
    records = tuple(map(dict, snapshot["records"]))
    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    summary = ", ".join(
        f"{counts[name]} {name}" for name in sorted(counts)
    ) or "no terminal units"
    state = "sealed" if snapshot["sealed"] else "live"
    _info(
        f"{state} catalogue: {summary}; {len(snapshot['running'])} running; "
        f"{len(snapshot['pending'])} pending"
    )

    failures: dict[str, list[tuple[str, ...]]] = {}
    for record in records:
        if str(record.get("status")) != "failed":
            continue
        failures.setdefault(_failure_signature(record), []).append(
            _record_names(record)
        )
    if failures:
        _bad(f"{sum(map(len, failures.values()))} failed {noun}(s) in "
             f"{len(failures)} causal class(es)")
        for signature, units in sorted(
            failures.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            names = tuple(
                name for unit_names in units for name in unit_names
            )
            _info(f"{len(units)} x {signature}")
            _info("  " + ", ".join(names[:6]))
    else:
        _ok(f"no hard {noun} failures")

    partials = [
        record for record in records
        if str(record.get("status")) == "partial"
    ]
    if partials:
        actions: dict[str, int] = {}
        unresolved_identities: list[tuple[str, str]] = []
        for record in partials:
            shortfall_kinds = tuple(dict.fromkeys(
                str(shortfall.get("kind") or "semantic/ABI")
                for shortfall in (
                    *(
                        shortfall
                        for accounting in record.get(
                            "repository_ssa_accounting", ()
                        )
                        for shortfall in accounting.get("shortfalls", ())
                    ),
                    *(
                        shortfall
                        for region in record.get("regions", ())
                        for shortfall in region.get("shortfalls", ())
                    ),
                )
            ))
            if shortfall_kinds:
                for action in shortfall_kinds:
                    actions[action] = actions.get(action, 0) + 1
            else:
                action = str(
                    record.get("control_frontier_action") or "semantic/ABI"
                )
                actions[action] = actions.get(action, 0) + 1
            for region in record.get("regions", ()):
                for shortfall in region.get("shortfalls", ()):
                    if shortfall.get("kind") != "unresolved-boundary-types":
                        continue
                    owner = ", ".join(_record_names(record))
                    for identity in shortfall.get("value_identities", ()):
                        tokens = identity.get("identity_token_chain") or ()
                        if tokens:
                            unresolved_identities.append((
                                owner, _identity_token_summary(tokens),
                            ))
        _bad(f"{len(partials)} partial SSA {noun}(s)")
        for action, count in sorted(
            actions.items(), key=lambda item: (-item[1], item[0])
        ):
            _info(f"{count} x {action}")
        if unresolved_identities:
            _info("deterministic boundary identities (representative):")
            for owner, identity in tuple(dict.fromkeys(
                unresolved_identities
            ))[:8]:
                _info(f"  {owner}: {identity}")
    else:
        _ok(f"no partial SSA {noun}s")

    unverified = [
        record for record in records
        if str(record.get("status")) == "compiled-unverified"
    ]
    if unverified:
        _bad(
            f"{len(unverified)} compiled {noun}(s) still require exact "
            "semantic and ABI verification"
        )
        names = tuple(
            name for record in unverified for name in _record_names(record)
        )
        if names:
            _info("  " + ", ".join(names[:8]))
    else:
        _ok(f"no unverified compiled {noun}s")

    verified = [
        record for record in records
        if str(record.get("status")) == "verified"
    ]
    if verified:
        probe_count = sum(
            int(region.get("probe_count") or 0)
            for record in verified
            for region in record.get("regions", ())
        )
        _ok(
            f"{len(verified)} verified {noun}(s); exact structural/semantic "
            f"probe count={probe_count}"
        )

    source_only = [
        record for record in records
        if str(record.get("status")) == "source-only"
    ]
    if source_only:
        _ok(
            f"{len(source_only)} source-only {noun}(s) are terminal; authored "
            "source remains authoritative"
        )
    if snapshot.get("subdivision_integral_count"):
        _info(
            f"{snapshot['subdivision_integral_count']} deterministic child "
            "integral(s) queued in subdivision-integrals.json"
        )
        _info(
            "NEXT: compile the queued subdivision plan; do not retry the "
            "resource-bound parent unchanged"
        )
    elif snapshot.get("kind") == "subdivision" and partials:
        _info(
            "NEXT: no deeper deterministic subdivision was published; "
            "repair the reported boundary type/identity/lowering semantics "
            "in the owning source, then rebuild its parent"
        )
    elif unverified:
        _info(
            "NEXT: run exact semantic and ABI verification before activating "
            "any compiled replacement"
        )
    if snapshot["running"]:
        _info(f"{len(snapshot['running'])} {noun}(s) are still compiling")
    return not (
        failures or partials or unverified or snapshot["pending"]
        or snapshot["running"]
    )


def stage_1_shortfalls(artifact: Any) -> bool:
    print("STAGE 1  lowering / emission shortfalls")
    shortfalls = tuple(getattr(artifact, "shortfalls", ()) or ())
    if not getattr(artifact, "complete", True) or shortfalls:
        _bad(f"{len(shortfalls)} shortfall(s); the SSA is not complete")
        for item in shortfalls[:10]:
            _info(str(item)[:160])
        return False
    _ok("artifact reports complete, no shortfalls")
    return True


def stage_2_ssa_wellformed(fn: Any) -> bool:
    print("STAGE 2  SSA well-formedness")
    healthy = True

    producers: dict[int, list[Any]] = {}
    for block in fn.blocks.values():
        for instr in block.instrs:
            if instr.res is not None:
                producers.setdefault(int(instr.res.id), []).append(instr.res)
    formals = {int(a.id): a for a in fn.args}

    # 2a -- duplicate producers by OBJECT, not by id.
    real_collisions = []
    inplace_reuse = []
    for vid, objs in producers.items():
        distinct = {id(o) for o in objs}
        formal = formals.get(vid)
        if formal is not None:
            if id(formal) in {id(o) for o in objs}:
                inplace_reuse.append(vid)
            else:
                real_collisions.append(vid)
        if len(distinct) > 1:
            real_collisions.append(vid)
    if real_collisions:
        healthy = False
        _bad(
            f"{len(sorted(set(real_collisions)))} id(s) produced by DIFFERENT "
            f"objects (true collision): {sorted(set(real_collisions))[:12]}"
        )
        _info("two distinct values sharing one id -- whichever renders last wins")
    else:
        _ok("no duplicate-producer collisions")
    if inplace_reuse:
        _ok(
            f"{len(inplace_reuse)} id(s) reuse their formal's cell in place "
            f"(same object) -- expected and fast: {sorted(inplace_reuse)[:12]}"
        )

    # 2d -- rank/shape disagreement.
    # A value can carry TWO statements of its own dimensionality: its
    # `.shape`, and its accounting (`program_abi_rank`, `program_abi_storage`).
    # When those disagree, every size decision downstream silently believes
    # the smaller one -- element counts, copy sizes, alloca sizes -- because
    # they read `.shape`. A rank-2 span with an empty shape is therefore
    # copied as ONE element, and a whole-array assignment quietly moves a
    # single number. That is not hypothetical: it is exactly how
    # `state.height = state.next_height + 0.0` came to update 1 cell of 16
    # in this program while every structural check above passed.
    # Rank>0 with an empty `.shape` is the NORMAL representation here: these
    # arrays are dynamically sized, and the Fortran backend resolves them
    # through the extents vector rather than a static shape. So the mere
    # disagreement is not the defect and flagging it would drown the reader.
    # The hazard is narrower: such a value standing as a REGION-CALL OUTPUT,
    # because the return-copy sizes itself from the static shape and so
    # moves exactly one element.
    all_values = {int(v.id): v for v in formals.values()}
    for block in fn.blocks.values():
        for instr in block.instrs:
            if instr.res is not None:
                all_values.setdefault(int(instr.res.id), instr.res)

    def declared_rank(value: Any) -> int:
        accounting = getattr(value, "accounting", None) or {}
        return max(
            int(accounting.get("program_abi_rank", 0) or 0),
            int(accounting.get("ssa_call_rank", 0) or 0),
            1 if str(accounting.get("program_abi_storage")) == "span" else 0,
        )

    sized_as_scalar = {}
    for block in fn.blocks.values():
        for instr in block.instrs:
            if str(instr.op) not in {"Call", "call"}:
                continue
            for out_id in map(int, instr.attributes.get("output_ids", ()) or ()):
                value = all_values.get(out_id)
                if value is None:
                    continue
                rank = declared_rank(value)
                if rank > 0 and not tuple(getattr(value, "shape", ()) or ()):
                    accounting = getattr(value, "accounting", None) or {}
                    sized_as_scalar[out_id] = (
                        str(accounting.get("program_abi_field") or "?"),
                        rank,
                        str(instr.attributes.get("callee", "?")).split("__")[-1],
                    )
    if sized_as_scalar:
        # NOT a failure any more. The LLVM backend now expands an
        # elementwise op whose result is a span into a loop sized from the
        # extents vector, so rank-with-empty-shape is handled rather than
        # silently truncated. Reported because it is still the shape that
        # once produced a one-element whole-array assignment, and because a
        # backend that has NOT learned this will truncate here -- but it is
        # informational, since flagging a fixed condition is how a checker
        # starts lying to its reader.
        _ok(
            f"{len(sized_as_scalar)} region-call output(s) are rank>0 with an "
            "empty .shape -- expanded elementwise from runtime extents"
        )
        for out_id, (field, rank, callee) in list(sized_as_scalar.items())[:10]:
            _info(f"id {out_id} (field {field}, rank {rank}) out of {callee}")
        _info(
            "If a NEW backend is added, this is the configuration to test "
            "first: sizing it from .shape moves one element of the array."
        )
    else:
        _ok("no region-call output is a rank>0 value sized as a scalar")

    # 2b -- phi arity.
    bad_phis = []
    for block_name, block in fn.blocks.items():
        for instr in block.instrs:
            if str(instr.op) in {"Phi", "phi"}:
                blocks = tuple(instr.attributes.get("incoming_blocks") or ())
                if blocks and len(blocks) != len(instr.args):
                    bad_phis.append((block_name, int(instr.res.id)))
    if bad_phis:
        healthy = False
        _bad(f"phi arity mismatch: {bad_phis[:10]}")
    else:
        _ok("every phi's incoming blocks match its operands")

    # 2c -- dangling operands.
    defined = set(producers) | set(formals)
    dangling = set()
    for block in fn.blocks.values():
        for instr in block.instrs:
            for a in instr.args:
                if int(a.id) not in defined:
                    dangling.add(int(a.id))
    if dangling:
        healthy = False
        _bad(f"operands with no definition: {sorted(dangling)[:12]}")
    else:
        _ok("every operand is a formal or is produced")

    # 2e -- scalar sequence-query results must dominate their consumers.
    # Generator materialization is represented by a loop plus a query Phi.
    # A global producer-set check cannot detect the query being scheduled
    # after a conditional that reads it: the id exists, but not yet on that
    # path.  Keep this deliberately narrow to the structural query ABI; a
    # broad dominance check would misdiagnose legitimate in-place cells and
    # output arguments as ordinary immutable SSA values.
    block_names = tuple(fn.blocks)
    if block_names:
        start = block_names[0]
        successors = {name: set() for name in block_names}
        for name, block in fn.blocks.items():
            if not block.instrs:
                continue
            terminator = block.instrs[-1]
            for key in ("target", "true_target", "false_target"):
                target = terminator.attributes.get(key)
                if isinstance(target, str) and target in fn.blocks:
                    successors[name].add(target)
        reachable = {start}
        pending = [start]
        while pending:
            current = pending.pop()
            for target in successors[current] - reachable:
                reachable.add(target)
                pending.append(target)
        predecessors = {name: set() for name in reachable}
        for source in reachable:
            for target in successors[source].intersection(reachable):
                predecessors[target].add(source)
        dominators = {
            name: ({name} if name == start else set(reachable))
            for name in reachable
        }
        while True:
            updated = {}
            for name in reachable:
                if name == start:
                    updated[name] = {name}
                elif predecessors[name]:
                    updated[name] = {name}.union(set.intersection(*(
                        dominators[parent]
                        for parent in predecessors[name]
                    )))
                else:
                    updated[name] = {name}
            if updated == dominators:
                break
            dominators = updated

        query_producers = {}
        for block_name, block in fn.blocks.items():
            for index, instr in enumerate(block.instrs):
                binding = str(instr.attributes.get("binding") or "")
                if (
                    str(instr.op).casefold() == "phi"
                    and binding.startswith("ssa_sequence_")
                    and instr.attributes.get("source_call_node_id") is not None
                    and instr.res is not None
                ):
                    query_producers[int(instr.res.id)] = (block_name, index)
        violations = []
        for value_id, (producer_block, producer_index) in query_producers.items():
            for block_name, block in fn.blocks.items():
                if block_name not in reachable:
                    continue
                for index, instr in enumerate(block.instrs):
                    if not any(int(argument.id) == value_id for argument in instr.args):
                        continue
                    if block_name == producer_block and index > producer_index:
                        continue
                    if (
                        block_name != producer_block
                        and producer_block in dominators[block_name]
                    ):
                        continue
                    violations.append((value_id, producer_block, block_name))
        if violations:
            healthy = False
            _bad(
                "sequence-query result(s) do not dominate every consumer: "
                f"{violations[:10]}"
            )
            _info(
                "the generator/query producer unit must be scheduled before "
                "the first conditional or region that reads its scalar result"
            )
        elif query_producers:
            _ok(
                f"{len(query_producers)} sequence-query result(s) dominate "
                "all reachable consumers"
            )

        # 2f -- structural sequence construction order and materialization.
        # These identities live in descriptor metadata, so ordinary operand
        # dominance cannot see an append/extend reading an arena before the
        # calls which populate it.  That exact shape compiled successfully in
        # the compiler's own ``build_module`` and emitted a plausible but
        # truncated WASM type section.
        sequence_producers: dict[int, list[tuple[str, int]]] = {}
        sequence_consumers: list[tuple[int, str, int, str]] = []
        for block_name, block in fn.blocks.items():
            for index, instr in enumerate(block.instrs):
                attributes = instr.attributes or {}
                operation = str(
                    attributes.get("ssa_sequence_operation") or ""
                )
                destination = attributes.get("sequence_id")
                if operation and destination is not None:
                    sequence_producers.setdefault(
                        int(destination), []
                    ).append((block_name, index))
                if operation == "append_joined_singleton":
                    continue
                for key in (
                    "source_sequence_id", "joined_source_sequence_id",
                ):
                    source = attributes.get(key)
                    if source is not None:
                        sequence_consumers.append((
                            int(source), block_name, index, operation,
                        ))

        internal_sequences = {
            int(sequence_id)
            for sequence_id, names
            in fn.metadata.get("sequence_value_names", ()) or ()
            if not tuple(names or ())
        }

        ordering_violations = []
        for source, block_name, consumer_index, operation in sequence_consumers:
            if source not in internal_sequences:
                continue
            local_producers = [
                producer_index
                for producer_block, producer_index
                in sequence_producers.get(source, ())
                if producer_block == block_name
            ]
            if local_producers and min(local_producers) > consumer_index:
                ordering_violations.append((
                    source, block_name, consumer_index,
                    min(local_producers), operation,
                ))
                continue
            later_blocks = [
                producer_block
                for producer_block, _producer_index
                in sequence_producers.get(source, ())
                if producer_block != block_name
                and producer_block in dominators
                and block_name in dominators[producer_block]
            ]
            if later_blocks and not local_producers:
                ordering_violations.append((
                    source, block_name, consumer_index,
                    later_blocks[0], operation,
                ))

        immutable_views = {
            int(item["source_sequence_id"])
            for item in fn.metadata.get("extraction_materializations", ()) or ()
            if isinstance(item, dict)
            and item.get("lowering") == "immutable-sequence-view"
            and item.get("source_sequence_id") is not None
        }
        unmaterialized_views = sorted({
            source
            for source, _block, _index, _operation in sequence_consumers
            if source in immutable_views
            and source not in sequence_producers
        })
        if ordering_violations:
            healthy = False
            _bad(
                "structural sequence source(s) are consumed before their "
                f"local producer calls: {ordering_violations[:10]}"
            )
            _info(
                "schedule the complete producer chain before the mutation "
                "which reads its resident arena"
            )
        if unmaterialized_views:
            healthy = False
            _bad(
                "immutable sequence view(s) are consumed without a producer "
                f"or scalar-singleton lowering: {unmaterialized_views[:12]}"
            )
            _info(
                "bytes([value]) inside a joined/generator sequence must append "
                "that scalar; an empty descriptor is not a materialization"
            )
        if not ordering_violations and not unmaterialized_views:
            _ok(
                "structural sequence sources are ordered and singleton views "
                "are materially lowered"
            )
    return healthy


def stage_3_inplace_safety(fn: Any, module_functions: dict) -> bool:
    print("STAGE 3  in-place aliasing safety")
    healthy = True
    calls = 0
    for block in fn.blocks.values():
        for instr in block.instrs:
            if str(instr.op) not in {"Call", "call"}:
                continue
            output_ids = tuple(map(int, instr.attributes.get("output_ids", ())))
            if not output_ids:
                continue
            calls += 1
            feed_ids = [int(a.id) for a in instr.args]
            callee_name = str(instr.attributes.get("callee") or "")

            # 3b -- two outputs on one id.
            if len(set(output_ids)) != len(output_ids):
                healthy = False
                dupes = [v for v in set(output_ids) if output_ids.count(v) > 1]
                _bad(
                    f"{callee_name.split('__')[-1]}: outputs repeat id(s) "
                    f"{dupes} -- the later write destroys the earlier"
                )

            # 3a -- in-place fusion (output id also a feed).
            fused = sorted(set(output_ids) & set(feed_ids))
            if fused:
                _ok(
                    f"{callee_name.split('__')[-1]}: {len(fused)} in-place "
                    f"out/feed fusion(s) {fused[:8]}"
                )
                # 3c -- order sensitivity inside the callee.
                callee = module_functions.get(callee_name)
                if callee is not None:
                    order: list[tuple[str, int]] = []
                    for cb in callee.blocks.values():
                        for ci in cb.instrs:
                            if ci.res is not None:
                                order.append(("w", int(ci.res.id)))
                            for ca in ci.args:
                                order.append(("r", int(ca.id)))
                    formal_ids = [int(a.id) for a in callee.args]
                    for position, vid in enumerate(output_ids):
                        if vid not in fused:
                            continue
                        try:
                            feed_pos = feed_ids.index(vid)
                        except ValueError:
                            continue
                        if feed_pos >= len(formal_ids):
                            continue
                        formal = formal_ids[feed_pos]
                        writes = [
                            i for i, (k, v) in enumerate(order)
                            if k == "w" and v == formal
                        ]
                        reads = [
                            i for i, (k, v) in enumerate(order)
                            if k == "r" and v == formal
                        ]
                        if writes and reads and max(reads) > min(writes):
                            healthy = False
                            _bad(
                                f"{callee_name.split('__')[-1]}: formal "
                                f"{formal} is READ after being WRITTEN while "
                                "fused in place -- order dependent"
                            )
    if calls and healthy:
        _ok(f"{calls} region call(s) checked; in-place fusions look safe")
    return healthy


def stage_4_observability(adv: Any, ids: tuple[int, ...]) -> None:
    print("STAGE 4  ABI observability")
    order = {int(v) for v in (adv.artifact.buffer_order or ())}
    _info(f"{len(order)} ids are in the public buffer ABI")
    for vid in ids:
        if vid in order:
            _ok(f"id {vid} IS observable")
        else:
            _bad(
                f"id {vid} is NOT observable (internal alloca). Reading it "
                "yields nothing; it is NOT zero."
            )


def stage_4_repository_boundary(fn: Any, ids: tuple[int, ...]) -> None:
    """Report what an SSA repository function exposes without inventing a backend ABI."""

    print("STAGE 4  repository function boundary")
    formals = {int(value.id) for value in fn.args}
    named_outputs = {
        int(value_id)
        for value_ids in dict(fn.metadata.get("named_outputs") or {}).values()
        for value_id in (
            value_ids if isinstance(value_ids, (tuple, list)) else (value_ids,)
        )
    }
    exposed = formals | named_outputs
    _info(
        f"{len(formals)} physical formal(s), {len(named_outputs)} named "
        "output value(s) at the repository boundary"
    )
    for value_id in ids:
        if value_id in exposed:
            _ok(f"id {value_id} is exposed by the repository function")
        else:
            _bad(
                f"id {value_id} is internal to repository SSA; this mode "
                "does not claim it is observable in any emitted backend"
            )


def load_repository_ssa(path: Path, entry: str | None):
    """Load a published repository-SSA tuple and select one function by name."""

    artifact_path = path.resolve()
    with artifact_path.open("rb") as stream:
        payload = pickle.load(stream)
    if isinstance(payload, tuple):
        module = payload[0]
        exports = tuple(payload[2]) if len(payload) > 2 else ()
    else:
        module = payload
        exports = ()
    functions = dict(getattr(module, "functions", {}) or {})
    if not functions:
        raise SystemExit(f"{artifact_path} contains no repository functions")
    if entry is None:
        candidates = tuple(name for name in exports if name in functions)
        if len(candidates) == 1:
            selected = candidates[0]
        elif len(functions) == 1:
            selected = next(iter(functions))
        else:
            raise SystemExit(
                "repository contains multiple functions; pass --entry NAME\n"
                + "available: " + ", ".join(sorted(functions)[:40])
            )
    elif entry in functions:
        selected = entry
    else:
        normalized = str(entry).replace(".", "_")
        candidates = tuple(
            name for name in functions
            if name.endswith(f"__{entry}")
            or name.endswith(f"__{normalized}")
            or name.split("__")[-1] in {str(entry), normalized}
            or str(
                getattr(functions[name], "metadata", {}).get(
                    "source_qualified_name", ""
                )
            ) == str(entry)
        )
        if len(candidates) != 1:
            raise SystemExit(
                f"could not select unique entry {entry!r}; matches: "
                + (", ".join(sorted(candidates)) or "none")
            )
        selected = candidates[0]

    receipt_path = artifact_path.with_name("unit.json")
    receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.is_file() else None
    )
    if receipt is None:
        artifact = SimpleNamespace(complete=True, shortfalls=())
        receipt_note = "no sibling unit.json; repository completeness is unproven"
    else:
        receipt_shortfalls = tuple(
            shortfall
            for accounting in receipt.get("repository_ssa_accounting") or ()
            for shortfall in accounting.get("shortfalls") or ()
        )
        artifact = SimpleNamespace(
            complete=bool(receipt.get("repository_ssa_complete", False)),
            shortfalls=(
                tuple(receipt.get("shortfalls") or ())
                or receipt_shortfalls
            ),
        )
        receipt_note = f"completeness receipt: {receipt_path}"
    return module, functions[selected], artifact, selected, receipt_note


def stage_5_influence(fn: Any, ids: tuple[int, ...]) -> None:
    """Dye-trace what reaches each suspect value."""
    print("STAGE 5  influence (dye)")
    if not ids:
        _info("no --ids given; skipped")
        return
    try:
        from src.compiler.influence_field import (
            InfluenceContract, field_from_ssa,
        )
    except Exception as error:  # pragma: no cover - diagnostic aid only
        _info(f"influence field unavailable: {error}")
        return

    name = fn.name

    class _Module:
        functions = {name: fn}

    field = field_from_ssa(
        _Module(), InfluenceContract(enabled=True), functions=[name],
    )
    transports = field.propagate()
    _info(f"{transports} transports over {name.split('__')[-1]}")

    readings: dict[int, dict[str, float]] = {}

    def locate(vid: int):
        for label, block in fn.blocks.items():
            for instr in block.instrs:
                if instr.res is not None and int(instr.res.id) == vid:
                    return (name, label, instr.res.id), label
        return None, None

    for vid in ids:
        key, label = locate(vid)
        if key is None:
            _info(f"id {vid}: no producing instruction (formal or absent)")
            continue
        reading = field.reading(key)
        parts = {
            category: (
                round(getattr(entry, "weight", 0.0), 6),
                round(getattr(entry, "hue", 0.0), 6),
            )
            for category, entry in (reading.categories or {}).items()
        }
        _info(f"id {vid} in {label}: {parts}")
        readings[vid] = {
            category: round(getattr(entry, "weight", 0.0), 6)
            for category, entry in (reading.categories or {}).items()
        }

    # The signal here is comparative, so do the comparison rather than
    # leaving it to the reader: identical weights between two values that
    # should be independent means they share an influence path.
    identical: list[tuple[int, int]] = []
    seen = list(readings.items())
    for index, (left, left_weights) in enumerate(seen):
        for right, right_weights in seen[index + 1:]:
            if left_weights and left_weights == right_weights:
                identical.append((left, right))
    if identical:
        for left, right in identical:
            _bad(
                f"ids {left} and {right} have IDENTICAL influence weights -- "
                "they share an influence path. Expected if they are two "
                "outputs of the SAME region call (the def-use view treats "
                "that call as one node); a real finding otherwise."
            )
    elif len(readings) > 1:
        _ok("no two selected ids share an identical influence profile")

    _info(
        "Read these COMPARATIVELY, never absolutely. Baked dominance is the "
        "norm in this program (dt/dx/gravity/limits feed nearly everything), "
        "so 'dominantly baked' is NOT by itself a defect -- verified: the "
        "known-CORRECT max_wave_speed reads dominantly baked too. What "
        "carries signal is a value whose profile differs from a comparable "
        "value that is known to work, or two supposedly-independent values "
        "whose weights match exactly."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Staged decision tree for translation defects. See "
            "tools/TRANSLATION_DEBUGGING.md for the full questionnaire."
        ),
        epilog=(
            "ids accept commas, spaces, and inclusive ranges: "
            "--ids 141 | --ids 141,47 | --ids 100-120 | --ids 100-120,141"
        ),
    )
    parser.add_argument(
        "--ids", default="",
        help="value ids to inspect; commas/spaces/ranges (a-b or a..b)",
    )
    parser.add_argument(
        "--stages", default="",
        help="run only these stages, e.g. --stages 2,3 (default: all)",
    )
    inlet = parser.add_mutually_exclusive_group()
    inlet.add_argument(
        "--repository-ssa", type=Path,
        help="inspect an existing repository-ssa.pkl instead of compiling fluid",
    )
    inlet.add_argument(
        "--compilation-unit", type=Path,
        help=(
            "inspect an isolated catalogue unit directory, including attempts "
            "which stopped before repository SSA was published"
        ),
    )
    inlet.add_argument(
        "--compilation-product", type=Path,
        help=(
            "summarize the causal frontier of a live or sealed isolated-unit "
            "catalogue without rebuilding it"
        ),
    )
    parser.add_argument(
        "--entry",
        help="repository function name (exact or unique authored-name suffix)",
    )
    args = parser.parse_args()
    ids = parse_id_spec(args.ids)
    wanted = set(parse_id_spec(args.stages)) if args.stages else set()

    def run(stage: int) -> bool:
        return not wanted or stage in wanted

    adv = None
    if args.compilation_product is not None:
        if args.entry is not None:
            parser.error("--entry is not used with --compilation-product")
        product_snapshot = load_compilation_product(args.compilation_product)
        print(f"product:    {product_snapshot['root']}")
        healthy = stage_0_compilation_product(product_snapshot)
        print()
        print(
            "VERDICT: catalogue frontier is clean"
            if healthy else
            "VERDICT: catalogue has terminal or still-pending frontiers"
        )
        return 0

    unit_snapshot = None
    repository_path = args.repository_ssa
    if args.compilation_unit is not None:
        unit_snapshot = load_compilation_unit(args.compilation_unit)
        print(f"unit:       {unit_snapshot['root']}")
        # Stage 0 is the inlet itself, not an optional SSA stage. Without it,
        # ``--stages 2`` on an unpublished worker would suppress the only
        # evidence explaining why Stage 2 cannot run.
        stage_0_compilation_unit(unit_snapshot)
        print()
        repository_path = unit_snapshot.get("repository")
        if repository_path is None:
            if unit_snapshot["state"] == "running":
                verdict = (
                    "meta-compilation is still running; no repository SSA yet"
                )
            elif unit_snapshot["state"] == "source-only":
                verdict = (
                    "source-only unit is terminal and safely retained"
                )
            else:
                verdict = "meta-compilation stopped before repository SSA"
            print("VERDICT:", verdict)
            return 0

    if repository_path is not None:
        selected_entry = args.entry or (
            None if unit_snapshot is None
            else unit_snapshot.get("qualified_name")
        )
        module, fn, artifact, selected, receipt_note = load_repository_ssa(
            repository_path, selected_entry,
        )
        module_functions = dict(module.functions)
        print(f"repository: {repository_path.resolve()}")
        print(f"entry:      {selected}")
        print(f"receipt:    {receipt_note}\n")
    else:
        if args.entry is not None:
            parser.error("--entry requires --repository-ssa or --compilation-unit")
        from src.compiler.symbolic_fluid_native_runtime import (
            compile_native_symbolic_fluid_advance,
        )

        build = ROOT / "build" / "diagnose-tmp"
        build.mkdir(parents=True, exist_ok=True)
        print(f"compiling into {build} ...\n")
        adv = compile_native_symbolic_fluid_advance(build)
        fn = adv.function
        artifact = adv.artifact
        module_functions = adv._module_functions or {}

    healthy = True
    if run(1):
        healthy &= stage_1_shortfalls(artifact)
        print()
    if run(2):
        healthy &= stage_2_ssa_wellformed(fn)
        print()
    if run(3):
        healthy &= stage_3_inplace_safety(fn, module_functions)
        print()
    if run(4) and ids:
        if adv is None:
            stage_4_repository_boundary(fn, ids)
        else:
            stage_4_observability(adv, ids)
        print()
    if run(5):
        stage_5_influence(fn, ids)
        print()
    print("VERDICT:", "no stage proved inconsistent" if healthy
          else "a stage above is provably inconsistent -- start there")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
