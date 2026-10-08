"""A value read after a loop on the ``break`` path rides the loop's exit edge.

The retry loop of ``step_with_dt_control_used`` in miniature (the one
``probe_dead_structural_dominance.py`` lowers), plus a ``break``:

* ``break-on-guard``     -- ``if dt < 0.05: break`` at the loop's own level.
* ``break-in-arm``       -- ``break`` ending the retry ``if`` arm after the
  arm rebinds ``dt``.
* ``break-else-arm`` / ``break-body-arm`` / ``break-for-else-arm`` -- the
  ``break`` ends one arm and the OTHER arm rebinds ``dt`` (the reducer lets
  that arm's bindings stand after the ``if`` unmerged, so the break must
  leave from its own arm or no later value dominates the exit edge:
  ``break-edge-value``).
* ``body-value-after``   -- ``tried = dt`` seeds two names from one pre-loop
  value; ``tried`` is rebound in the body and read after the loop (the site
  table keys each binding by its carried pair, not the shared seed).
* ``body-fresh-after``   -- a name first bound in the body (``accepted``, no
  pre-loop identity) in a ``while True`` retry loop whose ``else`` breaks.

OPEN_VARIANTS are reported, never counted: ``break-nested-guard`` is a
``break`` under an ``if`` nested in an arm, whose predicate is computed in
that arm; the loop-level lexical guard reads it where it does not dominate.

    python -u tools/compiler_probes/probe_break_edge_value.py [c|llvm|both]

Each variant goes through ``lower_ast_source_to_ssa`` under the real
``program_extraction.yaml`` contract, must have no operand whose definition
fails to dominate its use and no identity-concordance finding, is emitted on
the C and LLVM lanes and run natively against the Python function; results
must be bit-exact.
"""
from __future__ import annotations

import pathlib
import pickle
import subprocess
import sys
import tempfile
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "compiler_probes"))

PRELUDE = '''
import math


def advance(x):
    return x * 0.5 - 1.0

'''

_INPUTS = ((3.0, 1.0, 0.7), (3.0, 0.2, 0.5), (1.0, 0.4, 0.3), (9.0, 0.3, 0.02))

VARIANTS = {
    "break-on-guard": ('''
def root(x, dt, limit):
    step = 0
    while step < 4:
        v = advance(x)
        if (not math.isfinite(v) or v <= 0.0 or v >= dt):
            dt = dt * 0.5
        if limit is not None and math.isfinite(float(limit)) and float(limit) > 0.0:
            dt = min(dt, limit)
        if dt < 0.05:
            break
        step = step + 1
    return dt
''', _INPUTS),
    "break-in-arm": ('''
def root(x, dt, limit):
    step = 0
    while step < 4:
        v = advance(x)
        if (not math.isfinite(v) or v <= 0.0 or v >= dt):
            dt = dt * 0.5
            break
        if limit is not None and math.isfinite(float(limit)) and float(limit) > 0.0:
            dt = min(dt, limit)
        step = step + 1
    return dt
''', _INPUTS),
    "break-else-arm": ('''
def root(x, dt, limit):
    step = 0
    while step < 4:
        v = advance(x)
        if (not math.isfinite(v) or v <= 0.0 or v >= dt):
            dt = dt * 0.5
        else:
            break
        if limit is not None and math.isfinite(float(limit)) and float(limit) > 0.0:
            dt = min(dt, limit)
        step = step + 1
    return dt
''', _INPUTS),
    "break-body-arm": ('''
def root(x, dt, limit):
    step = 0
    while step < 4:
        v = advance(x)
        if math.isfinite(v) and v > 0.0 and v < dt:
            break
        else:
            dt = dt * 0.5
        step = step + 1
    return dt
''', _INPUTS),
    "break-for-else-arm": ('''
def root(x, dt, limit):
    acc = 0.0
    for step in range(5):
        v = advance(x)
        acc = acc + dt
        if (not math.isfinite(v) or v <= 0.0 or v >= dt):
            dt = dt * 0.5
        else:
            break
    return dt + acc
''', _INPUTS),
    "body-value-after": ('''
def root(x, dt, limit):
    step = 0
    tried = dt
    while step < 4:
        v = advance(x)
        tried = dt * 2.0 + v
        if (not math.isfinite(v) or v <= 0.0 or v >= dt):
            dt = dt * 0.5
        if limit is not None and math.isfinite(float(limit)) and float(limit) > 0.0:
            dt = min(dt, limit)
        if dt < 0.05:
            break
        step = step + 1
    return dt + tried
''', _INPUTS),
    "body-fresh-after": ('''
def root(x, dt, limit):
    retries = 0
    while True:
        v = advance(x)
        accepted = v * 2.0
        if (not math.isfinite(v) or v <= 0.0 or v >= dt):
            dt = dt * 0.5
        else:
            break
        retries = retries + 1
        if retries >= 4:
            break
    return dt + accepted
''', _INPUTS),
}

# Known open: a ``break`` guarded by an ``if`` nested in an arm whose predicate
# is computed in that arm (see the module docstring).  Reported, not counted.
OPEN_VARIANTS = {
    "break-nested-guard": ('''
def root(x, dt, limit):
    step = 0
    while step < 4:
        v = advance(x)
        if (not math.isfinite(v) or v <= 0.0 or v >= dt):
            dt = dt * 0.5
            if dt < 0.1:
                break
        if limit is not None and math.isfinite(float(limit)) and float(limit) > 0.0:
            dt = min(dt, limit)
        step = step + 1
    return dt
''', _INPUTS),
}

RUNNER = '''
import pickle, sys
import numpy as np
with open(sys.argv[1], "rb") as stream:
    artifact, feeds_by_probe, result_id = pickle.load(stream)
produced = []
for feeds in feeds_by_probe:
    result = artifact.prepare_execution({
        value_id: np.array([value], dtype="float64")
        for value_id, value in feeds.items()
    }).run()
    produced.append(float(np.asarray(result.buffers[result_id]).reshape(-1)[0]))
print(repr(produced))
'''


def main() -> int:
    import numpy as np

    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    lanes = sys.argv[1] if len(sys.argv) > 1 else "both"
    contract = REPO / "extraction_contracts" / "program_extraction.yaml"
    from probe_module_dominance import module_dominance_violations
    from src.compiler.identity_concordance import concordance_report

    failures = []
    open_failures = []
    every = {**VARIANTS, **OPEN_VARIANTS}
    for label, (body, probes) in every.items():
        bucket = open_failures if label in OPEN_VARIANTS else failures
        source = PRELUDE + body
        namespace: dict = {}
        exec(compile(source, "<authored>", "exec"), namespace)
        authored = namespace["root"]
        expected = [float(authored(*probe)) for probe in probes]
        prefix = "breakedge_" + label.replace("-", "_")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                module, outputs, _exports = lower_ast_source_to_ssa(
                    source, "root", name=prefix, extraction_contract=contract,
                    progress=lambda message: None,
                )
        except Exception as error:  # noqa: BLE001 - the probe reports it
            print(f"{label:18s} LOWER FAIL {type(error).__name__}: "
                  f"{str(error)[:300]}")
            bucket.append(label)
            continue
        violations = module_dominance_violations(module)
        dominance = sum(len(items) for items in violations.values())
        findings = concordance_report(module).splitlines()[0]
        print(f"{label:18s} dominance violations={dominance}; {findings}")
        if dominance or not findings.endswith(" 0 finding(s)"):
            bucket.append(f"{label}/structure")
        root = module.functions[f"{prefix}__root"]
        parameters = dict(root.metadata["parameter_names"])
        result_id = outputs[root.name][0].id
        feeds_by_probe = [
            {parameters[n]: v for n, v in zip(("x", "dt", "limit"), probe)
             if n in parameters}
            for probe in probes
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
            scratch = pathlib.Path(scratch)
            for lane in (("c", "llvm") if lanes == "both" else (lanes,)):
                if lane == "c":
                    artifact = emit_ssa_module_to_c(module, root.name)
                    complete, why = artifact.complete, artifact.shortfalls
                    if complete:
                        artifact.compile(scratch / prefix)
                else:
                    from src.compiler.ssa_llvm_backend import (
                        compile_artifact, emit_ssa_function_to_llvm,
                        prepare_artifact_execution,
                    )
                    llvm = emit_ssa_function_to_llvm(
                        module, root.name, entry_name=prefix)
                    complete, why = not llvm.shortfalls, llvm.shortfalls
                    if complete:
                        native = compile_artifact(
                            llvm, directory=scratch / prefix, optimization="O0")
                if not complete:
                    print(f"{label:18s} {lane:4s} EMIT FAIL {why}")
                    bucket.append(f"{label}/{lane}")
                    continue
                if lane == "c":
                    payload = scratch / f"{lane}.pkl"
                    payload.write_bytes(pickle.dumps(
                        (artifact, feeds_by_probe, result_id)))
                    done = subprocess.run(
                        [sys.executable, "-c", RUNNER, str(payload)],
                        capture_output=True, text=True, timeout=60)
                    if done.returncode:
                        print(f"{label:18s} {lane:4s} RUN FAIL "
                              f"{(done.stdout + done.stderr)[-400:]}")
                        bucket.append(f"{label}/{lane}")
                        continue
                    produced = eval(done.stdout.strip().splitlines()[-1])
                else:
                    produced = []
                    for feeds in feeds_by_probe:
                        run = prepare_artifact_execution(native, {
                            k: np.array([v], dtype="float64")
                            for k, v in feeds.items()}).run()
                        produced.append(float(
                            np.asarray(run.buffers[result_id]).reshape(-1)[0]))
                match = produced == expected
                print(f"{label:18s} {lane:4s} "
                      f"{'MATCH' if match else 'MISMATCH'} "
                      f"expected={expected} produced={produced}")
                if not match:
                    bucket.append(f"{label}/{lane}")
    if open_failures:
        print("OPEN (not counted): " + "; ".join(open_failures))
    print("FAILED: " + "; ".join(failures) if failures else "all MATCH")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
