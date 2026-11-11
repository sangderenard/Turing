# Divider -> Nodus compile host bridge (2026-11-10)

> Updated 2026-11-11: site-only account/workspace wiring is now implemented in
> isolated source, not deployed. The direct bridge CLI below is retired; it now
> refuses dispatch. Read [the current storage/wiring contract](../../../docs/SITE_COMPILER_ARTIFACTS_2026-11-11.md).
> Runtime dispatch requires the existing account workspace ledger, external
> user-space jobs root and configured site worker. Earlier launch snippets and
> the “not applied” wiring section below are historical, not current instructions.

`examples/meowdoku/divider_nodus_bridge.py` drains `source_divider`'s lease queue one piece at a time through a Nodus contract port
into the Nodus `TuringCompileHost` (`nodus/include/turing_compile_host_tool.h`). Tests: `tests/test_meowdoku_divider_nodus_bridge.py`
(fake endpoint on an ephemeral loopback port, real 72-byte frames, no Nodus build, no compiler).

## What it does

1. `source_divider.claim_next` leases ONE attempt (the bridge never claims while its own lease owner holds an unexpired lease).
2. `request_for_piece` writes `<jobs_dir>/<id>.request.json` atomically (temp + `os.replace`) in the schema of
   `nodus/tools/turing_row_tool_compiler.py --request`: `row_entry`, ordered rank-1 `columns`, `backend` (c|llvm|fortran), `output_dir`,
   explicit `extraction_contract`, `turing_root`, `product_name`, `source_text`, `allow_uncompiled: false`.
3. `submit_and_wait` submits one contract `(id, service_id, 1, 0)` using nodus' `contract_service_peer.EnginePeer` (the producer side;
   the compile host is the host side) and waits for the Route frame whose `result.payload` is the `CompileStatus`.
4. The status is recorded with `source_divider.complete` / `refuse` on the claimed attempt row.

## File convention

| file | owner |
|---|---|
| `<jobs_dir>/<id>.request.json` | bridge, written BEFORE the Submit |
| `<jobs_dir>/<id>.divider.json` | bridge: which (job, piece, backend, generation) owns the id |
| `<jobs_dir>/<id>/` product dir, `compile-host.result.json` | Nodus host |

Contract id = first 8 bytes of `sha256(job_id NUL piece_id NUL backend NUL generation)`, masked to 63 bits, nonzero. `generation` is
the lease generation (the attempt number). Collisions are resolved by the sidecar: an id owned by another tuple, or whose request or
product name already exists unowned, is skipped by incrementing. Use a DEDICATED compile jobs dir; if it is also the divider's state
`jobs` dir a job id made only of digits could equal a contract id (the probe skips it, but keep them apart).

Batched only (the owner's "no scalar cores" rule): the bridge never produces a rank-0 request. Columns follow
`nodus/tests/turing_nodus_runtime_shape_crossing_test.py`: every column `rank: 1, shape: null` (batch length arrives with the Nodus
handle), the output `shape_from` the first input, plus a per-request `<id>.extraction-contract.yaml` (the explicit base contract with
`program_abi` values: storage span, float64, rank 1, `AbstractTensor`). The driver has no per-row application of a scalar function
(`turing_row_tool_compiler.py:303-316` lowers the source as written), so SCALAR-annotated pieces (`x: float -> float`) are NOT
expressible and are refused with that reason. Expressible: one module-level function whose formals and return are all annotated
`AbstractTensor` (f64 only), no `*args/**kwargs/keyword-only`, at least one formal, backend c|llvm|fortran. Not verified by a real compile.

Status mapping: 1/2/3 -> `complete`; 16 -> `refused` RequestMissing; 17 -> `error` CompileFailed; 18 -> `error` AdoptFailed.
19 Cancelled, unknown codes, timeout, connect failure: NOT recorded; the lease is left (`reclaim_expired` requeues it with the next
generation, hence a new contract id).

## Pointing it at a canvas host

```
python tools/run_canvas.py --acquire PREFIX:N --compile-host PREFIX:HOST_ID:JOBS_DIR     (nodus; being added by another worker)
python examples/meowdoku/divider_nodus_bridge.py run-once --db <state>/sql/accounts.db --jobs-dir JOBS_DIR \
    --source-root <state>/jobs --port <contract port the canvas prints> --service-id <engine id> [--backend llvm] [--timeout 600]
```
Prints one JSON line (`{"state":"idle"}` when nothing is queued). Set the negotiator's `loan_ticks` large enough to cover a compile.
Jobs must be submitted with backends from c/llvm/fortran (e.g. `backends=("llvm",)`).

## Server wiring (one line, not applied)

`server.py` calls `hook(job_id)` after an accepted upload when `sessions.divider_hook` is set:

```python
sessions.divider_hook = divider_nodus_bridge.make_hook(sessions.state, divider_nodus_bridge.NodusCompileRoute("127.0.0.1", PORT, SERVICE_ID, JOBS_DIR, NODUS_TOOLS, source_root=sessions.state.jobs_dir))
```
`run_pending(db, paths, route, max_pieces=1)` is the underlying function. The hook BLOCKS the request thread up to the timeout.

## Not done

- The attempt is recorded on the claimed row's own backend (stage text `nodus-row-tool: <Status>`); a row with backend
  `nodus-row-tool` needs `source_divider.BACKENDS`/`submit` to allow it. `claim_next` has no backend filter, so a claimed attempt on
  an unserved backend (wasm/js) is refused `BackendNotServed`; use bridge-only jobs or a separate db.
- Only single-function AbstractTensor pieces; no scalar or multi-rank columns, multi-function SCC pieces, or cross-file imports.
- No `product_sha256`/`api_path`, no recomposition.json lowerings, no lowering_library registration for Nodus products.
- Never run against a live canvas host; no authentication beyond loopback; no restart durability beyond the host's receipt check.
- `run_canvas.py --compile-host` is not verified here.
