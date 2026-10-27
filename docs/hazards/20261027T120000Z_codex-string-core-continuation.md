2026-10-27 String Core continuation checkpoint (NOT ACCEPTED)

The full native persistent String Core manager has not passed. Component tests are evidence for individual fixes, not completion. The active goal remains full manager native acceptance with visible failures, supplied golden results, and Python/native timing.

## Checkouts and pushed changes

- Canonical: `C:/dev/Powershell/turing`, main. Mandatory AGENTS.md verbatim concordance quotations committed and pushed as `2ee30597`.
- Repair: `C:/dev/Powershell/turing-return-spans`, branch `codex/mapping-child-sequence-views`. All compiler/test changes committed and pushed through `b85b5951`.
- Clean integration: `C:/dev/tg-child-review`, detached at `b85b5951`.
- Draft PR: https://github.com/sangderenard/Turing/pull/949 . Updated description reports run 11 pending and no full-manager success.
- Canonical untracked `.claude/` contains a separate worktree and `out/` contains generated AbstractTensor compilation products. Neither is this task's source change; neither was added to Git. Preserve them.

## Last complete full-manager failure

Run 10 on clean `b0ea8040` failed after 502.63 seconds, exit 1, with nine final concordance findings. No native library was produced.

1. `repeated_manager` value239 was a receiver field-slot read in function_exit, while an earlier branch Call consumed it. Structural output recovery had recreated private-region Mul43 from a dependency closure and caused the slot reader to be placed at the duplicate. Recovery now uses the exact boundary roots at its original writer. No load hoist or dominance patch was added.
2. Eight ordinary VALUE_TYPE_ABI findings used stale or absent direct source-class evidence: six forked validate_batch_envelope parameter1 rows plus validate parameter4 and valid_atom parameter0. Readers now consume the current exact source-class projection; ABI publication includes its own matching current source-class Ref. Per-call lexical class scopes remain separate. Descriptor-backed bool Loads consume the exact sequence contract and matching target ABI.

Run 10 failure: `C:/dev/Powershell/string_core/build/native_manager_declared_product_10/units/repeated_manager/failure.json`.
Complete identity log: `C:/dev/tg-child-review/artifacts/identity_logs/repeated_manager.20261027T055555.failed.log.xz`.

## Current full run: do not duplicate it

Run 11 is ACTIVE, root exec session `72277`, worker PID36196. Latest observed progress: compiler stage11/11 finalization, elapsed395.71s. No failure.json existed at that observation. It has not been watched finish. Poll the existing session before launching another run. The process may finish after this report was written.

Output: `C:/dev/Powershell/string_core/build/native_manager_declared_product_11`.
Source: `C:/dev/Powershell/string_core/build/native_manager_acceptance/manager_workload.py`.
Invocation, workdir `C:/dev/tg-child-review`:

```powershell
$env:PYTHONPATH='C:\dev\Powershell\string_core\build;C:\dev\Powershell\string_core;C:\dev\tg-child-review'; python -u -m tools.compile_project_catalogue --source C:\dev\Powershell\string_core\build\native_manager_acceptance\manager_workload.py --entry repeated_manager --output C:\dev\Powershell\string_core\build\native_manager_declared_product_11 --jobs 1 --emit-native --extraction-contract C:\dev\Powershell\string_core\build\native_manager_acceptance\extraction.yaml --verification-cases C:\dev\Powershell\string_core\build\native_manager_acceptance\cases.json
```

The CLI supplies native verification and timing after successful admission/build. On a final source refusal, the new generic product path preserves `rejected-repository-ssa-{uuid}.pkl`, its SHA256, and explicit nonlinkable status in failure.json and terminal summary. Inspect that exact module/book rather than running another long lowering to recover evidence. This is diagnostic retention, never accepted source or native admission.

## Verified component results

- Latest integrated gate: 92 passed in48.33s, no deselection. Includes region-owned reads, frame aliases, multicall class projections, forwarded class views, input conversions, ABI receipts, operation shapes, rejected artifacts and source admission.
- Real two-call String Core source: zero findings, result7 on C/LLVM/Fortran, total120.54s.
- Real boolean-offset source: rejects envelope4, empty outputs and unchanged epoch, then result7 on all three backends, total62.20s.
- Repeated int/bool/int repro: zero findings and result2 on all three backends.
- Loop/mutation/late-read repro: results0/1000004/5000020 on C/LLVM/Fortran.
- Rejected artifact/admission gate:9 passed. Original refusal is preserved even if diagnostic storage fails.
- See preceding individual hazard receipts for exact commands, snapshots/hashes and writer observations. These are not full-manager acceptance.

## Acceptance and next actions

The full workload keeps one Vocabulary lifetime inside each native invocation, repeatedly mints Unicode and byte aggregates, verifies deduplication, checks five invalid requests, checks unchanged epoch for rejected requests, and checks domain/exact_atoms/cover_tokens lengths. It does not prove persistence between host invocations.

Cases rounds0/1/17/300/3000 must return0/7000175/119003791/2100321600/21027516000, respectively. Largest case performs9005 requests. Existing harness requests one warmup and three timed repetitions. Python goldens passed previously; rounds3000 single observed Python time0.1891165s is not a native comparison or speedup.

1. Watch run11 finish and report its actual refusal or native verification results immediately.
2. If it refuses, use the retained rejected SSA/book and complete log to identify the exact conflicting records and original writer; make a fast public-source repro before editing. Use only the concordance, with full owner and exact Ref. No source-name exceptions, dtype guesses, parallel ledgers, late repair passes or guard weakening.
3. If it passes, collect native library/API artifacts and all golden/timing rows. Full manager C/LLVM acceptance remains unproven: the project CLI currently builds Fortran only. Existing `ProjectCompilationProduct.load_repository_ssa('repeated_manager')` loads the accepted module without source lowering. Use existing C/LLVM artifact/runtime APIs on that module; do not create a replacement FFI or harness.
4. The C-lane read-only harness follow-up hit an account usage limit. Earlier lanes completed their bounded fixes and are idle. Use followup_task to wake a lane; send_message alone does not wake completed agents. Do not assume agents are still working.
5. Keep PR draft and goal active until measured full acceptance. Do not call this done based on component tests.

Separate known red frontier: bootstrap serial installation now reaches the Metrics record-union arm without a descriptor-backed side-by-side frame; the earlier shape-source loss is fixed. No dt_system edits. Sequence-helper suite retains15 failing test names measured unchanged on clean1c6a04ad. These are not successes and are not silently excluded.
