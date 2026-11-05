2026-10-24 LIFTED TAIL JUMPS SPLIT FROM #904 (MEASURED, WSL Ubuntu24):
Cherry-picked the tail-jump commit from #904 onto main c8bf22db as one
machine-lifting item. The original #904 branch now reverts this commit and
retains only its gate-script fix. The focused
tests/test_lifted_machine_ssa_execution.py passed 23/23 in Ubuntu24 on the
merged #904 head before the split. The first Linux gate attempt stopped during
part-ab collection because the new venv lacked setuptools; that is an
environment failure, not a gate comparison. The full Linux gate, machine
suites, and fresh-main comparison remain unverified for this split branch.
No authored-Python cNNN corpus program exercises a lifted binary transfer;
the focused machine test is the present regression. Do not merge as green.
