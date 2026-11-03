2026-11-03 Empty constructor-default audit on mobile dependencies (MEASURED):

The iOS mobile workflow installs requirements.txt, ziglang and pytest, but not
llvmlite.  At failed run 41678794048's 4aeec0b source and current main
2042cecd, calling the final numerical-constructor-default audit with an empty
IdentityBook raised ModuleNotFoundError from
c_backend_repository_ssa_reference.  The audit loaded the canonical C
repository SSA before checking whether either
NUMERICAL_CONSTRUCTOR_CODE_REFERENCE or NUMERICAL_CONSTRUCTOR_DEFAULT had any
rows.

The audit now reads those two concordance pages first and returns only when
both are empty.  If either page has evidence, it still loads the repository
SSA and runs the existing declaration, ancestry, admission, operand and stored
dtype checks.  No finding is suppressed for present evidence, no dtype is
selected, and no compiler or backend default changes.

In an isolated environment with requirements.txt, ziglang 0.16.0 and pytest,
and with llvmlite uninstalled, the exact workflow test command passed: 21
passed in 18.00 seconds.  The focused empty-page regression also passed in
that environment.  With llvmlite 0.46.0 installed, constructor-default and
numerical-schema tests passed together: 11 passed in 95.58 seconds.  Those
tests include a real omitted-default page, explicit-dtype absence, and forged
physical dtype, admission namespace and current dtype-operand refusals.

The test host is Linux, so the later Xcode/iPhoneOS link and IPA packaging
steps were not run.  No full bootstrap, compressor lowering, native parity,
deployment or dtype-frontier work was attempted.
