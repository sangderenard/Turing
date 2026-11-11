2026-11-11 PHOTO RETRY ACCOUNT TRANSACTION (MEASURED, Windows):

Accounts.claim caught duplicate asset ownership INSERT failures without rollback.
The persistent Accounts connection retained a write transaction. A second login
then failed in login_guard.record with sqlite3.OperationalError: database is locked;
the normal operator script audit writer was blocked by the same transaction.

Reproduced on a disposable database: first claim succeeds, repeated claim returns
False with in_transaction=True, and a separate writer times out. Fix 8a362b10
rolls back only the failed duplicate INSERT, preserving first-owner semantics.
Regression covers same-owner and other-owner repeats, unchanged ownership,
independent writer admission, and a later distinct claim. Canonical focused
accounts/tile/photo/guest-artifact suite: 35 passed (one cffi deprecation warning).

The code fix does not release a transaction already held by a running old server.
Normal authenticated cancellation and operator Status.ps1 were attempted and
reported blocked, not bypassed. A code deployment/recovery is still required.
No credentials, session records, access rules, or live database files were edited.
