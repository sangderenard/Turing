# Lichess status must not enroll an account

Measured on cloud Linux from main `f4a8503c4249e046bbec97daf5c22152a5ebfb43`.

`examples/meowdoku/lichess.py --status` called `LichessBot.connect()` before
checking the flag. For an account without title `BOT`, that method posts
`/api/bot/account/upgrade`, which the client documents as irreversible. The
status command also constructed `SelfPlay` unless `--random` was present.

The status branch now uses the existing `Api.get("/api/account")` and returns
before constructing a player or bot. Its JSON fields are unchanged. Normal
play still uses the existing connection and enrollment path.

## Verification

All new CLI tests mock the credential source and HTTP API and reject any
attempt to reach `urllib.request.urlopen`. No real token or account was used.

- Before the code change, `tests/test_meowdoku_lichess_status.py`: 6 failed,
  5 passed. All three non-bot status cases recorded the upgrade POST.
- After the change, the same file: 11 passed in 0.17 seconds. Cases cover
  bot/non-bot accounts, `--random`, `--challenge`, missing credentials, the
  existing play enrollment behavior, and normal CLI startup.
- `tests/test_meowdoku_dt_session.py`: 3 passed in 5.06 seconds.
- `tests/test_meowdoku_play.py`: 1 passed in 0.26 seconds.
- `tests/test_meowdoku_program.py`: 6 passed, 2 failed on both the candidate
  and an untouched worktree at the same base. The control-contract case
  stopped on missing `tqdm`; the batched-generator case stopped in the
  optional Torch backend with `NoneType` lacking `Tensor`. These are
  environment-limited results, not a green game-suite claim. An earlier
  sparse-checkout run also lacked `extraction_contracts`; those tracked
  files were materialized before the final comparison.
- `py_compile` for the changed source and new test, and `git diff --check`:
  passed.

Commands used `PYTHONPATH=.:examples`, Python 3.12, and `python -m pytest`
with `-q --tb=short`, bounded per test file. The environment has pytest 9.2.0,
NumPy, NetworkX 3.7 and llvmlite 0.50.0. It reports that the optional Nodus
tensor core is disconnected and uses the Python backends; no native or
compiler behavior is claimed. The full repository gate, live Lichess play,
account enrollment, browser/GPU tests, and Wasm compilation were not run.
