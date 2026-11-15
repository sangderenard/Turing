# Central checkout: branches made and committed by the service

An account changes a core repository without running Git and without write
access to the repository. The service makes the branch, holds the working
folder, and commits; the existing review gate lands it.

Source: `examples/meowdoku/central_checkout.py`. Routes: `GET /api/checkouts`
and `POST /api/checkouts/<operation>`, signed in with `dev_ui`. The SpeakToMe
MCP offers them as `checkouts_list` and `checkout_<operation>`.

## The flow

1. `checkout_open {repository, branch}` registers the branch to the account
   (the same SQL record `reviews_checkout` makes), creates it from the remote's
   default branch inside the live repository, and adds a Git worktree at
   `<state>/content/checkouts/<user id>/<repository>/<branch, / as __>`.
   Opening a branch the account already has returns the same checkout.
2. `checkout_files`, `checkout_read`, `checkout_write`, `checkout_remove`
   work on text files in that folder. `checkout_status` and `checkout_diff`
   show the change against the last commit.
3. `reviews_identity_set` declares the account's Git author once.
   `checkout_commit {message}` stages everything and commits as that author.
   Hooks run; nothing is skipped.
4. `reviews_submit` with the commit's full head submits it. The lander finds
   the branch in the live repository and lands it as before (fast-forward
   only).
5. `checkout_close` removes the working folder once everything is committed.
   The branch stays the account's.

## What is refused

- A path that is empty, absolute, contains `..`, names `.git` in any case,
  ends in a dot or space, contains `:` or `<>"|?*` or control characters, or
  names a Windows device (`con`, `aux`, `nul`, `com1`...).
- Any link or junction on the way to a file; the resolved path must stay in
  the working folder.
- Another account's checkout (404) or branch (409); a branch that already
  exists and is not the caller's; a default branch.
- Binary or non-UTF-8 files when reading; files over 2 MB; listings over 2000
  entries; diffs are cut at 400 kB.
- Closing with uncommitted changes; committing with nothing to commit or
  without a declared identity.

Every call, refused or not, is a row in the `checkout_calls` table (the last
20000 are kept).

## Not yet

- The working folders are not counted against the account's workspace quota
  (`workspace_accounting.py`).
- Commands cannot be run in a checkout, so tests and builds still need an
  operator.
- Text files only.
