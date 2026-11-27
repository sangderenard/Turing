"""Central checkout: a branch and its working folder, made and kept by the service for an account, over REST and MCP.

An account never runs Git in the core. After the SQL checkout (branch_review.register: the branch is this account's),
the service makes the branch from the remote's default branch in the core repository and a working folder for it in
the state root (``<state>/content/checkouts/<user id>/<repository>/<branch>``), a Git worktree of the core repository.
The account then works through declared operations only, each a name and typed arguments, never a command line:

    open    {repository, branch}            register the branch to yourself, make it and its working folder
    files   {repository, branch, path?}     list one folder of the working folder
    read    {repository, branch, path}      read one text file
    write   {repository, branch, path, text}   create or replace one text file
    remove  {repository, branch, path}      delete one file
    edit    {repository, branch, path, old, new, count?}   replace exact text (count times; once by default)
            {repository, branch, patch}     apply a unified diff (git diff, diff -u) or a Codex patch (*** Begin Patch)
    status  {repository, branch}            what differs from the branch's last commit
    diff    {repository, branch, path?}     the change in the tracked files, as a unified diff
    commit  {repository, branch, message}   commit everything, as the author identity this account declared
    merge   {repository, branch, from, message?}   bring another branch into this one (see [MERGE])
            {repository, branch, abort: true}      give up an unfinished merge
    close   {repository, branch}            remove the working folder (it must be committed); the branch stays

[MERGE] The one Git act beyond committing that an account reaches, and still a declared operation, not a command line.
``from`` must be the repository's default branch or a branch registered in this repository (branch_review: a central
checkout or a registered branch, anyone's), never this branch itself. The working folder must be committed first and
the account's author identity declared. A merge that applies is committed at once (both parents, the declared
identity). A merge that conflicts stays unfinished in the working folder: ``status`` names the commits it brings in
(``merging``) and the conflicted files; the account resolves them with ``edit`` or ``write`` and ends the merge with
``commit``, which refuses while any conflicted file still holds a conflict marker line; or gives it up with
``abort``. ``close`` refuses an unfinished merge. The merge is then an ordinary commit of this branch: it is
submitted, reviewed and landed like any other (it is how a review bundle's meta review is made, review_bundles.py).

[LOAN] Opening a turing or speaktome checkout also lends its owner a server for the branch (branch_servers.py); the answer
says which (``server``), or why there is none. Closing the checkout ends the loan.

Submitting is the existing review (POST /api/reviews {repository, branch, head}): the branch is in the core
repository, where the lander looks for it, so landing is unchanged. The cleaner removes a landed branch's worktree.

Only this account's own checkouts are reachable. A path is relative to the working folder; it may not leave it, name
``.git``, pass through a link or junction, or use a name Windows treats specially. Every call is recorded
(``checkout_calls``). The working folders belong to the service account: nobody else's Windows account writes them.
"""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path

from meowdoku import branch_gate, branch_review, branch_servers

MAX_FILE_BYTES = 10_000_000
MAX_LISTING = 2000
MAX_DIFF_BYTES = 400_000
MAX_MESSAGE = 5000
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
OPERATIONS = ("open", "files", "read", "write", "remove", "edit", "status", "diff", "commit", "merge", "close")
MAX_BRANCH = 200
CONFLICT_MARKERS = ("<<<<<<< ", ">>>>>>> ", "<<<<<<<\n", ">>>>>>>\n")

DDL = (
    "CREATE TABLE IF NOT EXISTS checkouts (repository TEXT NOT NULL, branch TEXT NOT NULL, user_id INTEGER NOT NULL, "
    "username TEXT, path TEXT NOT NULL, base TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY (repository, branch))",
    "CREATE TABLE IF NOT EXISTS checkout_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, user_id INTEGER, "
    "username TEXT, repository TEXT, branch TEXT, operation TEXT, path TEXT, bytes INTEGER, ok INTEGER NOT NULL, "
    "error TEXT)",
)
CALLS_KEEP = 20000


class CheckoutError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    branch_review.ensure(db)
    for statement in DDL:
        db.execute(statement)


def _git(path, *args, check=True) -> str:
    try:
        return branch_gate.git(path, *args, check=check)
    except branch_gate.GateError as error:
        raise CheckoutError(str(error), 409) from None


def _repository(policy, repository):
    declared = policy["repositories"].get(repository) if isinstance(repository, str) else None
    if declared is None:
        raise CheckoutError("repository must be one the policy names: " + ", ".join(sorted(policy["repositories"])), 404)
    return Path(declared["path"]), declared["default"], policy.get("remote", "origin")


def _row(db, repository, branch):
    row = db.execute("SELECT repository, branch, user_id, username, path, base, created_at FROM checkouts "
                     "WHERE repository = ? AND branch = ?", (repository, branch)).fetchone()
    keys = ("repository", "branch", "user_id", "username", "path", "base", "created_at")
    return None if row is None else dict(zip(keys, row))


def _mine(db, *, user_id, repository, branch) -> dict:
    """This account's open checkout of that branch, or a refusal that says so."""
    row = _row(db, repository, branch) if isinstance(repository, str) and isinstance(branch, str) else None
    if row is None or row["user_id"] != user_id:
        raise CheckoutError("you have no open checkout of that branch; open it first", 404)
    if not Path(row["path"]).is_dir():
        raise CheckoutError("the working folder of that checkout is gone; open it again", 410)
    return row


def _is_link(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def target(folder, relative, *, allow_root=False) -> Path:
    """The file or folder ``relative`` names inside ``folder``, or a refusal. Never outside it, never .git, never through
    a link or junction, never a name Windows reads as a device or a stream."""
    if not isinstance(relative, str) or len(relative) > 400:
        raise CheckoutError("path must be a relative path of at most 400 characters")
    cleaned = relative.replace("\\", "/").strip("/")
    folder = Path(folder)
    if not cleaned:
        if allow_root:
            return folder
        raise CheckoutError("path must name a file")
    parts = cleaned.split("/")
    for part in parts:
        if part in ("", ".", "..") or part.casefold() == ".git" or part != part.rstrip(" .") or ":" in part \
                or part.split(".")[0].upper() in RESERVED or any(ord(c) < 32 or c in '<>"|?*' for c in part):
            raise CheckoutError(f"path {relative!r} is not allowed here")
    current = folder
    for part in parts:
        current = current / part
        if _is_link(current):
            raise CheckoutError(f"path {relative!r} passes through a link")
    resolved, root = current.resolve(strict=False), folder.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise CheckoutError(f"path {relative!r} leaves the working folder")
    return current


def _record(db, who, repository, branch, operation, path=None, size=None, error=None) -> None:
    cursor = db.execute("INSERT INTO checkout_calls (at, user_id, username, repository, branch, operation, path, bytes, ok, "
                        "error) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (time.time(), who.get("user_id"), who.get("username"), repository, branch, operation,
                         None if path is None else str(path)[:400], size, int(error is None),
                         None if error is None else str(error)[:500]))
    db.execute("DELETE FROM checkout_calls WHERE id <= ?", (cursor.lastrowid - CALLS_KEEP,))


def _describe(row, head) -> dict:
    return {"repository": row["repository"], "branch": row["branch"], "base": row["base"], "head": head,
            "opened_at": row["created_at"], "owner": row["username"]}


def open_checkout(db, policy, root, *, user_id, username, repository, branch) -> dict:
    """Register the branch to this account (the SQL checkout) and make the branch and its working folder."""
    repo, default, remote = _repository(policy, repository)
    defaults = {declared["default"] for declared in policy["repositories"].values()}
    if not isinstance(branch, str) or branch in defaults:
        raise CheckoutError("choose a branch name of your own, not a default branch")
    held_before = branch_review.owner(db, repository, branch)
    exists = bool(_git(repo, "rev-parse", "-q", "--verify", f"refs/heads/{branch}", check=False)) or \
        bool(_git(repo, "rev-parse", "-q", "--verify", f"refs/remotes/{remote}/{branch}", check=False))
    if exists and (held_before is None or held_before["user_id"] != user_id):
        raise CheckoutError("that branch already exists and is not registered to you; choose a new name", 409)
    try:
        branch_review.register(db, user_id=user_id, username=username, repository=repository, branch=branch,
                               known=list(policy["repositories"]))
    except branch_review.ReviewError as error:
        raise CheckoutError(error.message, error.status) from None
    row = _row(db, repository, branch)
    if row is not None and Path(row["path"]).is_dir():
        return _describe(row, _git(row["path"], "rev-parse", "HEAD"))
    folder = Path(root) / str(int(user_id)) / repository / branch.replace("/", "__")
    if folder.exists():
        raise CheckoutError("a folder is already at that checkout's place; ask an operator to look at it", 409)
    if not _git(repo, "rev-parse", "-q", "--verify", f"refs/heads/{branch}", check=False):
        _git(repo, "fetch", "--quiet", remote, default)
        _git(repo, "branch", "--no-track", branch, f"refs/remotes/{remote}/{default}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "--quiet", str(folder), branch)
    db.execute("INSERT OR REPLACE INTO checkouts VALUES (?,?,?,?,?,?,?)",
               (repository, branch, int(user_id), username, str(folder), default, time.time()))
    return _describe(_row(db, repository, branch), _git(folder, "rev-parse", "HEAD"))


def files(row, path="") -> dict:
    folder = Path(row["path"])
    place = target(folder, path or "", allow_root=True)
    if not place.is_dir():
        raise CheckoutError("path must name a folder", 404)
    entries = []
    for child in sorted(place.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold())):
        if child.name.casefold() == ".git" or _is_link(child):
            continue
        entries.append({"name": child.name, "kind": "folder" if child.is_dir() else "file",
                        "bytes": None if child.is_dir() else child.stat().st_size})
        if len(entries) >= MAX_LISTING:
            break
    return {"path": place.relative_to(folder).as_posix() if place != folder else "", "entries": entries,
            "truncated": len(entries) >= MAX_LISTING}


def read(row, path) -> dict:
    place = target(row["path"], path)
    if not place.is_file():
        raise CheckoutError("no such file", 404)
    if place.stat().st_size > MAX_FILE_BYTES:
        raise CheckoutError(f"the file is larger than {MAX_FILE_BYTES} bytes", 413)
    data = place.read_bytes()
    if b"\0" in data:
        raise CheckoutError("that is a binary file; only text is offered", 415)
    try:
        return {"path": path, "text": data.decode("utf-8"), "bytes": len(data)}
    except UnicodeDecodeError:
        raise CheckoutError("that file is not UTF-8 text", 415) from None


def write(row, path, text) -> dict:
    if not isinstance(text, str):
        raise CheckoutError("text must be a string")
    data = text.encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise CheckoutError(f"text is larger than {MAX_FILE_BYTES} bytes", 413)
    place = target(row["path"], path)
    if place.is_dir():
        raise CheckoutError("path names a folder", 409)
    place.parent.mkdir(parents=True, exist_ok=True)
    target(row["path"], path)                     # again, now that the folders exist: none of them may be a link
    with open(place, "wb") as handle:             # the bytes as given: no newline translation
        handle.write(data)
    return {"path": path, "bytes": len(data)}


def remove(row, path) -> dict:
    place = target(row["path"], path)
    if not place.is_file():
        raise CheckoutError("no such file", 404)
    place.unlink()
    return {"path": path, "removed": True}


# ---------------------------------------------------------------- edit: a change described, not a whole file sent
# Three forms, one operation. The service applies each; the account never runs Git, and a large file never needs a
# whole-file write (the MCP caps a text argument at 65536 characters; server.py alone is about 290 kB).
#   {path, old, new, count?}   exact replacement: old must occur exactly count times (1 when omitted)
#   {patch} in unified diff form (diff -u, git diff): every path is checked first, then git apply --check, then git apply
#   {patch} in Codex's form (*** Begin Patch ... *** End Patch): Add, Delete and Update File (with Move to), each hunk
#           found by its context lines in order
# Nothing is written until the whole edit is known to apply.

def _text_of(place: Path) -> str:
    if not place.is_file():
        raise CheckoutError(f"no such file: {place.name}", 404)
    if place.stat().st_size > MAX_FILE_BYTES:
        raise CheckoutError(f"the file is larger than {MAX_FILE_BYTES} bytes", 413)
    data = place.read_bytes()
    if b"\0" in data:
        raise CheckoutError("that is a binary file; only text is edited", 415)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise CheckoutError("that file is not UTF-8 text", 415) from None


def _put(folder, relative, text: str) -> int:
    data = text.encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise CheckoutError(f"the edited file would be larger than {MAX_FILE_BYTES} bytes", 413)
    place = target(folder, relative)
    if place.is_dir():
        raise CheckoutError(f"{relative} names a folder", 409)
    place.parent.mkdir(parents=True, exist_ok=True)
    target(folder, relative)                      # again, now that the folders exist: none of them may be a link
    with open(place, "wb") as handle:             # the bytes as computed: no newline translation
        handle.write(data)
    return len(data)


def _replace_exact(text: str, old: str, new: str, count: int) -> str:
    found = text.count(old)
    if found == 0 and "\r\n" in text and "\r\n" not in old:      # the file has Windows line endings, the edit does not
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        found = text.count(old)
    if found != count:
        raise CheckoutError(f"old text occurs {found} time(s), not {count}: give more of its surroundings so it is "
                            f"unique, or the count it really has", 409)
    return text.replace(old, new)


def _unified_paths(patch: str) -> list:
    """Every path a unified diff names, without its a/ or b/ prefix; refuses what this edit does not change."""
    paths, prefixed = [], []
    for line in patch.splitlines():
        if line.startswith("GIT binary patch") or line.startswith("Binary files "):
            raise CheckoutError("binary patches are not applied here; only text", 415)
        if line.startswith(("new mode ", "old mode ", "new file mode ", "deleted file mode ")) and \
                line.split()[-1] in ("120000", "160000"):
            raise CheckoutError("a patch may not make or remove a link or a submodule", 400)
        if line.startswith(("--- ", "+++ ", "rename from ", "rename to ", "copy from ", "copy to ")):
            name = line.split(" ", 1)[1] if line.startswith(("--- ", "+++ ")) else line.split(" ", 2)[2]
            name = name.split("\t", 1)[0].strip()
            if name.startswith('"'):
                raise CheckoutError("quoted (escaped) paths are not applied here", 400)
            if name == "/dev/null":
                continue
            if line.startswith(("--- ", "+++ ")):            # only these carry the a/ b/ prefixes git diff writes
                prefixed.append(name.startswith(("a/", "b/")))
                name = name[2:] if name.startswith(("a/", "b/")) else name
            paths.append(name)
    if not paths:
        raise CheckoutError("that is not a unified diff: no --- / +++ file lines", 400)
    return sorted(set(paths)), bool(prefixed) and all(prefixed)


def _apply_unified(folder, patch: str) -> dict:
    import subprocess
    paths, prefixed = _unified_paths(patch)
    for relative in paths:
        target(folder, relative)                  # every path, before Git sees the patch
    strip = "-p1" if prefixed else "-p0"
    for check in (True, False):
        arguments = ["git", "-C", str(folder), "apply", strip, "--whitespace=nowarn"] + (["--check"] if check else []) + ["-"]
        done = subprocess.run(arguments, input=patch, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if done.returncode != 0:
            raise CheckoutError(f"the patch does not apply{' (nothing was changed)' if check else ''}: "
                                f"{(done.stderr or done.stdout).strip()[-600:]}", 409)
    return {"files": paths, "format": "unified", "bytes": len(patch.encode("utf-8"))}


def _codex_actions(patch: str) -> list:
    """Codex's patch as [(kind, path, move_to, lines)], kind add | delete | update."""
    lines = patch.replace("\r\n", "\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[0].strip() != "*** Begin Patch" or lines[-1].strip() != "*** End Patch":
        raise CheckoutError("a Codex patch begins with *** Begin Patch and ends with *** End Patch", 400)
    actions, current = [], None
    for line in lines[1:-1]:
        for kind, marker in (("add", "*** Add File: "), ("delete", "*** Delete File: "), ("update", "*** Update File: ")):
            if line.startswith(marker):
                current = [kind, line[len(marker):].strip(), None, []]
                actions.append(current)
                break
        else:
            if current is None:
                raise CheckoutError(f"patch line outside any file: {line[:80]!r}", 400)
            if line.startswith("*** Move to: ") and current[0] == "update":
                current[2] = line[len("*** Move to: "):].strip()
            else:
                current[3].append(line)
    if not actions:
        raise CheckoutError("the patch names no file", 400)
    return actions


def _codex_update(text: str, body: list, name: str) -> str:
    """The hunks of one Update File, applied in order: each @@ line may name a line to find first; each hunk's context
    and removed lines (' ', '-') must follow it, exactly or ignoring trailing spaces; added lines ('+') replace them."""
    crlf = "\r\n" in text
    ends_open = text.endswith(("\n", "\r\n"))
    lines = text.replace("\r\n", "\n").split("\n")
    if ends_open:
        lines.pop()
    hunks, current = [], None
    for line in body:
        if line.startswith("@@"):
            current = {"anchor": line[2:].strip(), "lines": [], "at_end": False}
            hunks.append(current)
        elif line.strip() == "*** End of File":
            if current is not None:
                current["at_end"] = True
        else:
            if current is None:
                current = {"anchor": "", "lines": [], "at_end": False}
                hunks.append(current)
            if line[:1] not in (" ", "-", "+") and line != "":
                raise CheckoutError(f"{name}: a hunk line starts with ' ', '-' or '+', not {line[:40]!r}", 400)
            current["lines"].append(line if line else " ")
    cursor = 0
    for number, hunk in enumerate(hunks, 1):
        if hunk["anchor"]:
            found = next((i for i in range(cursor, len(lines)) if lines[i].strip() == hunk["anchor"].strip()), None)
            if found is None:
                raise CheckoutError(f"{name}: hunk {number}'s @@ line {hunk['anchor'][:80]!r} is not in the file", 409)
            cursor = found + 1
        old = [l[1:] for l in hunk["lines"] if l[:1] in (" ", "-")]
        new = [l[1:] for l in hunk["lines"] if l[:1] in (" ", "+")]
        if not old:
            at = len(lines) if hunk["at_end"] or not hunk["anchor"] else cursor
            lines[at:at] = new
            cursor = at + len(new)
            continue
        start = None
        for same in (lambda a, b: a == b, lambda a, b: a.rstrip() == b.rstrip()):
            starts = range(len(lines) - len(old), cursor - 1, -1) if hunk["at_end"] else range(cursor, len(lines) - len(old) + 1)
            start = next((i for i in starts if all(same(lines[i + k], old[k]) for k in range(len(old)))), None)
            if start is not None:
                break
        if start is None:
            raise CheckoutError(f"{name}: hunk {number}'s context and removed lines are not in the file after the "
                                f"previous hunk (nothing was changed)", 409)
        lines[start:start + len(old)] = new
        cursor = start + len(new)
    joined = ("\r\n" if crlf else "\n").join(lines)
    return joined + ("\r\n" if crlf else "\n") if ends_open or not lines else joined


def _apply_codex(folder, patch: str) -> dict:
    actions = _codex_actions(patch)
    writes, deletes = {}, []
    for kind, name, move_to, body in actions:
        place = target(folder, name)
        if kind == "add":
            if place.exists():
                raise CheckoutError(f"{name} already exists; Update File changes it", 409)
            if any(line[:1] != "+" for line in body if line != ""):
                raise CheckoutError(f"{name}: every line of an added file starts with '+'", 400)
            writes[name] = "\n".join(line[1:] for line in body) + "\n"
        elif kind == "delete":
            _text_of(place)
            deletes.append(name)
        else:
            new_text = _codex_update(writes.get(name) if name in writes else _text_of(place), body, name)
            if move_to:
                if target(folder, move_to).exists():
                    raise CheckoutError(f"{move_to} already exists", 409)
                writes[move_to] = new_text
                deletes.append(name)
            else:
                writes[name] = new_text
    total = sum(_put(folder, name, text) for name, text in writes.items())     # only now: everything applied in memory
    for name in deletes:
        if name not in writes:
            target(folder, name).unlink()
    return {"files": sorted(set(writes) | set(deletes)), "format": "codex", "bytes": total}


def edit(row, body) -> dict:
    folder = row["path"]
    patch = body.get("patch")
    if patch is not None:
        if not isinstance(patch, str) or not patch.strip():
            raise CheckoutError("patch must be the text of a patch")
        if len(patch.encode("utf-8")) > MAX_FILE_BYTES:
            raise CheckoutError(f"the patch is larger than {MAX_FILE_BYTES} bytes", 413)
        if patch.lstrip().startswith("*** Begin Patch"):
            return _apply_codex(folder, patch)
        return _apply_unified(folder, patch)
    path, old, new, count = body.get("path"), body.get("old"), body.get("new"), body.get("count", 1)
    if not isinstance(old, str) or not old or not isinstance(new, str):
        raise CheckoutError("give path, old (the exact text there now) and new; or a patch")
    if old == new:
        raise CheckoutError("old and new are the same text")
    if type(count) is not int or count < 1:
        raise CheckoutError("count must be a whole number of at least 1")
    place = target(folder, path)
    written = _put(folder, path, _replace_exact(_text_of(place), old, new, count))
    return {"path": path, "files": [path], "format": "exact", "replaced": count, "bytes": written}


def _porcelain(folder) -> list:
    """``git status --porcelain`` lines as Git wrote them: the shared helper strips the output, which would cut the
    leading space of the first line's two-letter state (" M base.txt" read as "M base.txt")."""
    import subprocess
    done = subprocess.run(["git", "-C", str(folder), "status", "--porcelain=v1", "--untracked-files=all"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise CheckoutError(f"git status failed: {(done.stderr or done.stdout).strip()[-300:]}", 409)
    return [line for line in done.stdout.split("\n") if line]


def status(row) -> dict:
    lines = _porcelain(row["path"])
    return {"head": _git(row["path"], "rev-parse", "HEAD"),
            "changes": [{"state": line[:2], "path": line[3:]} for line in lines[:MAX_LISTING]],
            "truncated": len(lines) > MAX_LISTING, "merging": _merging(row["path"])}


def diff(row, path=None) -> dict:
    arguments = ["diff", "HEAD", "--no-color", "--no-ext-diff"]
    if path:
        target(row["path"], path)
        arguments += ["--", path.replace("\\", "/")]
    text = _git(row["path"], *arguments)
    return {"diff": text[:MAX_DIFF_BYTES], "truncated": len(text) > MAX_DIFF_BYTES,
            "note": "untracked files are listed by status, not here"}


def commit(db, row, *, user_id, message) -> dict:
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE:
        raise CheckoutError(f"message must be text of 1 to {MAX_MESSAGE} characters")
    declared = branch_review.identity(db, user_id)
    if not declared:
        raise CheckoutError("declare your Git author identity first (POST /api/reviews/identity, reviews_identity_set)", 409)
    folder = row["path"]
    merging = _merging(folder)
    if merging:                                    # [MERGE] ending an unfinished merge: no conflict may be left in it
        marked = [name for name in _unmerged(folder) if _has_marker(Path(folder) / name)]
        if marked:
            raise CheckoutError("these files still hold conflict markers: " + ", ".join(marked[:20]) +
                                "; resolve them with edit or write, or give the merge up (merge with abort)", 409)
    elif not _porcelain(folder):
        raise CheckoutError("nothing to commit", 409)
    _git(folder, "add", "--all")
    _git(folder, "-c", f"user.name={declared['name']}", "-c", f"user.email={declared['email']}", "commit", "--quiet",
         "-m", message)
    return {"head": _git(folder, "rev-parse", "HEAD"), "branch": row["branch"], **({"merged": merging} if merging else {}),
            "next": "submit it: reviews_submit (POST /api/reviews) with this repository, branch and head"}


# ---------------------------------------------------------------- merge: another branch brought into this one [MERGE]

def _merging(folder) -> list:
    """The commits an unfinished merge is bringing in (Git's MERGE_HEAD), or [] when none is."""
    place = Path(_git(folder, "rev-parse", "--git-path", "MERGE_HEAD"))
    place = place if place.is_absolute() else Path(folder) / place
    return place.read_text(encoding="ascii").split() if place.is_file() else []


def _unmerged(folder) -> list:
    """The files Git still records as conflicted in the unfinished merge."""
    text = _git(folder, "diff", "--name-only", "--diff-filter=U", check=False)
    return sorted({line.strip() for line in text.splitlines() if line.strip()})


def _has_marker(place: Path) -> bool:
    if not place.is_file():
        return False
    return any(line.startswith(CONFLICT_MARKERS) for line in
               place.read_bytes().decode("utf-8", "replace").replace("\r\n", "\n").splitlines(keepends=True))


def _source(db, policy, row, source) -> str:
    """The commit ``from`` names, or a refusal: the default branch, or a branch registered in this repository."""
    repo, default, remote = _repository(policy, row["repository"])
    if not isinstance(source, str) or not source or len(source) > MAX_BRANCH or source == row["branch"]:
        raise CheckoutError("from must name another branch of this repository")
    if _git(repo, "check-ref-format", "--branch", source, check=False) != source:
        raise CheckoutError(f"{source!r} is not a branch name", 400)
    if source != default and branch_review.owner(db, row["repository"], source) is None:
        raise CheckoutError("from must be the default branch or a registered branch of this repository", 404)
    _git(repo, "fetch", "--quiet", remote, default if source == default else source, check=False)
    places = ((f"refs/remotes/{remote}/{default}",) if source == default
              else (f"refs/heads/{source}", f"refs/remotes/{remote}/{source}"))
    for place in places:                           # a checkout commits to refs/heads; the default lands on the remote
        found = _git(repo, "rev-parse", "-q", "--verify", f"{place}^{{commit}}", check=False)
        if found:
            return found
    raise CheckoutError(f"there is no branch {source!r} in this repository", 404)


def merge(db, policy, row, *, user_id, source, message=None, abort=False) -> dict:
    import subprocess
    folder = row["path"]
    merging = _merging(folder)
    if abort:
        if not merging:
            raise CheckoutError("no merge is unfinished here", 409)
        _git(folder, "merge", "--abort")
        return {"aborted": merging, "head": _git(folder, "rev-parse", "HEAD")}
    if merging:
        raise CheckoutError("a merge is unfinished here: resolve it and commit, or merge with abort", 409)
    if _porcelain(folder):
        raise CheckoutError("the working folder has uncommitted changes: commit them first", 409)
    declared = branch_review.identity(db, user_id)
    if not declared:
        raise CheckoutError("declare your Git author identity first (POST /api/reviews/identity, reviews_identity_set)", 409)
    if message is not None and (not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE):
        raise CheckoutError(f"message must be text of 1 to {MAX_MESSAGE} characters")
    commit_id = _source(db, policy, row, source)
    before = _git(folder, "rev-parse", "HEAD")
    if subprocess.run(["git", "-C", str(folder), "merge-base", "--is-ancestor", commit_id, "HEAD"],
                      capture_output=True).returncode == 0:      # already contained: nothing to bring in
        return {"from": source, "commit": commit_id, "head": before, "already": True, "files": []}
    done = subprocess.run(["git", "-C", str(folder), "-c", f"user.name={declared['name']}",
                           "-c", f"user.email={declared['email']}", "merge", "--no-ff", "--no-edit",
                           "-m", message or f"Merge {source} into {row['branch']}", commit_id],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode == 0:
        return {"from": source, "commit": commit_id, "head": _git(folder, "rev-parse", "HEAD"), "files": [],
                "next": "submit it: reviews_submit (POST /api/reviews) with this repository, branch and head"}
    conflicts = _unmerged(folder)
    if _merging(folder) and conflicts:
        return {"from": source, "commit": commit_id, "head": before, "conflicts": conflicts, "files": conflicts,
                "next": "resolve each conflicted file with edit or write, then commit; or merge with abort"}
    if _merging(folder):
        _git(folder, "merge", "--abort", check=False)
    raise CheckoutError(f"the merge did not apply (nothing was changed): {(done.stderr or done.stdout).strip()[-600:]}", 409)


def close(db, policy, row) -> dict:
    if _merging(row["path"]):
        raise CheckoutError("a merge is unfinished here: commit it or give it up (merge with abort) first", 409)
    if _porcelain(row["path"]):
        raise CheckoutError("the working folder has uncommitted changes: commit them first", 409)
    repo, _, _ = _repository(policy, row["repository"])
    _git(repo, "worktree", "remove", row["path"])
    db.execute("DELETE FROM checkouts WHERE repository = ? AND branch = ?", (row["repository"], row["branch"]))
    return {"repository": row["repository"], "branch": row["branch"], "closed": True,
            "note": "the branch stays, registered to you; open it again to keep working"}


def listing(db, user_id) -> list:
    ensure(db)
    rows = db.execute("SELECT repository, branch, base, created_at FROM checkouts WHERE user_id = ? ORDER BY created_at DESC",
                      (user_id,)).fetchall()
    return [{"repository": r[0], "branch": r[1], "base": r[2], "opened_at": r[3]} for r in rows]


def run(db, policy, root, *, user_id, username, operation, body) -> dict:
    """One declared operation for the signed-in account; recorded whether it succeeds or not."""
    ensure(db)
    if operation not in OPERATIONS:
        raise CheckoutError("operation must be one of: " + ", ".join(OPERATIONS), 404)
    who = {"user_id": user_id, "username": username}
    repository, branch, path = body.get("repository"), body.get("branch"), body.get("path")
    try:
        if operation == "open":
            result = open_checkout(db, policy, root, user_id=user_id, username=username, repository=repository, branch=branch)
            result["server"] = branch_servers.lend(db, repository=repository, branch=branch, user_id=user_id)   # [LOAN]
        else:
            row = _mine(db, user_id=user_id, repository=repository, branch=branch)
            if operation == "files":
                result = files(row, path or "")
            elif operation == "read":
                result = read(row, path)
            elif operation == "write":
                result = write(row, path, body.get("text"))
            elif operation == "remove":
                result = remove(row, path)
            elif operation == "edit":
                result = edit(row, body)
            elif operation == "status":
                result = status(row)
            elif operation == "diff":
                result = diff(row, path)
            elif operation == "commit":
                result = commit(db, row, user_id=user_id, message=body.get("message"))
            elif operation == "merge":
                result = merge(db, policy, row, user_id=user_id, source=body.get("from"), message=body.get("message"),
                               abort=body.get("abort") is True)
            else:
                result = close(db, policy, row)
                ended = branch_servers.release(db, row["repository"], row["branch"])                       # [LOAN]
                if ended is not None:
                    result["server"] = {"ended": branch_servers.name(ended)}
    except CheckoutError as error:
        _record(db, who, repository, branch, operation, path, error=error.message)
        raise
    except OSError as error:
        _record(db, who, repository, branch, operation, path, error=f"{type(error).__name__}: {error}")
        raise CheckoutError(f"the service could not do that: {type(error).__name__}", 500) from None
    _record(db, who, repository, branch, operation, path or ",".join(result.get("files", [])) or None, size=result.get("bytes"))
    return {"operation": operation, **result}


def http_route(self, path, method, *, policy, accounts_db, root):
    """The central checkout over REST, for signed-in accounts holding dev_ui (the same as the reviews).

        GET  /api/checkouts                   your open checkouts
        POST /api/checkouts/<operation>       {repository, branch, ...}   one of OPERATIONS (see the module text)
    """
    import sqlite3
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    if "dev_ui" not in self._caps():
        return self._json({"error": "this needs dev_ui"}, 403)
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        if method == "GET" and path == "/api/checkouts":
            return self._json({"checkouts": listing(db, principal.user_id), "operations": list(OPERATIONS)})
        if method != "POST" or not path.startswith("/api/checkouts/"):
            return self._json({"error": "not found"}, 404)
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(3 * MAX_FILE_BYTES)
        if body is None:
            return
        if not isinstance(body, dict):
            return self._json({"error": "a JSON object is required"}, 400)
        operation = path[len("/api/checkouts/"):]
        result = run(db, policy, root, user_id=principal.user_id, username=principal.username, operation=operation, body=body)
        print(f"[checkout] {principal.username} {operation} {body.get('repository')} {body.get('branch')}", flush=True)
        return self._json(result)
    except CheckoutError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
