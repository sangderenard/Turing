"""Read and write the test baseline and hazards log, newest first.

The log used to be one file, TEST_BASELINE_AND_HAZARDS.md, where every round
prepended its entry to the top.  Two PRs that each prepended an entry always
conflicted.  New entries now live one file per entry in docs/hazards/, named

    YYYYMMDDTHHMMSSZ_<branch>.md

(UTC time of writing, then the branch with "/" spelled "-").  The name sorts
by time, and two branches can never write the same name.
TEST_BASELINE_AND_HAZARDS.md keeps every entry written before the split,
unchanged; it is read after the newest files.

    python tools/hazards_log.py            # newest 10 entries, then the old log's top
    python tools/hazards_log.py -n 3       # newest 3 entries only
    python tools/hazards_log.py --all      # every entry, then the whole old log
    python tools/hazards_log.py new        # create this branch's entry file, print its path
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY_DIR = ROOT / "docs" / "hazards"
LEGACY_LOG = ROOT / "TEST_BASELINE_AND_HAZARDS.md"
ENTRY_NAME = re.compile(r"^\d{8}T\d{6}Z_[A-Za-z0-9._-]+\.md$")
# The old log's own header ends at this line; its entries follow.
LEGACY_HEADER_END = "<!-- end of header: entries written before the split follow -->"


def entry_files(entry_dir: Path = ENTRY_DIR) -> list[Path]:
    """Entry files, newest first.  A file whose name breaks the pattern is an
    error, not something to skip: it would sort in the wrong place."""
    if not entry_dir.is_dir():
        return []
    files = []
    for path in entry_dir.iterdir():
        if path.suffix != ".md" or path.name == "README.md":
            continue
        if not ENTRY_NAME.match(path.name):
            raise SystemExit(
                f"{path}: entry name must be YYYYMMDDTHHMMSSZ_<branch>.md"
            )
        files.append(path)
    return sorted(files, key=lambda p: p.name, reverse=True)


def legacy_entries(text: str) -> str:
    """The old log without its header."""
    _, sep, rest = text.partition(LEGACY_HEADER_END)
    return rest.lstrip("\n") if sep else text


def entry_name(branch: str, when: _dt.datetime) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip("-") or "detached"
    return f"{when.strftime('%Y%m%dT%H%M%SZ')}_{slug}.md"


def _current_branch() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _print_entries(count: int | None, legacy_lines: int | None) -> None:
    files = entry_files()
    shown = files if count is None else files[:count]
    for path in shown:
        print(f"===== docs/hazards/{path.name}")
        print(path.read_text(encoding="utf-8").rstrip("\n"))
        print()
    if count is not None and len(files) > count:
        print(f"({len(files) - count} older entry files not shown; --all shows them)")
        return
    if legacy_lines == 0 or not LEGACY_LOG.is_file():
        return
    old = legacy_entries(LEGACY_LOG.read_text(encoding="utf-8"))
    lines = old.splitlines()
    print("===== TEST_BASELINE_AND_HAZARDS.md (entries written before the split)")
    print("\n".join(lines if legacy_lines is None else lines[:legacy_lines]))
    if legacy_lines is not None and len(lines) > legacy_lines:
        print(f"... ({len(lines) - legacy_lines} more lines; --all shows them)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", choices=["show", "new"], default="show")
    parser.add_argument("-n", type=int, default=10, help="entry files to show")
    parser.add_argument("--all", action="store_true", help="every entry and the whole old log")
    parser.add_argument("--branch", help="branch name for `new` (default: current branch)")
    args = parser.parse_args(argv)

    if args.command == "new":
        name = entry_name(args.branch or _current_branch(), _dt.datetime.now(_dt.timezone.utc))
        path = ENTRY_DIR / name
        if path.exists():
            raise SystemExit(f"{path} already exists")
        ENTRY_DIR.mkdir(parents=True, exist_ok=True)
        today = name[:4] + "-" + name[4:6] + "-" + name[6:8]
        path.write_text(f"{today} TITLE (MEASURED, where): \n", encoding="utf-8")
        print(path.relative_to(ROOT).as_posix())
        return 0

    if args.all:
        _print_entries(None, None)
    else:
        _print_entries(args.n, 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
