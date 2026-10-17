"""The book's page-taking calls take declared Page objects, never a name."""
from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src"

# ``IdentityBook.latest_ref(page, row)`` reads ``page.name``: a string literal
# in its first position is a ``'str' object has no attribute 'name'`` waiting
# for the call to run (it failed 23 compilation units at once).
LITERAL_PAGE = re.compile(r"""\.latest_ref\(\s*["']""")


def test_latest_ref_is_never_given_a_page_name_literal():
    offenders = []
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in LITERAL_PAGE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(SOURCE.parent)}:{line}")
    assert offenders == [], offenders
