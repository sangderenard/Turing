"""List every line that records a fact without naming its source.

Closes every identity book (``TURING_IDENTITY_LATCH=closed``), runs the named
test files in one process, and records, for each refused write, the first
caller frame outside identity_concordance.py: a raw page write
(``set``/``concord``/``revise``) or a post whose provenance is ``Unsourced``.
Usage (from the turing root):

    python -I docs/compiler_convergence/evidence/unsourced_write_sites.py tests/test_precompile_to_ssa.py ...
"""
import collections
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["TURING_IDENTITY_LATCH"] = "closed"
sys.path.insert(0, ROOT)

import pytest  # noqa: E402
import src.compiler.identity_concordance as ic  # noqa: E402

sites = collections.Counter()


DEPTH = int(os.environ.get("UNSOURCED_DEPTH", "4"))


def _book_class_lines():
    """Line range of ``class IdentityPage`` .. the first module-level ``def``
    after ``class IdentityBook`` in identity_concordance.py."""

    lines = open(ic.__file__, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines, 1) if l.startswith("class IdentityPage"))
    book = next(i for i, l in enumerate(lines, 1) if l.startswith("class IdentityBook"))
    end = next(i for i, l in enumerate(lines, 1) if i > book and l.startswith("def "))
    return start, end


BOOK_CLASSES = _book_class_lines()


def where():
    """The first ``DEPTH`` frames outside the book and this script, innermost
    first, plus the test that was running."""

    here = os.path.abspath(__file__)
    frames = []
    for frame in reversed(traceback.extract_stack()):
        if (
            frame.filename.endswith("identity_concordance.py")
            and BOOK_CLASSES[0] <= frame.lineno < BOOK_CLASSES[1]
        ):
            # Inside ``IdentityPage`` / ``IdentityBook`` themselves: the
            # writer is the caller.  Module-level helpers of the same file
            # (``declare_loop_scope``, ``record_proven_shape``) are writers.
            continue
        if os.path.abspath(frame.filename) == here:
            continue
        path = os.path.relpath(frame.filename, ROOT)
        frames.append(f"{path}:{frame.lineno} {frame.name}")
        if len(frames) >= DEPTH:
            break
    test = os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]
    return "\n           ".join(frames) + f"\n           [{test}]"


_raw = ic.IdentityBook._admit_raw_write


def admit(self, page, row):
    if self.latch is not ic.Latch.OPEN:
        sites[("raw write", page.name, where())] += 1
    return _raw(self, page, row)


ic.IdentityBook._admit_raw_write = admit
_post = ic.IdentityBook.post


def post(self, page, row, fact, *, stage, provenance, mode):
    try:
        return _post(self, page, row, fact, stage=stage,
                     provenance=provenance, mode=mode)
    except ic.ConcordanceRefusal as error:
        if "latch is CLOSED" in str(error):
            sites[("unsourced post", page.name, where())] += 1
        raise


ic.IdentityBook.post = post

for target in sys.argv[1:]:
    pytest.main([target, "-q", "--tb=no", "-p", "no:cacheprovider"])

print("\n=== lines that record without a source ===")
by_site = collections.defaultdict(lambda: collections.Counter())
for (kind, page, site), count in sites.items():
    by_site[site][f"{kind}:{page}"] += count
for site, pages in sorted(by_site.items(), key=lambda kv: -sum(kv[1].values())):
    detail = ", ".join(f"{k} x{v}" for k, v in pages.most_common(3))
    print(f"{sum(pages.values()):>5}  {site}\n         {detail}")
print(f"\n{len(by_site)} distinct lines")
