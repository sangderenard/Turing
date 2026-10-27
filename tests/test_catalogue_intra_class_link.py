"""A per-method catalogue unit links its intra-class callee.

``Toy.twice`` calls ``self.bump`` twice.  Compiled as its own catalogue unit
it links the completed ``Toy.bump`` unit.  Before the linked body carried its
owning identity book (``accepted_ssa_component``), the caller re-posted that
body's control rows in its own book, which does not hold them, and the unit
refused: ``physical structure lacks SSA identity: ('Toy_bump__bump@control:0',
...)``.  The same call linked fine inside the whole-class product.
"""

from __future__ import annotations

import json

import pytest

from tools.compile_project_catalogue import main as compile_catalogue


RETURNING = '''
class Toy:
    def __init__(self, n: int) -> None:
        self.n = n
        self.steps = 0

    def bump(self, k: int) -> int:
        self.steps += k
        return self.steps * self.n

    def twice(self, k: int) -> int:
        return self.bump(k) + self.bump(k)
'''

STATEMENT = '''
class Toy:
    def __init__(self, n: int) -> None:
        self.n = n
        self.steps = 0

    def bump(self, k: int) -> None:
        self.steps += k

    def twice(self, k: int) -> int:
        self.bump(k)
        self.bump(k)
        return self.steps
'''


@pytest.mark.parametrize("source", [RETURNING, STATEMENT], ids=["returning", "statement"])
def test_intra_class_call_links_in_the_catalogue(tmp_path, source):
    path = tmp_path / "toy.py"
    path.write_text(source, encoding="utf-8")
    output = tmp_path / "catalogue"
    compile_catalogue([
        "--source", str(path), "--output", str(output),
        "--emit-native", "--jobs", "1",
    ])
    manifest = json.loads((output / "manifest.json").read_text())
    units = {unit["qualified_name"]: unit for unit in manifest["units"]}
    for name in ("Toy.__init__", "Toy.bump", "Toy.twice"):
        assert units[name]["status"] == "complete", (name, units[name])
        assert units[name]["native_status"] == "complete", (name, units[name])
        assert units[name]["repository_ssa_complete"], name
