"""Keep the declared static export complete for the existing Studio module graph."""
from pathlib import Path
import ast
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "examples/meowdoku/web"
IMPORT = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s*)["'](\.[^"']+)["']""")
# compute.js loads this generated compiler product only for the selected WASM lane.
GENERATED = {"turing-wasm-shell.js"}


def declared_exports():
    tree = ast.parse((WEB.parent / "web_export.py").read_text(encoding="utf-8"))
    return {item.value for node in ast.walk(tree)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
            and node.target.id == "name" and isinstance(node.iter, ast.Tuple)
            for item in node.iter.elts if isinstance(item, ast.Constant)
            and isinstance(item.value, str)}


class StudioWebExports(unittest.TestCase):
    def test_studio_module_graph_is_exported(self):
        pending = [WEB / "marble_studio.js"]
        seen = set()
        while pending:
            path = pending.pop().resolve()
            name = path.relative_to(WEB.resolve()).as_posix()
            if name in seen or name in GENERATED:
                continue
            seen.add(name)
            self.assertTrue(path.is_file(), name)
            for spec in IMPORT.findall(path.read_text(encoding="utf-8")):
                dependency = path.parent / spec.split("?", 1)[0]
                if dependency.suffix == ".js":
                    pending.append(dependency)
        self.assertFalse(seen - declared_exports(),
                         "Unpublished Studio imports: " + str(sorted(seen - declared_exports())))

    def test_header_and_sql_issue_page_are_exported(self):
        self.assertTrue({"site-nav.js", "issues.html", "issues.js"} <= declared_exports())
        header = (WEB / "site-nav.js").read_text(encoding="utf-8")
        self.assertIn('/issues.html?new=1&source=', header)
        self.assertNotIn('gameButton.click()', header,
                         "A frozen game may still submit its report form to legacy /api/tickets")
        self.assertIn('/api/issues', (WEB / "issues.js").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
