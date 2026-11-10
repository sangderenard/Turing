"""The browser extraction retains the existing Python-authored control data."""

import ast
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_living_map_browser_control_declarations_are_the_authored_records():
    tree = ast.parse((ROOT / "src/compiler/abstract_ui_viewports.py").read_text(encoding="utf-8"))
    bindings = defaults = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_VIEWPORT_CONTROL_BINDINGS"
            for target in node.targets
        ):
            bindings = [
                {"action": ast.literal_eval(item.args[0]), "inputs": list(ast.literal_eval(item.args[1]))}
                for item in node.value.elts
            ]
        if isinstance(node, ast.ClassDef) and node.name == "ViewportControlPolicy":
            defaults = {
                item.target.id: ast.literal_eval(item.value)
                for item in node.body
                if isinstance(item, ast.AnnAssign) and item.value is not None and item.target.id != "bindings"
            }
    source = (ROOT / "examples/meowdoku/web/living_map_walk_controls.js").read_text(encoding="utf-8")
    for name, value in (("BINDINGS", bindings), ("POLICY_DEFAULTS", defaults)):
        match = re.search(r"export const LIVING_MAP_VIEWPORT_" + name + r" = freezeData\(([\s\S]*?)\);", source)
        assert match is not None
        assert json.loads(match.group(1)) == json.loads(json.dumps(value))
