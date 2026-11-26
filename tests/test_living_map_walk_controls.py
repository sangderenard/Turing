"""The browser extraction retains the page's authored control data.

The binding table and the resolved input contract come from the page's input
contract (configs/input/); the policy defaults are the ViewportControlPolicy
class defaults.
"""

import ast
import json
from pathlib import Path
import re

from src.compiler.abstract_ui_input_contract import load_input_contract


ROOT = Path(__file__).resolve().parents[1]


def test_living_map_browser_control_declarations_are_the_authored_records():
    contract = load_input_contract()
    bindings = [{"action": action, "inputs": list(inputs)} for action, inputs in contract.held_bindings()]
    tree = ast.parse((ROOT / "src/compiler/abstract_ui_viewports.py").read_text(encoding="utf-8"))
    defaults = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ViewportControlPolicy":
            defaults = {
                item.target.id: ast.literal_eval(item.value)
                for item in node.body
                if isinstance(item, ast.AnnAssign) and item.value is not None and item.target.id != "bindings"
            }
    source = (ROOT / "examples/meowdoku/web/living_map_walk_controls.js").read_text(encoding="utf-8")
    for name, value in (("VIEWPORT_BINDINGS", bindings), ("VIEWPORT_POLICY_DEFAULTS", defaults),
                        ("INPUT_CONTRACT", contract.to_data())):
        match = re.search(r"export const LIVING_MAP_" + name + r" = freezeData\(([\s\S]*?)\);\r?\n", source)
        assert match is not None, name
        assert json.loads(match.group(1)) == json.loads(json.dumps(value)), name
