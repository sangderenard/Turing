"""Input binding contracts: resolution, layering, the views built from them, refusals."""

from pathlib import Path
import shutil

import pytest

from src.compiler.abstract_ui_input_contract import (
    INPUT_CONTRACT_DIRECTORY, InputContractError, load_input_contract,
)
from src.compiler.abstract_ui_viewports import DEFAULT_VIEWPORT_CONTROL_BINDINGS
from src.compiler.abstract_ui_vehicles import VEHICLE_CONTROL_RECORD


def test_base_contract_is_first_person_only():
    base = load_input_contract("first_person.yaml")
    names = [action.name for action in base.actions]
    assert base.identity == "first-person"
    assert not [name for name in names if name.startswith("vehicle") or "differential" in name]
    assert base.controls == {}
    assert base.policy["rates"]["move_speed"] == 3.2
    assert [path for path, _ in base.files] == ["configs/input/first_person.yaml"]


def test_game_layer_extends_the_base():
    game = load_input_contract()
    assert game.identity == "creature-game"
    assert [path for path, _ in game.files] == [
        "configs/input/first_person.yaml", "configs/input/creature_game.yaml"]
    assert game.policy["rates"] == {"move_speed": 1.65, "run_multiplier": 2.0,
                                    "jump_speed": 3.6, "look_sensitivity": 0.0025}
    assert game.policy["captures"] == ["keyboard", "pointer", "gamepad"]
    assert len(game.held_bindings()) == 16
    commands = {action.name: action.keys() for action in game.commands()}
    assert commands["select-slot"] == {f"Digit{n}": n for n in range(1, 10)} | {"Digit0": 10}
    assert commands["back"] == {"Escape": None}
    assert commands["tool-mode-cycle"] == {"KeyM": None}
    assert commands["vehicle-mount-toggle"] == {"KeyV": None}
    assert game.action("back").resolves == ("exit-shader-only", "close-dialogue", "release-capture")


def test_views_are_built_from_the_contract():
    game = load_input_contract()
    policy = game.viewport_policy("world/controls", "actor")
    assert policy.move_speed == 1.65
    assert tuple((binding.action, binding.inputs) for binding in policy.bindings) == game.held_bindings()
    assert tuple((binding.action, binding.inputs) for binding in DEFAULT_VIEWPORT_CONTROL_BINDINGS) \
        == game.held_bindings()
    focus = game.focus_policy("world/focus", "actor").to_data()
    assert focus["switch_action"] == "secondary-action"
    assert focus["routes"] == ["game", "projected-pointer", "dialogue"]
    record = dict(VEHICLE_CONTROL_RECORD)
    assert {name: record[name] for name in game.controls} == game.controls
    data = game.to_data()
    assert data["widgets"]["joystick"]["drives"]["up"] == "move-forward"
    page = game.page_contract()
    assert page["kind"] == "input-binding" and page["resolved"] == data
    assert [entry["path"] for entry in page["files"]] == data["sources"]
    assert "schema: input-binding-contract-v0" in page["files"][1]["text"]


def _folder(tmp_path: Path, layer: str) -> Path:
    shutil.copy(INPUT_CONTRACT_DIRECTORY / "first_person.yaml", tmp_path / "first_person.yaml")
    (tmp_path / "layer.yaml").write_text(
        "schema: input-binding-contract-v0\nidentity: layer\nextends: first_person.yaml\n" + layer,
        encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(("layer", "refusal"), [
    ("actions:\n  fly: {inputs: ['keyboard:KeyF'], edge: press, routes: [game]}\n",
     "the page has no command 'fly'"),
    ("actions:\n  tool-mode-cycle: {inputs: ['gamepad:button-7'], edge: press, routes: [game]}\n",
     "a command is a key press"),
    ("actions:\n  tool-mode-cycle: {inputs: ['keyboard:KeyW'], edge: press, routes: [game]}\n",
     "keyboard:KeyW is bound to 'move-forward' and to 'tool-mode-cycle'"),
    ("actions:\n  tool-mode-cycle: {inputs: ['keyboard:KeyM'], edge: press}\n",
     "names the focus routes"),
    ("actions:\n  horn: {inputs: ['keyboard:KeyH'], routes: [game]}\n",
     "belong to commands"),
    ("actions:\n  horn: {inputs: ['mouse:left']}\n", "is not <keyboard|pointer|gamepad>"),
    ("controls:\n  throttle: '+fly -move-backward'\n", "reads 'fly', which is not a held action"),
    ("widgets:\n  hotbar: {kind: hotbar, slots: 9, select_action: select-slot}\n", "has 10 slots"),
    ("widgets:\n  sky: {kind: billboard}\n", "kind is among"),
    ("actions:\n  back: {inputs: ['keyboard:Escape'], edge: press, resolves: [quit]}\n",
     "resolves lists steps"),
])
def test_refusals(tmp_path, layer, refusal):
    with pytest.raises(InputContractError, match=refusal):
        load_input_contract("layer.yaml", directory=_folder(tmp_path, layer))


def test_extends_cycle_is_refused(tmp_path):
    (tmp_path / "a.yaml").write_text("schema: input-binding-contract-v0\nidentity: a\nextends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("schema: input-binding-contract-v0\nidentity: b\nextends: a.yaml\n")
    with pytest.raises(InputContractError, match="extends itself"):
        load_input_contract("a.yaml", directory=tmp_path)
