"""Input binding contracts: a page's actions, program controls and widgets.

A contract is a human-readable YAML file in ``configs/input/``. It declares
every device input a page answers, the program controls those actions fill,
and the page controls and HUD widgets the page renders. The page generator
reads nothing else for input: the viewport binding table
(``ViewportControlBinding``), the viewport policy, the control focus policy,
the key commands and the widgets are all views of one resolved contract.

A file may ``extend`` another file in the same folder. The layer replaces or
adds actions, controls and widgets by name and merges ``policy`` and ``focus``
key by key, so a game is a layer over the first-person base
(``creature_game.yaml`` over ``first_person.yaml``).

The format is documented at the top of ``configs/input/first_person.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


INPUT_CONTRACT_SCHEMA = "input-binding-contract-v0"
INPUT_CONTRACT_DIRECTORY = Path(__file__).resolve().parents[2] / "configs" / "input"
DEFAULT_INPUT_CONTRACT = "creature_game.yaml"

#: Devices an input may name.  ``keyboard:`` signals are KeyboardEvent.code.
INPUT_DEVICES = ("keyboard", "pointer", "gamepad")
#: The focus routes of ControlFocusPolicy.
FOCUS_ROUTES = ("game", "projected-pointer", "dialogue")
#: Commands (``edge: press``) the page generator has a handler for.  A command
#: outside this set would be a key the page cannot answer, so it is refused.
PAGE_COMMANDS = ("back", "select-slot", "tool-mode-cycle", "vehicle-mount-toggle")
#: The steps ``back`` may try, each implemented by the page.
BACK_STEPS = ("exit-shader-only", "close-dialogue", "release-capture")
#: Widget kinds the page generator renders.
WIDGET_KINDS = ("hotbar", "on-screen-joystick", "device-monitor", "contract-list")
JOYSTICK_DIRECTIONS = ("up", "down", "left", "right")
HOTBAR_SLOTS = 10

_RANGE = re.compile(r"^(keyboard:[A-Za-z]*?)([0-9]+|[A-Z])\.\.([A-Za-z]*?)([0-9]+|[A-Z])$")


class InputContractError(ValueError):
    """A contract file that does not say one exact thing."""


def _expand(source: str, entry: str) -> tuple[str, ...]:
    if ".." not in entry:
        return (entry,)
    match = _RANGE.match(entry)
    if match is None:
        raise InputContractError(f"{source}: range {entry!r} is not device:<prefix><first>..<prefix><last>")
    prefix, first, second_prefix, last = match.groups()
    if prefix.removeprefix("keyboard:") != second_prefix:
        raise InputContractError(f"{source}: range {entry!r} changes prefix")
    if first.isdigit() and last.isdigit():
        values = [str(value) for value in range(int(first), int(last) + 1)]
    elif first.isalpha() and last.isalpha():
        values = [chr(value) for value in range(ord(first), ord(last) + 1)]
    else:
        raise InputContractError(f"{source}: range {entry!r} mixes digits and letters")
    if len(values) < 2:
        raise InputContractError(f"{source}: range {entry!r} is empty or reversed")
    return tuple(prefix + value for value in values)


@dataclass(frozen=True, slots=True)
class InputAction:
    """One declared action: its expanded inputs and, for a command, its keys."""

    name: str
    inputs: tuple[str, ...]
    edge: str = "held"
    argument: str | None = None
    routes: tuple[str, ...] = ()
    resolves: tuple[str, ...] = ()

    @property
    def command(self) -> bool:
        return self.edge == "press"

    def keys(self) -> dict[str, int | None]:
        """KeyboardEvent.code -> argument value (the input's ordinal) or None."""

        return {
            item.removeprefix("keyboard:"): (index + 1 if self.argument else None)
            for index, item in enumerate(self.inputs)
        }

    def to_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {"action": self.name, "inputs": list(self.inputs), "edge": self.edge}
        if self.argument:
            data["argument"] = self.argument
        if self.routes:
            data["routes"] = list(self.routes)
        if self.resolves:
            data["resolves"] = list(self.resolves)
        return data


@dataclass(frozen=True, slots=True)
class InputContract:
    """A resolved contract: the named file with every layer it extends applied."""

    identity: str
    files: tuple[tuple[str, str], ...]      # (path relative to the repository, text), base first
    policy: Mapping[str, Any]
    focus: Mapping[str, Any]
    actions: tuple[InputAction, ...]
    controls: Mapping[str, str]
    widgets: Mapping[str, Mapping[str, Any]]

    def action(self, name: str) -> InputAction:
        for action in self.actions:
            if action.name == name:
                return action
        raise KeyError(name)

    def held_bindings(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """(action, inputs) for every held action, in declaration order."""

        return tuple((action.name, action.inputs) for action in self.actions if not action.command)

    def commands(self) -> tuple[InputAction, ...]:
        return tuple(action for action in self.actions if action.command)

    def widget(self, kind: str) -> Mapping[str, Any] | None:
        """The first declared widget of a kind, or None when the page has none."""

        return next((widget for widget in self.widgets.values() if widget["kind"] == kind), None)

    def viewport_policy(self, identity: str, actor: str | None):
        from .abstract_ui_viewports import ViewportControlBinding, ViewportControlPolicy

        policy = self.policy
        return ViewportControlPolicy(
            identity, actor,
            activation=policy["activation"],
            release=tuple(policy["release"]),
            captures=tuple(policy["captures"]),
            pointer_mode=policy["pointer_mode"],
            gamepad_selection=policy["gamepad_selection"],
            move_speed=float(policy["rates"]["move_speed"]),
            run_multiplier=float(policy["rates"]["run_multiplier"]),
            jump_speed=float(policy["rates"]["jump_speed"]),
            look_sensitivity=float(policy["rates"]["look_sensitivity"]),
            bindings=tuple(ViewportControlBinding(name, inputs) for name, inputs in self.held_bindings()),
        )

    def focus_policy(self, identity: str, actor: str | None):
        from .abstract_ui_control_focus import ControlFocusPolicy

        focus = self.focus
        return ControlFocusPolicy(
            identity, actor,
            initial=focus["initial"],
            routes=tuple(focus["routes"]),
            switch_action=focus["switch_action"],
            dialogue_priority=int(focus["dialogue"]["priority"]),
            return_rule=focus["dialogue"]["return_rule"],
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": INPUT_CONTRACT_SCHEMA,
            "identity": self.identity,
            "sources": [path for path, _ in self.files],
            "policy": _plain(self.policy),
            "focus": _plain(self.focus),
            "actions": [action.to_data() for action in self.actions],
            "commands": [
                {"action": action.name, "keys": action.keys(), "routes": list(action.routes),
                 "resolves": list(action.resolves)}
                for action in self.commands()
            ],
            "controls": dict(self.controls),
            "widgets": {name: _plain(widget) for name, widget in self.widgets.items()},
        }

    def page_contract(self) -> dict[str, Any]:
        """The entry the page lists in its Contracts section: files and resolved settings."""

        return {
            "kind": "input-binding",
            "identity": self.identity,
            "files": [{"path": path, "text": text} for path, text in self.files],
            "resolved": self.to_data(),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _merge(base: Mapping[str, Any], layer: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in layer.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(name: str, directory: Path, chain: tuple[str, ...]) -> tuple[dict[str, Any], tuple[tuple[str, str], ...]]:
    if name in chain:
        raise InputContractError(f"contract extends itself: {' -> '.join((*chain, name))}")
    path = directory / name
    if path.parent != directory or not path.is_file():
        raise InputContractError(f"no input contract {name!r} in {directory}")
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or raw.get("schema") != INPUT_CONTRACT_SCHEMA:
        raise InputContractError(f"{name}: schema must be {INPUT_CONTRACT_SCHEMA}")
    try:
        shown = path.resolve().relative_to(INPUT_CONTRACT_DIRECTORY.parents[1]).as_posix()
    except ValueError:
        shown = path.name
    if "extends" not in raw:
        return raw, ((shown, text),)
    base, files = _read(str(raw["extends"]), directory, (*chain, name))
    merged = {key: value for key, value in base.items() if key not in {"identity", "extends"}}
    for section in ("policy", "focus"):
        merged[section] = _merge(base.get(section) or {}, raw.get(section) or {})
    for section in ("actions", "controls", "widgets"):
        merged[section] = {**(base.get(section) or {}), **(raw.get(section) or {})}
    merged["identity"] = raw["identity"]
    return merged, (*files, (shown, text))


def _action(source: str, name: str, entry: Any) -> InputAction:
    where = f"{source}: action {name!r}"
    if not isinstance(entry, Mapping) or not entry.get("inputs"):
        raise InputContractError(f"{where} declares no inputs")
    unknown = set(entry) - {"inputs", "edge", "argument", "routes", "resolves"}
    if unknown:
        raise InputContractError(f"{where}: unknown fields {sorted(unknown)}")
    inputs = tuple(item for raw in entry["inputs"] for item in _expand(where, str(raw)))
    for item in inputs:
        device, _, signal = item.partition(":")
        if device not in INPUT_DEVICES or not signal:
            raise InputContractError(f"{where}: input {item!r} is not <{'|'.join(INPUT_DEVICES)}>:<signal>")
    if len(set(inputs)) != len(inputs):
        raise InputContractError(f"{where}: an input is listed twice")
    edge = entry.get("edge", "held")
    routes = tuple(entry.get("routes") or ())
    resolves = tuple(entry.get("resolves") or ())
    argument = entry.get("argument")
    if edge == "held":
        if routes or resolves or argument:
            raise InputContractError(f"{where}: routes, argument and resolves belong to commands (edge: press)")
        return InputAction(name, inputs)
    if edge != "press":
        raise InputContractError(f"{where}: edge is held or press, not {edge!r}")
    if name not in PAGE_COMMANDS:
        raise InputContractError(f"{where}: the page has no command {name!r} (it answers {', '.join(PAGE_COMMANDS)})")
    if any(not item.startswith("keyboard:") for item in inputs):
        raise InputContractError(f"{where}: a command is a key press; its inputs are keyboard inputs")
    if any(route not in FOCUS_ROUTES for route in routes):
        raise InputContractError(f"{where}: routes are among {', '.join(FOCUS_ROUTES)}")
    if name == "back":
        if routes or argument:
            raise InputContractError(f"{where}: back is routed by its resolves steps")
        if not resolves or any(step not in BACK_STEPS for step in resolves) or len(set(resolves)) != len(resolves):
            raise InputContractError(f"{where}: resolves lists steps among {', '.join(BACK_STEPS)}")
    else:
        if resolves:
            raise InputContractError(f"{where}: only back resolves steps")
        if not routes:
            raise InputContractError(f"{where}: a command names the focus routes it answers in")
    argument_name = None
    if argument is not None:
        if not isinstance(argument, Mapping) or len(argument) != 1 or list(argument.values()) != ["ordinal"]:
            raise InputContractError(f"{where}: argument is {{<name>: ordinal}}")
        argument_name = str(next(iter(argument)))
    return InputAction(name, inputs, edge, argument_name, routes, resolves)


def _validate(source: str, actions: tuple[InputAction, ...], controls: Mapping[str, Any],
              widgets: Mapping[str, Any], focus: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    from .abstract_ui_viewports import control_terms

    by_name = {action.name: action for action in actions}
    held = {action.name for action in actions if not action.command}
    keys: dict[str, str] = {}
    for action in actions:
        for item in action.inputs:
            if not item.startswith("keyboard:"):
                continue
            owner = keys.setdefault(item, action.name)
            if owner != action.name and (action.command or by_name[owner].command):
                raise InputContractError(f"{source}: {item} is bound to {owner!r} and to {action.name!r}; "
                                         "a command's key belongs to that command alone")
    for name, hookup in controls.items():
        for _, action in control_terms(str(hookup)):
            if action not in held:
                raise InputContractError(f"{source}: control {name!r} reads {action!r}, which is not a held action")
    if focus.get("switch_action") not in held:
        raise InputContractError(f"{source}: focus switch_action {focus.get('switch_action')!r} is not a held action")
    if set(focus.get("routes") or ()) - set(FOCUS_ROUTES) or focus.get("initial") not in (focus.get("routes") or ()):
        raise InputContractError(f"{source}: focus routes are among {', '.join(FOCUS_ROUTES)} and include initial")
    for key in ("activation", "release", "captures", "pointer_mode", "gamepad_selection", "rates"):
        if key not in policy:
            raise InputContractError(f"{source}: policy declares no {key}")
    for name, widget in widgets.items():
        kind = widget.get("kind") if isinstance(widget, Mapping) else None
        if kind not in WIDGET_KINDS:
            raise InputContractError(f"{source}: widget {name!r} kind is among {', '.join(WIDGET_KINDS)}")
        if kind == "hotbar":
            select = by_name.get(widget.get("select_action"))
            if widget.get("slots") != HOTBAR_SLOTS or select is None or not select.command \
                    or select.argument is None or len(select.inputs) != HOTBAR_SLOTS:
                raise InputContractError(
                    f"{source}: hotbar {name!r} has {HOTBAR_SLOTS} slots and a select_action command "
                    f"with one argument per slot")
        if kind == "on-screen-joystick":
            drives = widget.get("drives") or {}
            if set(drives) - set(JOYSTICK_DIRECTIONS) or any(action not in held for action in drives.values()):
                raise InputContractError(
                    f"{source}: joystick {name!r} drives held actions from {', '.join(JOYSTICK_DIRECTIONS)}")
    if sum(1 for widget in widgets.values() if widget.get("kind") == "on-screen-joystick") > 1:
        raise InputContractError(f"{source}: one on-screen joystick")


@lru_cache(maxsize=None)
def _load(name: str, directory: str) -> InputContract:
    root = Path(directory)
    raw, files = _read(name, root, ())
    source = files[-1][0]
    actions = tuple(_action(source, str(key), value) for key, value in (raw.get("actions") or {}).items())
    controls = {str(key): str(value) for key, value in (raw.get("controls") or {}).items()}
    widgets = {str(key): value for key, value in (raw.get("widgets") or {}).items()}
    focus, policy = raw.get("focus") or {}, raw.get("policy") or {}
    _validate(source, actions, controls, widgets, focus, policy)
    return InputContract(str(raw["identity"]), files, policy, focus, actions, controls, widgets)


def load_input_contract(name: str = DEFAULT_INPUT_CONTRACT, *, directory: Path | None = None) -> InputContract:
    """Read and resolve one contract file (and every file it extends)."""

    return _load(name, str((directory or INPUT_CONTRACT_DIRECTORY).resolve()))


__all__ = [
    "BACK_STEPS", "DEFAULT_INPUT_CONTRACT", "FOCUS_ROUTES", "INPUT_CONTRACT_DIRECTORY",
    "INPUT_CONTRACT_SCHEMA", "InputAction", "InputContract", "InputContractError",
    "PAGE_COMMANDS", "WIDGET_KINDS", "load_input_contract",
]
