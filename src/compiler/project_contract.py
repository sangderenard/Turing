"""Project contracts: what a dev page's project is, where it lives, what it holds.

A project is a folder with a human-readable ``project.yaml`` at its root (dev
pages ticket 9db5ea72, research case 8e062373). The contract names the
project, chooses its placement when the project is opened, and lists the
folders that make up its scope: the sources of the forward-facing product
(shaders and code, Nodus and Python files), never its binary products or the
published page. The scope is set by hand and may be edited later; every edit
is read through this loader.

Placement modes:

``branch``       the project is a branch, checked out centrally
                 (``placement.branch``: repository and name).
``standalone``   the project lives in the account's readable project space
                 (``placement.space``: a path under ``projects/``).
``branch-copy``  both: a branch, and a copy of it in the project space.

This module reads and validates only. Opening a project (the checkout, the
space, the first file) is a separate operation. It follows the input binding
contract's conventions (``abstract_ui_input_contract.py``): one schema
string, refusals that name the file and key, frozen records, ``to_data`` and
``page_contract`` for the page's Contracts section. The example is
``configs/projects/example_project.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import yaml

from .repo_world_scan import LANGUAGE_SUFFIXES


PROJECT_CONTRACT_SCHEMA = "project-contract-v0"
PROJECT_CONTRACT_FILE = "project.yaml"
PLACEMENT_MODES = ("branch", "standalone", "branch-copy")
#: The project space every standalone or branch-copy placement lives under.
SPACE_ROOT = "projects"

_TOP_KEYS = ("schema", "identity", "placement", "scope", "view")
_PLACEMENT_KEYS = ("mode", "branch", "space")
_SCOPE_KEYS = ("roots", "include", "exclude")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
#: The review gate's plain branch name (branch_review.BRANCH).
_BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}\Z")


class ProjectContractError(ValueError):
    """A project contract that does not say one exact thing."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """safe_load that refuses a mapping naming one key twice (YAML would keep the last)."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ProjectContractError(f"key {key!r} appears twice (line {key_node.start_mark.line + 1})")
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


@dataclass(frozen=True, slots=True)
class ProjectContract:
    """One resolved project contract."""

    identity: str
    path: str                                  # the file, as given to the loader
    text: str
    mode: str
    branch: Mapping[str, str] | None           # {repository, name} for branch and branch-copy
    space: str | None                          # projects/<...> for standalone and branch-copy
    roots: tuple[str, ...]
    languages: tuple[str, ...]
    exclude: tuple[str, ...]
    view: str

    @property
    def account(self) -> str:
        return self.identity.split("/", 1)[0]

    @property
    def project(self) -> str:
        return self.identity.split("/", 1)[1]

    def includes(self, path: str) -> bool:
        """Whether a repository-relative file is in the project's scope: under a root, of a declared
        language's suffix, and matched by no exclude pattern."""

        relative = PurePosixPath(str(path).replace("\\", "/"))
        text = relative.as_posix()
        if not any(text == root or text.startswith(root + "/") for root in self.roots):
            return False
        suffixes = {suffix for language in self.languages for suffix in LANGUAGE_SUFFIXES[language]}
        if relative.suffix.lower() not in suffixes:
            return False
        return not any(_excluded(text, pattern) for pattern in self.exclude)

    def to_data(self) -> dict[str, Any]:
        placement: dict[str, Any] = {"mode": self.mode}
        if self.branch is not None:
            placement["branch"] = dict(self.branch)
        if self.space is not None:
            placement["space"] = self.space
        return {
            "schema": PROJECT_CONTRACT_SCHEMA,
            "identity": self.identity,
            "placement": placement,
            "scope": {"roots": list(self.roots), "include": {"languages": list(self.languages)},
                      "exclude": list(self.exclude)},
            "view": {"name": self.view},
        }

    def page_contract(self) -> dict[str, Any]:
        """The entry the page lists in its Contracts section, shaped like the input contract's."""

        return {
            "kind": "project",
            "identity": self.identity,
            "files": [{"path": self.path, "text": self.text}],
            "resolved": self.to_data(),
        }


def _excluded(path: str, pattern: str) -> bool:
    if pattern.endswith("/"):                  # a folder: anything under it, at any depth
        folder = pattern.rstrip("/")
        return path.startswith(folder + "/") or ("/" + folder + "/") in ("/" + path)
    return fnmatchcase(path, pattern) or fnmatchcase(PurePosixPath(path).name, pattern)


def _mapping(source: str, key: str, value: Any, allowed: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectContractError(f"{source}: {key} must be a mapping")
    unknown = sorted(str(name) for name in value if name not in allowed)
    if unknown:
        raise ProjectContractError(f"{source}: {key} has unknown keys {unknown}; allowed: {list(allowed)}")
    return value


def _relative(source: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectContractError(f"{source}: {key} must be a non-empty relative path")
    text = value.strip().replace("\\", "/").rstrip("/")
    path = PurePosixPath(text)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", text) or ".." in path.parts or text in ("", "."):
        raise ProjectContractError(f"{source}: {key} {value!r} must be relative, without '..'")
    return path.as_posix()


def _strings(source: str, key: str, value: Any, *, required: bool) -> tuple[str, ...]:
    if value is None and not required:
        return ()
    if not isinstance(value, list) or (required and not value) or not all(isinstance(item, str) for item in value):
        raise ProjectContractError(f"{source}: {key} must be a {'non-empty ' if required else ''}list of strings")
    if len(set(value)) != len(value):
        raise ProjectContractError(f"{source}: {key} repeats an entry")
    return tuple(value)


def parse_project_contract(text: str, source: str = PROJECT_CONTRACT_FILE, *,
                           repositories: tuple[str, ...] | frozenset[str] | None = None) -> ProjectContract:
    """Read one contract from its text. ``repositories``, when given, is the set a branch may name
    (the branch policy's repositories)."""

    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ProjectContractError(f"{source}: not YAML: {error}") from None
    except ProjectContractError as error:
        raise ProjectContractError(f"{source}: {error}") from None
    raw = _mapping(source, "the contract", raw, _TOP_KEYS)
    if raw.get("schema") != PROJECT_CONTRACT_SCHEMA:
        raise ProjectContractError(f"{source}: schema must be {PROJECT_CONTRACT_SCHEMA!r}")
    identity = raw.get("identity")
    parts = identity.split("/") if isinstance(identity, str) else []
    if len(parts) != 2 or not all(_NAME.match(part) for part in parts):
        raise ProjectContractError(f"{source}: identity must be <account>/<project>, each a plain name")

    placement = _mapping(source, "placement", raw.get("placement"), _PLACEMENT_KEYS)
    mode = placement.get("mode")
    if mode not in PLACEMENT_MODES:
        raise ProjectContractError(f"{source}: placement.mode must be one of {list(PLACEMENT_MODES)}")
    wants_branch, wants_space = mode in ("branch", "branch-copy"), mode in ("standalone", "branch-copy")
    branch = None
    if wants_branch:
        given = _mapping(source, "placement.branch", placement.get("branch"), ("repository", "name"))
        repository, name = given.get("repository"), given.get("name")
        if not isinstance(repository, str) or not _NAME.match(repository):
            raise ProjectContractError(f"{source}: placement.branch.repository must be a repository name")
        if repositories is not None and repository not in repositories:
            raise ProjectContractError(f"{source}: placement.branch.repository {repository!r} is not one of "
                                       f"{sorted(repositories)}")
        if (not isinstance(name, str) or not _BRANCH.match(name) or ".." in name
                or name.endswith(("/", ".lock"))):
            raise ProjectContractError(f"{source}: placement.branch.name must be a plain branch name")
        branch = {"repository": repository, "name": name}
    elif "branch" in placement:
        raise ProjectContractError(f"{source}: placement.branch is not used by mode {mode!r}")
    space = None
    if wants_space:
        space = _relative(source, "placement.space", placement.get("space"))
        if PurePosixPath(space).parts[0] != SPACE_ROOT or len(PurePosixPath(space).parts) < 2:
            raise ProjectContractError(f"{source}: placement.space must be a folder under {SPACE_ROOT}/")
    elif "space" in placement:
        raise ProjectContractError(f"{source}: placement.space is not used by mode {mode!r}")

    scope = _mapping(source, "scope", raw.get("scope"), _SCOPE_KEYS)
    roots = tuple(_relative(source, "scope.roots", root)
                  for root in _strings(source, "scope.roots", scope.get("roots"), required=True))
    if len(set(roots)) != len(roots):
        raise ProjectContractError(f"{source}: scope.roots names one folder twice")
    include = _mapping(source, "scope.include", scope.get("include"), ("languages",))
    languages = _strings(source, "scope.include.languages", include.get("languages"), required=True)
    unknown = sorted(set(languages) - set(LANGUAGE_SUFFIXES))
    if unknown:
        raise ProjectContractError(f"{source}: scope.include.languages {unknown} are not languages the scanner "
                                   f"knows; known: {sorted(LANGUAGE_SUFFIXES)}")
    exclude = _strings(source, "scope.exclude", scope.get("exclude"), required=False)

    view = _mapping(source, "view", raw.get("view"), ("name",))
    view_name = view.get("name")
    if not isinstance(view_name, str) or not _NAME.match(view_name):
        raise ProjectContractError(f"{source}: view.name must be a plain name")
    return ProjectContract(identity=identity, path=source, text=text, mode=mode, branch=branch, space=space,
                           roots=roots, languages=languages, exclude=exclude, view=view_name)


def load_project_contract(path: str | Path, *,
                          repositories: tuple[str, ...] | frozenset[str] | None = None) -> ProjectContract:
    """Read and validate the contract file at ``path`` (a ``project.yaml``)."""

    file = Path(path)
    if file.name != PROJECT_CONTRACT_FILE:
        raise ProjectContractError(f"{file}: a project contract is named {PROJECT_CONTRACT_FILE}")
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as error:
        raise ProjectContractError(f"{file}: cannot be read: {error}") from None
    return parse_project_contract(text, file.as_posix(), repositories=repositories)
