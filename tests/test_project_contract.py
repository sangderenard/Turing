"""Project contracts (src/compiler/project_contract.py): one exact meaning or a refusal that names it."""
from pathlib import Path

import pytest
import yaml

from src.compiler.project_contract import (
    PROJECT_CONTRACT_SCHEMA, ProjectContractError, load_project_contract, parse_project_contract,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "projects" / "example_project.yaml"
BASE = {
    "schema": PROJECT_CONTRACT_SCHEMA,
    "identity": "alice/garden",
    "placement": {"mode": "branch", "branch": {"repository": "turing", "name": "alice/garden"}},
    "scope": {"roots": ["examples/garden"], "include": {"languages": ["python", "glsl"]}, "exclude": ["build/"]},
    "view": {"name": "sources"},
}


def parse(data, **kwargs):
    return parse_project_contract(yaml.safe_dump(data), "project.yaml", **kwargs)


def changed(path, value):
    data = yaml.safe_load(yaml.safe_dump(BASE))
    *parents, last = path.split(".")
    node = data
    for key in parents:
        node = node[key]
    if value is KeyError:
        del node[last]
    else:
        node[last] = value
    return data


def test_the_shipped_example_reads_and_round_trips():
    text = EXAMPLE.read_text(encoding="utf-8")
    contract = parse_project_contract(text, "configs/projects/example_project.yaml")
    assert contract.mode == "branch-copy" and contract.space == "projects/marble-garden"
    assert contract.branch == {"repository": "turing", "name": "alice/marble-garden"}
    assert (contract.account, contract.project) == ("alice", "marble-garden")
    again = parse_project_contract(yaml.safe_dump(contract.to_data()), "again")
    assert again.to_data() == contract.to_data()
    entry = contract.page_contract()
    assert entry["kind"] == "project" and entry["identity"] == "alice/marble-garden"
    assert entry["files"] == [{"path": "configs/projects/example_project.yaml", "text": text}]
    assert entry["resolved"] == contract.to_data()


def test_each_mode_uses_only_its_own_keys():
    assert parse(BASE).space is None
    standalone = changed("placement", {"mode": "standalone", "space": "projects/garden"})
    assert parse(standalone).branch is None and parse(standalone).space == "projects/garden"
    with pytest.raises(ProjectContractError, match="placement.space is not used by mode 'branch'"):
        parse(changed("placement.space", "projects/garden"))
    with pytest.raises(ProjectContractError, match="placement.branch is not used by mode 'standalone'"):
        parse(changed("placement", {"mode": "standalone", "space": "projects/g", "branch": BASE["placement"]["branch"]}))
    with pytest.raises(ProjectContractError, match="placement.space must be"):
        parse(changed("placement", {"mode": "branch-copy", "branch": BASE["placement"]["branch"]}))
    with pytest.raises(ProjectContractError, match="placement.mode must be one of"):
        parse(changed("placement.mode", "fork"))


@pytest.mark.parametrize("path, value, message", [
    ("schema", "project-contract-v9", "schema must be"),
    ("identity", "garden", "identity must be <account>/<project>"),
    ("identity", "alice/../bob", "identity must be"),
    ("extends", "base.yaml", "unknown keys \\['extends'\\]"),
    ("placement.branch.name", "a..b", "plain branch name"),
    ("placement.branch.name", "x.lock", "plain branch name"),
    ("placement.branch.repository", "../turing", "repository name"),
    ("scope.roots", [], "non-empty list"),
    ("scope.roots", ["/etc"], "must be relative"),
    ("scope.roots", ["C:/dev"], "must be relative"),
    ("scope.roots", ["examples/../secrets"], "must be relative"),
    ("scope.roots", ["a", "a/"], "names one folder twice"),
    ("scope.include.languages", ["python", "cobol"], "\\['cobol'\\] are not languages the scanner knows"),
    ("scope.include.languages", ["python", "python"], "repeats an entry"),
    ("scope.exclude", "build/", "list of strings"),
    ("scope.extra", 1, "scope has unknown keys"),
    ("view", KeyError, "view must be a mapping"),
    ("view.name", "two words", "view.name must be a plain name"),
])
def test_every_refusal_names_the_key(path, value, message):
    with pytest.raises(ProjectContractError, match=message):
        parse(changed(path, value))


def test_a_space_lives_under_projects_and_a_repository_may_be_checked():
    with pytest.raises(ProjectContractError, match="folder under projects/"):
        parse(changed("placement", {"mode": "standalone", "space": "elsewhere/garden"}))
    with pytest.raises(ProjectContractError, match="folder under projects/"):
        parse(changed("placement", {"mode": "standalone", "space": "projects"}))
    with pytest.raises(ProjectContractError, match="is not one of \\['speaktome'\\]"):
        parse(BASE, repositories=frozenset({"speaktome"}))
    assert parse(BASE, repositories=("turing", "speaktome")).branch["repository"] == "turing"


def test_a_key_given_twice_is_refused():
    text = yaml.safe_dump(BASE) + "identity: bob/garden\n"
    with pytest.raises(ProjectContractError, match="'identity' appears twice"):
        parse_project_contract(text, "project.yaml")


def test_scope_includes_sources_under_roots_only():
    contract = parse(changed("scope.exclude", ["build/", "*_generated.py"]))
    assert contract.includes("examples/garden/world.py")
    assert contract.includes("examples/garden/shaders/sky.frag")
    assert contract.includes("examples\\garden\\main.py")
    assert not contract.includes("examples/garden.py"), "a root is a folder, not a prefix"
    assert not contract.includes("examples/other/world.py")
    assert not contract.includes("examples/garden/page.js"), "javascript is not declared"
    assert not contract.includes("examples/garden/build/world.py")
    assert not contract.includes("examples/garden/deep/build/x.py")
    assert not contract.includes("examples/garden/world_generated.py")


def test_the_loader_reads_only_a_file_named_project_yaml(tmp_path):
    other = tmp_path / "contract.yaml"
    other.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    with pytest.raises(ProjectContractError, match="is named project.yaml"):
        load_project_contract(other)
    with pytest.raises(ProjectContractError, match="cannot be read"):
        load_project_contract(tmp_path / "missing" / "project.yaml")
    good = tmp_path / "project.yaml"
    good.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    assert load_project_contract(good).identity == "alice/garden"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["contract.yaml", "project.yaml"], \
        "reading creates nothing"
