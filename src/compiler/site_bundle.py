"""Build one compiler-inspection page as a versioned site bundle.

The homepage is only one page.  This module provides the general publishing
contract used for every other compiled Python source::

    site/programs/<slug>/versions/<version>/
        bundle.json
        index.html
        source/python_source/<uploaded name>.py
        source/<backend>/<emitted source>
        wasm/<module>.wasm
        math/sympy-process-model.json
        callables/<qualified-name>/index.html

Source-inspection bundles enumerate class methods first (grouped by class),
then module functions. Every callable receives an independent input/run
section in the parent shell and a dedicated nested inspection page.

``bundle.json`` is written last and the completed directory is renamed into
place atomically.  A gallery can therefore discover prepared pages by walking
the folder tree without relying on a separately maintained catalogue.
"""

from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import textwrap
from typing import Any, Mapping


BUNDLE_SCHEMA = "turing-program-bundle-v1"
BUNDLE_LAYOUT_VERSION = 1
BUILDER_VERSION = "site-bundle-v16"
DEFAULT_WASM_CARD_OPERATIONS = 2000
TURING_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLISH_ROOT = TURING_REPOSITORY_ROOT.parent

_SOURCE_EXTENSIONS = {
    "python_source": "py",
    "ssa": "ssa",
    "fortran": "f90",
    "spirv": "spvasm",
    "glsl": "comp.glsl",
    "webgl": "frag.glsl",
    "wat": "wat",
    "numpy": "py",
    "torch": "py",
    "abstract_tensor": "py",
}


@dataclass(frozen=True, slots=True)
class SourceContract:
    """Literal publishing instructions discovered without executing source."""

    entrypoint: str
    title: str
    slug: str
    feeds: Mapping[str, Any]
    feed_expressions: Mapping[str, str]
    width: int
    height: int
    probe_size: int
    backend: str
    remove_loops: bool
    unroll_limit: int
    state_feedback: Mapping[str, str]
    render_fps: float
    autostart: bool


@dataclass(frozen=True, slots=True)
class ProgramBundle:
    """A completed immutable program bundle."""

    directory: Path
    manifest_path: Path
    page_path: Path
    manifest: Mapping[str, Any]

    @property
    def url(self) -> str:
        return str(self.manifest["page"]["url"])


@dataclass(frozen=True, slots=True)
class InspectionCallable:
    """One source callable represented by its own shell run section."""

    owner: str | None
    name: str
    qualified_name: str
    kind: str
    signature: str
    parameters: tuple[Mapping[str, Any], ...]
    source: str


@dataclass(frozen=True, slots=True)
class _TextInspectionSubject:
    source: str
    source_name: str
    subject_name: str
    source_kind: str


def resolve_publish_root(destination: str | Path) -> Path:
    """Resolve a gallery root and reject Turing's source repository itself."""

    resolved = Path(destination).resolve()
    if resolved == TURING_REPOSITORY_ROOT:
        raise ValueError(
            "gallery bundles belong in the parent workspace root, not Turing"
        )
    return resolved


def static_gallery_items(destination: str | Path) -> list[dict[str, Any]]:
    """Return published manifests as links relative to the website root."""
    root = resolve_publish_root(destination)
    items: list[dict[str, Any]] = []
    for path in (root / "site" / "programs").glob("*/versions/*/bundle.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        program = manifest["program"]
        artifacts = manifest.get("artifacts", [])
        items.append({
            "slug": program["slug"],
            "title": program["title"],
            "entrypoint": program.get("entrypoint", "program"),
            "version": manifest["version"]["id"],
            "created_at": manifest.get("created_at", ""),
            "url": manifest["page"]["url"].lstrip("/"),
            "source": manifest.get("source", {}).get("filename", ""),
            "artifacts": len(artifacts),
            "bytes": sum(int(artifact.get("bytes", 0)) for artifact in artifacts),
            "latest": False,
        })
    items.sort(key=lambda item: (item["slug"], item["created_at"]), reverse=True)
    newest: set[str] = set()
    for item in items:
        if item["slug"] not in newest:
            item["latest"] = True
            newest.add(item["slug"])
    return items


def build_source_inspection_page(
    subject: str | Path | type | Any,
    destination: str | Path,
    *,
    title: str | None = None,
    resource_route: str = "/",
    callable_systems: Mapping[str, Any] | None = None,
    python_source_url: str = "",
    static_gallery: list[Mapping[str, Any]] | None = None,
) -> Path:
    """Render a file, class, or method with the website's HTML shell."""

    source, source_name, subject_name, source_kind = _inspection_source(subject)

    from ..common.tensors.topological_reducer import reduce_abstract_tensor_topology
    from ..transmogrifier.graph.graph_express2 import ProcessGraph
    from .precompile_to_ssa import lower_class_navigation_to_ssa
    from .shell_reference_tables import build_class_navigation_table
    from .shell_telemetry import summarize_process_graph
    from .wasm_html_shell import emit_html_shell

    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(source, filename=source_name))
        reduce_abstract_tensor_topology(graph)
    map_ir = dict(graph.G.graph.get("map_ir") or {})
    navigation = build_class_navigation_table(graph)
    navigation_mapping = navigation.to_mapping()
    systems = dict(callable_systems or _callable_system_mapping(source))
    page_by_identity = {
        str(item["identity"]): str(item.get("page_url") or "")
        for item in _iter_callable_systems(systems)
    }
    reference_by_identity: dict[str, int | None] = {}
    for record in navigation_mapping.get("classes", ()):
        for member in record.get("members", ()):
            identity = str(member["identity"])
            reference_by_identity[identity] = member.get("function_reference")
            if page_by_identity.get(identity):
                member["page_url"] = page_by_identity[identity]
    for item in _iter_callable_systems(systems):
        reference = reference_by_identity.get(str(item["identity"]))
        if reference is None:
            for candidate in (str(item["identity"]), str(item["name"])):
                try:
                    entry = graph.function_table.entry(candidate)
                except KeyError:
                    continue
                reference = int(entry.reference.address) if entry.graph is not None else None
                break
        item["function_reference"] = reference
        if python_source_url and item.get("kind") == "function":
            item["python_source_url"] = python_source_url
    map_ir["class_navigation"] = navigation_mapping
    map_ir["callable_systems"] = systems
    semantic = lower_class_navigation_to_ssa(navigation)
    map_ir["semantic_methods"] = tuple(
        {
            "function": function.name,
            "operations": tuple(dict.fromkeys(
                instruction.op
                for instruction in function.blocks["entry"].instrs
            )),
        }
        for function in semantic.functions.values()
    )
    map_ir["inspection"] = {
        "subject": subject_name,
        "source_kind": source_kind,
        "source_file": source_name,
        "title": title or subject_name,
        "python_source_url": python_source_url,
    }
    callable_items = list(_iter_callable_systems(systems))
    single_callable = callable_items[0] if len(callable_items) == 1 else None
    api = {
        "module": subject_name,
        "language": "python-ast",
        "entry": str(single_callable["identity"] if single_callable else "inspect"),
        "entry_points": [{
            "name": str(single_callable["identity"] if single_callable else "inspect"),
            "symbol": "inspect",
            "kind": "inspection",
            "parameters": list(single_callable.get("parameters", ())) if single_callable else [],
        }],
        "metadata": {"value_type": "i32", "element_bytes": 4},
    }
    shell = emit_html_shell(
        api,
        name="index",
        process_graph=summarize_process_graph(graph),
        origin_source=source,
        map_ir=map_ir,
        resource_route=resource_route,
        static_gallery=static_gallery,
    )
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return shell.write(output)


def _callable_parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[Mapping[str, Any], ...]:
    positional = (*function.args.posonlyargs, *function.args.args)
    defaults = (
        *(None for _ in range(len(positional) - len(function.args.defaults))),
        *function.args.defaults,
    )
    parameters: list[Mapping[str, Any]] = []
    for argument, default in zip(positional, defaults):
        parameter = {
            "name": argument.arg,
            "role": "input",
            "dtype": ast.unparse(argument.annotation) if argument.annotation else "python-object",
            "passing": "value",
        }
        if default is not None:
            parameter["default"] = ast.unparse(default)
        parameters.append(parameter)
    if function.args.vararg:
        argument = function.args.vararg
        parameters.append({
            "name": argument.arg,
            "role": "input",
            "dtype": ast.unparse(argument.annotation) if argument.annotation else "python-object",
            "passing": "variadic",
        })
    for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
        parameter = {
            "name": argument.arg,
            "role": "input",
            "dtype": ast.unparse(argument.annotation) if argument.annotation else "python-object",
            "passing": "keyword-only",
        }
        if default is not None:
            parameter["default"] = ast.unparse(default)
        parameters.append(parameter)
    if function.args.kwarg:
        argument = function.args.kwarg
        parameters.append({
            "name": argument.arg,
            "role": "input",
            "dtype": ast.unparse(argument.annotation) if argument.annotation else "python-object",
            "passing": "keyword-variadic",
        })
    return tuple(parameters)


def _inspection_callables(source: str) -> tuple[InspectionCallable, ...]:
    """List methods by class first, then module functions, preserving order."""

    module = ast.parse(source)
    targets: list[InspectionCallable] = []

    def append(function: ast.FunctionDef | ast.AsyncFunctionDef, owner: str | None) -> None:
        qualified = f"{owner}.{function.name}" if owner else function.name
        segment = ast.get_source_segment(source, function) or ast.unparse(function)
        targets.append(InspectionCallable(
            owner=owner,
            name=function.name,
            qualified_name=qualified,
            kind="method" if owner else "function",
            signature=f"{function.name}({ast.unparse(function.args)})",
            parameters=_callable_parameters(function),
            source=textwrap.dedent(segment),
        ))

    for definition in module.body:
        if not isinstance(definition, ast.ClassDef):
            continue
        for member in definition.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                append(member, definition.name)
    for definition in module.body:
        if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            append(definition, None)
    return tuple(targets)


def _callable_system_mapping(
    source: str,
    page_urls: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    urls = dict(page_urls or {})
    targets = _inspection_callables(source)

    def mapping(target: InspectionCallable) -> dict[str, Any]:
        return {
            "identity": target.qualified_name,
            "name": target.name,
            "kind": target.kind,
            "signature": target.signature,
            "parameters": [dict(item) for item in target.parameters],
            "page_url": urls.get(target.qualified_name, ""),
        }

    module = ast.parse(source)
    functions = [mapping(target) for target in targets if target.owner is None]
    symbols: list[dict[str, str]] = []
    for definition in module.body:
        if isinstance(definition, (ast.Import, ast.ImportFrom)):
            for alias in definition.names:
                symbols.append({
                    "name": alias.asname or alias.name,
                    "kind": "import",
                    "expression": ast.unparse(definition),
                })
            continue
        if isinstance(definition, ast.Assign):
            names = [
                target.id for target in definition.targets
                if isinstance(target, ast.Name)
            ]
            expression = ast.unparse(definition.value)
            symbols.extend(
                {"name": name, "kind": "binding", "expression": expression}
                for name in names
            )
            continue
        if isinstance(definition, ast.AnnAssign) and isinstance(definition.target, ast.Name):
            symbols.append({
                "name": definition.target.id,
                "kind": "symbolic" if definition.value is None else "binding",
                "expression": ast.unparse(definition),
            })

    classes: list[dict[str, Any]] = []
    owners = [node.name for node in module.body if isinstance(node, ast.ClassDef)]
    for owner in owners:
        classes.append({
            "identity": owner,
            "methods": [mapping(target) for target in targets if target.owner == owner],
        })
    return {
        "file_scope": {
            "identity": "module",
            "functions": functions,
            "symbols": symbols,
        },
        "classes": classes,
        "functions": functions,
    }


def _iter_callable_systems(systems: Mapping[str, Any]):
    for record in systems.get("classes", ()):
        yield from record.get("methods", ())
    yield from systems.get("functions", ())


def _inspection_source(
    subject: str | Path | type | Any | _TextInspectionSubject,
) -> tuple[str, str, str, str]:
    """Read an inspection subject without executing source-file contents."""

    if isinstance(subject, _TextInspectionSubject):
        return (
            subject.source,
            subject.source_name,
            subject.subject_name,
            subject.source_kind,
        )
    if isinstance(subject, Path) or (
        isinstance(subject, str) and Path(subject).is_file()
    ):
        path = Path(subject).resolve()
        source = path.read_text(encoding="utf-8")
        source_name = path.name
        subject_name = path.stem
        source_kind = "file"
    elif (
        inspect.isclass(subject)
        or inspect.isfunction(subject)
        or inspect.ismethod(subject)
    ):
        source = textwrap.dedent(inspect.getsource(subject))
        source_name = Path(
            inspect.getsourcefile(subject) or "source.py"
        ).name
        subject_name = str(getattr(
            subject, "__qualname__", getattr(subject, "__name__", "source")
        ))
        source_kind = "class" if inspect.isclass(subject) else "method"
    else:
        raise TypeError(
            "inspection subject must be a source file, class, or method"
        )
    return source, source_name, subject_name, source_kind


def build_source_inspection_bundle(
    subject: str | Path | type | Any,
    destination: str | Path,
    *,
    title: str | None = None,
    slug: str | None = None,
) -> ProgramBundle:
    """Publish a source/class/method AST inspection as a gallery bundle."""

    source, source_name, subject_name, source_kind = _inspection_source(subject)
    page_title = str(title or subject_name)
    page_slug = slugify(str(slug or subject_name))
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = json.dumps(
        {
            "builder": BUILDER_VERSION + "-ast",
            "source_sha256": source_digest,
            "subject": subject_name,
            "source_kind": source_kind,
            "title": page_title,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    version_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    version = f"v{BUNDLE_LAYOUT_VERSION}-{version_digest[:16]}"
    destination = resolve_publish_root(destination)
    versions = destination / "site" / "programs" / page_slug / "versions"
    final_directory = versions / version
    if final_directory.is_dir() and (final_directory / "bundle.json").is_file():
        return load_program_bundle(final_directory)
    versions.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=versions))
    try:
        route = f"/site/programs/{page_slug}/versions/{version}/"
        created_at = datetime.now(timezone.utc).isoformat()
        gallery = static_gallery_items(destination)
        for item in gallery:
            if item["slug"] == page_slug:
                item["latest"] = False
        gallery.insert(0, {
            "slug": page_slug,
            "title": page_title,
            "entrypoint": subject_name,
            "version": version,
            "created_at": created_at,
            "url": (route + "index.html").lstrip("/"),
            "source": source_name,
            "artifacts": 0,
            "bytes": 0,
            "latest": True,
        })
        python_source_url = "source/python_source/" + source_name
        targets = _inspection_callables(source)
        page_urls = {
            target.qualified_name: (
                "callables/" + slugify(target.qualified_name) + "/index.html"
            )
            for target in targets
        }
        callable_systems = _callable_system_mapping(source, page_urls)
        for target in targets:
            callable_directory = temporary / "callables" / slugify(target.qualified_name)
            callable_source_name = slugify(target.qualified_name) + ".py"
            target_mapping = {
                "identity": target.qualified_name,
                "name": target.name,
                "kind": target.kind,
                "signature": target.signature,
                "parameters": [dict(item) for item in target.parameters],
                "page_url": page_urls[target.qualified_name],
            }
            target_systems = (
                {
                    "classes": [{"identity": target.owner, "methods": [target_mapping]}],
                    "functions": [],
                }
                if target.owner
                else {"classes": [], "functions": [target_mapping]}
            )
            build_source_inspection_page(
                _TextInspectionSubject(
                    source=target.source,
                    source_name=callable_source_name,
                    subject_name=target.qualified_name,
                    source_kind=target.kind,
                ),
                callable_directory,
                title=target.qualified_name,
                resource_route="./",
                callable_systems=target_systems,
                python_source_url="../../source/python_source/" + source_name,
                static_gallery=gallery,
            )
            callable_source = callable_directory / "source" / callable_source_name
            callable_source.parent.mkdir(parents=True, exist_ok=True)
            callable_source.write_text(target.source, encoding="utf-8")
        page_path = build_source_inspection_page(
            subject,
            temporary,
            title=page_title,
            resource_route="./",
            callable_systems=callable_systems,
            python_source_url=python_source_url,
            static_gallery=gallery,
        )
        source_relative = Path("source") / "python_source" / source_name
        published_source = temporary / source_relative
        published_source.parent.mkdir(parents=True, exist_ok=True)
        published_source.write_text(source, encoding="utf-8")
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "layout_version": BUNDLE_LAYOUT_VERSION,
            "program": {
                "slug": page_slug,
                "title": page_title,
                "entrypoint": subject_name,
                "kind": "python-ast-inspection",
            },
            "version": {
                "id": version,
                "source_sha256": source_digest,
                "builder": BUILDER_VERSION + "-ast",
            },
            "created_at": created_at,
            "page": {"path": page_path.name, "url": route + page_path.name},
            "source": {
                "path": source_relative.as_posix(),
                "filename": source_name,
            },
            "compiler": {
                "mode": "python-ast-inspection",
                "source_kind": source_kind,
                "callable_systems": len(targets),
            },
            "artifacts": _artifact_inventory(temporary),
        }
        (temporary / "bundle.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if final_directory.exists():
            existing = load_program_bundle(final_directory)
            shutil.rmtree(temporary)
            return existing
        temporary.replace(final_directory)
        return load_program_bundle(final_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def slugify(value: str) -> str:
    """Return a conservative URL/path component."""

    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if not slug:
        raise ValueError("program slug is empty after normalization")
    return slug[:80].rstrip("-")


def _literal_page_config(module: ast.Module) -> dict[str, Any]:
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(isinstance(target, ast.Name) and target.id == "TURING_PAGE" for target in targets):
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (TypeError, ValueError) as error:
            raise ValueError("TURING_PAGE must be a literal dictionary") from error
        if not isinstance(value, dict):
            raise ValueError("TURING_PAGE must be a literal dictionary")
        return value
    return {}


def discover_source_contract(
    source: str,
    *,
    entrypoint: str | None = None,
    title: str | None = None,
    slug: str | None = None,
    probes: Mapping[str, Any] | None = None,
) -> SourceContract:
    """Inspect a Python module and select its page contract.

    A source file may declare a literal ``TURING_PAGE`` dictionary.  Request
    arguments override it.  No source code is imported or executed here.
    """

    module = ast.parse(source)
    config = _literal_page_config(module)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    public = [node for node in functions if not node.name.startswith("_")]
    if not public:
        raise ValueError("source defines no public top-level function")
    selected = entrypoint or config.get("entrypoint")
    if selected is None:
        names = {node.name for node in public}
        selected = next((name for name in ("render", "kernel") if name in names), public[-1].name)
    function = next((node for node in functions if node.name == selected), None)
    if function is None:
        raise ValueError(f"entrypoint {selected!r} is not a top-level function")
    if function.args.vararg or function.args.kwarg:
        raise ValueError("entrypoint *args and **kwargs cannot define a page ABI")

    configured_feeds = config.get("feeds", {})
    if not isinstance(configured_feeds, dict):
        raise ValueError("TURING_PAGE['feeds'] must be a mapping")
    merged_feeds = {**configured_feeds, **dict(probes or {})}
    parameters = [
        *(argument.arg for argument in function.args.posonlyargs),
        *(argument.arg for argument in function.args.args),
        *(argument.arg for argument in function.args.kwonlyargs),
    ]
    unknown = set(merged_feeds) - set(parameters)
    if unknown:
        raise ValueError(f"probe values name unknown parameters: {sorted(unknown)!r}")

    expressions = config.get("feed_expressions", {})
    if not isinstance(expressions, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in expressions.items()
    ):
        raise ValueError("TURING_PAGE['feed_expressions'] must map names to strings")
    unknown_expressions = set(expressions) - set(parameters)
    if unknown_expressions:
        raise ValueError(
            f"feed expressions name unknown parameters: {sorted(unknown_expressions)!r}"
        )

    state_feedback = config.get("state_feedback", {})
    if not isinstance(state_feedback, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in state_feedback.items()
    ):
        raise ValueError(
            "TURING_PAGE['state_feedback'] must map input names to output names"
        )
    unknown_state_inputs = set(state_feedback) - set(parameters)
    if unknown_state_inputs:
        raise ValueError(
            "state feedback names unknown input parameters: "
            f"{sorted(unknown_state_inputs)!r}"
        )

    page_title = str(title or config.get("title") or selected.replace("_", " ").title())
    page_slug = slugify(str(slug or config.get("slug") or selected))
    width = int(config.get("width", 64))
    height = int(config.get("height", 40))
    probe_size = int(config.get("probe_size", 4))
    unroll_limit = int(config.get("unroll_limit", 4096))
    render_fps = float(config.get("render_fps", 30.0))
    if min(width, height, probe_size) < 1:
        raise ValueError("width, height, and probe_size must be positive")
    if unroll_limit < 0:
        raise ValueError("unroll_limit must be non-negative")
    if not math.isfinite(render_fps) or render_fps <= 0.0:
        raise ValueError("render_fps must be finite and positive")
    return SourceContract(
        entrypoint=str(selected),
        title=page_title,
        slug=page_slug,
        feeds=merged_feeds,
        feed_expressions=dict(expressions),
        width=width,
        height=height,
        probe_size=probe_size,
        backend=str(config.get("backend", "c")),
        remove_loops=bool(config.get("remove_loops", True)),
        unroll_limit=unroll_limit,
        state_feedback=dict(state_feedback),
        render_fps=render_fps,
        autostart=bool(config.get("autostart", False)),
    )


def _probe_value(specification: Any, size: int):
    import numpy as np

    if isinstance(specification, Mapping):
        if set(specification) == {"literal"}:
            return specification["literal"]
        if set(specification) == {"values"}:
            return np.asarray(specification["values"], dtype=np.float64)
        raise ValueError("a feed mapping must contain exactly 'literal' or 'values'")
    if isinstance(specification, (list, tuple)):
        return np.asarray(specification, dtype=np.float64)
    if specification is None:
        return np.zeros(size, dtype=np.float64)
    if isinstance(specification, (bool, int, float)):
        return np.full(size, specification, dtype=np.float64)
    raise TypeError(f"unsupported probe value {specification!r}")


def _entrypoint_parameters(source: str, entrypoint: str) -> tuple[str, ...]:
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint
    )
    return tuple(
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    )


def _content_version(
    source: str,
    contract: SourceContract,
    *,
    presentation_shader: str | None = None,
    presentation_document: str | None = None,
    shader_configuration: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = {
        "builder": BUILDER_VERSION,
        "source_sha256": source_digest,
        "entrypoint": contract.entrypoint,
        "feeds": contract.feeds,
        "feed_expressions": contract.feed_expressions,
        "backend": contract.backend,
        "remove_loops": contract.remove_loops,
        "unroll_limit": contract.unroll_limit,
        "state_feedback": dict(contract.state_feedback),
        "render_fps": contract.render_fps,
        "autostart": contract.autostart,
        "presentation_shader_sha256": (
            hashlib.sha256(presentation_shader.encode("utf-8")).hexdigest()
            if presentation_shader is not None else None
        ),
        "presentation_document_sha256": (
            hashlib.sha256(presentation_document.encode("utf-8")).hexdigest()
            if presentation_document is not None else None
        ),
        "shader_configuration": dict(shader_configuration or {}),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"v{BUNDLE_LAYOUT_VERSION}-{digest[:16]}", source_digest


def _write_sources(
    directory: Path,
    source: str,
    filename: str,
    sources: Any,
    *,
    presentation_shader: str | None = None,
) -> list[dict[str, Any]]:
    entries = [{
        "language": "python_source",
        "title": "Original Python",
        "source": source,
        "available": True,
        "reason": "",
        "highlight": "python",
        "lines": source.count("\n") + 1,
        "filename": Path(filename).name,
    }]
    if presentation_shader is not None:
        entries.append({
            "language": "webgl",
            "title": "WebGL presentation shader",
            "source": presentation_shader,
            "available": True,
            "reason": "",
            "highlight": "c",
            "lines": presentation_shader.count("\n") + 1,
            "filename": "surface.frag.glsl",
            "role": "shader-surface",
        })
    if sources is not None:
        entries.extend(sources.to_mapping()["sources"])
    published: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        body = str(item.pop("source", "") or "")
        if item.get("available"):
            language = str(item["language"])
            extension = _SOURCE_EXTENSIONS.get(language, "txt")
            output_name = str(item.get("filename") or f"{language}.{extension}")
            role = str(item.get("role") or "")
            relative = (
                Path("source") / "roles" / role / output_name
                if role == "shader-surface"
                else Path("source") / language / output_name
            )
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Backend source hashes are verification identities.  Preserve
            # the emitter's LF bytes even on Windows so the published file is
            # byte-for-byte the artifact that was compiled and checked.
            path.write_text(body, encoding="utf-8", newline="\n")
            item.update({
                "url": relative.as_posix(),
                "filename": output_name,
                "bytes": len(body.encode("utf-8")),
            })
        published.append(item)
    return published


def _shader_execution_descriptor(
    published_sources: list[dict[str, Any]],
    shell_io: Mapping[str, Any] | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Select the explicitly-role-marked browser presentation shader."""

    for source in published_sources:
        if (
            source.get("language") == "webgl"
            and source.get("role") == "shader-surface"
            and source.get("available")
            and source.get("url")
        ):
            descriptor = {
                "url": str(source["url"]),
                "language": "webgl2-glsl-es",
                "stage": "fragment",
                "role": "shader-surface",
                "autostart": True,
                "execution": {
                    "continuous": True,
                    "prefer_contiguous": True,
                },
            }
            if shell_io:
                descriptor["io"] = dict(shell_io)
            if configuration:
                descriptor["configuration"] = dict(configuration)
            return descriptor
    # Desktop GLSL is deliberately not a fallback. Its SSBO binding/channel
    # arena needs the dedicated memory handler documented in the GLSL
    # ingestion layer and cannot be executed by a WebGL canvas.
    return None


def _artifact_inventory(directory: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == "bundle.json":
            continue
        body = path.read_bytes()
        inventory.append({
            "path": path.relative_to(directory).as_posix(),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        })
    return inventory


def load_program_bundle(directory: Path) -> ProgramBundle:
    manifest_path = directory / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"unsupported bundle schema in {manifest_path}")
    return ProgramBundle(
        directory=directory,
        manifest_path=manifest_path,
        page_path=directory / str(manifest["page"]["path"]),
        manifest=manifest,
    )


def build_program_bundle(
    source: str,
    destination: str | Path,
    *,
    source_filename: str = "program.py",
    entrypoint: str | None = None,
    title: str | None = None,
    slug: str | None = None,
    probes: Mapping[str, Any] | None = None,
    include_backends: bool = True,
    include_mathematics: bool = True,
    presentation_shader: str | None = None,
    presentation_document: str | None = None,
    shader_configuration: Mapping[str, Any] | None = None,
) -> ProgramBundle:
    """Compile source and atomically publish its complete versioned bundle."""

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    contract = discover_source_contract(
        source, entrypoint=entrypoint, title=title, slug=slug, probes=probes
    )
    version, source_digest = _content_version(
        source,
        contract,
        presentation_shader=presentation_shader,
        presentation_document=presentation_document,
        shader_configuration=shader_configuration,
    )
    destination = resolve_publish_root(destination)
    versions = destination / "site" / "programs" / contract.slug / "versions"
    final_directory = versions / version
    if final_directory.is_dir() and (final_directory / "bundle.json").is_file():
        return load_program_bundle(final_directory)
    versions.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=versions))
    compile_log = io.StringIO()
    try:
        from ..common.tensors.accelerator_backends.aot_compile import (
            compile_ast_aot,
            project_public_numerical_program,
        )
        from ..common.tensors.topological_reducer import reduce_abstract_tensor_topology
        from ..transmogrifier.graph.graph_express2 import ProcessGraph
        from .backend_sources import collect_backend_sources
        from .fused_program_wasm_backend import emit_wasm_module, required_steps
        from .shell_telemetry import TelemetryChannel, summarize_process_graph
        from .sympy_math_renderer import render_reduced_program_mathematics
        from .wasm_class_coordinator import (
            build_class_inventory,
            emit_wasm_class_coordinator,
        )
        from .wasm_class_modules import (
            build_embedded_class_graph,
            emit_class_modules,
            partition_reduced_program,
        )
        from .wasm_html_shell import emit_html_shell

        parameter_names = _entrypoint_parameters(source, contract.entrypoint)
        feeds = {
            name: _probe_value(contract.feeds.get(name), contract.probe_size)
            for name in parameter_names
        }
        channel = TelemetryChannel(name=f"program:{contract.slug}")
        with contextlib.redirect_stdout(compile_log), contextlib.redirect_stderr(compile_log):
            graph = ProcessGraph(materialize_memory=False)
            graph.build_from_ast(ast.parse(source))
            reduce_abstract_tensor_topology(graph)
            aot = compile_ast_aot(
                source,
                contract.entrypoint,
                feeds,
                backend=contract.backend,
                remove_loops=contract.remove_loops,
                unroll_limit=contract.unroll_limit,
                precompile_only=True,
                python_bindings=globals(),
            )
            # This extraction is deliberately late in compilation.  ``aot``
            # already represents Python AST -> ProcessGraph -> planned
            # control/map/numerical compilation.  ``program`` is only the
            # internal numerical member required by this particular Wasm
            # emitter; it is not the source compiler, an application API, or
            # evidence that the Python recompiler is numerics-only.
            program = project_public_numerical_program(aot)
            module = emit_wasm_module(
                program, name=contract.entrypoint, dtype="float64"
            )
            if contract.state_feedback:
                entry = module.api.entry_points[0]
                input_names = {
                    item.name for item in entry.parameters if item.role == "input"
                }
                output_names = {
                    item.name for item in entry.parameters if item.role == "output"
                }
                missing_inputs = set(contract.state_feedback) - input_names
                missing_outputs = set(contract.state_feedback.values()) - output_names
                if missing_inputs or missing_outputs:
                    raise ValueError(
                        "compiled state feedback does not match the Python ABI; "
                        f"missing inputs={sorted(missing_inputs)!r}; "
                        f"missing outputs={sorted(missing_outputs)!r}"
                    )
                module = replace(
                    module,
                    api=replace(
                        module.api,
                        metadata={
                            **dict(module.api.metadata),
                            "state_feedback": dict(contract.state_feedback),
                            "render_fps": float(contract.render_fps),
                            "autostart": bool(contract.autostart),
                        },
                    ),
                )
        if not module.complete:
            raise RuntimeError(module.shortfall_report())

        wasm_relative = Path("wasm") / f"{module.name}.wasm"
        wasm_path = temporary / wasm_relative
        wasm_path.parent.mkdir(parents=True, exist_ok=True)
        wasm_path.write_bytes(module.binary)

        card_directory = Path("wasm") / f"size-{DEFAULT_WASM_CARD_OPERATIONS}"
        specs = partition_reduced_program(
            program,
            chunk_size=DEFAULT_WASM_CARD_OPERATIONS,
            owner_name=contract.entrypoint,
        )
        card_modules = emit_class_modules(
            specs, dtype="float64", link_calls=False, shared_memory=True
        )
        incomplete_cards = [
            card_modules[spec.index]
            for spec in specs
            if not card_modules[spec.index].complete
        ]
        if incomplete_cards:
            raise RuntimeError("\n".join(
                card.shortfall_report() for card in incomplete_cards
            ))
        card_manifest = build_embedded_class_graph(
            specs,
            card_modules,
            program,
            entrypoint=contract.entrypoint,
            embed_binaries=False,
            module_dir=card_directory.as_posix(),
        )
        inventory = build_class_inventory(card_manifest)
        coordinator = emit_wasm_class_coordinator(
            inventory,
            name=f"{contract.entrypoint}_coordinator_{DEFAULT_WASM_CARD_OPERATIONS}",
        )
        card_manifest.update({
            "region_steps": DEFAULT_WASM_CARD_OPERATIONS,
            "class_inventory": inventory.to_mapping(),
            "coordinator": {
                "name": coordinator.name,
                "url": (card_directory / f"{coordinator.name}.wasm").as_posix(),
                "entry": "run_range",
                "memory_import": {"module": "env", "field": "memory"},
                "method_count": len(inventory.methods),
            },
        })
        card_output_directory = temporary / card_directory
        card_output_directory.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            (card_output_directory / f"{spec.module_name}.wasm").write_bytes(
                card_modules[spec.index].binary
            )
        (card_output_directory / f"{coordinator.name}.wasm").write_bytes(
            coordinator.binary
        )
        (card_output_directory / "class-inventory.json").write_text(
            json.dumps(inventory.to_mapping(), indent=2), encoding="utf-8"
        )

        sources = None
        if include_backends:
            with contextlib.redirect_stdout(compile_log), contextlib.redirect_stderr(compile_log):
                sources = collect_backend_sources(
                    aot,
                    numerical_name=contract.entrypoint,
                    control_name=f"{contract.entrypoint}_control",
                    channel=channel,
                    wasm_source=module.source,
                    program=program,
                )
        published_sources = _write_sources(
            temporary,
            source,
            Path(source_filename).name,
            sources,
            presentation_shader=presentation_shader,
        )
        effective_shader_configuration = dict(shader_configuration or {})
        if presentation_document is not None:
            document_relative = (
                Path("source") / "roles" / "shader-document" / "index.html"
            )
            document_path = temporary / document_relative
            document_path.parent.mkdir(parents=True, exist_ok=True)
            document_path.write_text(
                presentation_document,
                encoding="utf-8",
                newline="\n",
            )
            effective_shader_configuration["document_url"] = (
                document_relative.as_posix()
            )

        fortran_verification = None
        if sources is not None:
            fortran_source = next(
                (
                    item for item in sources.sources
                    if item.language == "fortran" and item.available
                ),
                None,
            )
            fortran_artifact = (
                None if fortran_source is None else fortran_source.artifact
            )
            if fortran_artifact is not None:
                from ..common.tensors.fused_ir import ordered_feed_ids
                from .fortran_fidelity import verify_fortran_module
                from .ssa_fortran_backend import fortran_compiler

                source_path = temporary / "source" / "fortran" / "fortran.f90"
                api_path = source_path.with_suffix(".api.yaml")
                fortran_artifact.api.write(api_path)
                if fortran_compiler() is not None:
                    feed_origins = dict(
                        (getattr(program, "extras", None) or {}).get(
                            "capture_feed_origins", {}
                        )
                    )
                    feed_values = {}
                    for feed_index, feed_id in enumerate(ordered_feed_ids(program)):
                        origin = feed_origins.get(
                            feed_id, feed_origins.get(str(feed_id), {})
                        )
                        parameter_name = origin.get("binding_name")
                        if parameter_name not in feeds:
                            parameter_name = parameter_names[feed_index]
                        feed_values[feed_id] = feeds[parameter_name]
                    proof = verify_fortran_module(
                        fortran_artifact,
                        program,
                        feed_values,
                        temporary / "native" / "fortran",
                        entrypoint=contract.entrypoint,
                    )
                    proof_relative = Path("verification") / "fortran-fidelity.json"
                    proof_path = temporary / proof_relative
                    proof_path.parent.mkdir(parents=True, exist_ok=True)
                    proof_path.write_text(
                        json.dumps(proof, indent=2, allow_nan=True),
                        encoding="utf-8",
                    )
                    fortran_verification = {
                        "passed": True,
                        "case_count": proof["case_count"],
                        "source_sha256": proof["source_sha256"],
                        "url": proof_relative.as_posix(),
                    }
                else:
                    fortran_verification = {
                        "passed": None,
                        "reason": "no Fortran compiler was available at build time",
                    }

        mathematics = None
        math_error = ""
        if include_mathematics:
            try:
                document = render_reduced_program_mathematics(
                    program,
                    input_names=parameter_names,
                    program_name=contract.entrypoint,
                )
                payload = document.to_mapping()
                relative = Path("math") / "sympy-process-model.json"
                path = temporary / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(payload, separators=(",", ":")), encoding="utf-8"
                )
                mathematics = {
                    key: value for key, value in payload.items() if key != "equations"
                }
                mathematics.update({
                    "url": relative.as_posix(),
                    "filename": relative.name,
                    "bytes": path.stat().st_size,
                })
            except Exception as error:  # page generation survives optional projection refusal
                math_error = f"{type(error).__name__}: {error}"

        entry = module.api.entry_points[0]
        contiguous = {
            "name": module.name,
            "url": wasm_relative.as_posix(),
            "entry": module.api.entry,
            "inputs": [item.name for item in entry.parameters if item.role == "input"],
            "outputs": [item.name for item in entry.parameters if item.role == "output"],
            "value_type": module.api.metadata.get("value_type", "f64"),
            "element_bytes": module.api.metadata.get("element_bytes", 8),
            "memory_export": module.api.metadata.get("memory_export", "memory"),
            "reserved_bytes": module.api.metadata.get("reserved_bytes", 0),
            "operation_count": len(required_steps(program)),
        }
        route = f"/site/programs/{contract.slug}/versions/{version}/"
        shader_execution = _shader_execution_descriptor(
            published_sources,
            (module.api.metadata or {}).get("shell_io"),
            effective_shader_configuration,
        )
        shell = emit_html_shell(
            module.api,
            name="index",
            telemetry=channel,
            process_graph=summarize_process_graph(graph),
            origin_source="",
            feed_expressions=contract.feed_expressions,
            build_parameters={
                "bundle schema": BUNDLE_SCHEMA,
                "content version": version,
                "source SHA-256": source_digest,
                "steps": len(required_steps(program)),
            },
            default_width=contract.width,
            default_height=contract.height,
            backend_sources=published_sources,
            mathematics=mathematics,
            map_ir=aot.map_ir,
            class_graph={
                **card_manifest,
                "variants": {str(DEFAULT_WASM_CARD_OPERATIONS): card_manifest},
                "contiguous": contiguous,
                "runtime_version": BUILDER_VERSION,
            },
            resource_route=route,
            shader_execution=shader_execution,
        )
        page_path = shell.write(temporary)

        log_text = compile_log.getvalue().strip()
        if log_text:
            log_path = temporary / "build" / "compiler.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(log_text + "\n", encoding="utf-8")

        manifest = {
            "schema": BUNDLE_SCHEMA,
            "layout_version": BUNDLE_LAYOUT_VERSION,
            "program": {
                "slug": contract.slug,
                "title": contract.title,
                "entrypoint": contract.entrypoint,
            },
            "version": {
                "id": version,
                "source_sha256": source_digest,
                "builder": BUILDER_VERSION,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "page": {
                "path": "index.html",
                "url": route + "index.html",
                "mode": "shader-execution" if shader_execution else "inspection",
                "shader": shader_execution,
            },
            "source": {
                "path": f"source/python_source/{Path(source_filename).name}",
                "filename": Path(source_filename).name,
            },
            "compiler": {
                "backend": contract.backend,
                "remove_loops": contract.remove_loops,
                "unroll_limit": contract.unroll_limit,
                "mathematics_error": math_error,
                "fortran_verification": fortran_verification,
            },
            "artifacts": _artifact_inventory(temporary),
        }
        manifest_path = temporary / "bundle.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if final_directory.exists():
            existing = load_program_bundle(final_directory)
            shutil.rmtree(temporary)
            return existing
        temporary.replace(final_directory)
        return load_program_bundle(final_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "BUNDLE_LAYOUT_VERSION",
    "BUNDLE_SCHEMA",
    "BUILDER_VERSION",
    "DEFAULT_PUBLISH_ROOT",
    "ProgramBundle",
    "SourceContract",
    "TURING_REPOSITORY_ROOT",
    "build_program_bundle",
    "build_source_inspection_bundle",
    "build_source_inspection_page",
    "discover_source_contract",
    "load_program_bundle",
    "resolve_publish_root",
    "slugify",
]
