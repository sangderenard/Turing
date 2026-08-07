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
from functools import lru_cache
import hashlib
import importlib
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
import threading
import time
from typing import Any, Callable, Mapping, Sequence


BUNDLE_SCHEMA = "turing-program-bundle-v1"
BUNDLE_LAYOUT_VERSION = 1
BUILDER_VERSION = "site-bundle-v18"
DEFAULT_WASM_CARD_OPERATIONS = 2000
PROGRAM_BAKE_MODES = frozenset({"one_shot", "whole_program"})
PROGRAM_SCHEDULE_PREFERENCES = frozenset({"asap", "alap"})
TURING_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLISH_ROOT = TURING_REPOSITORY_ROOT.parent

# The default whole-screen passthrough every published page compiles,
# unconditionally, whether or not it is ever used -- see the compile step
# in ``build_program_bundle`` for why.
#
# The ``+ 0.0`` is load-bearing, not decoration: a bare ``return red, green,
# blue`` traces zero operations (AbstractTensor capture only records
# computed steps), so ``project_public_numerical_program`` finds no output
# to name and every backend reports "no outputs" as a shortfall -- this
# passthrough has never actually compiled without it. The no-op arithmetic
# forces one real captured step per output, same visual result, and is what
# makes ``passthrough_module.complete`` (and its WebGPU sibling) ever True.
_PASSTHROUGH_SOURCE = (
    "def turing_passthrough(red, green, blue):\n"
    "    return red + 0.0, green + 0.0, blue + 0.0\n"
)

_SOURCE_EXTENSIONS = {
    "python_source": "py",
    "ssa": "ssa",
    "fortran": "f90",
    "spirv": "spvasm",
    "glsl": "comp.glsl",
    "webgl": "frag.glsl",
    "webgpu": "compute.wgsl",
    "wat": "wat",
    "numpy": "py",
    "torch": "py",
    "abstract_tensor": "py",
}

# A published bundle is a product of both authored source and the compiler
# implementation which lowered it.  Keep this list on the actual bundle path:
# unrelated repository work must not churn immutable URLs, while a change to
# any stage capable of changing the emitted ABI/control/module bytes must.
_BUNDLE_COMPILER_IMPLEMENTATION_FILES = (
    Path(__file__),
    Path(__file__).with_name("backend_sources.py"),
    Path(__file__).with_name("fused_program_wasm_backend.py"),
    Path(__file__).with_name("glsl_deployment_strategy.py"),
    Path(__file__).with_name("loop_composer.py"),
    Path(__file__).with_name("precompile_to_ssa.py"),
    Path(__file__).with_name("process_graph_fusion.py"),
    Path(__file__).with_name("ssa_webgpu_backend.py"),
    Path(__file__).with_name("shader_stages.py"),
    Path(__file__).with_name("fused_program_webgl_backend.py"),
    Path(__file__).with_name("machine_targets.py"),
    Path(__file__).with_name("wasm_class_coordinator.py"),
    Path(__file__).with_name("wasm_class_modules.py"),
    Path(__file__).with_name("wasm_html_shell.py"),
    Path(__file__).parents[1] / "common" / "tensors" / "abstraction.py",
    Path(__file__).parents[1] / "common" / "tensors" / "topological_reducer.py",
    Path(__file__).parents[1] / "common" / "tensors"
    / "accelerator_backends" / "aot_compile.py",
    Path(__file__).parents[1] / "common" / "tensors"
    / "accelerator_backends" / "c_primitive_program.py",
    Path(__file__).parents[1] / "transmogrifier" / "graph"
    / "graph_deep_compiler.py",
    Path(__file__).parents[1] / "transmogrifier" / "graph"
    / "graph_express2.py",
    Path(__file__).parents[1] / "transmogrifier" / "operator_defs.py",
)


@lru_cache(maxsize=1)
def _bundle_compiler_digest() -> str:
    """Fingerprint code that can change a generated program bundle."""

    digest = hashlib.sha256()
    for path in _BUNDLE_COMPILER_IMPLEMENTATION_FILES:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"bundle compiler implementation file is missing: {resolved}"
            )
        digest.update(resolved.relative_to(TURING_REPOSITORY_ROOT).as_posix().encode(
            "utf-8"
        ))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
    bake_mode: str
    schedule_preference: str
    remove_loops: bool
    unroll_limit: int
    state_feedback: Mapping[str, str]
    render_fps: float
    autostart: bool
    presentation_entrypoint: str | None
    shader_configuration: Mapping[str, Any]
    audio_configuration: Mapping[str, Any]
    constant_map: Mapping[str, Any]
    mutable_parameters: tuple[str, ...]


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


def refresh_static_gallery(destination: str | Path) -> Path:
    """Refresh the root shell's offline gallery from published manifests.

    The loopback server discovers manifests dynamically, while GitHub Pages has
    no server-side catalogue.  Keeping this small rewrite beside bundle
    publication gives prebuilt and compiler-produced interiors the same public
    discovery path.
    """
    root = resolve_publish_root(destination)
    page = root / "index.html"
    html = page.read_text(encoding="utf-8")
    replacement = "const STATIC_GALLERY = " + json.dumps(
        static_gallery_items(root), default=str,
    ) + ";"
    updated, count = re.subn(
        r"const STATIC_GALLERY = \[.*?\];", lambda _match: replacement, html,
        count=1, flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError(f"root shell has no unique STATIC_GALLERY marker: {page}")
    page.write_text(updated, encoding="utf-8")
    return page


def publish_prebuilt_program_bundle(
    *,
    destination: str | Path,
    slug: str,
    title: str,
    entrypoint: str,
    html: str,
    source_filename: str,
    source: str | bytes,
    artifacts: Mapping[str, str | bytes] | None = None,
    runtime: Mapping[str, Any] | None = None,
    refresh_gallery: bool = False,
) -> ProgramBundle:
    """Publish an already-assembled program interior through the common ABI.

    This is the common seam for Dream documents, native-shell products, and
    other programs whose HTML/runtime artifacts already exist.  It deliberately
    shares the immutable version layout and manifest inventory used by compiled
    Python bundles instead of introducing a second gallery format.
    """
    root = resolve_publish_root(destination)
    page_slug = slugify(slug)
    source_body = source.encode("utf-8") if isinstance(source, str) else bytes(source)
    clean_html = "\n".join(line.rstrip() for line in html.splitlines())
    if html.endswith(("\n", "\r")):
        clean_html += "\n"
    bodies: dict[str, bytes] = {
        "index.html": clean_html.encode("utf-8"),
        f"source/{source_filename}": source_body,
    }
    for relative, body in (artifacts or {}).items():
        normalized = Path(relative).as_posix().lstrip("/")
        if not normalized or normalized.startswith("../") or "/../" in normalized:
            raise ValueError(f"bundle artifact escapes its version directory: {relative}")
        if normalized in bodies or normalized == "bundle.json":
            raise ValueError(f"duplicate or reserved bundle artifact: {relative}")
        bodies[normalized] = body.encode("utf-8") if isinstance(body, str) else bytes(body)

    identity = hashlib.sha256()
    identity.update(BUILDER_VERSION.encode("utf-8"))
    for relative, body in sorted(bodies.items()):
        identity.update(relative.encode("utf-8"))
        identity.update(b"\0")
        identity.update(body)
        identity.update(b"\0")
    identity.update(json.dumps(runtime or {}, sort_keys=True, default=str).encode("utf-8"))
    version_id = "v1-" + identity.hexdigest()[:16]
    versions = root / "site" / "programs" / page_slug / "versions"
    final_directory = versions / version_id
    if final_directory.is_dir() and (final_directory / "bundle.json").is_file():
        bundle = load_program_bundle(final_directory)
        if refresh_gallery:
            refresh_static_gallery(root)
        return bundle

    versions.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=versions))
    try:
        for relative, body in bodies.items():
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        route = f"/site/programs/{page_slug}/versions/{version_id}/index.html"
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "layout_version": BUNDLE_LAYOUT_VERSION,
            "program": {
                "slug": page_slug, "title": title, "entrypoint": entrypoint,
            },
            "version": {
                "id": version_id,
                "source_sha256": hashlib.sha256(source_body).hexdigest(),
                "builder": BUILDER_VERSION,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "page": {"path": "index.html", "url": route},
            "source": {
                "path": f"source/{source_filename}",
                "filename": source_filename,
            },
            "compiler": {"backend": "prebuilt-program-interior"},
            "runtime": dict(runtime or {}),
            "artifacts": _artifact_inventory(temporary),
        }
        (temporary / "bundle.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        os.replace(temporary, final_directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    bundle = load_program_bundle(final_directory)
    if refresh_gallery:
        refresh_static_gallery(root)
    return bundle


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


def _with_heartbeat(
    progress: "Callable[[str], None] | None",
    label: str,
    call: "Callable[[], Any]",
    *,
    interval: float = 1.0,
) -> Any:
    """Run one call that cannot itself be subdivided (a single C-level
    ``ast.parse``/``ast.literal_eval`` invocation) while a background
    thread keeps emitting proof of life on its own timer. There is no real
    sub-step to report from inside a call like that -- this does not
    invent one -- it only makes silence during it distinguishable from a
    hang by continuing to report elapsed time until the call returns.
    """

    if progress is None:
        return call()
    started = time.monotonic()
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(interval):
            progress(f"{label}: still running ({time.monotonic() - started:.1f}s elapsed)")

    heartbeat = threading.Thread(target=_beat, daemon=True)
    heartbeat.start()
    try:
        return call()
    finally:
        stop.set()
        heartbeat.join()
        progress(f"{label}: finished ({time.monotonic() - started:.1f}s total)")


def _literal_page_config(
    module: ast.Module,
    progress: "Callable[[str], None] | None" = None,
) -> dict[str, Any]:
    if progress is not None:
        progress(f"scanning {len(module.body)} top-level statements for TURING_PAGE")
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(isinstance(target, ast.Name) and target.id == "TURING_PAGE" for target in targets):
            continue
        if progress is not None:
            progress("found TURING_PAGE assignment; literal_eval-ing its value")
        try:
            value = _with_heartbeat(
                progress, "TURING_PAGE literal_eval",
                lambda: ast.literal_eval(statement.value),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("TURING_PAGE must be a literal dictionary") from error
        if not isinstance(value, dict):
            raise ValueError("TURING_PAGE must be a literal dictionary")
        if progress is not None:
            progress(f"TURING_PAGE literal_eval complete: {len(value)} top-level keys")
        return value
    if progress is not None:
        progress("no TURING_PAGE assignment found")
    return {}


def discover_source_contract(
    source: str,
    *,
    entrypoint: str | None = None,
    title: str | None = None,
    slug: str | None = None,
    probes: Mapping[str, Any] | None = None,
    bake_mode: str | None = None,
    schedule_preference: str | None = None,
    progress: "Callable[[str], None] | None" = None,
) -> SourceContract:
    """Inspect a Python module and select its page contract.

    A source file may declare a literal ``TURING_PAGE`` dictionary.  Request
    arguments override it.  No source code is imported or executed here.
    """

    module = _with_heartbeat(progress, "ast.parse", lambda: ast.parse(source))
    config = _literal_page_config(module, progress)
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

    constant_map = config.get("constants", config.get("constant_map", {}))
    if not isinstance(constant_map, dict):
        raise ValueError("TURING_PAGE['constants'] must be a mapping")
    unknown_constants = set(constant_map) - set(parameters)
    if unknown_constants:
        raise ValueError(
            "configured constants name unknown parameters: "
            f"{sorted(unknown_constants)!r}"
        )
    if progress is not None:
        progress(f"validating {len(constant_map)} configured constants as literals")
    for name, value in constant_map.items():
        if progress is not None:
            size = len(value) if isinstance(value, (list, tuple, dict, set)) else 1
            progress(f"validating constant {name!r} ({size} element(s))")
        try:
            _with_heartbeat(
                progress, f"validating constant {name!r}",
                lambda value=value: ast.literal_eval(ast.parse(repr(value), mode="eval").body),
            )
        except (SyntaxError, ValueError, TypeError) as error:
            raise ValueError(
                f"configured constant {name!r} must be a Python literal"
            ) from error
        if progress is not None:
            progress(f"constant {name!r} validated")

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

    presentation_entrypoint = config.get("presentation_entrypoint")
    if presentation_entrypoint is not None:
        presentation_entrypoint = str(presentation_entrypoint)
        presentation = next(
            (node for node in functions if node.name == presentation_entrypoint), None
        )
        if presentation is None:
            raise ValueError(
                f"presentation entrypoint {presentation_entrypoint!r} is not a "
                "top-level function"
            )
    shader_config = config.get("shader_configuration", {})
    if not isinstance(shader_config, dict):
        raise ValueError("TURING_PAGE['shader_configuration'] must be a mapping")
    audio_config = config.get("audio", {})
    if not isinstance(audio_config, dict):
        raise ValueError("TURING_PAGE['audio'] must be a mapping")

    page_title = str(title or config.get("title") or selected.replace("_", " ").title())
    page_slug = slugify(str(slug or config.get("slug") or selected))
    width = int(config.get("width", 64))
    height = int(config.get("height", 40))
    probe_size = int(config.get("probe_size", 4))
    unroll_limit = int(config.get("unroll_limit", 4096))
    selected_bake_mode = str(
        bake_mode if bake_mode is not None
        else config.get("bake_mode", "whole_program")
    ).strip().lower().replace("-", "_")
    selected_schedule_preference = str(
        schedule_preference if schedule_preference is not None
        else config.get("schedule_preference", "alap")
    ).strip().lower()
    render_fps = float(config.get("render_fps", 30.0))
    if min(width, height, probe_size) < 1:
        raise ValueError("width, height, and probe_size must be positive")
    if unroll_limit < 0:
        raise ValueError("unroll_limit must be non-negative")
    if selected_bake_mode not in PROGRAM_BAKE_MODES:
        raise ValueError(
            "TURING_PAGE['bake_mode'] must be 'one_shot' or "
            f"'whole_program', not {selected_bake_mode!r}"
        )
    forbidden_constants = set(constant_map) & set(state_feedback)
    if forbidden_constants:
        raise ValueError(
            "state-feedback parameters are mutable and cannot be configured "
            "as constants: " + ", ".join(sorted(forbidden_constants))
        )
    if selected_schedule_preference not in PROGRAM_SCHEDULE_PREFERENCES:
        raise ValueError(
            "TURING_PAGE['schedule_preference'] must be 'asap' or "
            f"'alap', not {selected_schedule_preference!r}"
        )
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
        bake_mode=selected_bake_mode,
        schedule_preference=selected_schedule_preference,
        remove_loops=bool(config.get("remove_loops", True)),
        unroll_limit=unroll_limit,
        state_feedback=dict(state_feedback),
        render_fps=render_fps,
        autostart=bool(config.get("autostart", False)),
        presentation_entrypoint=presentation_entrypoint,
        shader_configuration=dict(shader_config),
        audio_configuration=dict(audio_config),
        constant_map=dict(constant_map),
        mutable_parameters=tuple(state_feedback),
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
    include_backends: bool = True,
    backend_targets: Sequence[str] | None = None,
) -> tuple[str, str]:
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = {
        "builder": BUILDER_VERSION,
        "compiler_implementation_sha256": _bundle_compiler_digest(),
        "source_sha256": source_digest,
        "entrypoint": contract.entrypoint,
        "feeds": contract.feeds,
        "feed_expressions": contract.feed_expressions,
        "backend": contract.backend,
        "bake_mode": contract.bake_mode,
        "schedule_preference": contract.schedule_preference,
        "remove_loops": contract.remove_loops,
        "unroll_limit": contract.unroll_limit,
        "state_feedback": dict(contract.state_feedback),
        "render_fps": contract.render_fps,
        "autostart": contract.autostart,
        "presentation_entrypoint": contract.presentation_entrypoint,
        "contract_shader_configuration": dict(contract.shader_configuration),
        "audio_configuration": dict(contract.audio_configuration),
        "constant_map": dict(contract.constant_map),
        "presentation_shader_sha256": (
            hashlib.sha256(presentation_shader.encode("utf-8")).hexdigest()
            if presentation_shader is not None else None
        ),
        "presentation_document_sha256": (
            hashlib.sha256(presentation_document.encode("utf-8")).hexdigest()
            if presentation_document is not None else None
        ),
        "shader_configuration": dict(shader_configuration or {}),
        # Different backend selections produce different published source
        # tabs from the same compilation, so they need different content
        # versions too -- otherwise a second build with a narrower
        # backend_targets list would silently reuse the first build's
        # directory (and its wider set of tabs) instead of actually
        # reflecting the request.
        "include_backends": include_backends,
        "backend_targets": sorted(str(item).lower() for item in backend_targets or ()),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"v{BUNDLE_LAYOUT_VERSION}-{digest[:16]}", source_digest, digest[:16]


_COMPILER_VERSION_NUMBER = int(re.search(r"(\d+)$", BUILDER_VERSION).group(1))
_SEQUENTIAL_VERSION_PATTERN = re.compile(
    rf"^v{_COMPILER_VERSION_NUMBER}\.(\d{{3}})-"
)


def _next_sequential_version(versions_dir: Path, content_hash16: str) -> str:
    """A fresh, never-reused version name for --full-style forced rebuilds.

    Format: ``v<compiler version>.<iteration>-<date>-<content hash>``. The
    compiler-version number comes from ``BUILDER_VERSION`` (the same value
    every bundle already records under ``compiler.implementation_sha256``'s
    neighbor fields); the iteration counter is scoped to that compiler
    version and resets when it changes, so "how many times has this program
    been rebuilt under the current compiler" is legible directly from the
    version name. Unlike the default content-addressed version (same input
    -> same directory, see ``_content_version``), this always produces a
    new, previously-unused name -- deliberately: a forced rebuild is asking
    for an additional version on record, not a lookup.
    """

    highest = 0
    if versions_dir.is_dir():
        for entry in versions_dir.iterdir():
            match = _SEQUENTIAL_VERSION_PATTERN.match(entry.name)
            if match:
                highest = max(highest, int(match.group(1)))
    iteration = highest + 1
    date_stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"v{_COMPILER_VERSION_NUMBER}.{iteration:03d}-{date_stamp}-{content_hash16}"


def _write_sources(
    directory: Path,
    source: str,
    filename: str,
    sources: Any,
    *,
    presentation_shader: str | None = None,
    presentation_shader_wgsl: str | None = None,
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
    if presentation_shader_wgsl is not None:
        entries.append({
            "language": "webgpu",
            "title": "WebGPU presentation shader",
            "source": presentation_shader_wgsl,
            "available": True,
            "reason": "",
            "highlight": "rust",
            "lines": presentation_shader_wgsl.count("\n") + 1,
            "filename": "surface.compute.wgsl",
            "role": "shader-surface",
        })
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


# Priority order for the published page's live execution surface: WebGPU
# compute first (real dispatch, no draw-buffer cap), WebGL 2 fragment-raster
# second (the only path older browsers have), plain 2D canvas last (needs no
# GPU shading language at all -- just paints the WASM numeric output, see
# wasm_html_shell.py's canvas2d branch). Which one a page actually runs is a
# runtime feature-detection choice in the browser (navigator.gpu, then a
# webgl2 context, then always-available 2D); this only decides which
# candidates exist to choose from and in what order.
_SHADER_SURFACE_LANGUAGES: tuple[tuple[str, str, str], ...] = (
    ("webgpu", "wgsl", "compute"),
    ("webgl", "webgl2-glsl-es", "fragment"),
)


def _shader_execution_descriptor(
    published_sources: list[dict[str, Any]],
    shell_io: Mapping[str, Any] | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build the priority-ordered list of shader-surface candidates.

    The returned mapping's top level still describes the highest-priority
    candidate (``url``/``language``/``stage``/``role``) for existing readers
    (the manifest, the HTML shell's inspection panel); ``candidates`` adds
    the full ordered list, including a GPU-free ``canvas2d`` entry that
    needs no published source at all, so the page always has *something* to
    fall back to.
    """

    candidates: list[dict[str, Any]] = []
    by_language = {
        str(source.get("language")): source for source in published_sources
        if source.get("role") == "shader-surface" and source.get("available")
        and source.get("url")
    }
    for source_language, wire_language, stage in _SHADER_SURFACE_LANGUAGES:
        source = by_language.get(source_language)
        if source is None:
            continue
        candidates.append({
            "url": str(source["url"]),
            "language": wire_language,
            "stage": stage,
            "role": "shader-surface",
        })
    if not candidates:
        # Desktop GLSL is deliberately never a candidate here. Its SSBO
        # binding/channel arena needs the dedicated memory handler
        # documented in the GLSL ingestion layer and cannot be executed by
        # a browser canvas of any kind. No compiled shader source at all
        # means no shader-execution page, same as before this function grew
        # a candidate list -- a bare 2D-canvas fallback is only offered
        # below as a *last resort after* a real candidate, never as the
        # sole reason to switch a page into execution mode.
        return None
    # Always-available last resort once at least one real shader source
    # exists: paint the WASM numeric output straight to a 2D canvas, no
    # shader compilation of any kind. wasm_html_shell.py's runtime tries
    # this only if every GPU-backed candidate above it fails to acquire a
    # context.
    if shell_io:
        candidates.append({
            "url": None,
            "language": "canvas2d",
            "stage": "none",
            "role": "shader-surface",
        })
    descriptor = dict(candidates[0])
    descriptor["autostart"] = True
    descriptor["execution"] = {"continuous": True, "prefer_contiguous": True}
    descriptor["candidates"] = candidates
    if shell_io:
        descriptor["io"] = dict(shell_io)
    if configuration:
        descriptor["configuration"] = dict(configuration)
    return descriptor


def _write_program_origin(
    destination: Path,
    contract: SourceContract,
    source: str,
    source_filename: str,
    *,
    backend_targets: Sequence[str] | None,
    include_backends: bool,
    include_mathematics: bool,
) -> Path:
    """Record how to rebuild this *program* from scratch, program-scoped.

    One file per slug (``site/programs/<slug>/origin.json``), not one per
    version -- a program's origin (its source, entrypoint, probes, backend
    selection) is a property of the program, not of any single compiled
    snapshot, and a program can carry many versions under one origin. The
    full source text is embedded directly rather than referencing a
    version's published copy, so regeneration is self-contained even when a
    program has zero, one, or many versions on disk: there is exactly one
    place to look, independent of which version (if any) is newest.
    Overwritten on every build with the same slug, so it always reflects
    the most recent recipe used for that program.
    """

    origin_path = destination / "site" / "programs" / contract.slug / "origin.json"
    origin_path.parent.mkdir(parents=True, exist_ok=True)
    origin = {
        "slug": contract.slug,
        "title": contract.title,
        "entrypoint": contract.entrypoint,
        "source": source,
        "source_filename": Path(source_filename).name,
        "probes": contract.feeds,
        "backend_targets": (
            sorted(str(item).lower() for item in backend_targets)
            if backend_targets else None
        ),
        "include_backends": include_backends,
        "include_mathematics": include_mathematics,
        "bake_mode": contract.bake_mode,
        "schedule_preference": contract.schedule_preference,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    origin_path.write_text(json.dumps(origin, indent=2), encoding="utf-8")
    return origin_path


def _region_target_capabilities(
    region_programs: Mapping[int, Any],
) -> dict[str, list[str]]:
    """Which registered ``machine_targets`` entries can serve each region.

    Reports capability (``machine_targets.targets_for``), not a chosen
    assignment -- there is no planner here deciding which target a region
    should actually run on, only data about which ones honestly could.
    """

    from . import machine_targets

    return {
        str(region_index): sorted(machine_targets.targets_for(region_program))
        for region_index, region_program in region_programs.items()
    }


def _passthrough_candidates(
    *descriptors: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Same top-level-plus-candidates shape as ``_shader_execution_descriptor``,
    for the always-compiled passthrough shader. Never autostarted; nothing
    subscribes to it by default (see ``passthrough_script`` in
    ``wasm_html_shell.py``), so this only needs to expose every language a
    caller could reach for, webgpu first.
    """

    candidates = [item for item in descriptors if item is not None]
    if not candidates:
        return None
    result = dict(candidates[0])
    result["candidates"] = candidates
    return result


def _write_audio_asset(
    directory: Path,
    configuration: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Materialize a trusted Python audio producer into a shell asset contract."""

    if not configuration:
        return None
    generator = str(configuration.get("generator", ""))
    module_name, separator, callable_name = generator.partition(":")
    if not separator or not module_name or not callable_name:
        raise ValueError("TURING_PAGE['audio'].generator must be 'module:callable'")
    arguments = configuration.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise ValueError("TURING_PAGE['audio'].arguments must be a mapping")
    producer = getattr(importlib.import_module(module_name), callable_name)
    track = producer(**dict(arguments))
    wave = track.wave_bytes()
    features = {
        str(name): [float(value) for value in values]
        for name, values in dict(track.feature_feeds).items()
    }
    audio_relative = Path("audio") / "managed-time-source.wav"
    features_relative = Path("audio") / "spectral-features.json"
    audio_path = directory / audio_relative
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(wave)
    features_relative_path = directory / features_relative
    features_relative_path.write_text(
        json.dumps({
            "feature_fps": int(track.feature_fps),
            "duration": float(track.duration),
            "feeds": features,
        }, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        "audio_url": audio_relative.as_posix(),
        "features_url": features_relative.as_posix(),
        "sample_rate": int(track.sample_rate),
        "feature_fps": int(track.feature_fps),
        "duration": float(track.duration),
        "managed_time_output": str(
            configuration.get("managed_time_output", "next_time")
        ),
        "pan_output": str(configuration.get("pan_output", "")),
        "pan_range": list(configuration.get("pan_range", (-1.0, 1.0))),
    }


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


def _execute_module_scope(
    source: str,
    source_filename: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the program's module-level statements and return its globals.

    A compiled entrypoint routinely refers to names its own module bound
    above it -- a configured object, a lookup table, a constant built once
    at import. Those are *definitions*, established before anything runs,
    not runtime inputs. Without executing module scope they reach the
    compiler as unbound externals, and the build fails asking to be fed a
    value the program already computed for itself (``missing ProcessGraph
    input 'HEAD' ... binding_kind='external'``).

    This is the compile-time setup phase: statements run top to bottom,
    definitions are accepted, and the resulting namespace becomes the
    entrypoint's static bindings. It is what ``import`` does, on a module
    this function is compiling anyway.

    A module that fails to execute is not fatal here. Its namespace is
    simply whatever bound before the failure, and compilation proceeds to
    report what it cannot resolve -- a partial result that names the gap,
    rather than a stop that hides every other finding behind the first one.
    """

    namespace: dict[str, Any] = {
        "__name__": "__turing_program__",
        "__file__": str(source_filename),
        "__builtins__": __builtins__,
    }
    try:
        exec(compile(source, str(source_filename), "exec"), namespace)
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        if progress is not None:
            progress(
                f"module scope stopped early ({type(error).__name__}: "
                f"{error}); continuing with the names bound so far"
            )
    else:
        if progress is not None:
            progress(f"module scope bound {len(namespace)} names")
    namespace.pop("__builtins__", None)
    return namespace


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
    backend_targets: Sequence[str] | None = None,
    include_mathematics: bool = True,
    presentation_shader: str | None = None,
    presentation_document: str | None = None,
    shader_configuration: Mapping[str, Any] | None = None,
    bake_mode: str | None = None,
    schedule_preference: str | None = None,
    progress_sink: Callable[[Any], None] | None = None,
    force_new_version: bool = False,
) -> ProgramBundle:
    """Compile source and atomically publish its complete versioned bundle.

    ``force_new_version``, if True, skips the default content-addressed
    idempotency (same source + config -> same directory, see
    ``_content_version``) and instead always publishes an additional,
    previously-unused version under this program's ``versions/`` directory
    (see ``_next_sequential_version``) -- what ``publish_bundles.py --full``
    wants: process the origin again into one more version on record, not a
    lookup that might just hand back what's already there.

    ``progress_sink``, if given, is subscribed to the same
    ``shell_telemetry.TelemetryChannel`` that is baked into the published
    page (``emit_html_shell(telemetry=...)``): every record this function
    emits while compiling reaches the sink immediately, live, as well as
    ending up in the page's own console log the next time it is opened. A
    caller that wants terminal visibility during the build passes a
    printing sink; a caller that doesn't care can leave it ``None`` and the
    records still travel with the page.

    ``backend_targets``, if given, restricts the published source tabs to
    those languages (e.g. ``("glsl", "webgpu")``) rather than every backend
    ``collect_backend_sources`` can serve. Matched case-insensitively against
    each source's ``language``; the original Python source tab is always
    kept regardless. Ignored when ``include_backends`` is False (there is
    nothing to restrict). Unmatched names are silently inert rather than an
    error -- a target list that includes a backend this program can't reach
    (say, ``webgl`` for a bitwise-only kernel) is a normal, expected outcome,
    not a caller mistake.
    """

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # The channel is created before contract discovery, not after, on
    # purpose: parsing and ``ast.literal_eval``-ing a source file whose
    # ``TURING_PAGE`` bakes in large literal tensor constants (a
    # fixed-size grid's worth of floats, spelled out as source text) is
    # not instant, and it used to be the one compile stage with no
    # progress record at all -- the silent gap ran out before anyone
    # watching could tell the difference between "compiling" and "hung."
    from .shell_telemetry import TelemetryChannel

    channel = TelemetryChannel(name="program:pending")
    if progress_sink is not None:
        channel.subscribe(progress_sink)
    channel.log(f"parsing source ({len(source)} bytes)", path="contract")
    contract = discover_source_contract(
        source,
        entrypoint=entrypoint,
        title=title,
        slug=slug,
        probes=probes,
        bake_mode=bake_mode,
        schedule_preference=schedule_preference,
        progress=lambda message: channel.log(message, path="contract"),
    )
    channel.name = f"program:{contract.slug}"
    channel.log(
        f"discovered contract for {contract.entrypoint!r}",
        path="contract", slug=contract.slug,
    )
    program_bindings = _execute_module_scope(
        source,
        source_filename,
        progress=lambda message: channel.log(message, path="module-scope"),
    )
    contract_shader_configuration = {
        **dict(contract.shader_configuration),
        **dict(shader_configuration or {}),
    }
    version, source_digest, content_hash16 = _content_version(
        source,
        contract,
        presentation_shader=presentation_shader,
        presentation_document=presentation_document,
        shader_configuration=contract_shader_configuration,
        include_backends=include_backends,
        backend_targets=backend_targets,
    )
    destination = resolve_publish_root(destination)
    versions = destination / "site" / "programs" / contract.slug / "versions"
    if force_new_version:
        version = _next_sequential_version(versions, content_hash16)
        final_directory = versions / version
        # No idempotency shortcut here on purpose -- _next_sequential_version
        # never hands back a name already on disk, so there is nothing to
        # look up, only a new version to publish.
    else:
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
        from .backend_sources import BackendSourceSet, collect_backend_sources
        from .fused_program_wasm_backend import emit_wasm_module, required_steps
        from .fused_program_webgl_backend import emit_webgl_fragment_module
        from .shell_telemetry import summarize_process_graph
        from .sympy_math_renderer import render_reduced_program_mathematics
        from .wasm_class_coordinator import (
            build_browser_thread_plan,
            build_class_inventory,
            emit_wasm_class_coordinator,
            emit_wasm_control_coordinator,
        )
        from .wasm_class_modules import (
            build_embedded_class_graph,
            emit_class_modules,
            emit_control_region_modules,
            fused_program_extent_effect,
            partition_reduced_program,
            partition_threaded_wasm_program,
        )
        from .wasm_html_shell import emit_html_shell

        parameter_names = _entrypoint_parameters(source, contract.entrypoint)
        feeds = {
            name: _probe_value(contract.feeds.get(name), contract.probe_size)
            for name in parameter_names
        }
        with contextlib.redirect_stdout(compile_log), contextlib.redirect_stderr(compile_log):
            # ``channel.timed(...)`` only emits a record when the block
            # finishes (success or failure) -- it never announces that it
            # started, which made a slow block look identical to nothing
            # having happened yet. An explicit "starting" log plus a
            # heartbeat on the one genuinely long, opaque call (AOT compile
            # can run for minutes with no sub-step to report) closes that
            # gap the same way ast.parse's heartbeat did.
            channel.log("build process graph starting", path="process_graph")
            with channel.timed("build process graph", path="process_graph"):
                graph = ProcessGraph(materialize_memory=False)
                channel.log("process_graph: build_from_ast starting", path="process_graph")
                graph.build_from_ast(
                    ast.parse(source),
                    resolve_unresolved_parents=True,
                    progress=lambda message: channel.log(message, path="process_graph"),
                )
                channel.log("process_graph: build_from_ast finished", path="process_graph")
                channel.log("process_graph: reduce_abstract_tensor_topology starting", path="process_graph")
                reduce_abstract_tensor_topology(graph)
                channel.log("process_graph: reduce_abstract_tensor_topology finished", path="process_graph")
            channel.log(
                "AOT compile starting", path="aot",
                entrypoint=contract.entrypoint,
            )
            with channel.timed("AOT compile", path="aot", entrypoint=contract.entrypoint):
                channel.log("AOT compile starting", path="aot", entrypoint=contract.entrypoint)
                aot = compile_ast_aot(
                    source,
                    contract.entrypoint,
                    feeds,
                    backend=contract.backend,
                    remove_loops=contract.remove_loops,
                    unroll_limit=contract.unroll_limit,
                    precompile_only=True,
                    python_bindings={**globals(), **program_bindings},
                    bake_mode=contract.bake_mode,
                    schedule_preference=contract.schedule_preference,
                    constant_map=contract.constant_map,
                    mutable_parameters=contract.mutable_parameters,
                    progress=lambda message: channel.log(message, path="aot"),
                )
                channel.log("AOT compile finished", path="aot", entrypoint=contract.entrypoint)
            if (
                contract.bake_mode == "whole_program"
                and aot.control_shortfalls
            ):
                details = "; ".join(
                    f"{item['function']} loop {item['loop_node_id']}: "
                    + ", ".join(item["blockers"])
                    for item in aot.control_shortfalls
                )
                raise RuntimeError(
                    "WebAssembly whole-program bake refused an incomplete "
                    "control lowering; emitting the discovery-time numerical "
                    f"trace would not be the source program ({details})"
                )
            # This extraction is deliberately late in compilation.  ``aot``
            # already represents Python AST -> ProcessGraph -> planned
            # control/map/numerical compilation.  ``program`` is only the
            # internal numerical member required by this particular Wasm
            # emitter; it is not the source compiler, an application API, or
            # evidence that the Python recompiler is numerics-only.
            program = project_public_numerical_program(aot)
            channel.log("emitting wasm module", path="wasm", entrypoint=contract.entrypoint)
            module = emit_wasm_module(
                program, name=contract.entrypoint, dtype="float64"
            )
            channel.log("wasm module emitted", path="wasm", operations=len(required_steps(program)))
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

            effective_presentation_shader = presentation_shader
            # No WebGPU analog for the hand-authored ``presentation_shader``
            # passthrough parameter -- that path publishes raw, caller-supplied
            # GLSL text (demo scripts that author their own shader), not
            # something this compiler emits, so there's nothing here to lower
            # to WGSL. Only the compiler-generated paths below get one.
            effective_presentation_shader_wgsl: str | None = None
            effective_shader_configuration = dict(contract_shader_configuration)
            if contract.presentation_entrypoint is not None:
                channel.log(
                    "compiling presentation shader", path="presentation",
                    entrypoint=contract.presentation_entrypoint,
                )
                presentation_parameters = _entrypoint_parameters(
                    source, contract.presentation_entrypoint
                )
                # Mirror the main entrypoint's own feed derivation exactly
                # (``feeds = {name: _probe_value(contract.feeds.get(name), ...)}``
                # above): a presentation parameter that is also named in
                # ``TURING_PAGE["feeds"]`` gets that configured value/array,
                # not a blanket per-pixel probe. Without this, a presentation
                # entrypoint could never be handed a real fixed-size array
                # (e.g. one row per tracked element) -- every parameter would
                # be forced to the same uniform probe shape, which is what
                # made a hand-authored GLSL loop look like the only way to
                # reach per-element data from this compile path.
                presentation_feeds = {
                    parameter: _probe_value(
                        contract.feeds.get(parameter, 0.0), contract.probe_size
                    )
                    for parameter in presentation_parameters
                }
                presentation_aot = compile_ast_aot(
                    source,
                    contract.presentation_entrypoint,
                    presentation_feeds,
                    backend=contract.backend,
                    remove_loops=contract.remove_loops,
                    unroll_limit=contract.unroll_limit,
                    precompile_only=True,
                    python_bindings={**globals(), **program_bindings},
                    bake_mode=contract.bake_mode,
                    schedule_preference=contract.schedule_preference,
                    progress=lambda message: channel.log(message, path="presentation"),
                )
                presentation_program = project_public_numerical_program(
                    presentation_aot
                )
                presentation_module = emit_webgl_fragment_module(
                    presentation_program,
                    name=contract.presentation_entrypoint,
                    output_layout="rgba",
                    input_sampling="normalized",
                )
                if not presentation_module.complete:
                    raise RuntimeError(
                        "WebGL presentation emission shortfalls:\n" + "\n".join(
                            "- " + item.format()
                            for item in presentation_module.shortfalls
                        )
                    )
                effective_presentation_shader = presentation_module.source
                channel.log("presentation shader compiled", path="presentation")
                origins = dict(
                    (getattr(presentation_program, "extras", None) or {}).get(
                        "capture_feed_origins", {}
                    )
                )
                bindings = {}
                for item in presentation_module.api.metadata["feed_bindings"]:
                    value_id = item["value_id"]
                    origin = origins.get(value_id, origins.get(str(value_id), {}))
                    input_name = origin.get("binding_name")
                    if input_name is None and value_id < len(presentation_parameters):
                        input_name = presentation_parameters[value_id]
                    bindings[item["uniform"]] = str(input_name)
                effective_shader_configuration["output_feed_bindings"] = bindings
        if not module.complete:
            raise RuntimeError(module.shortfall_report())

        # A trivial identity shader (display red/green/blue as-is) compiled
        # for every page, unconditionally -- not because this page needs it,
        # but so the shell always has *some* auto-compiled whole-screen
        # passthrough surface on hand for anything that wants one later,
        # without waiting on a page author to remember to declare one. Never
        # autostarted; a page's own presentation_entrypoint (if any) is what
        # actually runs by default.
        channel.log("compiling default passthrough shader", path="passthrough")
        # float32, not _probe_value's default float64: this is a display-only
        # identity map, precision is irrelevant, and WebGPU core has no f64 at
        # all -- ssa_webgpu_backend.py reports float64 as a real shortfall
        # (correctly, for a program that actually needs the precision), which
        # would otherwise make the always-available passthrough's WebGPU
        # sibling permanently incomplete for no reason that matters here.
        passthrough_feeds = {
            name: _probe_value(0.0, contract.probe_size).astype("float32")
            for name in ("red", "green", "blue")
        }
        passthrough_aot = compile_ast_aot(
            _PASSTHROUGH_SOURCE,
            "turing_passthrough",
            passthrough_feeds,
            backend=contract.backend,
            remove_loops=True,
            unroll_limit=contract.unroll_limit,
            precompile_only=True,
            python_bindings={**globals(), **program_bindings},
        )
        passthrough_program = project_public_numerical_program(passthrough_aot)
        passthrough_module = emit_webgl_fragment_module(
            passthrough_program, name="turing_passthrough",
            output_layout="rgba", input_sampling="normalized",
        )
        # Same identity program, compiled through the WebGPU compute path too
        # -- ssa_webgpu_backend.py's compute+present JS runtime is what a
        # published page actually prefers (see _shader_execution_descriptor's
        # webgpu-first priority order), so the always-available passthrough
        # needs a real WGSL sibling, not just the WebGL one.
        passthrough_wgsl_module = None
        try:
            from .precompile_to_ssa import lower_fused_program_to_ssa as _lower_passthrough
            from .ssa_webgpu_backend import emit_module as _emit_passthrough_wgsl
            from ..transmogrifier.ssa import IRModule as _PassthroughIRModule

            passthrough_function, passthrough_lowering_shortfalls = _lower_passthrough(
                passthrough_program, function_name="turing_passthrough",
            )
            if not passthrough_lowering_shortfalls:
                passthrough_returned = next(
                    (
                        instruction.args
                        for block in passthrough_function.blocks.values()
                        for instruction in block.instrs
                        if instruction.op in {"Ret", "ret", "Return", "return"}
                    ),
                    (),
                )
                passthrough_count = max(
                    (
                        int(size)
                        for value in passthrough_function.args
                        for size in value.shape
                    ),
                    default=1,
                )
                passthrough_wgsl_module = _emit_passthrough_wgsl(
                    _PassthroughIRModule({"turing_passthrough": passthrough_function}),
                    name="turing_passthrough",
                    outputs={"turing_passthrough": passthrough_returned},
                    count=passthrough_count,
                )
        except Exception as error:
            channel.error(
                f"default WebGPU passthrough shader raised: "
                f"{type(error).__name__}: {error}",
                path="passthrough",
            )
        passthrough_descriptor: dict[str, Any] | None = None
        passthrough_wgsl_descriptor: dict[str, Any] | None = None
        if passthrough_module.complete:
            passthrough_relative = Path("source") / "webgl" / "passthrough.frag.glsl"
            passthrough_path = temporary / passthrough_relative
            passthrough_path.parent.mkdir(parents=True, exist_ok=True)
            passthrough_path.write_text(passthrough_module.source, encoding="utf-8")
            passthrough_descriptor = {
                "url": passthrough_relative.as_posix(),
                "language": "webgl2-glsl-es",
                "role": "passthrough-surface",
                "inputs": ["red", "green", "blue"],
                "autostart": False,
                "note": (
                    "Always-available whole-screen identity shader. Not used "
                    "by this page's default presentation; present for "
                    "anything that wants a plain texture passthrough."
                ),
            }
            channel.log("default passthrough shader compiled", path="passthrough")
        else:
            channel.error(
                "default passthrough shader had shortfalls", path="passthrough",
                shortfalls=[item.format() for item in passthrough_module.shortfalls],
            )
        if passthrough_wgsl_module is not None and passthrough_wgsl_module.complete:
            passthrough_wgsl_relative = Path("source") / "webgpu" / "passthrough.compute.wgsl"
            passthrough_wgsl_path = temporary / passthrough_wgsl_relative
            passthrough_wgsl_path.parent.mkdir(parents=True, exist_ok=True)
            passthrough_wgsl_path.write_text(passthrough_wgsl_module.source, encoding="utf-8")
            passthrough_wgsl_descriptor = {
                "url": passthrough_wgsl_relative.as_posix(),
                "language": "wgsl",
                "role": "passthrough-surface",
                "inputs": ["red", "green", "blue"],
                "autostart": False,
                "note": (
                    "Always-available whole-screen identity shader, WebGPU "
                    "compute variant. Not used by this page's default "
                    "presentation; present for anything that wants a plain "
                    "texture passthrough."
                ),
            }
            channel.log("default WebGPU passthrough shader compiled", path="passthrough")
        elif passthrough_wgsl_module is not None:
            channel.error(
                "default WebGPU passthrough shader had shortfalls", path="passthrough",
                shortfalls=[item.format() for item in passthrough_wgsl_module.shortfalls],
            )

        # "Shader support" is mandatory, not an opt-in a page author has to
        # remember: if no presentation_entrypoint was authored, and the main
        # entrypoint itself already names outputs red/green/blue, the always
        # -compiled passthrough becomes this page's active presentation --
        # tensor math stays entirely in the (unrestricted) WASM kernel, and
        # nothing but a trivial per-pixel display ever has to be expressed
        # as WebGL-safe scalar code. A page author never authors a shader
        # function unless they want something other than plain display.
        if (
            contract.presentation_entrypoint is None
            and passthrough_module.complete
            and effective_presentation_shader is None
        ):
            output_names = {
                item.name
                for item in module.api.entry_points[0].parameters
                if item.role == "output"
            }
            if {"red", "green", "blue"} <= output_names:
                channel.log(
                    "no presentation_entrypoint authored; using the default "
                    "passthrough as this page's shader support",
                    path="presentation",
                )
                effective_presentation_shader = passthrough_module.source
                if passthrough_wgsl_module is not None and passthrough_wgsl_module.complete:
                    effective_presentation_shader_wgsl = passthrough_wgsl_module.source
                effective_shader_configuration["output_feed_bindings"] = {
                    item["uniform"]: name
                    for item, name in zip(
                        passthrough_module.api.metadata["feed_bindings"],
                        ("red", "green", "blue"),
                    )
                }

        card_directory = Path("wasm") / f"size-{DEFAULT_WASM_CARD_OPERATIONS}"
        channel.log("partitioning program into wasm regions", path="regions")
        real_control = (
            aot.shell_control_program
            if contract.bake_mode == "whole_program"
            and aot.shell_control_program is not None
            and aot.shell_control_program.region_indices
            and aot.region_programs
            else None
        )
        effective_region_programs = dict(aot.region_programs)
        thread_topology = None
        if real_control is not None and len(real_control.region_indices) == 1:
            source_region = int(real_control.region_indices[0])
            source_program = effective_region_programs.get(source_region)
            if source_program is not None:
                channel.log("attempting threaded wasm partition", path="regions")
                threaded = partition_threaded_wasm_program(
                    # The public projection above has already collapsed
                    # byte-for-byte duplicate observational definitions and
                    # restored the exact source output contract. The raw
                    # capture intentionally retains those observations and is
                    # therefore not a valid partition input.
                    program,
                    max_nodes_per_region=64,
                    schedule_preference=contract.schedule_preference,
                )
                if threaded is not None:
                    real_control, effective_region_programs, thread_topology = threaded
                    channel.log(
                        "threaded partition found", path="regions",
                        waves=len(thread_topology.get("rewrite_history", ())) if thread_topology else 0,
                    )
                else:
                    channel.log("threaded partition unavailable, staying single-region", path="regions")
        if real_control is not None:
            channel.log(
                "emitting control region modules", path="regions",
                regions=len(real_control.region_indices),
            )
            card_modules, card_manifest = emit_control_region_modules(
                real_control,
                effective_region_programs,
                owner_name=contract.entrypoint,
                module_dir=card_directory.as_posix(),
                dtype="float64",
            )
            channel.log("control region modules emitted", path="regions", modules=len(card_modules))
            producer = {
                int(value_id): (
                    entry["name"], str(output_name)
                )
                for region, region_program in effective_region_programs.items()
                if int(region) in card_modules
                for entry in card_manifest["modules"]
                if int(entry["region_index"]) == int(region)
                for output_name, value_id in getattr(
                    region_program, "program", region_program
                ).outputs.items()
            }
            logical_outputs = {}
            for output_name in aot.function_outputs:
                identities = tuple(aot.identity_table.get(output_name, ()))
                binding = next(
                    (
                        producer[int(value_id)]
                        for value_id in reversed(identities)
                        if int(value_id) in producer
                    ),
                    None,
                )
                if binding is not None:
                    logical_outputs[str(output_name)] = list(binding)
            if not logical_outputs:
                for entry in reversed(card_manifest["modules"]):
                    for output_name in entry["outputs"]:
                        logical_outputs.setdefault(
                            str(output_name), [entry["name"], output_name]
                        )
            card_manifest["logical_outputs"] = logical_outputs
            channel.log("building class inventory", path="regions")
            inventory = build_class_inventory(card_manifest)
            channel.log("class inventory built", path="regions", methods=len(inventory.methods))
            field_slots_by_key = {
                field.key: int(field.index) for field in inventory.fields
            }
            control_value_slots = {
                int(value_id): field_slots_by_key[key]
                for value_id, key in card_manifest.get(
                    "value_bindings", {}
                ).items()
                if key in field_slots_by_key
            }
            region_methods = {
                int(entry["region_index"]): int(method.index)
                for entry, method in zip(
                    card_manifest["modules"], inventory.methods
                )
            }
            channel.log("emitting wasm control coordinator", path="regions")
            coordinator = emit_wasm_control_coordinator(
                inventory,
                real_control,
                region_methods=region_methods,
                value_slots=control_value_slots,
                region_signatures={
                    int(region): (
                        tuple(sorted(map(int, region_program.feeds))),
                        tuple(map(int, region_program.outputs.values())),
                    )
                    for region, region_program in effective_region_programs.items()
                },
                name=(
                    f"{contract.entrypoint}_control_"
                    f"{DEFAULT_WASM_CARD_OPERATIONS}"
                ),
            )
            channel.log("wasm control coordinator emitted", path="regions")
            specs = ()
            contiguous = None
            channel.log("building browser thread plan", path="regions")
            thread_plan = build_browser_thread_plan(
                real_control,
                region_methods,
                region_extent_effects={
                    int(region): fused_program_extent_effect(region_program)
                    for region, region_program in effective_region_programs.items()
                },
            )
            channel.log("browser thread plan built", path="regions")
        else:
            channel.log("partitioning reduced program (no parallel regions)", path="regions")
            specs = partition_reduced_program(
                program,
                chunk_size=DEFAULT_WASM_CARD_OPERATIONS,
                owner_name=contract.entrypoint,
            )
            channel.log("emitting class modules", path="regions", chunks=len(specs))
            card_modules = emit_class_modules(
                specs, dtype="float64", link_calls=False, shared_memory=True
            )
            channel.log("class modules emitted", path="regions")
            incomplete_cards = [
                card_modules[spec.index]
                for spec in specs
                if not card_modules[spec.index].complete
            ]
            if incomplete_cards:
                raise RuntimeError("\n".join(
                    card.shortfall_report() for card in incomplete_cards
                ))
            channel.log("building embedded class graph", path="regions")
            card_manifest = build_embedded_class_graph(
                specs,
                card_modules,
                program,
                entrypoint=contract.entrypoint,
                embed_binaries=False,
                module_dir=card_directory.as_posix(),
            )
            channel.log("building class inventory", path="regions")
            inventory = build_class_inventory(card_manifest)
            channel.log("emitting wasm class coordinator", path="regions")
            coordinator = emit_wasm_class_coordinator(
                inventory,
                name=(
                    f"{contract.entrypoint}_coordinator_"
                    f"{DEFAULT_WASM_CARD_OPERATIONS}"
                ),
            )
            channel.log("wasm class coordinator emitted", path="regions")
            thread_plan = None
            wasm_relative = Path("wasm") / f"{module.name}.wasm"
            wasm_path = temporary / wasm_relative
            wasm_path.parent.mkdir(parents=True, exist_ok=True)
            wasm_path.write_bytes(module.binary)
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
        card_manifest.update({
            "region_steps": DEFAULT_WASM_CARD_OPERATIONS,
            "class_inventory": inventory.to_mapping(),
            "coordinator": {
                "name": coordinator.name,
                "url": (card_directory / f"{coordinator.name}.wasm").as_posix(),
                "entry": "run_range",
                "memory_import": {"module": "env", "field": "memory"},
                "method_count": len(inventory.methods),
                "supports_ranges": real_control is None,
            },
            "thread_deployment": thread_plan,
            "thread_topology": thread_topology,
        })
        card_output_directory = temporary / card_directory
        card_output_directory.mkdir(parents=True, exist_ok=True)
        if real_control is not None:
            for region, region_module in card_modules.items():
                entry = next(
                    item for item in card_manifest["modules"]
                    if int(item["region_index"]) == int(region)
                )
                (card_output_directory / f"{entry['name']}.wasm").write_bytes(
                    region_module.binary
                )
        else:
            for spec in specs:
                (card_output_directory / f"{spec.module_name}.wasm").write_bytes(
                    card_modules[spec.index].binary
                )
        (card_output_directory / f"{coordinator.name}.wasm").write_bytes(
            coordinator.binary
        )
        if real_control is not None:
            # Publish the whole-program control kernel at the conventional
            # Wasm root as well. It imports the region kernels listed by the
            # class manifest; it is not the old flattened numerical module.
            wasm_root = temporary / "wasm"
            wasm_root.mkdir(parents=True, exist_ok=True)
            (wasm_root / f"{coordinator.name}.wasm").write_bytes(
                coordinator.binary
            )
        (card_output_directory / "class-inventory.json").write_text(
            json.dumps(inventory.to_mapping(), indent=2), encoding="utf-8"
        )
        channel.log(
            "wasm regions emitted", path="regions",
            regions=len(inventory.methods), parallel=thread_topology is not None,
        )

        sources = None
        if include_backends:
            channel.log("collecting backend sources", path="sources")
            with contextlib.redirect_stdout(compile_log), contextlib.redirect_stderr(compile_log):
                sources = collect_backend_sources(
                    aot,
                    numerical_name=contract.entrypoint,
                    control_name=f"{contract.entrypoint}_control",
                    channel=channel,
                    wasm_source=(
                        coordinator.wat
                        if real_control is not None
                        else module.source
                    ),
                    program=program,
                )
            if backend_targets:
                wanted = {str(item).lower() for item in backend_targets}
                sources = BackendSourceSet(tuple(
                    item for item in sources.sources
                    if item.language.lower() in wanted
                ))
                channel.log(
                    "restricted to requested backend targets", path="sources",
                    targets=sorted(wanted), kept=len(sources.sources),
                )
            channel.log(
                "backend sources collected", path="sources",
                languages=len(sources.sources) if sources is not None else 0,
            )
        channel.log("writing sources to bundle directory", path="sources")
        published_sources = _write_sources(
            temporary,
            source,
            Path(source_filename).name,
            sources,
            presentation_shader=effective_presentation_shader,
            presentation_shader_wgsl=effective_presentation_shader_wgsl,
        )
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
                channel.log("checking for a fortran compiler", path="fortran")
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
                    channel.log("compiling and verifying native fortran module", path="fortran")
                    proof = verify_fortran_module(
                        fortran_artifact,
                        program,
                        feed_values,
                        temporary / "native" / "fortran",
                        entrypoint=contract.entrypoint,
                    )
                    channel.log(
                        "fortran verification finished", path="fortran",
                        case_count=proof["case_count"],
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

        # Run the exact WebAssembly the bundle ships against the same NumPy
        # rendering the Fortran path uses -- the numpy source published under
        # this bundle's own backend tab.  Exported-memory numeric modules only;
        # imported-memory region modules are driven by the control path.
        wasm_verification = None
        if module.binary is not None and not (
            module.api.metadata or {}
        ).get("shared_memory_import"):
            from ..common.tensors.fused_ir import ordered_feed_ids
            from .wasm_fidelity import node_runtime, verify_wasm_module

            channel.log("checking for a node runtime", path="wasm")
            if node_runtime() is not None:
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
                channel.log(
                    "running emitted wasm against the numpy reference", path="wasm"
                )
                proof = verify_wasm_module(
                    module,
                    program,
                    feed_values,
                    temporary / "native" / "wasm",
                    entrypoint=contract.entrypoint,
                )
                channel.log(
                    "wasm verification finished", path="wasm",
                    case_count=proof["case_count"],
                )
                proof_relative = Path("verification") / "wasm-fidelity.json"
                proof_path = temporary / proof_relative
                proof_path.parent.mkdir(parents=True, exist_ok=True)
                proof_path.write_text(
                    json.dumps(proof, indent=2, allow_nan=True),
                    encoding="utf-8",
                )
                wasm_verification = {
                    "passed": True,
                    "case_count": proof["case_count"],
                    "binary_sha256": proof["binary_sha256"],
                    "url": proof_relative.as_posix(),
                }
            else:
                wasm_verification = {
                    "passed": None,
                    "reason": "no node runtime was available at build time",
                }

        mathematics = None
        math_error = ""
        if include_mathematics:
            channel.log("rendering sympy mathematics model", path="mathematics")
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
                channel.log("sympy mathematics model rendered", path="mathematics")
            except Exception as error:  # page generation survives optional projection refusal
                math_error = f"{type(error).__name__}: {error}"
                channel.error(f"sympy mathematics model failed: {math_error}", path="mathematics")

        route = f"/site/programs/{contract.slug}/versions/{version}/"
        channel.log("building shader execution descriptor", path="shader")
        shader_execution = _shader_execution_descriptor(
            published_sources,
            (module.api.metadata or {}).get("shell_io"),
            effective_shader_configuration,
        )
        channel.log("writing audio asset", path="audio")
        audio_runtime = _write_audio_asset(
            temporary, contract.audio_configuration
        )
        channel.log("rendering html shell", path="shell")
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
            audio_runtime=audio_runtime,
            passthrough_shader=_passthrough_candidates(
                passthrough_wgsl_descriptor, passthrough_descriptor,
            ),
        )
        channel.log("html shell rendered, writing page to disk", path="shell")
        page_path = shell.write(temporary)
        channel.log("page written", path="shell", page=str(page_path))

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
                # "shader support" is the mandatory pairing this page always
                # has: a compiled WASM numeric program (tensor math,
                # unrestricted) plus a thin GLSL presentation of its named
                # outputs -- the passthrough by default, an authored
                # presentation_entrypoint only when a page wants something
                # other than plain display. "shader" is kept for existing
                # readers; "shader_support" is the concept to check going
                # forward, since it is never null once the numeric program
                # names red/green/blue outputs.
                "shader_support": {
                    "numeric": {"entrypoint": contract.entrypoint},
                    "presentation": shader_execution,
                },
                "audio": audio_runtime,
            },
            "source": {
                "path": f"source/python_source/{Path(source_filename).name}",
                "filename": Path(source_filename).name,
            },
            "compiler": {
                "backend": contract.backend,
                "bake_mode": contract.bake_mode,
                "schedule_preference": contract.schedule_preference,
                "program_record_mode": aot.program_record_mode,
                "constant_map": dict(aot.constant_map),
                "mutable_parameters": list(aot.mutable_parameters),
                "thread_topology": thread_topology,
                "remove_loops": contract.remove_loops,
                "unroll_limit": contract.unroll_limit,
                "implementation_sha256": _bundle_compiler_digest(),
                "mathematics_error": math_error,
                "fortran_verification": fortran_verification,
                "wasm_verification": wasm_verification,
                # Per-subgraph *capability*, not an assignment: which
                # registered machine_targets entries could serve each region
                # as-is, keyed by region index. This is the raw material a
                # distributed-dispatch planner needs (which client capability
                # profiles could even accept a given region's binary) -- it
                # is not that planner. Nothing here picks one target per
                # region; that decision doesn't exist yet.
                "region_target_capabilities": _region_target_capabilities(
                    effective_region_programs
                ),
            },
            "artifacts": _artifact_inventory(temporary),
        }
        manifest_path = temporary / "bundle.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _write_program_origin(
            destination, contract, source, source_filename,
            backend_targets=backend_targets, include_backends=include_backends,
            include_mathematics=include_mathematics,
        )
        if final_directory.exists():
            channel.log("identical version already published", path="bundle", version=version)
            existing = load_program_bundle(final_directory)
            shutil.rmtree(temporary)
            return existing
        temporary.replace(final_directory)
        channel.log("bundle published", path="bundle", version=version, route=route)
        return load_program_bundle(final_directory)
    except Exception as error:
        channel.exception(error, path="bundle", phase="build_program_bundle")
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
