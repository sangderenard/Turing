"""Sentinel-structured polyglot programs loaded through the card graph.

A dream document is scanned as bytes before any language frontend runs. Each
payload is then decoded and handed to its declared language handler. Shader
blocks are intrinsically device deployments, while parallel directives group
ordinary or shader blocks without embedding synchronization in their source.
"""

from __future__ import annotations

import ast
import base64
import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
import io
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .card_graph import build_card_graph


SCHEMA = "turing.dream-document.v1"
_OPEN = re.compile(
    rb"/\*@turing\.(segment|shader|parallel)\.v1[ \t]*\r?\n(.*?)@end\*/",
    re.DOTALL,
)
_CLOSE = b"/*@turing.end*/"


class DreamDocumentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DreamLanguageTranslation:
    route: str
    translation_table: str
    execution_target: str
    executable: bool


DREAM_LANGUAGE_TRANSLATIONS: Mapping[str, DreamLanguageTranslation] = (
    MappingProxyType({
        "python": DreamLanguageTranslation(
            "ast", "ProcessGraph.build_from_ast", "process-graph-aot", False,
        ),
        "sympy": DreamLanguageTranslation(
            "sympy", "SYMPY_PROCESS_GRAPH_TRANSLATIONS", "symbolic-process-graph", False,
        ),
        "javascript": DreamLanguageTranslation(
            "javascript-ast", "acorn.parse", "browser-javascript", True,
        ),
        "glsl": DreamLanguageTranslation(
            "glsl-ssa", "GLSL_*_TO_SSA", "shader-device", True,
        ),
        "wgsl": DreamLanguageTranslation(
            "source", "DREAM_SOURCE_LANGUAGE_TRANSLATIONS", "shader-device", True,
        ),
        "cpp": DreamLanguageTranslation(
            "cpp-shell", "cpp_shell_desugar.desugar_cpp_shell",
            "process-graph-aot", False,
        ),
    })
)


@dataclass(frozen=True, slots=True)
class DreamSectionCompilation:
    block: str
    language: str
    route: str
    translation_table: str
    execution_target: str
    executable: bool
    graph: Mapping[str, Any]
    shortfalls: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "language": self.language,
            "route": self.route,
            "translation_table": self.translation_table,
            "execution_target": self.execution_target,
            "executable": self.executable,
            "graph": dict(self.graph),
            "shortfalls": list(self.shortfalls),
        }


def _json_graph_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {
            str(key): _json_graph_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_graph_value(item) for item in value]
    return repr(value)


def _process_graph_mapping(graph: Any, *, route: str) -> dict[str, Any]:
    def graph_body(subject: Any) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": str(node_id),
                    **{
                        key: _json_graph_value(value)
                        for key, value in data.items()
                        if key not in {"expr_obj", "tensor"}
                    },
                }
                for node_id, data in subject.G.nodes(data=True)
            ],
            "edges": [
                {"from": str(source), "to": str(target)}
                for source, target in subject.G.edges()
            ],
            "roots": [str(root) for root in subject.roots],
        }

    body = graph_body(graph)
    return {
        "schema": "turing.dream-section-graph.v1",
        "route": route,
        **body,
        "functions": [
            {
                "reference": entry.reference.address,
                "name": entry.name,
                "qualified_name": entry.qualified_name,
                "state": entry.state.value,
                "recursive": entry.recursive,
                "graph": (
                    None if entry.graph is None else graph_body(entry.graph)
                ),
            }
            for entry in graph.function_table
        ],
    }


def _ssa_module_graph_mapping(module: Any, *, route: str) -> dict[str, Any]:
    nodes = []
    edges = []
    roots = []
    for function_name, function in module.functions.items():
        function_id = f"function:{function_name}"
        roots.append(function_id)
        nodes.append({
            "id": function_id,
            "type": "Function",
            "name": function_name,
        })
        value_nodes: set[str] = set()

        def add_value(value: Any) -> str:
            value_id = f"{function_id}:value:{value.id}"
            if value_id not in value_nodes:
                value_nodes.add(value_id)
                nodes.append({
                    "id": value_id,
                    "type": "SSAValue",
                    "dtype": value.dtype,
                    "shape": list(value.shape),
                    "device": value.device,
                })
            return value_id

        for argument in function.args:
            value_id = add_value(argument)
            edges.append({"from": function_id, "to": value_id, "role": "argument"})
        for block_name, basic_block in function.blocks.items():
            for index, instruction in enumerate(basic_block.instrs):
                instruction_id = f"{function_id}:{block_name}:instruction:{index}"
                nodes.append({
                    "id": instruction_id,
                    "type": "SSAInstruction",
                    "op": instruction.op,
                    "attributes": _json_graph_value(instruction.attributes),
                    "source_span": _json_graph_value(instruction.source_span),
                })
                edges.append({
                    "from": function_id,
                    "to": instruction_id,
                    "role": "contains",
                })
                for argument_index, argument in enumerate(instruction.args):
                    edges.append({
                        "from": add_value(argument),
                        "to": instruction_id,
                        "role": f"argument:{argument_index}",
                    })
                if instruction.res is not None:
                    edges.append({
                        "from": instruction_id,
                        "to": add_value(instruction.res),
                        "role": "result",
                    })
    return {
        "schema": "turing.dream-section-graph.v1",
        "route": route,
        "nodes": nodes,
        "edges": edges,
        "roots": roots,
    }


def _source_section_graph(block: "DreamBlock") -> dict[str, Any]:
    lines = block.payload.splitlines()
    nodes = [{
        "id": "section",
        "type": "SourceSection",
        "language": block.language,
    }]
    edges = []
    previous: str | None = None
    for number, text in enumerate(lines, 1):
        identity = f"line:{number}"
        nodes.append({
            "id": identity,
            "type": "SourceLine",
            "line": number,
            "text": text,
        })
        edges.append({"from": "section", "to": identity, "role": "contains"})
        if previous is not None:
            edges.append({"from": previous, "to": identity, "role": "next"})
        previous = identity
    return {
        "schema": "turing.dream-section-graph.v1",
        "route": "source",
        "nodes": nodes,
        "edges": edges,
        "roots": ["section"],
    }


_JS_AST_PARSER = Path(__file__).resolve().parent / "vendor" / "js_ast_parse.js"

_JS_TOP_LEVEL_DECLARATION_ID_KEY = {
    "FunctionDeclaration": "id",
    "ClassDeclaration": "id",
}


def _run_javascript_ast_parser(source: str) -> dict[str, Any]:
    import json
    import subprocess

    try:
        result = subprocess.run(
            ["node", str(_JS_AST_PARSER)],
            input=source.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise DreamDocumentError(
            "javascript Dream sections require a `node` executable on PATH "
            "to build their AST dependency graph"
        ) from error
    if result.returncode != 0:
        raise DreamDocumentError(
            "javascript source failed to parse: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return json.loads(result.stdout.decode("utf-8"))


def _js_top_level_declarations(
    program_body: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    declarations: dict[str, dict[str, Any]] = {}
    for statement in program_body:
        node_type = statement.get("type")
        id_key = _JS_TOP_LEVEL_DECLARATION_ID_KEY.get(node_type)
        if id_key is not None:
            identifier = statement.get(id_key) or {}
            name = identifier.get("name")
            if name:
                declarations[name] = statement
        elif node_type == "VariableDeclaration":
            for declarator in statement.get("declarations", ()):
                target = declarator.get("id") or {}
                if target.get("type") == "Identifier" and target.get("name"):
                    declarations[target["name"]] = statement
    return declarations


def _js_collect_identifier_references(
    node: Any, names: set[str], found: set[str],
) -> None:
    if isinstance(node, dict):
        if node.get("type") == "Identifier" and node.get("name") in names:
            found.add(node["name"])
        for key, value in node.items():
            if key in {"start", "end", "type", "name"}:
                continue
            _js_collect_identifier_references(value, names, found)
    elif isinstance(node, list):
        for item in node:
            _js_collect_identifier_references(item, names, found)


def _javascript_dependency_graph(source: str, *, identity: str) -> Any:
    """Build a top-level declaration dependency graph from a JS AST.

    Each module-level function/class/variable declaration becomes a node;
    an edge records that one declaration's body references another
    declaration's name. This mirrors the Python `ast` route's use of
    `ProcessGraph` as the shared graph representation the rest of the
    Dream compiler pipeline already knows how to lower and schedule.
    """

    from ..transmogrifier.graph.graph_express2 import ProcessGraph

    program = _run_javascript_ast_parser(source)
    declarations = _js_top_level_declarations(program.get("body", []))

    graph = ProcessGraph(materialize_memory=False)
    for name, statement in declarations.items():
        graph.G.add_node(
            name,
            type=statement.get("type"),
            language="javascript",
            block=identity,
        )
    other_names = set(declarations)
    for name, statement in declarations.items():
        referenced: set[str] = set()
        _js_collect_identifier_references(
            statement, other_names - {name}, referenced,
        )
        for target in referenced:
            graph.G.add_edge(name, target, role="depends-on")
    graph.roots = list(declarations)
    return graph


def _header(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise DreamDocumentError("sentinel headers must be restricted ASCII") from error
    result: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise DreamDocumentError(f"sentinel header line {number} has no '='")
        key, value = (part.strip() for part in stripped.split("=", 1))
        if not key or key in result:
            raise DreamDocumentError(f"invalid or duplicate sentinel key {key!r}")
        result[key] = value
    return result


def _names(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class DreamBlock:
    identity: str
    kind: str
    language: str
    payload: str
    encoding: str
    stage: str | None = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    decorations: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    content_hash: str = ""

    @property
    def gpu_deployment(self) -> bool:
        return self.kind == "shader"


@dataclass(frozen=True, slots=True)
class DreamParallelDeployment:
    identity: str
    members: tuple[str, ...]
    join: str


@dataclass(frozen=True, slots=True)
class DreamDocument:
    blocks: tuple[DreamBlock, ...]
    parallel: tuple[DreamParallelDeployment, ...] = ()
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        identities = [block.identity for block in self.blocks]
        if len(set(identities)) != len(identities):
            raise DreamDocumentError("dream block identities must be unique")
        known = set(identities)
        deployed: set[str] = set()
        for group in self.parallel:
            if not group.members or len(set(group.members)) != len(group.members):
                raise DreamDocumentError(f"parallel group {group.identity!r} has invalid members")
            missing = set(group.members) - known
            if missing:
                raise DreamDocumentError(
                    f"parallel group {group.identity!r} references unknown blocks {sorted(missing)!r}"
                )
            overlap = deployed & set(group.members)
            if overlap:
                raise DreamDocumentError(f"blocks belong to multiple parallel groups: {sorted(overlap)!r}")
            deployed.update(group.members)

    def block(self, identity: str) -> DreamBlock:
        for block in self.blocks:
            if block.identity == identity:
                return block
        raise KeyError(identity)

    def card_graph(self) -> dict[str, Any]:
        """Project blocks into the existing lazy card/read-head graph ABI."""

        modules = [
            {
                "name": block.identity,
                "entry": block.decorations.get("entry", "main"),
                "inputs": list(block.inputs),
                "outputs": list(block.outputs),
                "value_type": block.decorations.get("dtype", "f32"),
                "url": f"dream://{block.identity}",
                "cache_key": block.content_hash,
                "language": block.language,
                "stage": block.stage,
            }
            for block in self.blocks
        ]
        edges = [
            {
                "from": {"module": left.identity, "output": left.outputs[0] if left.outputs else "completion"},
                "to": {"module": right.identity, "input": right.inputs[0] if right.inputs else "trigger"},
            }
            for left, right in zip(self.blocks, self.blocks[1:])
        ]
        graph = build_card_graph({}, {
            "name": "dream-document",
            "modules": modules,
            "edges": edges,
            "class_inventory": {"methods": [{"module": block.identity} for block in self.blocks]},
            "external_link_policy": {
                "execution": "dream-document-read-head",
                "rebind": "every-block",
            },
        })
        graph["document_schema"] = self.schema
        graph["parallel_deployments"] = [
            {"id": group.identity, "members": list(group.members), "join": group.join}
            for group in self.parallel
        ]
        graph["block_metadata"] = {
            block.identity: {
                "kind": block.kind,
                "language": block.language,
                "stage": block.stage,
                "encoding": block.encoding,
                "content_hash": block.content_hash,
                "gpu_deployment": block.gpu_deployment,
                "decorations": dict(block.decorations),
            }
            for block in self.blocks
        }
        return graph

    def compile_sections(self) -> tuple[DreamSectionCompilation, ...]:
        """Graph every block and select its declared language route."""

        from ..transmogrifier.graph.graph_express2 import ProcessGraph

        compiled = []
        for block in self.blocks:
            translation = DREAM_LANGUAGE_TRANSLATIONS.get(block.language)
            if translation is None:
                translation = DreamLanguageTranslation(
                    "source", "DREAM_SOURCE_LANGUAGE_TRANSLATIONS",
                    "source-string-interpreter", False,
                )
            shortfalls = []
            if translation.route == "ast":
                from ..common.tensors.topological_reducer import (
                    reduce_abstract_tensor_topology,
                )

                graph = ProcessGraph(materialize_memory=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    graph.build_from_ast(ast.parse(
                        block.payload, filename=f"<dream:{block.identity}>",
                    ))
                    reduce_abstract_tensor_topology(graph)
                mapping = _process_graph_mapping(graph, route="ast")
            elif translation.route == "sympy":
                import sympy
                from .symbolic_process_graph import ingest_sympy_expression

                graph = ProcessGraph(materialize_memory=False)
                ingest_sympy_expression(
                    graph, sympy.sympify(block.payload, evaluate=False), strict=True,
                )
                mapping = _process_graph_mapping(graph, route="sympy")
            elif translation.route == "glsl-ssa":
                from .glsl_source_ingestion import lower_glsl_source_to_ssa

                lowering = lower_glsl_source_to_ssa(
                    block.payload,
                    function_name=f"dream_block_{block.identity.replace('-', '_')}",
                )
                mapping = _ssa_module_graph_mapping(
                    lowering.module, route="glsl-ssa",
                )
                shortfalls.extend(item.format() for item in lowering.shortfalls)
            elif translation.route == "javascript-ast":
                graph = _javascript_dependency_graph(
                    block.payload, identity=block.identity,
                )
                mapping = _process_graph_mapping(graph, route="javascript-ast")
            elif translation.route == "cpp-shell":
                from pycparser import c_parser

                from .cpp_shell_desugar import desugar_cpp_shell
                from ..transmogrifier.graph.oop_language_translations import (
                    install_c_role_schemas,
                )

                desugared = desugar_cpp_shell(block.payload)
                tree = c_parser.CParser().parse(desugared)
                graph = ProcessGraph(materialize_memory=False)
                install_c_role_schemas(graph)
                graph.build_graph(tree)
                mapping = _process_graph_mapping(graph, route="cpp-shell")
            else:
                mapping = _source_section_graph(block)
            if not translation.executable:
                shortfalls.append(
                    f"{translation.execution_target} has no executable Dream shell artifact"
                )
            compiled.append(DreamSectionCompilation(
                block=block.identity,
                language=block.language,
                route=translation.route,
                translation_table=translation.translation_table,
                execution_target=translation.execution_target,
                executable=translation.executable,
                graph=MappingProxyType(mapping),
                shortfalls=tuple(shortfalls),
            ))
        return tuple(compiled)

    def lower_to_ssa(self) -> "DreamSSALowering":
        """Lower block calls and every dispatch frame into repository SSA."""

        from ..transmogrifier.ssa import (
            BasicBlock, Function, IRModule, Instr, SSADeploymentLane,
            SSADeploymentRegion, SSAValue,
        )
        from .deployment_frame import DeploymentJoin, DeploymentJoinMode

        group_memberships = {
            member: (group_index, lane_index)
            for group_index, group in enumerate(self.parallel)
            for lane_index, member in enumerate(group.members)
        }
        shader_region_ids = {
            block.identity: len(self.parallel) + index
            for index, block in enumerate(
                item for item in self.blocks if item.gpu_deployment
            )
        }
        functions: dict[str, Any] = {}
        root_instructions = []
        root_results = []
        next_value_id = 0
        call_index_by_block: dict[str, int] = {}
        for block in self.blocks:
            arguments = [
                SSAValue(next_value_id + index, block.decorations.get("dtype", "f32"))
                for index, _name in enumerate(block.inputs)
            ]
            next_value_id += len(arguments)
            outputs = [
                SSAValue(next_value_id + index, block.decorations.get("dtype", "f32"))
                for index, _name in enumerate(block.outputs)
            ]
            next_value_id += len(outputs)
            block_instructions = []
            for output_index, output in enumerate(outputs or (None,)):
                block_instructions.append(Instr(
                    "Call", arguments, output,
                    attributes={
                        "callee": f"__dream_{block.language}_{block.stage or 'host'}__",
                        "dream_block": block.identity,
                        "language": block.language,
                        "stage": block.stage,
                        "device_deployment": block.gpu_deployment,
                        "output_port": output_index if outputs else None,
                        "component_abi": block.decorations.get("abi"),
                    },
                ))
            block_instructions.append(Instr("Ret", outputs, None))
            functions[f"dream_block_{block.identity.replace('-', '_')}"] = Function(
                f"dream_block_{block.identity.replace('-', '_')}",
                arguments,
                {"entry": BasicBlock("entry", block_instructions)},
                metadata={
                    "kind": "dream-document-block",
                    "identity": block.identity,
                    "language": block.language,
                    "stage": block.stage,
                    "content_hash": block.content_hash,
                },
            )
            memberships = []
            if block.identity in group_memberships:
                memberships.append(group_memberships[block.identity])
            if block.identity in shader_region_ids:
                memberships.append((shader_region_ids[block.identity], 0))
            root_result = outputs[0] if outputs else None
            call_index_by_block[block.identity] = len(root_instructions)
            root_instructions.append(Instr(
                "Call", [], root_result,
                attributes={
                    "callee": f"dream_block_{block.identity.replace('-', '_')}",
                    "dream_block": block.identity,
                    "deployment_memberships": tuple(memberships),
                },
            ))
            if root_result is not None:
                root_results.append(root_result)
        root_instructions.append(Instr("Ret", root_results[-1:], None))

        regions = []
        for region_id, group in enumerate(self.parallel):
            regions.append(SSADeploymentRegion(
                region_id=region_id,
                function="dream_main",
                kind="parallel_deployment",
                schedule="independent_lanes",
                lanes=tuple(
                    SSADeploymentLane(
                        index=lane_index,
                        instruction_sites=(("entry", call_index_by_block[member]),),
                        callees=(f"dream_block_{member.replace('-', '_')}",),
                    )
                    for lane_index, member in enumerate(group.members)
                ),
                origin="dream-document",
                join=DeploymentJoin(DeploymentJoinMode.BARRIER),
            ))
        for block in self.blocks:
            if not block.gpu_deployment:
                continue
            region_id = shader_region_ids[block.identity]
            regions.append(SSADeploymentRegion(
                region_id=region_id,
                function="dream_main",
                kind="device_dispatch",
                schedule=f"{block.language}:{block.stage}",
                lanes=(SSADeploymentLane(
                    index=0,
                    instruction_sites=(("entry", call_index_by_block[block.identity]),),
                    callees=(f"dream_block_{block.identity.replace('-', '_')}",),
                ),),
                origin="dream-document",
                join=DeploymentJoin(DeploymentJoinMode.BARRIER),
            ))
        root = Function(
            "dream_main", [], {"entry": BasicBlock("entry", root_instructions)},
            metadata={
                "kind": "dream-document-root",
                "deployment_regions": tuple(regions),
                "card_graph": self.card_graph(),
            },
        )
        functions["dream_main"] = root
        module = IRModule(
            functions,
            deployment_table={"dream_main": tuple(regions)},
        )
        return DreamSSALowering(module, tuple(regions))

    def display_handoff(self) -> "DreamDisplayHandoff | None":
        owners = [
            block for block in self.blocks
            if block.gpu_deployment
            and block.stage == "fragment"
            and block.decorations.get("display-owner") == "program-interior"
        ]
        if not owners:
            return None
        if len(owners) != 1:
            raise DreamDocumentError("a dream document must have exactly one interior display owner")
        controllers = [
            block for block in self.blocks
            if block.decorations.get("role") == "display-controller"
        ]
        if len(controllers) != 1:
            raise DreamDocumentError("interior display ownership requires one display controller")
        owner = owners[0]
        controller = controllers[0]
        return DreamDisplayHandoff(
            owner.identity,
            owner.decorations.get("context", "webgl2"),
            owner.payload,
            controller.payload,
            controller.decorations.get("entry", "installTuringDisplay"),
            tuple(
                block for block in self.blocks
                if block.gpu_deployment and block.stage == "compute"
            ),
        )


def parse_dream_document(source: bytes | str) -> DreamDocument:
    raw = source.encode("utf-8") if isinstance(source, str) else bytes(source)
    position = 0
    blocks: list[DreamBlock] = []
    groups: list[DreamParallelDeployment] = []
    while position < len(raw):
        match = _OPEN.search(raw, position)
        if match is None:
            if raw[position:].strip():
                raise DreamDocumentError("unframed bytes remain after the final dream block")
            break
        if raw[position:match.start()].strip():
            raise DreamDocumentError("every non-whitespace byte must belong to a sentinel block")
        kind = match.group(1).decode("ascii")
        metadata = _header(match.group(2))
        identity = metadata.pop("id", "").strip()
        if not identity:
            raise DreamDocumentError(f"{kind} sentinel requires an id")
        if kind == "parallel":
            groups.append(DreamParallelDeployment(
                identity, _names(metadata.get("members")), metadata.get("join", "barrier"),
            ))
            position = match.end()
            continue
        close = raw.find(_CLOSE, match.end())
        if close < 0:
            raise DreamDocumentError(f"block {identity!r} has no closing sentinel")
        payload_bytes = raw[match.end():close]
        if payload_bytes.startswith(b"\r\n"):
            payload_bytes = payload_bytes[2:]
        elif payload_bytes.startswith(b"\n"):
            payload_bytes = payload_bytes[1:]
        if payload_bytes.endswith(b"\r\n"):
            payload_bytes = payload_bytes[:-2]
        elif payload_bytes.endswith(b"\n"):
            payload_bytes = payload_bytes[:-1]
        encoding = metadata.pop("encoding", "utf-8")
        try:
            payload = payload_bytes.decode(encoding)
        except (LookupError, UnicodeDecodeError) as error:
            raise DreamDocumentError(
                f"block {identity!r} cannot be decoded as {encoding!r}"
            ) from error
        expected_hash = metadata.pop("sha256", None)
        content_hash = sha256(payload_bytes).hexdigest()
        if expected_hash and expected_hash.lower() != content_hash:
            raise DreamDocumentError(f"block {identity!r} failed its sha256 sentinel")
        language = metadata.pop("language", "glsl" if kind == "shader" else "")
        if not language:
            raise DreamDocumentError(f"block {identity!r} requires a language")
        stage = metadata.pop("stage", None)
        if kind == "shader" and stage not in {"compute", "fragment"}:
            raise DreamDocumentError(f"shader {identity!r} requires compute or fragment stage")
        blocks.append(DreamBlock(
            identity=identity,
            kind=kind,
            language=language,
            payload=payload,
            encoding=encoding,
            stage=stage,
            inputs=_names(metadata.pop("inputs", None)),
            outputs=_names(metadata.pop("outputs", None)),
            decorations=MappingProxyType(dict(metadata)),
            content_hash=content_hash,
        ))
        position = close + len(_CLOSE)
    return DreamDocument(tuple(blocks), tuple(groups))


def load_dream_document(path: str | Path) -> DreamDocument:
    return parse_dream_document(Path(path).read_bytes())


@dataclass(frozen=True, slots=True)
class DreamExecutionRecord:
    block: str
    language: str
    stage: str | None
    gpu_active: bool
    result: Any = None


@dataclass(frozen=True, slots=True)
class DreamSSALowering:
    module: Any
    deployment_regions: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class DreamDisplayHandoff:
    owner: str
    context: str
    fragment_source: str
    controller_source: str
    controller_entry: str
    compute_blocks: tuple[DreamBlock, ...] = ()

    def to_shader_execution(self) -> dict[str, Any]:
        return {
            "role": "shader-surface",
            "display_ownership": "program-interior",
            "context": self.context,
            "candidates": [{
                "language": "webgl2-glsl-es",
                "role": "shader-surface",
            }],
            "interior": {
                "owner": self.owner,
                "fragment_source": self.fragment_source,
                "controller_source": self.controller_source,
                "controller_entry": self.controller_entry,
                "compute": [
                    {
                        "id": block.identity,
                        "language": block.language,
                        "stage": block.stage,
                        "source": block.payload,
                        "decorations": dict(block.decorations),
                    }
                    for block in self.compute_blocks
                ],
            },
        }


class DreamRuntime:
    """Card-graph read head for a dream document.

    Parallel members are submitted together and joined as a group. No shared
    state lock is manufactured: blocks must communicate through their declared
    ports or intentionally share a caller-provided arena.
    """

    def __init__(self, document: DreamDocument) -> None:
        self.document = document
        self.graph = document.card_graph()

    def run(
        self,
        handlers: Mapping[str, Callable[[DreamBlock], Any]],
        *,
        shader_deployer: Callable[[DreamBlock], Any] | None = None,
        gpu_indicator: Callable[[bool, DreamBlock], None] | None = None,
        maximum_workers: int | None = None,
    ) -> tuple[DreamExecutionRecord, ...]:
        group_by_member = {
            member: group
            for group in self.document.parallel
            for member in group.members
        }
        completed_groups: set[str] = set()
        records: list[DreamExecutionRecord] = []

        def execute(block: DreamBlock) -> DreamExecutionRecord:
            if block.gpu_deployment:
                if shader_deployer is None:
                    raise DreamDocumentError(
                        f"shader {block.identity!r} has no device deployer"
                    )
                if gpu_indicator is not None:
                    gpu_indicator(True, block)
                try:
                    result = shader_deployer(block)
                finally:
                    if gpu_indicator is not None:
                        gpu_indicator(False, block)
                return DreamExecutionRecord(
                    block.identity, block.language, block.stage, True, result,
                )
            handler = handlers.get(block.language)
            if handler is None:
                raise DreamDocumentError(
                    f"block {block.identity!r} has no handler for {block.language!r}"
                )
            return DreamExecutionRecord(
                block.identity, block.language, block.stage, False, handler(block),
            )

        for block in self.document.blocks:
            group = group_by_member.get(block.identity)
            if group is None:
                records.append(execute(block))
                continue
            if group.identity in completed_groups:
                continue
            members = [self.document.block(identity) for identity in group.members]
            with ThreadPoolExecutor(
                max_workers=maximum_workers or len(members),
                thread_name_prefix=f"dream-{group.identity}",
            ) as pool:
                futures = [pool.submit(execute, member) for member in members]
                records.extend(future.result() for future in futures)
            completed_groups.add(group.identity)
        return tuple(records)


def python_exec_handler(namespace: dict[str, Any] | None = None):
    """Return an explicit opt-in handler for trusted Python dream blocks."""

    scope = {} if namespace is None else namespace

    def execute(block: DreamBlock) -> Any:
        compiled = compile(block.payload, f"<dream:{block.identity}>", "exec")
        exec(compiled, scope, scope)
        return scope.get("result")

    return execute


def emit_dream_html_shell(document: DreamDocument, *, name: str = "dream_program"):
    """Build the standard shell with presentation owned by the document."""

    from .compiled_program_api import CompiledProgramAPI, EntryPoint, Parameter
    from .shell_io import (
        ShellIOManifest, ShellIORequest, SystemPort, attach_shell_io,
    )
    from .wasm_html_shell import emit_html_shell

    handoff = document.display_handoff()
    if handoff is None:
        raise DreamDocumentError(
            "HTML shell emission requires a fragment block promising interior display ownership"
        )
    section_compilations = document.compile_sections()
    api = CompiledProgramAPI(
        module=name,
        language="dream-document",
        entry="dream_main",
        entry_points=(
            EntryPoint("dream_main", "dream_main", "control"),
            EntryPoint("load_subject", "load_subject", "control", (
                Parameter(
                    "subject_bytes", "input", "u8", "uint8_t", "c_uint8",
                    "reference", source_name="binary_bytes",
                ),
                Parameter(
                    "subject_length", "input", "i64", "int64_t", "c_int64",
                    "value", source_name="binary_length",
                ),
            )),
        ),
        metadata={
            "document_schema": document.schema,
            "display_ownership": "program-interior",
            "section_compilations": [
                {
                    key: value
                    for key, value in compilation.to_mapping().items()
                    if key != "graph"
                }
                for compilation in section_compilations
            ],
            "deployment_regions": [
                {
                    "region_id": region.region_id,
                    "kind": region.kind,
                    "schedule": region.schedule,
                    "lanes": len(region.lanes),
                    "join": region.join.mode.value,
                }
                for region in document.lower_to_ssa().deployment_regions
            ],
        },
    )
    api = attach_shell_io(api, ShellIOManifest(
        (ShellIORequest.create("files"),),
        system_ports=(SystemPort.create(
            "subject-binary", "file", "input",
            entry_point="load_subject",
            fields={"data": "binary_bytes", "length": "binary_length"},
            attributes={
                "accept": ".exe,.dll,application/vnd.microsoft.portable-executable,application/octet-stream",
                "purpose": "machine-subject",
            },
        ),),
    ))
    source = "\n\n".join(
        f"/* dream block: {block.identity} ({block.language}:{block.stage or 'host'}) */\n{block.payload}"
        for block in document.blocks
    )
    return emit_html_shell(
        api,
        source=source,
        origin_source=source,
        map_ir={
            "card_graph": document.card_graph(),
            "section_graphs": {
                compilation.block: dict(compilation.graph)
                for compilation in section_compilations
            },
        },
        shader_execution=handoff.to_shader_execution(),
        name=f"{name}_shell",
    )


def embed_machine_snapshot(artifact, snapshot: bytes):
    """Publish one validated machine snapshot when a generated page opens."""

    from .wasm_html_shell import HtmlShell

    payload = bytes(snapshot)
    if payload[:8] != b"TMSNAP01":
        raise ValueError("embedded machine state is not a TMSNAP01 snapshot")
    encoded = base64.b64encode(payload).decode("ascii")
    bootstrap = f"""
<script id="turing-embedded-machine-snapshot">
document.addEventListener("DOMContentLoaded", () => {{
  const encoded = "{encoded}";
  const raw = atob(encoded);
  const snapshot = Uint8Array.from(raw, character => character.charCodeAt(0));
  if (!window.TuringMachineSnapshots?.publish) {{
    throw new Error("the interior display did not install TuringMachineSnapshots");
  }}
  window.TuringMachineSnapshots.publish(snapshot);
}});
</script>
"""
    if "</body>" not in artifact.html:
        raise ValueError("generated HTML shell has no body terminator")
    return HtmlShell(
        artifact.name,
        artifact.html.replace("</body>", bootstrap + "</body>", 1),
        artifact.embedded,
    )


def embed_machine_snapshot_replay(artifact, snapshots: tuple[bytes, ...] | list[bytes]):
    """Embed a finite reversible tape for interaction on a static website.

    The native owner remains the path for arbitrary binaries and external
    calls.  This replay makes a published bundle genuinely controllable without
    inventing server endpoints on GitHub Pages: forward/backward controls move
    over snapshots produced by the real executor from the shipped subject.
    """
    from .wasm_html_shell import HtmlShell

    payloads = tuple(bytes(snapshot) for snapshot in snapshots)
    if not payloads or any(payload[:8] != b"TMSNAP01" for payload in payloads):
        raise ValueError("embedded machine replay requires TMSNAP01 snapshots")
    encoded = [base64.b64encode(payload).decode("ascii") for payload in payloads]
    import json

    bootstrap = f"""
<script id="turing-embedded-machine-replay">
document.addEventListener("DOMContentLoaded", () => {{
  const api = window.TuringMachineSnapshots;
  if (!api?.publish) throw new Error("machine snapshot liaison is unavailable");
  api.disconnect();
  let frames = {json.dumps(encoded)}.map(encoded => {{
    const raw = atob(encoded);
    return Uint8Array.from(raw, character => character.charCodeAt(0));
  }});
  let index = 0;
  let direction = 0;
  let timer = 0;
  let generation = BigInt(api.generation || 0);
  let interval = 120;
  const publish = () => {{
    const frame = frames[index].slice();
    const view = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
    view.setBigUint64(16, ++generation, true);
    view.setInt32(24, direction, true);
    api.publish(frame);
  }};
  const stop = () => {{ if (timer) clearInterval(timer); timer = 0; }};
  const step = delta => {{
    index = Math.max(0, Math.min(frames.length - 1, index + delta));
    if ((delta > 0 && index === frames.length - 1) || (delta < 0 && index === 0)) {{
      direction = 0; stop();
    }}
    publish();
  }};
  const run = value => {{
    direction = value; stop(); publish();
    timer = setInterval(() => step(direction), interval);
  }};
  api.sendControl = async (action, value = null) => {{
    if (action === "pause") {{ direction = 0; stop(); publish(); }}
    else if (action === "step_forward") {{ direction = 0; stop(); step(1); }}
    else if (action === "step_backward") {{ direction = 0; stop(); step(-1); }}
    else if (action === "forward") run(1);
    else if (action === "backward") run(-1);
    else if (action === "speed") {{
      interval = Math.max(16, Math.min(1000, 1000000 / Math.max(1, Number(value))));
      if (direction) run(direction);
    }} else throw new Error("unknown embedded replay control: " + action);
  }};
  api.sendTerminalInput = async value => document.dispatchEvent(new CustomEvent(
    "turing-embedded-terminal-input", {{detail: String(value)}}
  ));
  api.loadSubject = async () => {{
    throw new Error("arbitrary subject loading requires the native Python owner");
  }};
  api.localReplay = {{
    get frames() {{ return frames.length; }},
    get index() {{ return index; }},
        copyFrames() {{ return frames.map(frame => frame.slice()); }},
    replaceFrames(nextFrames) {{
      if (!Array.isArray(nextFrames) || !nextFrames.length) {{
        throw new Error("embedded replay replacement requires at least one frame");
      }}
      stop();
      frames = nextFrames.map(frame => new Uint8Array(frame).slice());
      index = 0;
      direction = 0;
      publish();
    }},
  }};
  document.documentElement.dataset.machineTransport = "embedded-replay";
  publish();
}});
</script>
"""
    if "</body>" not in artifact.html:
        raise ValueError("generated HTML shell has no body terminator")
    return HtmlShell(
        artifact.name,
        artifact.html.replace("</body>", bootstrap + "</body>", 1),
        artifact.embedded,
    )


def embed_machine_wasm_block_bootstrap(artifact, descriptor: Mapping[str, Any]):
    """Execute and authenticate one published recompiled block in the browser."""
    import json

    from .wasm_html_shell import HtmlShell

    contract = dict(descriptor)
    required = {
        "module", "state", "guest", "plan", "journal_bytes",
        "expected_first_witness",
    }
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError("machine Wasm bootstrap is missing " + ", ".join(missing))
    bootstrap = f"""
<script id="turing-recompiled-machine-block">
document.addEventListener("DOMContentLoaded", async () => {{
  const contract = {json.dumps(contract, default=str)};
  const status = document.documentElement.dataset;
  const api = window.TuringMachineSnapshots;
  if (!api?.publish) throw new Error("machine snapshot liaison is unavailable");
  status.recompiledMachineBlock = "loading";
  try {{
    const fetchBytes = async path => {{
      const response = await fetch(new URL(path, document.baseURI), {{cache: "no-store"}});
      if (!response.ok) throw new Error(path + " returned HTTP " + response.status);
      return new Uint8Array(await response.arrayBuffer());
    }};
    const [moduleBytes, stateBytes, guestBytes, planResponse] = await Promise.all([
      fetchBytes(contract.module), fetchBytes(contract.state), fetchBytes(contract.guest),
      fetch(new URL(contract.plan, document.baseURI), {{cache: "no-store"}}),
    ]);
    if (!planResponse.ok) throw new Error(contract.plan + " returned HTTP " + planResponse.status);
    const plan = await planResponse.json();
    const {{instance}} = await WebAssembly.instantiate(moduleBytes, {{}});
    const memory = instance.exports.memory;
    const run = instance.exports.run;
    if (!(memory instanceof WebAssembly.Memory) || typeof run !== "function") {{
      throw new Error("recompiled machine block does not export memory/run");
    }}
    const stateOffset = 0;
    const journalOffset = 1024;
    const guestOffset = Math.ceil((journalOffset + Number(contract.journal_bytes)) / 4096) * 4096;
    const requiredBytes = Math.max(
      stateOffset + stateBytes.byteLength,
      journalOffset + Number(contract.journal_bytes),
      guestOffset + guestBytes.byteLength,
    );
    if (requiredBytes > memory.buffer.byteLength) {{
      memory.grow(Math.ceil((requiredBytes - memory.buffer.byteLength) / 65536));
    }}
    new Uint8Array(memory.buffer, stateOffset, stateBytes.byteLength).set(stateBytes);
    new Uint8Array(memory.buffer, guestOffset, guestBytes.byteLength).set(guestBytes);
    run(stateOffset, journalOffset, guestOffset);
    const journal = new Uint8Array(
      memory.buffer, journalOffset, Number(contract.journal_bytes),
    ).slice();
    const view = new DataView(journal.buffer, journal.byteOffset, journal.byteLength);
    for (const witness of plan.witnesses) {{
      const base = Number(witness.operation_index) * Number(plan.journal_stride);
      if (
        view.getBigUint64(base, true) !== BigInt(witness.address)
        || view.getBigUint64(base + 8, true) !== BigInt(witness.semantic_id)
        || view.getBigUint64(base + 16, true)
          !== BigInt("0x" + witness.encoded_sha256.slice(0, 16))
      ) throw new Error("browser Wasm journal provenance witness disagrees with its plan");
    }}
    const effectful = plan.witnesses.some(witness => {{
      const base = Number(witness.operation_index) * Number(plan.journal_stride);
      return view.getBigUint64(base + Number(plan.journal_effect_offset), true) !== 0n;
    }});
    const sourceSnapshot = api.current?.slice();
    if (!effectful && sourceSnapshot && api.localReplay?.replaceFrames) {{
      const sourceView = new DataView(
        sourceSnapshot.buffer, sourceSnapshot.byteOffset, sourceSnapshot.byteLength,
      );
      const registerCount = sourceView.getUint32(44, true);
      const registerOffset = sourceView.getUint32(52, true);
      if (registerCount < 54) throw new Error("machine snapshot register ABI is truncated");
      const projected = [sourceSnapshot];
      for (const witness of plan.witnesses) {{
        const frame = sourceSnapshot.slice();
        const frameView = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
        const record = Number(witness.operation_index) * Number(plan.journal_stride);
        const state = record + Number(plan.journal_state_offset);
        const copyQword = (snapshotIndex, stateIndex) => frameView.setBigUint64(
          registerOffset + snapshotIndex * 8,
          view.getBigUint64(state + stateIndex * 8, true), true,
        );
        for (let index = 0; index < 16; index++) copyQword(index, index);
        copyQword(16, 16); // RIP
        copyQword(17, 17); // RFLAGS
        for (let index = 0; index < 32; index++) copyQword(20 + index, 19 + index);
        copyQword(52, 18); // architectural step count
        frameView.setInt32(24, 1, true);
        frameView.setBigUint64(
          32, sourceView.getBigUint64(32, true) + BigInt(projected.length), true,
        );
        projected.push(frame);
      }}
            const retained = api.localReplay.copyFrames();
            api.localReplay.replaceFrames([
                ...projected,
                ...retained.slice(projected.length),
            ]);
      status.recompiledMachineProjection = "ready";
      const requestedSteps = Number(
        new URLSearchParams(location.search).get("recompiled-step") || 0,
      );
      if (Number.isInteger(requestedSteps) && requestedSteps > 0) {{
        const boundedSteps = Math.min(requestedSteps, projected.length - 1);
        for (let index = 0; index < boundedSteps; index++) {{
          await api.sendControl("step_forward");
        }}
        status.recompiledMachineStep = String(boundedSteps);
      }}
    }} else {{
      status.recompiledMachineProjection = effectful ? "effectful" : "unavailable";
    }}
    window.TuringRecompiledMachineBlock = Object.freeze({{
      contract, plan, instance, journal, effectful,
    }});
    status.recompiledMachineBlock = "ready";
    document.dispatchEvent(new CustomEvent(
      "turing-recompiled-machine-block", {{detail: {{contract, plan}}}},
    ));
  }} catch (error) {{
    status.recompiledMachineBlock = "error";
    status.recompiledMachineBlockError = error instanceof Error ? error.message : String(error);
    console.error("recompiled machine block bootstrap failed", error);
  }}
}});
</script>
"""
    if "</body>" not in artifact.html:
        raise ValueError("generated HTML shell has no body terminator")
    return HtmlShell(
        artifact.name,
        artifact.html.replace("</body>", bootstrap + "</body>", 1),
        artifact.embedded,
    )


def embed_machine_snapshot_stream(
    artifact,
    *,
    snapshot_endpoint: str = "/snapshot",
    input_endpoint: str = "/input",
    control_endpoint: str = "/control",
    subject_endpoint: str = "/subject",
    interval_milliseconds: int = 16,
):
    """Connect a generated interior display to its same-origin native host."""

    import json

    from .wasm_html_shell import HtmlShell

    interval = int(interval_milliseconds)
    if interval < 4:
        raise ValueError("machine snapshot polling interval must be at least 4 ms")
    snapshot_json = json.dumps(str(snapshot_endpoint))
    input_json = json.dumps(str(input_endpoint))
    control_json = json.dumps(str(control_endpoint))
    subject_json = json.dumps(str(subject_endpoint))
    bootstrap = f"""
<script id="turing-live-machine-snapshot">
document.addEventListener("DOMContentLoaded", () => {{
  if (!window.TuringMachineSnapshots?.connect) {{
    throw new Error("the interior display did not install TuringMachineSnapshots");
  }}
  window.TuringMachineSnapshots.connect({snapshot_json}, {{
    inputEndpoint: {input_json}, controlEndpoint: {control_json},
    subjectEndpoint: {subject_json}, interval: {interval}
  }});
  if (new URLSearchParams(location.search).get("snapshot-once") === "1") {{
    let unsubscribe = () => {{}};
    unsubscribe = window.TuringMachineSnapshots.subscribe(() => {{
      unsubscribe();
      window.TuringMachineSnapshots.disconnect();
      document.documentElement.dataset.snapshotOnceReady = "true";
    }});
  }}
}});
</script>
"""
    if "</body>" not in artifact.html:
        raise ValueError("generated HTML shell has no body terminator")
    return HtmlShell(
        artifact.name,
        artifact.html.replace("</body>", bootstrap + "</body>", 1),
        artifact.embedded,
    )


def main(argv: list[str] | None = None) -> int:
    """Inspect or reference-run one dream document through its card graph."""

    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true", help="print the card graph")
    mode.add_argument(
        "--run-reference", action="store_true",
        help="execute trusted Python and simulate in-place shader deployment",
    )
    mode.add_argument(
        "--emit-shell", type=Path, metavar="HTML",
        help="write a launchable shell that hands its context to the interior display",
    )
    parser.add_argument(
        "--machine-tape", type=Path, metavar="TAPE",
        help="embed the tape's latest dependency-validated TMSNAP01 state",
    )
    arguments = parser.parse_args(argv)
    if arguments.machine_tape is not None and arguments.emit_shell is None:
        parser.error("--machine-tape requires --emit-shell")
    document = load_dream_document(arguments.document)
    if arguments.inspect:
        print(json.dumps(document.card_graph(), indent=2, sort_keys=True))
        return 0
    if arguments.emit_shell is not None:
        artifact = emit_dream_html_shell(
            document, name=arguments.document.stem,
        )
        if arguments.machine_tape is not None:
            from .machine_state_buffer import build_machine_state_snapshot
            if arguments.machine_tape.is_dir():
                from .machine_tape_segments import SegmentedMachineTapeStore

                tape = SegmentedMachineTapeStore(arguments.machine_tape)
            else:
                from .machine_system_tape import MachineSystemTape

                tape = MachineSystemTape.read(arguments.machine_tape)
            states = []
            positions = []
            colors = []
            for core in range(tape.core_count):
                sequence = tape.latest_sequence(core)
                states.append(tape.resume_state(core, sequence=sequence))
                positions.append(int(tape.record(sequence)["position"]))
                colors.append(tape.annotation_color_rgba8(sequence, core=core))
            snapshot = build_machine_state_snapshot(
                states, positions=positions, annotation_colors=colors,
                maximum_output_bytes=1024 * 1024,
            )
            artifact = embed_machine_snapshot(artifact, snapshot)
        output = arguments.emit_shell
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(artifact.html, encoding="utf-8")
        print(output)
        return 0

    namespace: dict[str, Any] = {}

    def deploy(block: DreamBlock) -> str:
        # The reference runner proves scheduling and the device boundary. A
        # live OpenGL host replaces this callback without changing the file.
        return f"reference-deployed {block.language}:{block.stage}"

    def indicator(active: bool, block: DreamBlock) -> None:
        print(f"GPU {'ACTIVE' if active else 'IDLE'} | {block.identity}")

    records = DreamRuntime(document).run(
        {
            "python": python_exec_handler(namespace),
            "javascript": lambda block: f"deferred-browser-block:{block.identity}",
        },
        shader_deployer=deploy,
        gpu_indicator=indicator,
    )
    def display_result(value: Any) -> str:
        if isinstance(value, Mapping):
            return "{" + ", ".join(map(str, value.keys())) + "}"
        if isinstance(value, (list, tuple)) and len(value) > 8:
            return f"{type(value).__name__}[{len(value)}]"
        return repr(value)

    for record in records:
        print(
            f"{record.block}: {record.language}"
            f"{':' + record.stage if record.stage else ''} -> "
            f"{display_result(record.result)}"
        )
    return 0


__all__ = [
    "DreamBlock", "DreamDocument", "DreamDocumentError",
    "DreamDisplayHandoff", "DreamExecutionRecord", "DreamParallelDeployment",
    "DreamRuntime", "DreamSSALowering",
    "load_dream_document", "parse_dream_document", "python_exec_handler",
    "emit_dream_html_shell", "embed_machine_snapshot",
    "embed_machine_snapshot_stream", "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised as a module
    raise SystemExit(main())
