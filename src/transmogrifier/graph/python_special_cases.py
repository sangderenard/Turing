"""Python-only semantic overlays for structural ProcessGraph ingestion.

Source extraction and graph interpretation are deliberately separate. The
extraction contract resolves a live callable while that identity is still
available and attaches its receipt to the corresponding ``ast.Call``. This
module consumes that receipt at the graph boundary without admitting source.
"""

from __future__ import annotations

import ast
import copy
from typing import Any, Mapping

from .node_special_cases import SpecialCase, context_scope_statements
from .python_identity_programs import resolve_python_identity
from ...common.tensors.operator_catalog import canonical_operator_name


_EXTRACTION_ACTIONS = frozenset({
    "ingest_python",
    "intrinsic",
    "python_host_call",
    "use_native",
    "decompile_machine",
    "reject",
})

_SHELL_FILE_HELPERS = {
    "__turing_shell_file_open": "open",
    "__turing_shell_file_read": "read",
    "__turing_shell_file_write": "write",
    "__turing_shell_file_seek": "seek",
    "__turing_shell_file_tell": "tell",
    "__turing_shell_file_flush": "flush",
    "__turing_shell_file_close": "close",
}


def _shell_file_receipt(operation: str) -> dict[str, Any]:
    """Receipt attached to one compiler-created file-broker operation."""

    return {
        "action": "python_host_call",
        "rule_id": "python-filesystem-context-to-shell-file-broker",
        "identity": f"turing.shell.files.{operation}",
        "classification": "shell-resource-operation",
        "parameters": {
            "execution": "shell_io.file_broker",
            "shell_capability": "files",
            "shell_abi": "turing-shell-io-abi.files",
            "operation": str(operation),
        },
    }


def _shell_file_call(
    operation: str,
    arguments: list[ast.expr],
    source: ast.AST,
) -> ast.Call:
    call = ast.Call(
        func=ast.Name(
            id=f"__turing_shell_file_{operation}", ctx=ast.Load(),
        ),
        args=arguments,
        keywords=[],
    )
    call._extraction_contract = _shell_file_receipt(operation)
    call._turing_shell_file_context = {
        "schema": "turing.python-shell-file-context.v1",
        "operation": str(operation),
        "cleanup_policy": "ordered-scope-exit",
        "exception_policy": "shell-status",
    }
    return ast.copy_location(call, source)


class _ShellFileBodyRewriter(ast.NodeTransformer):
    """Replace method calls on one lexical stream by handle operations."""

    _METHODS = frozenset({
        "read", "write", "seek", "tell", "flush", "close",
    })

    def __init__(self, stream_name: str, handle_name: str) -> None:
        self.stream_name = str(stream_name)
        self.handle_name = str(handle_name)
        self.unsupported = False
        self.calls: list[ast.Call] = []

    def visit_Call(self, node):  # noqa: N802
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == self.stream_name
        ):
            if function.attr not in self._METHODS or node.keywords:
                self.unsupported = True
                return node
            arguments = [self.visit(copy.deepcopy(item)) for item in node.args]
            replacement = _shell_file_call(
                function.attr,
                [ast.Name(id=self.handle_name, ctx=ast.Load()), *arguments],
                node,
            )
            # Arguments are visited before the outer call is created, which
            # records nested stream operations in Python evaluation order.
            self.calls.append(replacement)
            return replacement
        return self.generic_visit(node)

    def visit_Name(self, node):  # noqa: N802
        # A stream passed to another function (pickle.dump(obj, stream), for
        # example) needs that function's stream protocol lowered first. Keep
        # the enclosing With intact instead of changing what the argument is.
        if isinstance(node.ctx, ast.Load) and node.id == self.stream_name:
            self.unsupported = True
        return node


class _ShellFileContextLowerer(ast.NodeTransformer):
    """Turn a resolved Python file scope into ordered shell operations.

    This deliberately handles only lexical stream-method use. A stream that
    escapes into an arbitrary call is retained as ``With`` so the compiler
    reports the still-missing protocol rather than silently passing an integer
    handle where Python promised a file object.
    """

    def __init__(self) -> None:
        self.counter = 0
        self.lowered: list[dict[str, Any]] = []
        self.function_stack: list[str] = []

    def visit_FunctionDef(self, node):  # noqa: N802
        self.function_stack.append(str(node.name))
        try:
            return self.generic_visit(node)
        finally:
            self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    @staticmethod
    def _plan_operand(node: ast.expr) -> dict[str, Any]:
        if isinstance(node, ast.Name):
            return {"kind": "name", "name": str(node.id)}
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (str, int, float, bool, type(None))
        ):
            return {"kind": "literal", "value": node.value}
        if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
            return {"kind": "bytes", "hex": node.value.hex()}
        return {"kind": "source", "expression": ast.unparse(node)}

    @staticmethod
    def _direct_call_results(statements: list[ast.stmt]) -> dict[int, str]:
        results: dict[int, str] = {}
        for statement in statements:
            for candidate in ast.walk(statement):
                if (
                    isinstance(candidate, ast.Assign)
                    and len(candidate.targets) == 1
                    and isinstance(candidate.targets[0], ast.Name)
                    and isinstance(candidate.value, ast.Call)
                ):
                    results[id(candidate.value)] = candidate.targets[0].id
                elif (
                    isinstance(candidate, ast.AnnAssign)
                    and isinstance(candidate.target, ast.Name)
                    and isinstance(candidate.value, ast.Call)
                ):
                    results[id(candidate.value)] = candidate.target.id
        return results

    @staticmethod
    def _file_context(call: ast.AST) -> bool:
        if not isinstance(call, ast.Call):
            return False
        receipt = extraction_receipt(call)
        return bool(
            receipt is not None
            and receipt["action"] == "python_host_call"
            and receipt["parameters"].get("execution")
            == "shell_io.file_broker"
        )

    @staticmethod
    def _path_and_mode(call: ast.Call) -> tuple[ast.expr, ast.expr] | None:
        identity = str(
            (extraction_receipt(call) or {}).get("identity") or ""
        )
        keyword_mode = next((
            keyword.value for keyword in call.keywords
            if keyword.arg == "mode"
        ), None)
        if isinstance(call.func, ast.Attribute) and identity.startswith(
            "pathlib."
        ):
            path = copy.deepcopy(call.func.value)
            mode = (
                copy.deepcopy(call.args[0]) if call.args
                else copy.deepcopy(keyword_mode)
                if keyword_mode is not None else ast.Constant("r")
            )
            return path, mode
        if identity in {"builtins.open", "io.open", "_io.open"}:
            if not call.args:
                return None
            path = copy.deepcopy(call.args[0])
            mode = (
                copy.deepcopy(call.args[1]) if len(call.args) > 1
                else copy.deepcopy(keyword_mode)
                if keyword_mode is not None else ast.Constant("r")
            )
            return path, mode
        return None

    def visit_With(self, node):  # noqa: N802
        node = self.generic_visit(node)
        if len(node.items) != 1:
            return node
        item = node.items[0]
        if (
            not self._file_context(item.context_expr)
            or not isinstance(item.optional_vars, ast.Name)
        ):
            return node
        path_mode = self._path_and_mode(item.context_expr)
        if path_mode is None:
            return node
        # The broker surface is byte-oriented. Python text streams also own
        # encoding, newline translation, and incremental decoder state; those
        # semantics need their own shell operation and must not be silently
        # approximated by C stdio. Dynamic modes likewise remain explicit.
        if not (
            isinstance(path_mode[1], ast.Constant)
            and isinstance(path_mode[1].value, str)
            and "b" in path_mode[1].value
        ):
            return node
        self.counter += 1
        # Keep the minted handle in the ordinary authored-name domain. The
        # lexical reducer reserves double-underscore spellings for language
        # and compiler implementation identities, so a dunder temporary can
        # be treated as an external implementation binding instead of the
        # local SSA value this scope just produced.
        handle_name = f"turing_file_handle_{self.counter}"
        rewriter = _ShellFileBodyRewriter(item.optional_vars.id, handle_name)
        body = [rewriter.visit(copy.deepcopy(statement)) for statement in node.body]
        if rewriter.unsupported:
            return node
        opened = _shell_file_call(
            "open", [path_mode[0], path_mode[1]], item.context_expr,
        )
        closed = _shell_file_call(
            "close", [ast.Name(id=handle_name, ctx=ast.Load())], node,
        )
        scope = (
            f"file-scope:{int(getattr(node, 'lineno', -1))}:"
            f"{int(getattr(node, 'col_offset', -1))}:{self.counter}"
        )
        ordered_calls = [opened, *rewriter.calls, closed]
        for sequence, call in enumerate(ordered_calls):
            call._turing_shell_file_context.update({
                "scope": scope,
                "sequence": int(sequence),
            })
        receipt = extraction_receipt(item.context_expr) or {}
        direct_results = self._direct_call_results(body)
        operation_results = {
            id(opened): handle_name,
            **direct_results,
        }
        self.lowered.append({
            "schema": "turing.python-shell-file-context.v1",
            "identity": receipt.get("identity"),
            "function": (
                self.function_stack[-1] if self.function_stack else None
            ),
            "handle": handle_name,
            "stream": item.optional_vars.id,
            "cleanup_policy": "ordered-scope-exit",
            "scope": scope,
            # These identities are retained as plan provenance. They are not
            # expected to materialize as repository-SSA instructions: the
            # enclosing shell consumes the ordered operation records instead.
            "operation_identities": tuple(
                str((_call._extraction_contract or {}).get("identity") or "")
                for _call in ordered_calls
            ),
            "operations": tuple({
                "operation": str(
                    (_call._extraction_contract or {}).get(
                        "parameters", {}
                    ).get("operation") or ""
                ),
                "sequence": int(sequence),
                "arguments": tuple(
                    self._plan_operand(argument) for argument in _call.args
                ),
                "result": operation_results.get(id(_call)),
                "source_span": {
                    "line": int(getattr(_call, "lineno", -1)),
                    "column": int(getattr(_call, "col_offset", -1)),
                },
            } for sequence, _call in enumerate(ordered_calls)),
        })
        return [
            ast.copy_location(ast.Assign(
                targets=[ast.Name(id=handle_name, ctx=ast.Store())],
                value=opened,
            ), node),
            *body,
            ast.copy_location(ast.Expr(value=closed), node),
        ]


def method_receiver_formal(definition: Any) -> str | None:
    """The formal Python binds a method's receiver to, as the source spells it.

    Python passes the instance (or, for a classmethod, the class) as the first
    positional argument, whatever that parameter is named; a staticmethod has
    no receiver.  The caller states that ``definition`` is a class member.
    """

    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    if any(
        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
        for decorator in definition.decorator_list
    ):
        return None
    positional = (*definition.args.posonlyargs, *definition.args.args)
    return positional[0].arg if positional else None


def lower_python_shell_file_contexts(tree: ast.AST) -> ast.AST:
    """Lower resolved, non-escaping Python file contexts in ``tree``."""

    lowerer = _ShellFileContextLowerer()
    lowerer.visit(tree)
    if lowerer.lowered:
        tree._turing_shell_file_contexts = tuple(lowerer.lowered)
    ast.fix_missing_locations(tree)
    return tree


def extraction_receipt(node: Any) -> dict[str, Any] | None:
    """Return a detached, minimally validated call receipt if one exists."""

    value = getattr(node, "_extraction_contract", None)
    if not isinstance(value, Mapping):
        return None
    receipt = dict(value)
    action = str(receipt.get("action") or "")
    if action not in _EXTRACTION_ACTIONS:
        return None
    receipt["action"] = action
    receipt["parameters"] = dict(receipt.get("parameters") or {})
    return receipt


def _receipt_attributes(receipt: Mapping[str, Any]) -> dict[str, Any]:
    attributes = {
        "extraction_contract": dict(receipt),
        "extraction_action": receipt["action"],
        "extraction_rule": receipt.get("rule_id"),
        "extraction_identity": receipt.get("identity"),
        "extraction_classification": receipt.get("classification"),
    }
    if receipt["action"] == "intrinsic":
        parameters = dict(receipt.get("parameters") or {})
        attributes["backend_intrinsic_candidate"] = {
            "semantic_identity": receipt.get("identity"),
            "lowering_namespace": parameters.get("lowering_namespace"),
            "ingested_fallback": bool(
                parameters.get("ingest_fallback_source", False)
            ),
        }
    return attributes


def _call_spelling(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _named_integer_origin(value: Any, path: str) -> dict[str, Any] | None:
    """Describe the CPython named-int wrapper category, if ``value`` is one."""

    value_type = type(value)
    if (
        value_type is int
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value_type.__dict__.get("__reduce__", object()) is not None
    ):
        return None
    symbolic_name = getattr(value, "name", None)
    if not isinstance(symbolic_name, str) or not symbolic_name:
        return None
    return {
        "schema": "turing.python-named-integer.v1",
        "path": str(path),
        "module": str(value_type.__module__),
        "type": str(value_type.__qualname__),
        "name": symbolic_name,
        "integer_value": int(value),
    }


def canonicalize_python_static_data(
    value: Any,
    *,
    path: str,
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    """Canonicalize named integers inside one static Python value tree."""

    origin = _named_integer_origin(value, path)
    if origin is not None:
        return int(value), (origin,)

    if isinstance(value, tuple):
        values = []
        origins = []
        for index, item in enumerate(value):
            canonical, nested = canonicalize_python_static_data(
                item,
                path=f"{path}[{index}]",
            )
            values.append(canonical)
            origins.extend(nested)
        return (tuple(values), tuple(origins)) if origins else (value, ())
    if isinstance(value, list):
        values = []
        origins = []
        for index, item in enumerate(value):
            canonical, nested = canonicalize_python_static_data(
                item,
                path=f"{path}[{index}]",
            )
            values.append(canonical)
            origins.extend(nested)
        return (values, tuple(origins)) if origins else (value, ())
    if isinstance(value, (set, frozenset)):
        values = []
        origins = []
        ordered_items = sorted(
            value,
            key=lambda item: (
                str(type(item).__module__),
                str(type(item).__qualname__),
                repr(item),
            ),
        )
        for index, item in enumerate(ordered_items):
            canonical, nested = canonicalize_python_static_data(
                item,
                path=f"{path}[{index}]",
            )
            values.append(canonical)
            origins.extend(nested)
        if not origins:
            return value, ()
        container = frozenset if isinstance(value, frozenset) else set
        return container(values), tuple(origins)
    if isinstance(value, dict):
        values = {}
        origins = []
        for index, (key, item) in enumerate(value.items()):
            canonical_key, key_origins = canonicalize_python_static_data(
                key,
                path=f"{path}.key[{index}]",
            )
            canonical_item, item_origins = canonicalize_python_static_data(
                item,
                path=f"{path}[{key!r}]",
            )
            values[canonical_key] = canonical_item
            origins.extend(key_origins)
            origins.extend(item_origins)
        return (values, tuple(origins)) if origins else (value, ())
    return value, ()


def canonicalize_python_static_bindings(
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a persistence-safe copy of a Python static environment."""

    return {
        name: canonicalize_python_static_data(
            value,
            path=str(name),
        )[0]
        for name, value in bindings.items()
    }


def is_python_static_literal(value: Any) -> bool:
    """Whether ``value`` is exact literal data a ProcessGraph may persist."""

    if value is None or value is Ellipsis or isinstance(
        value, (bool, bytes, complex, float, int, str),
    ):
        return True
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(is_python_static_literal(item) for item in value)
    if isinstance(value, dict):
        return all(
            is_python_static_literal(key) and is_python_static_literal(item)
            for key, item in value.items()
        )
    return False


def interpret_python_static_value(
    value: Any,
    *,
    path: str,
) -> SpecialCase | None:
    """Reduce a Python-only named integer wrapper to a plain graph constant.

    CPython uses private ``int`` subclasses for symbolic constants in a few
    source modules.  Some of them deliberately set ``__reduce__ = None``;
    retaining such a live wrapper in a resolved ProcessGraph makes ordinary
    graph serialization try to call that non-callable reducer.  The wrapper
    contributes no runtime behavior: its integer value is the program value
    and its name/type are source provenance.

    Keep this recognition at the Python ingestion boundary.  The reducer uses
    the returned special case to create the ordinary ``Constant`` leaf at the
    exact ``Name``/``Attribute`` occurrence and redirects that occurrence to
    it.  No native boundary is introduced and source pursuit is unchanged.
    """

    canonical, origins = canonicalize_python_static_data(value, path=path)
    if not origins:
        return None
    return SpecialCase(
        "Constant",
        {
            "value": canonical,
            "python_static_origins": origins,
        },
        canonical,
    )


_THREADING_CONSTRUCTORS = {"threading.Thread": "thread", "threading.Condition": "condition", "threading.Barrier": "barrier",
                 "threading.Event": "event", "collections.deque": "byte_queue", "_collections.deque": "byte_queue"}
_THREADING_METHODS = {
    "thread": {"start", "join", "is_alive"},
    "condition": {"acquire", "release", "wait", "notify", "notify_all"},
    "barrier": {"wait", "abort", "reset"},
    "event": {"set", "clear", "wait", "is_set"},
}
_MUTEX_CONSTRUCTORS = {
    "threading.Lock": False, "_thread.allocate_lock": False,
    "threading.RLock": True, "_thread.RLock": True,
}
_MUTEX_METHOD_IDENTITIES = {
    "_thread.lock.acquire": "mutex", "_thread.lock.release": "mutex", "_thread.lock.locked": "mutex",
    "_thread.RLock.acquire": "rmutex", "_thread.RLock.release": "rmutex",
}


def python_dispatch_entry_expressions(call: ast.Call, identity: str) -> tuple[ast.expr, ...]:
    """Source entry references owned by a resolved Python dispatch construct."""
    if identity != "threading.Thread":
        return ()
    targets = tuple(keyword.value for keyword in call.keywords if keyword.arg == "target")
    if targets:
        return targets
    # Thread(group=None, target=None, ...); group is not an entry point.
    return (call.args[1],) if len(call.args) > 1 else ()


def lower_python_threading(tree: ast.AST, *, lower_resources: bool = True) -> ast.AST:
    # Standalone syntax callers and real source ingestion use the same
    # source-root writer. Existing stamped occurrences retain their owner.
    from .graph_express2 import post_source_span, _annotate_visual_source_owners, _source_module_identity
    if isinstance(tree, ast.Module) and not hasattr(tree, '_turing_source_span_row'):
        _annotate_visual_source_owners(tree, module_name=_source_module_identity(tree))
    class Lowerer(ast.NodeTransformer):
        def __init__(self):
            self.bindings = {}
            self.records = []
            self.counter = 0
            self.names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

        def operation(self, kind, action, args, keywords, source, origin):
            operation = f"{kind}_{action}"
            call = ast.copy_location(ast.Call(
                func=ast.Name(id=f"turing_dispatch_{operation}", ctx=ast.Load()),
                args=args, keywords=keywords,
            ), source)
            record = {"operation": operation, "source_identity": origin,
                      "line": getattr(source, "lineno", None),
                      "column": getattr(source, "col_offset", None),
                      "requires_concurrent_progress": True}
            if kind == "sync":
                from ...compiler.synchronization import SYNCHRONIZATION_SPECS
                spec = SYNCHRONIZATION_SPECS[action]
                record["requires_concurrent_progress"] = spec.progress
                record["required_capability"] = (
                    "compiler_ordering" if spec.domain == "compiler" else
                    "memory_ordering" if spec.domain == "memory" else "communicating_tasks"
                )
            # A ``Thread`` in authored source is a request for CONCURRENT
            # PROGRESS, not a request for an operating-system thread. Taken
            # literally it would mean spawn-and-destroy per unit of work,
            # which is the one disposition the deployment runtime never
            # wants: ``turing_pool`` exists precisely because workers should
            # start once and park between frames ("already up and waiting
            # for jobs"), and ``HostDeploymentPool`` says the same on the
            # host side. So the authored spawn lowers to a JOB SUBMISSION
            # against an already-running pool, and the authored join lowers
            # to that frame's completion wait.
            #
            # Stating it here rather than leaving it to each backend keeps
            # one answer: a backend that reads ``dispatch_disposition``
            # cannot decide to spawn, and the serial disposition (zero
            # workers, caller drains) remains the identical claim loop
            # rather than a second implementation.
            if kind == "thread":
                record["dispatch_disposition"] = "pooled_job"
                record["pool_residency"] = "persistent"
            call._turing_dispatch_operation = record
            parameters = {"lowering_namespace": "dispatcher",
                          "operation": operation,
                          "required_capability": record.get("required_capability", "communicating_tasks")}
            if kind == "thread":
                parameters["dispatch_disposition"] = "pooled_job"
                parameters["pool_residency"] = "persistent"
            call._extraction_contract = {
                "action": "intrinsic", "identity": f"turing.dispatch.{operation}",
                "rule_id": "python-threading-to-dispatcher",
                "classification": "dispatcher-operation",
                "parameters": parameters,
            }
            self.records.append(record)
            return call

        def visit_Call(self, node):
            existing = getattr(node, '_turing_dispatch_operation', None)
            if existing is not None:
                # Re-entering the ingestion visitor after exception conversion
                # preserves the operation, visits only operands, and retains
                # its summary fact. Its synthetic func is never a Python callee.
                node.args = [self.visit(argument) for argument in node.args]
                node.keywords = [ast.keyword(arg=keyword.arg, value=self.visit(keyword.value))
                                 for keyword in node.keywords]
                self.records.append(existing)
                return node
            receipt = extraction_receipt(node) or {}
            identity = receipt.get("identity")
            # Rejected extraction decisions never become an authorized intrinsic.
            if receipt.get("action") == "reject":
                return self.generic_visit(node)
            node = self.generic_visit(node)
            if identity in _MUTEX_CONSTRUCTORS:
                if node.args or node.keywords:
                    raise ValueError("mutex constructor arguments require an explicit supported policy")
                from ...compiler.synchronization import Scope
                return self.operation("sync", "mutex_create", [
                    ast.copy_location(ast.Constant(int(Scope.OBJECT)), node),
                    ast.copy_location(ast.Constant(_MUTEX_CONSTRUCTORS[identity]), node),
                ], [], node, identity)
            from ...compiler.synchronization import synchronization_identity
            synchronization = synchronization_identity(identity)
            if synchronization is not None:
                return self.operation("sync", synchronization, node.args, node.keywords, node, identity)
            if identity == 'threading.Barrier':
                keywords = [ast.keyword(arg='participants' if keyword.arg == 'parties' else keyword.arg,
                                        value=keyword.value) for keyword in node.keywords]
                return self.operation('barrier', 'create', node.args, keywords, node, identity)
            if identity in _THREADING_CONSTRUCTORS:
                return self.operation(_THREADING_CONSTRUCTORS[identity], "create", node.args,
                                      node.keywords, node, identity)
            # Condition installs acquire/release from its lock on the
            # instance, so class-member source discovery cannot resolve them.
            # A locally constructed/aliased handle is the exact receiver proof.
            if isinstance(node.func, ast.Attribute):
                kind = self.bindings.get(node.func.value.id) if isinstance(node.func.value, ast.Name) else None
                if kind is None:
                    kind = _MUTEX_METHOD_IDENTITIES.get(identity)
                mutex_methods = {"acquire", "release", "locked"} if kind == "mutex" else {"acquire", "release"}
                if kind in {"mutex", "rmutex"} and node.func.attr in mutex_methods:
                    keywords = list(node.keywords)
                    if node.func.attr == "acquire":
                        supplied = {keyword.arg for keyword in keywords}
                        for index, (name, default) in enumerate((("blocking", True), ("timeout", -1))):
                            if len(node.args) <= index and name not in supplied:
                                keywords.append(ast.keyword(arg=name, value=ast.copy_location(ast.Constant(default), node)))
                    return self.operation("sync", "mutex_" + node.func.attr,
                                          [node.func.value, *node.args], keywords, node,
                                          "threading." + ("RLock" if kind == "rmutex" else "Lock") + "." + node.func.attr)
                if kind in _THREADING_METHODS and node.func.attr in _THREADING_METHODS[kind]:
                    return self.operation(kind, node.func.attr, [node.func.value, *node.args],
                                          node.keywords, node,
                                          f"threading.{kind.title()}.{node.func.attr}")
            for kind, methods in _THREADING_METHODS.items():
                prefix = f"threading.{kind.title()}."
                if isinstance(identity, str) and identity.startswith(prefix):
                    method = identity[len(prefix):]
                    if method in methods and isinstance(node.func, ast.Attribute):
                        return self.operation(kind, method, [node.func.value, *node.args],
                                              node.keywords, node, identity)
            return node

        def visit_Assign(self, node):
            node.value = self.visit(node.value)
            record = getattr(node.value, "_turing_dispatch_operation", {})
            operation = record.get("operation", "")
            kind = operation[:-7] if operation.endswith("_create") else None
            if operation == "sync_mutex_create":
                kind = "rmutex" if _MUTEX_CONSTRUCTORS.get(record.get("source_identity"), False) else "mutex"
            if isinstance(node.value, ast.Name):
                kind = self.bindings.get(node.value.id)
            for target in node.targets:
                for member in ast.walk(target):
                    if isinstance(member, ast.Name):
                        self.bindings.pop(member.id, None)
                if isinstance(target, ast.Name) and kind:
                    self.bindings[target.id] = kind
            return node

        def visit_FunctionDef(self, node):
            previous = self.bindings
            self.bindings = dict(previous)
            # Parameters and local assignments shadow outer bindings even before
            # their first assignment. Nested definitions have their own scope.
            pending = list(node.body)
            locals_ = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args,
                                          *node.args.kwonlyargs)}
            if node.args.vararg: locals_.add(node.args.vararg.arg)
            if node.args.kwarg: locals_.add(node.args.kwarg.arg)
            while pending:
                member = pending.pop()
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    locals_.add(member.name)
                    continue
                if isinstance(member, ast.Name) and isinstance(member.ctx, ast.Store):
                    locals_.add(member.id)
                pending.extend(ast.iter_child_nodes(member))
            for name in locals_: self.bindings.pop(name, None)
            node.body = self.statements(node.body)
            self.bindings = previous
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

        def statements(self, body):
            result = []
            for statement in body:
                lowered = self.visit(statement)
                result.extend(lowered if isinstance(lowered, list) else [lowered])
            return result

        def visit_If(self, node):
            node.test = self.visit(node.test)
            before = dict(self.bindings)
            node.body = self.statements(node.body)
            yes = dict(self.bindings)
            self.bindings = dict(before)
            node.orelse = self.statements(node.orelse)
            self.bindings = {name: kind for name, kind in yes.items()
                             if self.bindings.get(name) == kind}
            return node

        def visit_With(self, node):
            recognized = all(isinstance(item.context_expr, ast.Name)
                             and self.bindings.get(item.context_expr.id) in {"condition", "mutex", "rmutex"}
                             for item in node.items)
            if not recognized:
                return self.generic_visit(node)
            from ...compiler.concordance_declarations import SOURCE_RESOURCE_CONVERSION, INGESTION
            from ...compiler.identity_concordance import current_identity_book, Derived, Mode
            # Concord the authored scope before the first call-only pass
            # rewrites anything in its body. Later resource conversion reads
            # these exact recipe cells, not a new hash of rewritten syntax.
            conversions = getattr(node, '_turing_resource_conversions', None)
            kinds = tuple(self.bindings[item.context_expr.id] for item in node.items)
            if conversions is None:
                source = post_source_span(None, node)
                if source is None:
                    raise ValueError('resource scope has no ingested source occurrence')
                conversions = tuple(current_identity_book().post(
                    SOURCE_RESOURCE_CONVERSION, (source, ordinal),
                    (kind, 'condition_acquire' if kind == 'condition' else 'sync_mutex_acquire',
                     'condition_release' if kind == 'condition' else 'sync_mutex_release'), stage=INGESTION,
                    provenance=Derived((source,)), mode=Mode.CONCORD)
                    for ordinal, kind in enumerate(kinds))
                node._turing_resource_conversions = conversions
            if not lower_resources:
                return self.generic_visit(node)
            body = self.statements(node.body)
            for ordinal, item in reversed(tuple(enumerate(node.items))):
                conversion = conversions[ordinal]
                # Bind once, before entry. A body may reassign the source name;
                # cleanup must still release the condition actually acquired.
                while True:
                    self.counter += 1
                    name = f"turing_condition_scope_{self.counter}"
                    if name not in self.names:
                        self.names.add(name)
                        break
                capture = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                                     value=copy.deepcopy(item.context_expr))
                handle = ast.Name(id=name, ctx=ast.Load())
                kind = kinds[ordinal]
                owner = "threading.Condition" if kind == "condition" else "threading.RLock" if kind == "rmutex" else "threading.Lock"
                keywords = ([] if kind == "condition" else [
                    ast.keyword(arg="blocking", value=ast.Constant(True)),
                    ast.keyword(arg="timeout", value=ast.Constant(-1))])
                enter = self.operation("condition" if kind == "condition" else "sync",
                                       "acquire" if kind == "condition" else "mutex_acquire",
                                       [copy.deepcopy(handle)], keywords, node, owner + ".__enter__")
                acquire = (ast.Expr(value=enter) if item.optional_vars is None else
                           ast.Assign(targets=[copy.deepcopy(item.optional_vars)], value=enter))
                release = self.operation("condition" if kind == "condition" else "sync",
                                         "release" if kind == "condition" else "mutex_release", [handle], [], node,
                                         owner + ".__exit__")
                body = context_scope_statements(
                    [capture, acquire], body, cleanup=[ast.Expr(value=release)],
                )
                scope = body[2]
                scope._turing_resource_scope = name
                scope._turing_resource_contract = conversion
                scope._turing_resource_operations = (enter, release)
                for root in (capture, acquire, scope):
                    for generated in ast.walk(root):
                        if isinstance(generated, (ast.expr, ast.stmt)) and getattr(generated, '_turing_resource_conversion', None) is None:
                            generated._turing_resource_conversion = conversion
            return [ast.copy_location(statement, node) for statement in body]

        def visit_For(self, node):
            # Loop-carried/rebound receiver identities need control-aware
            # proof. Do not promote the final syntactic assignment to a fact
            # about every iteration or the zero-iteration exit.
            names = {member.id for member in ast.walk(node)
                     if isinstance(member, ast.Name) and isinstance(member.ctx, ast.Store)}
            for name in names: self.bindings.pop(name, None)
            node = self.generic_visit(node)
            for name in names: self.bindings.pop(name, None)
            return node

        visit_While = visit_For
        visit_Try = visit_For

    lowerer = Lowerer()
    tree = lowerer.visit(tree)
    ast.fix_missing_locations(tree)
    # Preserve lexical control ownership independently of source locations.
    # In particular, generated acquire and finally-release share a location
    # but must never be scheduled as adjacent straight-line calls.
    record_python_control_scopes(tree)
    tree._turing_dispatch_operations = tuple(lowerer.records)
    return tree


def record_python_control_scopes(node, scopes=()):
    """Retain source control ownership, including late-bound source calls."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        scopes = ()
    if isinstance(node, ast.Call):
        node._turing_control_scopes = scopes
    operation = getattr(node, "_turing_dispatch_operation", None)
    if operation is not None:
        operation["control_scopes"] = scopes
    for field, value in ast.iter_fields(node):
        nested = scopes
        if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.AsyncFor,
                             ast.While, ast.Try, ast.TryStar, ast.With,
                             ast.AsyncWith, ast.BoolOp, ast.comprehension,
                             ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            nested += ((type(node).__name__, getattr(node, "lineno", None), field),)
        for child in value if isinstance(value, list) else (value,):
            if isinstance(child, ast.AST):
                record_python_control_scopes(child, nested)


def interpret_python_special_case(node: Any) -> SpecialCase | None:
    """Classify Python syntax without performing callable source discovery.

    A non-terminal ``Call`` overlay retains the ordinary call role schema and
    authored argument edges. Its receipt states whether callee source was
    admitted, retained at a boundary, decompiled, or rejected.
    """

    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        attributes = {"attribute": node.attr}
        canonical = canonical_operator_name(node.attr)
        # ``real`` and ``imag`` are property spellings in NumPy/Torch but
        # ordinary abstract tensor operators in the compiler.  Record the
        # candidate at ingestion without asserting that an arbitrary object's
        # equally named field is a tensor operation; grounding proves that
        # later from the receiver's tensor descriptor.
        if canonical in {"real", "imag"}:
            attributes["tensor_candidate"] = canonical
        return SpecialCase("GetAttr", attributes, None)

    if not isinstance(node, ast.Call):
        return None

    receipt = extraction_receipt(node)
    if (receipt or {}).get("action") != "reject" and (receipt or {}).get("identity") in _THREADING_CONSTRUCTORS and not getattr(node, "_turing_dispatch_operation", None):
        # Source pursuit can deliver an initializer's exact boundary receipt
        # only when its function graph is extracted. Use the same frontend
        # replacement there, retaining the original occurrence identity.
        scopes = getattr(node, "_turing_control_scopes", ())
        lowered = lower_python_threading(ast.Expression(body=node)).body
        node._turing_dispatch_operation = dict(lowered._turing_dispatch_operation)
        node._turing_dispatch_operation["control_scopes"] = scopes
        node._extraction_contract = lowered._extraction_contract
        receipt = extraction_receipt(node)
    attributes = _receipt_attributes(receipt) if receipt is not None else {}
    dispatch_operation = getattr(node, "_turing_dispatch_operation", None)
    if dispatch_operation is not None:
        attributes.update({
            "dispatch_operation": dict(dispatch_operation),
            **({"extraction_identity": dispatch_operation['source_identity']}
               if dispatch_operation['operation'].startswith('barrier_') else {}),
            "ordered_effect": True,
            "deployment_owner": "dispatcher",
            "required_capability": dispatch_operation.get("required_capability", "communicating_tasks"),
        })
        # Dispatcher arguments are still authored expressions. A terminal
        # overlay dropped e.g. ACQUIRE_RELEASE | WORKGROUP_MEMORY entirely;
        # lexical reduction could recreate simple literals but not that
        # missing expression. Keep the incumbent Call role traversal.
        from ...compiler.thread_operations import PROCESS_OPERATORS
        return SpecialCase(
            PROCESS_OPERATORS.get(dispatch_operation["operation"], "Call"),
            attributes, None, terminal=False,
            # The Python callable has been replaced by the semantic operation.
            # Its source identity remains in the extraction receipt; only
            # actual operands belong in the executable dependency graph.
            role_schema={"up": {"args": "many", "keywords": "many"}, "down": {}},
        )
    spelling = _call_spelling(node)
    identity = receipt.get("identity") if receipt is not None else None
    shell_operation = (
        _SHELL_FILE_HELPERS.get(spelling)
        if spelling is not None else None
    )
    if shell_operation is not None:
        attributes.update({
            "shell_operation": shell_operation,
            "callee": f"turing_shell_file_{shell_operation}",
            "argument_count": len(node.args),
            "ordered_effect": True,
            "shell_boundary": True,
            "deployment_owner": "shell_io.file_broker",
            "shell_file_context": dict(
                getattr(node, "_turing_shell_file_context", {}) or {}
            ),
        })
        if (
            shell_operation == "open"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            attributes["file_mode"] = str(node.args[1].value)
        if (
            shell_operation == "seek"
            and len(node.args) >= 3
            and isinstance(node.args[2], ast.Constant)
            and isinstance(node.args[2].value, int)
        ):
            attributes["seek_origin"] = int(node.args[2].value)
        return SpecialCase(
            "Call",
            attributes,
            None,
            terminal=False,
            role_schema={
                "up": {"args": "many", "keywords": "many"},
                "down": {},
            },
        )
    program = resolve_python_identity(identity)
    if program is not None:
        attributes.update({
            "python_identity_program": program.mapping(),
            "python_replacement_kind": program.kind,
        })
        operator = program.direct_operator
        if operator is not None:
            attributes.update(program.direct_attributes)
            attributes["argument_count"] = len(node.args)
            return SpecialCase(operator, attributes, None, terminal=False)

    # Preserve the pre-contract convenience behavior for isolated structural
    # ingestion. Governed compilation always selects by resolved identity.
    if receipt is None and spelling in {"float", "int", "bool"}:
        return SpecialCase(
            spelling, {"cast": spelling}, None, terminal=False,
        )
    if receipt is None and spelling == "print":
        return SpecialCase(
            "stream_publish",
            {"stream": "text", "argument_count": len(node.args)},
            None,
            terminal=False,
        )

    if receipt is None:
        return None

    # Terminal with respect to source pursuit, but not argument dataflow.
    return SpecialCase("Call", attributes, None, terminal=False)


__all__ = [
    "canonicalize_python_static_bindings",
    "canonicalize_python_static_data",
    "extraction_receipt",
    "interpret_python_special_case",
    "interpret_python_static_value",
    "lower_python_shell_file_contexts",
    "lower_python_threading",
    "python_dispatch_entry_expressions",
]
