"""Source-owned native cuts shared by Python initialization and ingestion.

The tree records lexical ownership, not a guessed runtime call graph. ABI and
semantic proof remain owned by ProjectCompilationProduct and its installers.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import types


_CUTS = {}
_REBUILT = {}
_INSTALLED = {}


def source_tree(source_path, source_module):
    from .project_compilation_product import _authored_definitions, authored_definition_sha256
    path = Path(source_path).resolve()
    source = path.read_text(encoding="utf-8")
    definitions = _authored_definitions(source)
    owners = {id(definition.node): name for name, definition in definitions.items()}
    nodes = {}
    for name, definition in definitions.items():
        parent = None
        for ancestor in reversed(definition.ancestors):
            if id(ancestor) in owners:
                parent = owners[id(ancestor)]
                break
            if isinstance(ancestor, ast.ClassDef):
                parent = ancestor.name
                nodes.setdefault(parent, {"qualified_name": parent, "parent": None,
                                          "kind": "class", "available": []})
                break
        nodes[name] = {"qualified_name": name, "parent": parent, "kind": definition.kind,
                       "line": definition.node.lineno,
                       "definition_sha256": authored_definition_sha256(source, name),
                       "available": []}
    return {"source": str(path), "module": source_module,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "nodes": nodes}


def select_tree_cuts(tree):
    """An available enclosure supersedes all of its lexical descendants."""
    nodes = tree["nodes"]
    selected = {}
    for name, node in nodes.items():
        if not node["available"]:
            continue
        parent = node["parent"]
        while parent is not None and not nodes[parent]["available"]:
            parent = nodes[parent]["parent"]
        if parent is None:
            selected[name] = node["available"][-1]
    return selected


def nested_authored_callable(module, qualified_name):
    """Recover a capture-free nested code object without running its enclosure.

    A captured cell or dynamic default requires an explicit closure ABI. Do not
    invent a value for it while replaying a native verifier at process startup.
    """
    from .project_compilation_product import _authored_definitions
    path = Path(module.__file__).resolve()
    source = path.read_text(encoding="utf-8")
    node = _authored_definitions(source)[qualified_name].node
    code = compile(source, str(path), "exec")

    def find(current):
        for constant in current.co_consts:
            if not isinstance(constant, types.CodeType):
                continue
            if constant.co_qualname == qualified_name:
                return constant
            found = find(constant)
            if found is not None:
                return found
        return None

    code = find(code)
    if code is None or code.co_freevars:
        raise ValueError(f"native nested cut needs a declared closure-cell ABI: {qualified_name}")
    if node.decorator_list:
        raise ValueError(f"native nested cut needs its runtime decorator binding: {qualified_name}")
    try:
        defaults = tuple(ast.literal_eval(value) for value in node.args.defaults)
        kwdefaults = {argument.arg: ast.literal_eval(value)
                      for argument, value in zip(node.args.kwonlyargs, node.args.kw_defaults)
                      if value is not None}
    except (ValueError, TypeError):
        raise ValueError(f"native nested cut needs its runtime default binding: {qualified_name}") from None
    function = types.FunctionType(code, vars(module), node.name, defaults or None)
    function.__kwdefaults__ = kwdefaults
    function.__module__ = module.__name__
    function.__qualname__ = qualified_name
    return function


def register_cut(module, name, deployed, definition_sha256):
    native = getattr(deployed, "__turing_deployed_callable__", deployed)
    receipt = getattr(native, "__turing_native_verification__", {})
    if (receipt.get("status") != "verified" or receipt.get("qualified_name") != name
            or receipt.get("native_probe_count", 0) < 1 or receipt.get("fallback_probe_count") != 0):
        raise ValueError("source cuts require an exclusively native verified callable")
    deployed.__turing_cut_definition_sha256__ = definition_sha256
    _CUTS[(module, name)] = deployed


def _binding_name(module, name):
    return "__turing_native_cut_" + hashlib.sha256((module + ":" + name).encode()).hexdigest()[:24]


def splice_source(source, module, *, entrypoint=None, bindings=None):
    """Replace accepted bodies by calls to their declared native pieces.

    Preserve authored headers, decorators and locations. Only the requested
    root and its enclosing definitions reveal source for recompilation.
    """
    from .project_compilation_product import _authored_definitions, authored_definition_sha256
    definitions = _authored_definitions(source)
    lines = source.splitlines(keepends=True)
    selected_bindings = dict(bindings or {})
    edits = []
    for (owner, name), deployed in tuple(_CUTS.items()):
        if owner != module or name not in definitions:
            continue
        if authored_definition_sha256(source, name) != deployed.__turing_cut_definition_sha256__:
            continue
        if entrypoint is not None and (name == entrypoint or entrypoint.startswith(name + ".<locals>.")):
            continue
        from .extraction_contract import llvm_piece_of
        if llvm_piece_of(deployed) is None:
            continue  # Python installation alone does not declare a source-link ABI.
        definition = definitions[name]
        node = definition.node
        if node.args.vararg or node.args.kwarg or isinstance(node, ast.AsyncFunctionDef):
            raise ValueError(f"native source cut lacks a finite synchronous call ABI: {name}")
        symbol = _binding_name(module, name)
        selected_bindings[symbol] = deployed
        call = ast.Call(ast.Name(symbol, ast.Load()),
                        [ast.Name(argument.arg, ast.Load()) for argument in (*node.args.posonlyargs, *node.args.args)],
                        [ast.keyword(argument.arg, ast.Name(argument.arg, ast.Load()))
                         for argument in node.args.kwonlyargs])
        first_statement = node.body[0]
        if (len(node.body) > 1 and isinstance(first_statement, ast.Expr)
                and isinstance(first_statement.value, ast.Constant) and isinstance(first_statement.value.value, str)):
            first_statement = node.body[1]
        first, last = first_statement.lineno, node.body[-1].end_lineno
        prefix = lines[first - 1].encode("utf-8")[:first_statement.col_offset].decode("utf-8")
        edits.append((first, last, prefix + "return " + ast.unparse(call) + "\n"))
    if not edits:
        return source, selected_bindings
    # Selected tree cuts form an antichain, so their body spans never overlap.
    for first, last, replacement in sorted(edits, reverse=True):
        lines[first - 1:last] = [replacement] + ["\n"] * (last - first)
    rewritten = "".join(lines)
    ast.parse(rewritten)
    return rewritten, selected_bindings


def initialize_nested_cuts(module):
    """Rebuild Python enclosures without executing definitions or decorators."""
    from .project_compilation_product import _authored_definitions
    source = Path(module.__file__).read_text(encoding="utf-8")
    definitions = _authored_definitions(source)
    for (owner, name), deployed in tuple(_CUTS.items()):
        if owner != module.__name__ or ".<locals>." not in name:
            continue
        # The first importable enclosure owns all nested creation sites.
        root_name = name.split(".<locals>.", 1)[0]
        root_parts = root_name.split(".")
        target = module
        for component in root_parts[:-1]:
            target = getattr(target, component)
        descriptor = inspect.getattr_static(target, root_parts[-1])
        authored = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
        if isinstance(authored, property):
            authored = authored.fget
        authored = getattr(authored, "__turing_authored_source_callable__", authored)
        key = (module.__name__, root_name)
        original = _REBUILT.get(key, (target, descriptor))[1]
        original_function = original.__func__ if isinstance(original, (staticmethod, classmethod)) else original
        if isinstance(original_function, property):
            original_function = original_function.fget
        rewritten, bindings = splice_source(source, module.__name__, entrypoint=root_name)
        module.__dict__.update({name: value for name, value in bindings.items() if name.startswith("__turing_native_cut_")})
        code = compile(rewritten, str(Path(module.__file__).resolve()), "exec")
        def find(current):
            for constant in current.co_consts:
                if isinstance(constant, types.CodeType):
                    if constant.co_qualname == root_name:
                        return constant
                    found = find(constant)
                    if found is not None:
                        return found
            return None
        code = find(code)
        if code is None or code.co_freevars != original_function.__code__.co_freevars:
            raise ValueError(f"source splice changes enclosing closure ownership: {root_name}")
        rebuilt = types.FunctionType(code, vars(module), original_function.__name__,
                                     original_function.__defaults__, original_function.__closure__)
        rebuilt.__kwdefaults__ = original_function.__kwdefaults__
        rebuilt.__annotations__ = original_function.__annotations__
        rebuilt.__qualname__ = original_function.__qualname__
        rebuilt.__module__ = module.__name__
        rebuilt.__turing_authored_source_callable__ = original_function
        _REBUILT[key] = (target, original)
        replacement = (staticmethod(rebuilt) if isinstance(original, staticmethod) else
                       classmethod(rebuilt) if isinstance(original, classmethod) else
                       property(rebuilt, original.fset, original.fdel, original.__doc__)
                       if isinstance(original, property) else rebuilt)
        setattr(target, root_parts[-1], replacement)


def clear_source_cuts():
    for (_module, name), (owner, descriptor) in tuple(_REBUILT.items()):
        setattr(owner, name.rsplit(".", 1)[-1], descriptor)
    _REBUILT.clear()
    _CUTS.clear()


def deployment_source_trees(paths):
    """Describe only receipt-backed products over their exact authored revision."""
    from .project_compilation_product import open_project_compilation_product
    from .compiler_bootstrap_runtime import _ACTIVATION_ADAPTERS, _file_sha256
    import json
    trees = {}
    for path in paths:
        product = open_project_compilation_product(path)
        source = Path(str(product.manifest.get("source") or ""))
        if not source.is_file() or _file_sha256(source) != product.manifest.get("source_sha256"):
            raise ValueError(f"source tree product has a stale authored revision: {product.root}")
        for name, link in product.links.items():
            module = link.get("source_module")
            if not module or link.get("kind") == "source-region-integral":
                continue
            library = product.root / str(link.get("native_library") or "")
            api = product.root / str(link.get("native_api") or "")
            receipt_path = library.parent / "native-verification.json"
            if not library.is_file() or not api.is_file() or not receipt_path.is_file():
                continue
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("status") != "verified" or
                receipt.get("activation_adapter") not in _ACTIVATION_ADAPTERS or
                receipt.get("source_sha256") != product.manifest["source_sha256"] or
                receipt.get("qualified_name") != name or
                receipt.get("native_probe_count", 0) < 1 or receipt.get("fallback_probe_count") != 0 or
                receipt.get("library_sha256") != _file_sha256(library) or
                receipt.get("api_sha256") != _file_sha256(api)):
                continue
            key = (module, str(source.resolve()))
            tree = trees.setdefault(key, source_tree(source, module))
            node = tree["nodes"].get(name)
            if node is None or (link.get("authored_source_sha256") and
                                link["authored_source_sha256"] != node.get("definition_sha256")):
                raise ValueError(f"native cut does not match its authored declaration: {module}.{name}")
            node["available"].append({"product": str(product.root), "qualified_name": name,
                                      "verification_sha256": _file_sha256(receipt_path)})
    result = []
    for tree in trees.values():
        tree["selected_cuts"] = select_tree_cuts(tree)
        result.append(tree)
    return result


def initialize_source_tree(paths):
    """Install maximal verified cuts; provenance bootstrap keeps its module gate."""
    from .compiler_bootstrap_runtime import (
        _ACTIVATION_ADAPTERS, _ACTIVE_DEPLOYMENTS, CompilerBootstrapActivation,
        PROVENANCE_CORE_MODULE,
    )
    from .project_compilation_product import open_project_compilation_product
    import importlib
    import json
    paths = tuple(Path(path).resolve() for path in paths)
    trees = deployment_source_trees(paths)
    if any(tree["module"] == PROVENANCE_CORE_MODULE for tree in trees):
        raise ValueError("provenance core installation requires the complete module activation gate")
    products = {str(path): open_project_compilation_product(path) for path in paths}
    prepared = []
    for tree in trees:
        for name, cut in tree["selected_cuts"].items():
            product = products[cut["product"]]
            library = product.root / product.links[name]["native_library"]
            receipt = json.loads((library.parent / "native-verification.json").read_text())
            adapter = receipt["activation_adapter"]
            owner, deployed = _ACTIVATION_ADAPTERS[adapter](product, name)
            prepared.append((tree, name, product, owner, deployed, adapter))
    old_cuts, old_rebuilt, old_installed = dict(_CUTS), dict(_REBUILT), dict(_INSTALLED)
    old_deployments = dict(_ACTIVE_DEPLOYMENTS)
    snapshots = []
    for key, (owner, _original, _installed) in old_installed.items():
        attribute = key[1].rsplit(".", 1)[-1]
        snapshots.append((owner, attribute, inspect.getattr_static(owner, attribute)))
    for (_module, name), (owner, _original) in old_rebuilt.items():
        attribute = name.rsplit(".", 1)[-1]
        snapshots.append((owner, attribute, inspect.getattr_static(owner, attribute)))
    for _tree, name, _product, owner, _deployed, _adapter in prepared:
        attribute = name.rsplit(".", 1)[-1]
        snapshots.append((owner, attribute, inspect.getattr_static(owner, attribute)))
    namespaces = [(importlib.import_module(tree["module"]), {
        key: value for key, value in vars(importlib.import_module(tree["module"])).items()
        if key.startswith("__turing_native_cut_")}) for tree in trees]
    try:
        # All native proofs have succeeded before any previous tree is dismantled.
        clear_source_cuts()
        for key, (owner, descriptor, installed) in tuple(_INSTALLED.items()):
            attribute = key[1].rsplit(".", 1)[-1]
            if inspect.getattr_static(owner, attribute) is installed:
                setattr(owner, attribute, descriptor)
            _ACTIVE_DEPLOYMENTS.pop(key, None)
        _INSTALLED.clear()
        activations = []
        for tree, name, product, owner, deployed, adapter in prepared:
            attribute = name.rsplit(".", 1)[-1]
            original = inspect.getattr_static(owner, attribute)
            installed = product.install_callable(name, owner, deployed, targeted_source_fallback=True)
            key = (str(product.root), name)
            _INSTALLED[key] = (owner, original, inspect.getattr_static(owner, attribute))
            _ACTIVE_DEPLOYMENTS[key] = deployed
            register_cut(tree["module"], name, installed, tree["nodes"][name]["definition_sha256"])
            verification = deployed.__turing_native_verification__
            activations.append(CompilerBootstrapActivation(
                str(product.root), name, adapter, verification["status"],
                verification["native_probe_count"], verification["fallback_probe_count"],
            ).to_mapping())
        for tree in trees:
            initialize_nested_cuts(importlib.import_module(tree["module"]))
        return {"schema": "turing.initialized-source-tree.v1", "trees": trees,
                "activations": activations}

    except BaseException:
        for owner, attribute, value in reversed(snapshots):
            setattr(owner, attribute, value)
        for module, bindings in namespaces:
            for key in tuple(vars(module)):
                if key.startswith("__turing_native_cut_") and key not in bindings:
                    del module.__dict__[key]
            module.__dict__.update(bindings)
        _CUTS.clear()
        _CUTS.update(old_cuts)
        _REBUILT.clear()
        _REBUILT.update(old_rebuilt)
        _INSTALLED.clear()
        _INSTALLED.update(old_installed)
        _ACTIVE_DEPLOYMENTS.clear()
        _ACTIVE_DEPLOYMENTS.update(old_deployments)
        raise


def initialize_registered_source_tree(registry_path=None):
    from .compiler_bootstrap_runtime import registered_compiler_bootstrap_product_paths
    return initialize_source_tree(registered_compiler_bootstrap_product_paths(registry_path))
