"""Generic compiled products for class-like numerical object surfaces.

The product boundary deliberately joins existing compiler authorities instead
of creating a library-specific compiler: ``KernelBank`` owns verified
parametric/specialized forward artifacts, ProcessGraph owns analytical graph
inversion, and repository SSA plus LLVM own native reverse compilation.
Publication is atomic and fails closed unless every method has both its
parametric forward and its compiled parametric VJP.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Callable, Mapping, Sequence

from .kernel_bank import CompiledVariant, KernelBank, KernelSpec
from .llvm_training_runtime import (
    BrowserGraphReverse,
    NativeGraphReverse,
    compile_graph_reverse_to_wasm,
    compile_native_graph_reverse,
    native_artifact_record,
)


STANDARD_OBJECT_SCHEMA = "turing.standard-object-product.v1"


@dataclass(frozen=True, slots=True)
class StandardProperty:
    name: str
    kind: str
    mutable: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "mutable": bool(self.mutable),
        }


@dataclass(frozen=True)
class StandardObjectSurface:
    """Authored class surface and one representative whole-object entry."""

    source: str
    entrypoint: str
    feeds: Mapping[str, Any]


@dataclass(frozen=True)
class MethodGraphCapture:
    """A fresh AbstractTensor method run and its differentiation contract."""

    output: Any
    bindings: Mapping[str, Any]
    wrt_value_ids: tuple[int, ...]


@dataclass(frozen=True)
class StandardMethod:
    name: str
    kernel: KernelSpec
    capture_graph: Callable[[], MethodGraphCapture]
    specializations: tuple[Mapping[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.name != self.kernel.name:
            raise ValueError(
                f"method {self.name!r} must name kernel {self.kernel.name!r}"
            )


@dataclass(frozen=True)
class StandardObject:
    name: str
    identity: str
    methods: tuple[StandardMethod, ...]
    properties: tuple[StandardProperty, ...] = ()
    surface: StandardObjectSurface | None = None

    def __post_init__(self) -> None:
        method_names = tuple(method.name for method in self.methods)
        property_names = tuple(prop.name for prop in self.properties)
        if not self.name or not self.identity:
            raise ValueError("standard object requires a name and identity")
        if not method_names:
            raise ValueError(f"standard object {self.identity!r} has no methods")
        if len(set(method_names)) != len(method_names):
            raise ValueError(f"standard object repeats methods: {method_names!r}")
        if len(set(property_names)) != len(property_names):
            raise ValueError(f"standard object repeats properties: {property_names!r}")


@dataclass(frozen=True)
class CompiledStandardMethod:
    spec: StandardMethod
    parametric_forward: CompiledVariant
    parametric_reverse: NativeGraphReverse
    browser_reverse: BrowserGraphReverse
    specialized_forwards: tuple[CompiledVariant, ...]


@dataclass(frozen=True)
class StandardObjectProduct:
    directory: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    methods: Mapping[str, CompiledStandardMethod]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def _variant_record(variant: CompiledVariant) -> dict[str, Any]:
    manifest = json.loads(
        (variant.directory / "manifest.json").read_text(encoding="utf-8")
    )
    return {
        "key": variant.key,
        "specialized": dict(variant.specialized),
        "manifest": str((variant.directory / "manifest.json").resolve()),
        "verification": dict(manifest.get("verification") or {}),
    }


def _reverse_artifact_record(
    reverse: NativeGraphReverse, root: Path,
) -> dict[str, Any]:
    record = native_artifact_record(reverse.artifact)
    record["library_path"] = Path(record["library_path"]).relative_to(root).as_posix()
    return record


def _browser_reverse_record(
    reverse: BrowserGraphReverse, root: Path,
) -> dict[str, Any]:
    return {
        "name": reverse.name,
        "wasm_path": reverse.wasm_path.relative_to(root).as_posix(),
        "buffer_order": list(reverse.buffer_order),
        "buffer_shapes": [list(shape) for shape in reverse.buffer_shapes],
        "buffer_dtypes": list(reverse.buffer_dtypes),
        "extent_order": [list(item) for item in reverse.extent_order],
    }


def _browser_reverse_javascript() -> str:
    return r'''const TYPES={double:[Float64Array,8],i32:[Int32Array,4],i64:[BigInt64Array,8],i1:[Uint8Array,1],ptr:[Uint32Array,4]};
const align=(value,boundary)=>(value+boundary-1)&~(boundary-1);
const count=shape=>shape.length?shape.reduce((a,b)=>a*Number(b),1):1;

export class CompiledObjectReverse {
  static async load(base=new URL("./",import.meta.url)) {
    const manifest=await(await fetch(new URL("manifest.json",base))).json();
    return new CompiledObjectReverse(base,manifest);
  }
  constructor(base,manifest){this.base=base;this.manifest=manifest;this.instances=new Map();this.methods=manifest.methods.map(item=>item.name);}
  async instance(method){
    if(this.instances.has(method))return this.instances.get(method);
    const artifact=this.manifest.artifacts[method].browser_parametric_reverse;
    const bytes=await(await fetch(new URL(artifact.wasm_path,this.base))).arrayBuffer();
    const instance=(await WebAssembly.instantiate(bytes,{})).instance;
    const value={artifact,instance};this.instances.set(method,value);return value;
  }
  async vjp(method,upstream,bindings={}){
    const semantic=this.manifest.methods.find(item=>item.name===method);
    if(!semantic)throw new Error(`unknown compiled reverse ${method}`);
    const expected=Object.keys(semantic.reverse_input_value_ids).sort();
    const received=Object.keys(bindings).sort();
    if(JSON.stringify(expected)!==JSON.stringify(received))throw new Error(`${method} reverse bindings must be ${expected.join(", ")}`);
    const {artifact,instance}=await this.instance(method),feeds=new Map();
    for(const [name,id] of Object.entries(semantic.reverse_input_value_ids))feeds.set(Number(id),bindings[name]);
    const outputIds=semantic.reverse_output_value_ids.map(Number);
    const upstreams=outputIds.length===1?[upstream]:upstream;
    if(!Array.isArray(upstreams)||upstreams.length!==outputIds.length)throw new Error(`${method} reverse needs ${outputIds.length} upstream value(s)`);
    for(let i=0;i<outputIds.length;i++)feeds.set(Number(semantic.reverse_seed_value_ids[String(outputIds[i])]),upstreams[i]);
    const order=artifact.buffer_order.map(Number),layouts=new Map();
    let cursor=align(Number(instance.exports.__heap_base.value),8);
    const pointerOffset=cursor;cursor+=order.length*4;
    const extentOffset=align(cursor,4);cursor=extentOffset+artifact.extent_order.length*4;
    for(let i=0;i<order.length;i++){
      const dtype=artifact.buffer_dtypes[i],shape=artifact.buffer_shapes[i],[Ctor,width]=TYPES[dtype];
      if(!Ctor)throw new Error(`unsupported reverse ABI dtype ${dtype}`);
      cursor=align(cursor,width);layouts.set(order[i],{offset:cursor,Ctor,shape,size:count(shape)});cursor+=count(shape)*width;
    }
    const missing=cursor-instance.exports.memory.buffer.byteLength;
    if(missing>0)instance.exports.memory.grow(Math.ceil(missing/65536));
    const memory=instance.exports.memory.buffer,pointers=new Uint32Array(memory,pointerOffset,order.length);
    order.forEach((id,index)=>{const layout=layouts.get(id);pointers[index]=layout.offset;const view=new layout.Ctor(memory,layout.offset,layout.size);if(feeds.has(id)){const value=feeds.get(id);if(layout.Ctor===BigInt64Array)view.set(Array.from(value?.length===undefined?[value]:value,BigInt));else view.set(value?.length===undefined?[value]:value);}});
    const extents=new Int32Array(memory,extentOffset,artifact.extent_order.length);
    artifact.extent_order.forEach(([id,kind,axis],index)=>{const layout=layouts.get(Number(id)),shape=layout.shape;extents[index]=kind==="rank"?shape.length:(kind==="numel"||kind==="element_count")?layout.size:(axis!==null&&axis!==undefined)?Number(shape[Number(axis)]):0;});
    instance.exports[artifact.name](pointerOffset,extentOffset);
    const gradients={};
    for(const [name,inputId] of Object.entries(semantic.reverse_input_value_ids)){const gradientId=Number(semantic.reverse_gradient_value_ids[String(inputId)]),layout=layouts.get(gradientId),view=new layout.Ctor(memory,layout.offset,layout.size),copy=new layout.Ctor(view);gradients[name]=layout.shape.length?copy:copy[0];}
    return gradients;
  }
}
export default CompiledObjectReverse;
'''


def _compile_object_surface(
    surface: StandardObjectSurface,
    *,
    identity: str,
    root: Path,
) -> dict[str, Any]:
    from .fortran_c_shell import lower_ast_source_to_ssa

    module, outputs, exports = lower_ast_source_to_ssa(
        surface.source,
        surface.entrypoint,
        name=identity.replace(".", "_"),
        runtime_closure_only=False,
    )
    if module is None:
        raise RuntimeError(f"{identity} whole-object lowering produced no SSA")
    unresolved = tuple(
        record
        for records in module.call_table.values()
        for record in records
        if record.resolution == "unresolved"
    )
    if unresolved:
        raise RuntimeError(
            f"{identity} whole-object surface has unresolved calls: "
            f"{tuple((item.caller, item.callee_symbol) for item in unresolved)!r}"
        )
    artifact_path = root / "object-surface.pkl"
    with artifact_path.open("wb") as stream:
        pickle.dump((module, outputs, exports), stream, protocol=5)
    return {
        "source_sha256": hashlib.sha256(surface.source.encode("utf-8")).hexdigest(),
        "entrypoint": surface.entrypoint,
        "feed_names": sorted(surface.feeds),
        "artifact": artifact_path.relative_to(root).as_posix(),
        "functions": sorted(module.functions),
        "exports": list(exports),
        "classes": sorted(
            str(record.identity)
            for record in getattr(
                getattr(module, "class_table", None), "classes", ()
            )
        ),
        "call_count": sum(len(records) for records in module.call_table.values()),
        "unresolved_call_count": 0,
    }


def cook_standard_object(
    spec: StandardObject,
    *,
    directory: str | Path,
    contract: str | None = None,
) -> StandardObjectProduct:
    """Compile and publish one complete standard numerical object.

    Ordering is an invariant: every parametric forward is admitted first,
    every explicit-seed graph reverse is then compiled to a native library,
    and only then are optional exact parameter rows admitted.  Consequently a
    specialization can never make an otherwise incomplete method publishable.
    """

    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bank = KernelBank(root / "forward-bank", {
        method.name: method.kernel for method in spec.methods
    })
    surface_record = (
        _compile_object_surface(spec.surface, identity=spec.identity, root=root)
        if spec.surface is not None else None
    )
    compiled: dict[str, CompiledStandardMethod] = {}

    for method in spec.methods:
        forward = bank.get(
            method.name, contract=contract, specialized=None,
        )
        capture = method.capture_graph()
        reverse = compile_native_graph_reverse(
            capture.output,
            bindings=capture.bindings,
            wrt_value_ids=capture.wrt_value_ids,
            name=f"{spec.identity.replace('.', '_')}__{method.name}__vjp",
            directory=root / "reverse" / method.name,
            unit_output_seed=False,
        )
        browser_reverse = compile_graph_reverse_to_wasm(
            reverse,
            directory=root / "browser-reverse" / method.name,
        )
        specializations = tuple(
            bank.get(
                method.name,
                contract=contract,
                specialized=dict(parameters),
            )
            for parameters in method.specializations
        )
        compiled[method.name] = CompiledStandardMethod(
            method, forward, reverse, browser_reverse, specializations,
        )

    browser_loader_source = _browser_reverse_javascript()
    semantic_manifest = {
        "schema": STANDARD_OBJECT_SCHEMA,
        "object": {
            "name": spec.name,
            "identity": spec.identity,
            "properties": [prop.to_mapping() for prop in spec.properties],
            **({"surface": surface_record} if surface_record is not None else {}),
        },
        "contract": contract or "develop",
        "publication_invariant": {
            "parametric_forward_required": True,
            "parametric_graph_reverse_required": True,
            "reverse_must_be_backend_compiled": True,
            "browser_reverse_must_be_compiled": True,
            "specializations_are_optional_overlays": True,
        },
        "browser_loader_source_sha256": hashlib.sha256(
            browser_loader_source.encode("utf-8")
        ).hexdigest(),
        "methods": [
            {
                "name": method.name,
                "source_sha256": hashlib.sha256(
                    method.kernel.source.encode("utf-8")
                ).hexdigest(),
                "parametric_forward_key": compiled[method.name].parametric_forward.key,
                "reverse_input_value_ids": {
                    str(key): int(value) for key, value in
                    compiled[method.name].parametric_reverse.input_value_ids.items()
                },
                "reverse_output_value_ids": list(
                    compiled[method.name].parametric_reverse.output_value_ids
                ),
                "reverse_gradient_value_ids": {
                    str(key): int(value) for key, value in
                    compiled[method.name].parametric_reverse.gradient_value_ids.items()
                },
                "reverse_seed_value_ids": {
                    str(key): int(value) for key, value in
                    compiled[method.name].parametric_reverse.seed_value_ids.items()
                },
                "specializations": [
                    dict(variant.specialized)
                    for variant in compiled[method.name].specialized_forwards
                ],
            }
            for method in spec.methods
        ],
    }
    product_id = hashlib.sha256(_canonical(semantic_manifest)).hexdigest()
    browser_loader = root / "compiled-reverse.js"
    browser_loader.write_text(
        browser_loader_source, encoding="utf-8", newline="\n",
    )
    manifest = {
        **semantic_manifest,
        "product_id": product_id,
        "browser_loader": browser_loader.relative_to(root).as_posix(),
        "artifacts": {
            method.name: {
                "parametric_forward": _variant_record(
                    compiled[method.name].parametric_forward
                ),
                "parametric_reverse": _reverse_artifact_record(
                    compiled[method.name].parametric_reverse, root
                ),
                "browser_parametric_reverse": _browser_reverse_record(
                    compiled[method.name].browser_reverse, root
                ),
                "specialized_forwards": [
                    _variant_record(variant) for variant in
                    compiled[method.name].specialized_forwards
                ],
            }
            for method in spec.methods
        },
    }
    manifest_path = root / "manifest.json"
    temporary = root / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(manifest_path)
    return StandardObjectProduct(root, manifest_path, manifest, compiled)


def mathematical_sublibrary_object(
    library: Any,
    *,
    kernels: Mapping[str, KernelSpec],
    graph_captures: Mapping[str, Callable[[], MethodGraphCapture]],
    specializations: Mapping[str, Sequence[Mapping[str, int]]] | None = None,
) -> StandardObject:
    """Adapt a canonical mathematical sublibrary without copying its catalog."""

    names = tuple(method.name for method in library.methods)
    missing_kernels = set(names) - set(kernels)
    missing_captures = set(names) - set(graph_captures)
    extras = (set(kernels) | set(graph_captures)) - set(names)
    if missing_kernels or missing_captures or extras:
        raise ValueError(
            "mathematical object implementation does not equal its catalog: "
            f"missing_kernels={sorted(missing_kernels)!r}, "
            f"missing_graph_captures={sorted(missing_captures)!r}, "
            f"extras={sorted(extras)!r}"
        )
    matrix = dict(specializations or {})
    return StandardObject(
        name=str(library.name),
        identity=str(library.identity),
        methods=tuple(
            StandardMethod(
                name,
                kernels[name],
                graph_captures[name],
                tuple(matrix.get(name, ())),
            )
            for name in names
        ),
    )


def standard_object_from_source(
    *,
    name: str,
    identity: str,
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    kernels: Mapping[str, KernelSpec],
    graph_captures: Mapping[str, Callable[[], MethodGraphCapture]],
    specializations: Mapping[str, Sequence[Mapping[str, int]]] | None = None,
) -> StandardObject:
    """Batch-ingest one authored class surface and its numerical recipes."""

    tree = ast.parse(source)
    classes = tuple(node for node in tree.body if isinstance(node, ast.ClassDef))
    if len(classes) != 1:
        raise ValueError(
            "standard-object source must contain exactly one public class; "
            f"found {tuple(node.name for node in classes)!r}"
        )
    class_node = classes[0]
    methods = tuple(
        node.name for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
        and not any(
            isinstance(decorator, ast.Name) and decorator.id == "property"
            for decorator in node.decorator_list
        )
    )
    properties: list[StandardProperty] = []
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(decorator, ast.Name) and decorator.id == "property"
            for decorator in node.decorator_list
        ):
            properties.append(StandardProperty(node.name, "property", False))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            properties.append(StandardProperty(
                node.target.id, ast.unparse(node.annotation), node.value is not None,
            ))
    expected = set(methods)
    missing_kernels = expected - set(kernels)
    missing_captures = expected - set(graph_captures)
    extras = (set(kernels) | set(graph_captures)) - expected
    if missing_kernels or missing_captures or extras:
        raise ValueError(
            "standard-object implementations do not equal the authored surface: "
            f"missing_kernels={sorted(missing_kernels)!r}, "
            f"missing_graph_captures={sorted(missing_captures)!r}, "
            f"extras={sorted(extras)!r}"
        )
    matrix = dict(specializations or {})
    return StandardObject(
        name=name,
        identity=identity,
        methods=tuple(
            StandardMethod(
                method, kernels[method], graph_captures[method],
                tuple(matrix.get(method, ())),
            )
            for method in methods
        ),
        properties=tuple(properties),
        surface=StandardObjectSurface(source, entrypoint, dict(feeds)),
    )
