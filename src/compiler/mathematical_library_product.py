"""Emit the outer mathematical-library shell and its library subproducts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from ..common.tensors.mathematical_library import TURING_MATHEMATICAL_LIBRARY
from .blas_server import BLASServerProduct, build_blas_server
from .numpy_mathematical_library import emit_numpy_mathematical_library
from .standard_object_blas import blas_standard_object
from .standard_object_trigonometry import trigonometry_standard_object
from .standard_object_product import StandardObjectProduct, cook_standard_object
from .wasm_binary import CodeBuilder, build_module


PRODUCT_SCHEMA = "turing.mathematical-library-product.v1"
MATRIX_SCHEMA = "turing.mathematical-library-matrix.v1"


class MathematicalLibraryProductError(RuntimeError):
    """The outer product cannot be emitted without misrepresenting coverage."""


@dataclass(frozen=True, slots=True)
class MathematicalLibraryProduct:
    directory: Path
    manifest_path: Path
    matrix_path: Path
    python_loader: Path
    numpy_loader: Path
    javascript_path: Path
    wasm_path: Path
    demo_path: Path
    blas: BLASServerProduct
    blas_object: StandardObjectProduct
    trigonometry_object: StandardObjectProduct
    manifest: Mapping[str, Any]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _prepare_root(root: Path) -> Path:
    marker = root / ".turing-mathematical-library-build"
    if root.exists() and any(root.iterdir()):
        owned = marker.is_file()
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            try:
                owned = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ).get("schema") == PRODUCT_SCHEMA
            except (OSError, ValueError):
                owned = False
        if not owned:
            raise MathematicalLibraryProductError(
                f"refusing to replace non-mathematical-library directory: {root}"
            )
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(PRODUCT_SCHEMA, encoding="ascii")
    return marker


def _coverage(blas: BLASServerProduct) -> list[dict[str, Any]]:
    deployed = {
        surface: frozenset(roles)
        for surface, roles in blas.manifest["surface_roles"].items()
    }
    deployed["python_numpy"] = frozenset(
        method.identity
        for method in TURING_MATHEMATICAL_LIBRARY.library("blas").methods
    )
    records = []
    for method in TURING_MATHEMATICAL_LIBRARY.library("blas").methods:
        realizations = {
            surface: {
                "status": (
                    "packaged" if method.identity in deployed[surface]
                    else "semantic-only"
                ),
                **({"product": "libraries/blas"}
                   if method.identity in deployed[surface] else {}),
                **({
                    "reason": "no durable deployment product is registered yet",
                } if method.identity not in deployed[surface] else {}),
            }
            for surface in ("native", "python", "python_numpy", "webgpu")
        }
        records.append({
            "method": method.name,
            "identity": method.identity,
            "source_sha256": method.source_sha256,
            "realizations": realizations,
        })
    return records


def _python_loader() -> str:
    return '''"""Generated Python view of one Turing mathematical library."""
from __future__ import annotations
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


class CompiledObjectReverse:
    """Generated Python ABI for the product's compiled method VJPs."""

    _DTYPES = {
        "double": np.float64, "i32": np.int32, "i64": np.int64,
        "i1": np.bool_, "ptr": np.uintp,
    }

    def __init__(self, root, record):
        self.root = Path(root)
        self.record = record
        self.methods = tuple(item["name"] for item in record["methods"])
        self._methods = {item["name"]: item for item in record["methods"]}
        self._artifacts = record["artifacts"]
        self._libraries = {}

    def vjp(self, method, upstream, **bindings):
        method = str(method)
        try:
            semantic = self._methods[method]
            artifact = self._artifacts[method]["parametric_reverse"]
        except KeyError as error:
            raise KeyError(f"unknown compiled reverse {method!r}") from error
        expected = set(semantic["reverse_input_value_ids"])
        if set(bindings) != expected:
            raise ValueError(
                f"{method} reverse bindings must be {sorted(expected)!r}; "
                f"received {sorted(bindings)!r}"
            )
        feeds = {
            int(semantic["reverse_input_value_ids"][name]): value
            for name, value in bindings.items()
        }
        output_ids = tuple(map(int, semantic["reverse_output_value_ids"]))
        upstreams = (
            tuple(upstream) if isinstance(upstream, (tuple, list))
            else (upstream,)
        )
        if len(upstreams) != len(output_ids):
            raise ValueError(
                f"{method} reverse needs {len(output_ids)} upstream value(s)"
            )
        seed_ids = {
            int(key): int(value)
            for key, value in semantic["reverse_seed_value_ids"].items()
        }
        feeds.update({
            seed_ids[output_id]: value
            for output_id, value in zip(output_ids, upstreams)
        })
        order = tuple(map(int, artifact["buffer_order"]))
        dtypes = tuple(artifact.get("buffer_dtypes") or ("double",) * len(order))
        buffers = {}
        for value_id, shape, dtype in zip(order, artifact["buffer_shapes"], dtypes):
            numpy_dtype = self._DTYPES[str(dtype)]
            if value_id in feeds:
                value = np.asarray(feeds[value_id], dtype=numpy_dtype)
                buffers[value_id] = (
                    np.ascontiguousarray(value) if value.ndim else value.copy()
                )
            else:
                buffers[value_id] = np.zeros(tuple(shape) or (), dtype=numpy_dtype)
        pointers = (ctypes.c_void_p * len(order))(*(
            ctypes.c_void_p(int(buffers[value_id].ctypes.data))
            for value_id in order
        ))
        extents = []
        for value_id, kind, axis in artifact["extent_order"]:
            value = buffers[int(value_id)]
            if kind in ("numel", "element_count"):
                extents.append(int(value.size))
            elif kind == "rank":
                extents.append(int(value.ndim))
            elif kind in ("dim", "shape") and axis is not None:
                extents.append(int(value.shape[int(axis)]))
            elif kind == "shape" and value.ndim == 0:
                extents.append(0)
            else:
                raise ValueError(f"cannot derive reverse extent {(value_id, kind, axis)!r}")
        library = self._libraries.get(method)
        if library is None:
            library = ctypes.CDLL(str(self.root / artifact["library_path"]))
            self._libraries[method] = library
        entry = getattr(library, artifact["name"])
        entry.restype = None
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32),
        ]
        extent_array = (ctypes.c_int32 * len(extents))(*extents)
        entry(pointers, extent_array)
        gradients = {
            int(key): int(value)
            for key, value in semantic["reverse_gradient_value_ids"].items()
        }
        return {
            name: buffers[gradients[int(value_id)]].copy()
            for name, value_id in semantic["reverse_input_value_ids"].items()
        }


class CompiledStandardObject:
    """One source-bearing standard object, deployment matrix, and installer."""

    _DTYPES = CompiledObjectReverse._DTYPES

    def __init__(self, root, record):
        self.root = Path(root)
        self.record = record
        self.methods = tuple(item["name"] for item in record["methods"])
        self.sources = {
            item["name"]: item["source"] for item in record["methods"]
        }
        self.deployment_matrix = tuple(record["deployment_matrix"])
        self._artifacts = record["artifacts"]
        self._libraries = {}
        self._installations = {}

    def authored_source(self, method):
        try:
            return self.sources[str(method)]
        except KeyError as error:
            raise KeyError(f"unknown authored method {method!r}") from error

    def select(self, method, parameters=None):
        requested = dict(parameters or {})
        rows = [
            row for row in self.deployment_matrix
            if row["method"] == method and row["parameters"] == requested
        ]
        if not rows and requested:
            rows = [
                row for row in self.deployment_matrix
                if row["method"] == method and not row["parameters"]
            ]
        if len(rows) != 1:
            raise RuntimeError(
                f"{method} deployment matrix has {len(rows)} matches for "
                f"{requested!r}"
            )
        row = rows[0]
        if not row["parameters"]:
            return self._artifacts[method]["parametric_forward"]
        return next(
            record
            for record in self._artifacts[method]["specialized_forwards"]
            if record["key"] == row["key"]
        )

    def __getattr__(self, method):
        if method not in self.methods:
            raise AttributeError(method)

        def invoke(*args, **bindings):
            deployment_parameters = bindings.pop("_deployment_parameters", None)
            artifact = self.select(method, deployment_parameters)
            expected = tuple(artifact["input_value_ids"])
            if len(args) > len(expected):
                raise TypeError(f"{method} accepts {len(expected)} inputs")
            bound = dict(zip(expected, args))
            overlap = set(bound) & set(bindings)
            if overlap:
                raise TypeError(f"{method} repeats bindings {sorted(overlap)!r}")
            bound.update(bindings)
            if set(bound) != set(expected):
                raise TypeError(
                    f"{method} bindings must be {sorted(expected)!r}"
                )
            return self._call(method, artifact, bound)

        return invoke

    @staticmethod
    def _installed_argument(value):
        return value.data if hasattr(value, "ensure_tensor") else value

    @staticmethod
    def _installed_result(template, value):
        if isinstance(value, tuple):
            return tuple(
                CompiledStandardObject._installed_result(template, item)
                for item in value
            )
        return template.ensure_tensor(value)

    def install(self, host):
        """Replace declared host operators with this object's deployer."""

        if host in self._installations:
            return self
        originals = {}
        installable = {
            item["name"] for item in self.record["methods"]
            if item.get("installation") == "instance_operator"
        }
        for method in installable:
            originals[method] = (
                method in vars(host), getattr(host, method),
            )

            def deployed(template, *args, __method=method, **kwargs):
                raw_args = tuple(
                    self._installed_argument(value)
                    for value in (template, *args)
                )
                raw_kwargs = {
                    name: self._installed_argument(value)
                    for name, value in kwargs.items()
                }
                result = getattr(self, __method)(*raw_args, **raw_kwargs)
                return self._installed_result(template, result)

            setattr(host, method, deployed)
        self._installations[host] = originals
        packs = tuple(getattr(host, "_installed_operator_packs", ()))
        setattr(host, "_installed_operator_packs", (*packs, self))
        return self

    def uninstall(self, host):
        originals = self._installations.pop(host, None)
        if originals is None:
            return self
        for method, (owned, original) in originals.items():
            if owned:
                setattr(host, method, original)
            else:
                delattr(host, method)
        packs = tuple(
            pack for pack in getattr(host, "_installed_operator_packs", ())
            if pack is not self
        )
        setattr(host, "_installed_operator_packs", packs)
        return self

    def _call(self, method, record, bindings):
        if record.get("kind") != "captured_graph":
            raise RuntimeError(f"{method} is not a captured-graph forward")
        artifact = record["artifact"]
        feeds = {
            int(record["input_value_ids"][name]): value
            for name, value in bindings.items()
        }
        order = tuple(map(int, artifact["buffer_order"]))
        dtypes = tuple(artifact.get("buffer_dtypes") or ("double",) * len(order))
        buffers = {}
        for value_id, shape, dtype in zip(order, artifact["buffer_shapes"], dtypes):
            numpy_dtype = self._DTYPES[str(dtype)]
            if value_id in feeds:
                value = np.asarray(feeds[value_id], dtype=numpy_dtype)
                buffers[value_id] = np.ascontiguousarray(value).copy()
            else:
                buffers[value_id] = np.zeros(tuple(shape) or (), dtype=numpy_dtype)
        pointers = (ctypes.c_void_p * len(order))(*(
            ctypes.c_void_p(int(buffers[value_id].ctypes.data))
            for value_id in order
        ))
        extents = []
        for value_id, kind, axis in artifact["extent_order"]:
            value = buffers[int(value_id)]
            if kind in ("numel", "element_count"):
                extents.append(int(value.size))
            elif kind == "rank":
                extents.append(int(value.ndim))
            elif kind in ("dim", "shape") and axis is not None:
                extents.append(int(value.shape[int(axis)]))
            else:
                raise ValueError(f"cannot derive forward extent {(value_id, kind, axis)!r}")
        library = self._libraries.get(method)
        if library is None:
            library = ctypes.CDLL(str(self.root / artifact["library_path"]))
            self._libraries[method] = library
        entry = getattr(library, artifact["name"])
        entry.restype = None
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32),
        ]
        extent_array = (ctypes.c_int32 * len(extents))(*extents)
        entry(pointers, extent_array)
        outputs = tuple(
            buffers[int(value_id)].copy()
            for value_id in record["output_value_ids"]
        )
        return outputs[0] if len(outputs) == 1 else outputs


class TuringMathematicalLibrary:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        matrix_bytes = (self.root / self.manifest["matrix"]).read_bytes()
        if hashlib.sha256(matrix_bytes).hexdigest() != self.manifest["matrix_sha256"]:
            raise RuntimeError("mathematical-library matrix identity mismatch")
        self.matrix = json.loads(matrix_bytes)
        product = self.matrix["products"]["blas"]
        object_path = self.root / product["standard_object"]["path"]
        object_record = json.loads(
            (object_path / "manifest.json").read_text(encoding="utf-8")
        )
        self.blas_reverse = CompiledObjectReverse(object_path, object_record)
        loader = self.root / product["path"] / "python" / "turing_blas_server.py"
        spec = importlib.util.spec_from_file_location("packaged_turing_blas", loader)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError("generated BLAS loader cannot be imported")
        spec.loader.exec_module(module)
        self.blas = module.load(self.root / product["path"])
        self.blas.reverse = self.blas_reverse
        self.blas.vjp = self.blas_reverse.vjp
        trig_product = self.matrix["products"].get("trigonometry")
        if trig_product is not None:
            trig_path = self.root / trig_product["standard_object"]["path"]
            trig_record = json.loads(
                (trig_path / "manifest.json").read_text(encoding="utf-8")
            )
            self.trigonometry = CompiledStandardObject(trig_path, trig_record)
            self.trigonometry_reverse = CompiledObjectReverse(
                trig_path, trig_record,
            )
            self.trigonometry.reverse = self.trigonometry_reverse
            self.trigonometry.vjp = self.trigonometry_reverse.vjp
        numpy_loader = self.root / self.manifest["surfaces"]["python"]["numpy"]["module"]
        numpy_spec = importlib.util.spec_from_file_location(
            "packaged_turing_numpy_math", numpy_loader,
        )
        numpy_module = importlib.util.module_from_spec(numpy_spec)
        if numpy_spec.loader is None:
            raise RuntimeError("generated NumPy mathematical library cannot be imported")
        numpy_spec.loader.exec_module(numpy_module)
        self.numpy = numpy_module.load()

    @property
    def libraries(self):
        return tuple(self.matrix["products"])

    def install(self, host, attribute="math", implementation="numpy"):
        providers = {"numpy": self.numpy, "native": self}
        try:
            provider = providers[str(implementation)]
        except KeyError as error:
            raise ValueError("implementation must be 'numpy' or 'native'") from error
        hook = getattr(host, "install_mathematical_library", None)
        if hook is not None:
            hook(provider)
            for name in provider.libraries:
                sublibrary = getattr(provider, name, None)
                installer = getattr(sublibrary, "install", None)
                if installer is not None:
                    installer(host)
            return provider
        setattr(host, str(attribute), provider)
        return provider

    def close(self):
        self.blas.close()


def load(root=None):
    return TuringMathematicalLibrary(root)
'''


def _javascript(matrix_size: int, digest: str) -> str:
    return f'''const MATRIX_BYTES={matrix_size};
const MATRIX_SHA256="{digest}";
const hex=bytes=>[...new Uint8Array(bytes)].map(v=>v.toString(16).padStart(2,"0")).join("");

export class TuringMathematicalLibrary {{
  static async load(base=new URL("./",import.meta.url)) {{
    const wasm=await(await fetch(new URL("mathematical-library.wasm",base))).arrayBuffer();
    const instance=(await WebAssembly.instantiate(wasm,{{}})).instance;
    const bytes=new Uint8Array(instance.exports.memory.buffer,0,MATRIX_BYTES).slice();
    const digest=hex(await crypto.subtle.digest("SHA-256",bytes));
    if(digest!==MATRIX_SHA256)throw new Error("mathematical-library matrix identity mismatch");
    const matrix=JSON.parse(new TextDecoder().decode(bytes));
    const module=await import(new URL("../libraries/blas/web/blas-server.js",base));
    const blas=await module.TuringBLASServer.load(new URL("../libraries/blas/web/",base));
    const reverseModule=await import(new URL("../objects/blas/compiled-reverse.js",base));
    const reverse=await reverseModule.CompiledObjectReverse.load(new URL("../objects/blas/",base));
    blas.reverse=reverse;blas.vjp=reverse.vjp.bind(reverse);
    return new TuringMathematicalLibrary(matrix,blas,reverse);
  }}
  constructor(matrix,blas,reverse){{this.matrix=matrix;this.blas=blas;this.blasReverse=reverse;}}
  get libraries(){{return Object.keys(this.matrix.products);}}
  install(target=globalThis,options={{}}){{const primary=options.name??"turingMath";target[primary]=this;target.tensorMath=this;target.turingBLAS=this.blas;return this;}}
}}

export default TuringMathematicalLibrary;
'''


def _browser_installer() -> str:
    return '''(function installTuringMathematicalLibrary() {
  "use strict";
  const script = document.currentScript;
  if (!script) throw new Error("Turing math installer must run as a classic script");
  const configured = script.dataset.turingMathBase;
  const base = configured
    ? new URL(configured, document.baseURI)
    : new URL("./", script.src);
  const name = script.dataset.turingMathGlobal || "turingMath";
  const ready = import(new URL("mathematical-library.js", base).href)
    .then(({TuringMathematicalLibrary}) => TuringMathematicalLibrary.load(base))
    .then((library) => {
      library.install(globalThis, {name});
      globalThis.dispatchEvent(new CustomEvent(
        "turing-math-ready", {detail: {library, name, base: base.href}},
      ));
      return library;
    });
  globalThis.turingMathReady = ready;
})();
'''


def _browser_template() -> str:
    return '''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Turing tensor math</title></head>
<body>
<pre id="result">Loading Turing tensor math…</pre>

<!-- Set both paths relative to this HTML document. The product installs
     window.turingMath, window.tensorMath, and window.turingBLAS page-wide. -->
<script src="./install-turing-math.js" data-turing-math-base="./"></script>
<script type="module">
  const math = await window.turingMathReady;
  const a = new Float32Array([1, 2, 3, 4]);
  const b = new Float32Array([5, 6, 7, 8]);
  const c = await math.blas.gemm(a, b, {m: 2, n: 2, k: 2});
  document.querySelector("#result").textContent = JSON.stringify([...c]);
</script>
</body>
</html>
'''


def _demo() -> str:
    return '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Turing mathematical library</title>
<style>
:root{color-scheme:dark;--ink:#eaf2f8;--muted:#91a7b8;--panel:#0d1924;--line:#203646;--cyan:#65d6e8;--green:#76e6a2;--red:#ff8a8a}
*{box-sizing:border-box}body{background:#081019;color:var(--ink);font:14px ui-monospace,SFMono-Regular,Consolas,monospace;max-width:1160px;margin:0 auto;padding:32px 22px 70px}h1{font-size:28px;margin-bottom:5px}h2{font-size:17px;margin-top:30px}.lead,.muted{color:var(--muted)}.chips{display:flex;gap:7px;flex-wrap:wrap;margin:16px 0}.chip{border:1px solid var(--line);background:var(--panel);border-radius:99px;padding:6px 10px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}.card,pre,.notice{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:15px}.card b{display:block;color:var(--cyan);margin-bottom:6px}.toolbar{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:14px 0}button{padding:9px 12px;background:var(--cyan);color:#061017;border:0;border-radius:6px;font:inherit;font-weight:bold;cursor:pointer}button.secondary{background:#203646;color:var(--ink)}button:disabled{opacity:.45;cursor:wait}input{width:72px;padding:8px;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px}table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:normal}.ok{color:var(--green)}.error{color:var(--red)}pre{overflow:auto;white-space:pre-wrap;line-height:1.5}code{color:#c9f4fb}.notice{display:none;border-color:#785353;color:#ffd2d2}
</style>
</head>
<body>
<h1>Turing mathematical library</h1>
<p class="lead">A packaged, installable numerical surface—not a single GEMM demo.</p>
<div class="chips"><span class="chip">scal</span><span class="chip">axpy</span><span class="chip">dot</span><span class="chip">gemv</span><span class="chip">gemm</span><span class="chip">rot</span></div>
<div id="fileNotice" class="notice"><b>This page was opened as a file.</b> Browsers block module, WASM, and shader fetches from <code>file://</code>. Run <code>python -m http.server 8767 --directory &lt;product-directory&gt;</code>, then open <code>http://127.0.0.1:8767/web/</code>.</div>
<section class="grid">
  <div class="card"><b>Forward surface</b><span id="forwardSummary">Loading WebGPU deployments…</span></div>
  <div class="card"><b>Backward surface</b><span id="reverseSummary">Loading compiled WASM VJPs…</span></div>
  <div class="card"><b>Product identity</b><span id="productIdentity">Loading deterministic matrix…</span></div>
</section>
<h2>Packaged method matrix</h2>
<table><thead><tr><th>Method</th><th>Forward</th><th>Backward</th><th>Specialization</th></tr></thead><tbody id="methods"><tr><td colspan="4">Loading…</td></tr></tbody></table>
<h2>Browser benchmark</h2>
<p class="muted">Runs the same installed object API a site imports. GEMM reports both the compiler-retained source shader and selected optimized shader.</p>
<div class="toolbar"><label>iterations <input id="iterations" type="number" min="1" max="100" value="5"></label><button id="forward" disabled>Run all forwards</button><button id="backward" class="secondary" disabled>Run all compiled backwards</button></div>
<table><thead><tr><th>Operation</th><th>Shape</th><th>Realization</th><th>Mean ms</th><th>Checksum</th></tr></thead><tbody id="results"><tr><td colspan="5">No runs yet.</td></tr></tbody></table>
<h2>Site-wide installation</h2>
<pre><code>&lt;script src="./install-turing-math.js" data-turing-math-base="./"&gt;&lt;/script&gt;
&lt;script type="module"&gt;
  const math = await window.turingMathReady;
  const output = await math.blas.gemm(a, b, {m, n, k});
  const gradients = await math.blas.vjp("gemm", upstream, {a, b, c, alpha, beta});
&lt;/script&gt;</code></pre>
<pre id="status">Initializing library…</pre>
<script src="./install-turing-math.js" data-turing-math-base="./"></script>
<script type="module">
const byId=id=>document.getElementById(id),status=byId("status"),body=byId("methods"),results=byId("results");
if(location.protocol==="file:")byId("fileNotice").style.display="block";
const elements=value=>value?.length===undefined?[value]:value;
const checksum=value=>Array.isArray(value)?value.reduce((sum,item)=>sum+checksum(item),0):Array.from(elements(value),Number).reduce((a,b)=>a+b,0);
const shapeText=shape=>Object.entries(shape).map(([key,value])=>`${key}=${value}`).join(" × ");
const values=(length,phase=0)=>Float32Array.from({length},(_,i)=>((i+phase)%17-8)/9);
const reverseValues=(shape,phase=0)=>{const length=shape.length?shape.reduce((a,b)=>a*Number(b),1):1,data=Float64Array.from({length},(_,i)=>((i+phase)%11-5)/7);return shape.length?data:data[0];};
let library;
try{
  library=await globalThis.turingMathReady;
  const blas=library.blas,reverse=library.blasReverse,objectMethods=reverse.manifest.methods;
  byId("forwardSummary").innerHTML=`<span class="ok">${blas.deployedMethods.length} WebGPU methods</span><br>${blas.pipelines.size} pipelines materialized lazily`;
  byId("reverseSummary").innerHTML=`<span class="ok">${objectMethods.length} compiled VJPs</span><br>standalone WebAssembly`;
  byId("productIdentity").textContent=library.matrix.products.blas.product_id;
  body.innerHTML=objectMethods.map(method=>{const records=blas.matrix.webgpu_prebakes[method.name]??[],shape=records[0]?.problem_shape??{};return `<tr><td><b>${method.name}</b></td><td class="ok">WebGPU shader</td><td class="ok">compiled WASM VJP</td><td>${shapeText(shape)||"parametric"}</td></tr>`;}).join("");
  byId("forward").disabled=false;byId("backward").disabled=false;
  status.textContent=`Installed window.turingMath, window.tensorMath, and window.turingBLAS.\nLibrary product: ${JSON.stringify(library.libraries)}\nMethods: ${blas.methods.join(", ")}`;
}catch(error){status.className="error";status.textContent=`Library initialization failed: ${error.message}\n\nIf this is a file:// URL, serve the product over HTTP first.`;}

async function forwardCase(method,variant="fast"){
  const blas=library.blas,record=blas.matrix.webgpu_prebakes[method][0],shape=record.problem_shape;
  if(method==="scal")return [await blas.scal(values(shape.n),1.25),shape];
  if(method==="axpy")return [await blas.axpy(values(shape.n),values(shape.n,3),.75),shape];
  if(method==="dot")return [await blas.dot(values(shape.n),values(shape.n,3)),shape];
  if(method==="gemv")return [await blas.gemv(values(shape.m*shape.n),values(shape.n,3),{...shape,alpha:.75,beta:.25,y:values(shape.m,5)}),shape];
  if(method==="rot")return [await blas.rot(values(shape.n),values(shape.n,3),.8,.6),shape];
  if(method==="gemm")return [await blas.gemm(values(shape.m*shape.k),values(shape.k*shape.n,3),{...shape,alpha:.75,beta:.25,c:values(shape.m*shape.n,5),variant}),shape];
  throw new Error(`no demo fixture for ${method}`);
}
async function timedForward(method,variant,iterations){
  await forwardCase(method,variant);let output,shape,start=performance.now();
  for(let i=0;i<iterations;i++)[output,shape]=await forwardCase(method,variant);
  return {operation:method,shape:shapeText(shape),realization:method==="gemm"?(variant==="source"?"source SSA shader":"selected optimized shader"):"selected WebGPU shader",ms:(performance.now()-start)/iterations,sum:checksum(output)};
}
async function timedReverse(method,iterations){
  const reverse=library.blasReverse,semantic=reverse.manifest.methods.find(item=>item.name===method),artifact=reverse.manifest.artifacts[method].browser_parametric_reverse,index=new Map(artifact.buffer_order.map((id,i)=>[Number(id),i])),bindings={};
  Object.entries(semantic.reverse_input_value_ids).forEach(([name,id],slot)=>{bindings[name]=reverseValues(artifact.buffer_shapes[index.get(Number(id))],slot);});
  const seeds=semantic.reverse_output_value_ids.map((id,slot)=>reverseValues(artifact.buffer_shapes[index.get(Number(semantic.reverse_seed_value_ids[String(id)]))],slot+2)),upstream=seeds.length===1?seeds[0]:seeds;
  let gradients;const start=performance.now();for(let i=0;i<iterations;i++)gradients=await library.blas.vjp(method,upstream,bindings);
  const shapes=Object.entries(semantic.reverse_input_value_ids).map(([name,id])=>`${name}:${artifact.buffer_shapes[index.get(Number(id))].join("×")||"scalar"}`).join(" ");
  return {operation:`${method}.vjp`,shape:shapes,realization:"compiled WebAssembly reverse",ms:(performance.now()-start)/iterations,sum:checksum(Object.values(gradients))};
}
const render=rows=>{results.innerHTML=rows.map(row=>`<tr><td>${row.operation}</td><td>${row.shape}</td><td>${row.realization}</td><td>${row.ms.toFixed(3)}</td><td>${row.sum.toFixed(5)}</td></tr>`).join("");};
async function run(button,job){button.disabled=true;status.textContent="Running…";try{const rows=await job();render(rows);status.textContent=`Completed ${rows.length} library realizations.`;}catch(error){status.className="error";status.textContent=error.stack??error.message;}finally{button.disabled=false;}}
byId("forward").onclick=()=>run(byId("forward"),async()=>{const iterations=Math.max(1,Number(byId("iterations").value)||1),rows=[];for(const method of library.blas.deployedMethods){if(method==="gemm")rows.push(await timedForward(method,"source",iterations));rows.push(await timedForward(method,"fast",iterations));}return rows;});
byId("backward").onclick=()=>run(byId("backward"),async()=>{const iterations=Math.max(1,Number(byId("iterations").value)||1),rows=[];for(const method of library.blasReverse.methods)rows.push(await timedReverse(method,iterations));return rows;});
</script>
</body></html>'''


def _readme(product_id: str, blas: BLASServerProduct) -> str:
    methods = ", ".join(blas.manifest["methods"])
    trigonometry_methods = ", ".join(
        method.name
        for method in TURING_MATHEMATICAL_LIBRARY.library(
            "trigonometry"
        ).methods
    )
    return f"""# Turing mathematical library

Product identity: `{product_id}`.

This is the outer mathematical-library product. It contains synchronized BLAS
and trigonometry subunits. The semantic BLAS catalog contains {methods}. The
trigonometry object is ingested from the existing AbstractTensor surface and
contains {trigonometry_methods}. Deployment coverage is recorded per method
rather than inferred from which files happen to exist.

## Python

```python
from python import load
math = load()
result = math.blas.gemm(a, b)
gradients = math.blas.vjp("gemm", upstream, a=a, b=b)
wave = math.trigonometry.sin(x)
wave_gradient = math.trigonometry.vjp("sin", upstream, value=x)
numpy_math = math.install(MyNumericalHost)  # standalone NumPy is the default
native_math = math.install(AnotherHost, implementation="native")
math.close()
```

Native installation is active deployment, not a side namespace: each standard
object replaces the host instance operators it declares. For example, the
trigonometry pack replaces `AbstractTensor.sin`, selects a row from its own
deployment matrix, and invokes that baked artifact. The very same pack retains
the exact authored definition through
`math.trigonometry.authored_source("sin")`; `uninstall()` (also used by
`AbstractTensor.use_semantic_mathematical_library()`) restores the displaced
operator.

`math.numpy` is a standalone NumPy class whose numerical bodies were manifested
from the canonical AbstractTensor graphs by the compiler. `math.blas` is the
separate native-DLL realization. In a Turing checkout the NumPy class can
bootstrap `AbstractTensor` while preserving the default graph-building
implementation:

```python
from python import load_numpy

numpy_math = load_numpy()  # this file needs only NumPy at runtime

from src.common.tensors.abstraction import AbstractTensor

math.install(AbstractTensor)
result = AbstractTensor.blas.gemm(a, b)
AbstractTensor.use_semantic_mathematical_library()
```

## Native

Include `native/turing_mathematical_library.h` for the packed BLAS ABI. The
trigonometry standard object publishes one native forward DLL and one fully
compiled native graph-reverse DLL per existing method under
`objects/trigonometry`; all binaries and target information are listed in
`manifest.json`.

## Browser

Serve this product directory over HTTP and open `web/index.html`. The outer
JavaScript object exposes `library.blas`, backed by the matrix-bearing outer
WASM coordinator and the BLAS subunit's own verified WASM/shader assemblage.
For embedding, copy `web/embed-template.html` and adjust the two relative paths
on its installer script. The script immediately publishes `turingMathReady`;
once resolved, `turingMath`, `tensorMath`, and `turingBLAS` are page-wide.
Every BLAS method also exposes its fully compiled graph reverse through
`await library.blas.vjp(method, upstream, bindings)`. The browser calls the
packaged WebAssembly reverse directly; it does not interpret graph metadata.
"""


def build_mathematical_library_product(
    bank: Any,
    gemm_shapes: Iterable[int | Iterable[int]],
    directory: str | Path,
    *,
    contract: str | None = "fast",
    cores: int | None = None,
    candidate_sizes: tuple[int, ...] = (16, 32, 64, 128, 256),
) -> MathematicalLibraryProduct:
    """Build the outer shell and its synchronized mathematical subunits."""

    root = Path(directory).resolve()
    marker = _prepare_root(root)
    gemm_shapes = tuple(gemm_shapes)
    blas = build_blas_server(
        bank, gemm_shapes, root / "libraries" / "blas",
        contract=contract, cores=cores, candidate_sizes=candidate_sizes,
    )
    normalized_gemm_shapes = tuple(
        (int(shape), int(shape), int(shape))
        if isinstance(shape, int) else tuple(map(int, shape))
        for shape in gemm_shapes
    )
    blas_object = cook_standard_object(
        blas_standard_object(specializations={
            "gemm": tuple(
                {"m": m, "n": n, "k": k}
                for m, n, k in normalized_gemm_shapes
            ),
        }),
        directory=root / "objects" / "blas",
        contract=contract,
    )
    trigonometry_object = cook_standard_object(
        trigonometry_standard_object(),
        directory=root / "objects" / "trigonometry",
        contract=contract,
        reverse_backends=("native",),
    )
    numpy_source, numpy_receipt = emit_numpy_mathematical_library()
    browser_installer = _browser_installer()
    browser_template = _browser_template()
    matrix = {
        "schema": MATRIX_SCHEMA,
        "catalog": TURING_MATHEMATICAL_LIBRARY.to_mapping(include_source=True),
        "products": {
            "blas": {
                "path": "libraries/blas",
                "product_id": blas.manifest["product_id"],
                "matrix_sha256": blas.manifest["server_matrix_sha256"],
                "coverage": _coverage(blas),
                "standard_object": {
                    "path": "objects/blas",
                    "product_id": blas_object.manifest["product_id"],
                    "parametric_forward": "required",
                    "compiled_graph_reverse": "required",
                },
            },
            "trigonometry": {
                "path": "objects/trigonometry",
                "product_id": trigonometry_object.manifest["product_id"],
                "coverage": [
                    {
                        "method": method["name"],
                        "identity": f"trigonometry.{method['name']}",
                        "realizations": {
                            "native_forward": "packaged",
                            "native_reverse": "packaged",
                            "wasm_reverse": "not-selected",
                            "webgpu": "not-yet-packaged",
                        },
                    }
                    for method in trigonometry_object.manifest["methods"]
                ],
                "standard_object": {
                    "path": "objects/trigonometry",
                    "product_id": trigonometry_object.manifest["product_id"],
                    "parametric_forward": "required",
                    "compiled_graph_reverse": "required",
                },
            },
        },
        "python_realizations": {
            "numpy": numpy_receipt,
            "native": {
                "provider": "libraries/blas/python/turing_blas_server.py",
                "matrix_sha256": blas.manifest["server_matrix_sha256"],
            },
        },
        "browser_installation": {
            "installer": "web/install-turing-math.js",
            "template": "web/embed-template.html",
            "source_sha256": _sha(browser_installer.encode("utf-8")),
            "globals": ["turingMathReady", "turingMath", "tensorMath", "turingBLAS"],
        },
    }
    matrix_bytes = _canonical(matrix)
    matrix_sha = _sha(matrix_bytes)
    matrix_path = root / "mathematical-library-matrix.json"
    matrix_path.write_bytes(matrix_bytes)

    python_root = root / "python"
    native_root = root / "native"
    web_root = root / "web"
    for path in (python_root, native_root, web_root):
        path.mkdir(parents=True, exist_ok=True)
    python_loader = python_root / "turing_mathematical_library.py"
    python_loader.write_text(_python_loader(), encoding="utf-8", newline="\n")
    numpy_loader = python_root / "turing_numpy_mathematical_library.py"
    numpy_loader.write_text(numpy_source, encoding="utf-8", newline="\n")
    (python_root / "__init__.py").write_text(
        "from .turing_mathematical_library import TuringMathematicalLibrary, load\n"
        "from .turing_numpy_mathematical_library import (\n"
        "    NumPyBLAS, NumPyMathematicalLibrary, load as load_numpy,\n"
        ")\n",
        encoding="utf-8", newline="\n",
    )
    native_header = native_root / "turing_mathematical_library.h"
    native_header.write_text(
        """#ifndef TURING_MATHEMATICAL_LIBRARY_H
#define TURING_MATHEMATICAL_LIBRARY_H
#include "../libraries/blas/native/turing_blas_server.h"
#endif
""",
        encoding="utf-8", newline="\n",
    )
    wasm_path = web_root / "mathematical-library.wasm"
    wasm_path.write_bytes(build_module(
        function_name="turing_mathematical_library_matrix",
        parameter_types=[], body=CodeBuilder("i32", 0),
        memory_pages=max(1, (len(matrix_bytes) + 65535) // 65536),
        data=matrix_bytes,
    ))
    javascript_path = web_root / "mathematical-library.js"
    javascript_path.write_text(
        _javascript(len(matrix_bytes), matrix_sha),
        encoding="utf-8", newline="\n",
    )
    installer_path = web_root / "install-turing-math.js"
    installer_path.write_text(browser_installer, encoding="utf-8", newline="\n")
    template_path = web_root / "embed-template.html"
    template_path.write_text(browser_template, encoding="utf-8", newline="\n")
    demo_path = web_root / "index.html"
    demo_path.write_text(_demo(), encoding="utf-8", newline="\n")
    readme_path = root / "README.md"
    readme_path.write_text(
        _readme(matrix_sha, blas), encoding="utf-8", newline="\n",
    )
    marker.unlink()

    artifacts = {}
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda item: item.as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        artifacts[relative] = {
            "sha256": _sha(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema": PRODUCT_SCHEMA,
        "product_id": matrix_sha,
        "matrix": matrix_path.relative_to(root).as_posix(),
        "matrix_sha256": matrix_sha,
        "libraries": {
            "blas": {
                "path": "libraries/blas",
                "product_id": blas.manifest["product_id"],
                "methods": blas.manifest["methods"],
                "deployed_roles": blas.manifest["deployed_roles"],
                "standard_object": {
                    "manifest": "objects/blas/manifest.json",
                    "product_id": blas_object.manifest["product_id"],
                    "methods": [
                        method["name"] for method in blas_object.manifest["methods"]
                    ],
                },
            },
            "trigonometry": {
                "path": "objects/trigonometry",
                "product_id": trigonometry_object.manifest["product_id"],
                "methods": [
                    method["name"]
                    for method in trigonometry_object.manifest["methods"]
                ],
                "standard_object": {
                    "manifest": "objects/trigonometry/manifest.json",
                    "product_id": trigonometry_object.manifest["product_id"],
                    "methods": [
                        method["name"]
                        for method in trigonometry_object.manifest["methods"]
                    ],
                },
            },
        },
        "surfaces": {
            "native": {
                "header": native_header.relative_to(root).as_posix(),
                "libraries": {
                    "blas": {
                        **blas.manifest["surfaces"]["native"],
                        "library": (
                            "libraries/blas/"
                            + blas.manifest["surfaces"]["native"]["library"]
                        ),
                        "header": (
                            "libraries/blas/"
                            + blas.manifest["surfaces"]["native"]["header"]
                        ),
                    },
                    "trigonometry": {
                        "object": "objects/trigonometry/manifest.json",
                        "forward_artifacts": "per-method native DLL",
                        "reverse_artifacts": "per-method native DLL",
                        "python_runtime_dependency": False,
                    },
                },
            },
            "python": {
                "loader": python_loader.relative_to(root).as_posix(),
                "entry": "load",
                "installation_entry": "TuringMathematicalLibrary.install",
                "default_installation": "numpy",
                "numpy": {
                    "module": numpy_loader.relative_to(root).as_posix(),
                    "entry": "load",
                    "class": "NumPyMathematicalLibrary",
                    "compiler": numpy_receipt["compiler"],
                    "source_sha256": numpy_receipt["module_source_sha256"],
                },
                "native": {
                    "selector": "native",
                    "provider": "TuringMathematicalLibrary",
                },
            },
            "web": {
                "wasm": wasm_path.relative_to(root).as_posix(),
                "javascript": javascript_path.relative_to(root).as_posix(),
                "installer": installer_path.relative_to(root).as_posix(),
                "template": template_path.relative_to(root).as_posix(),
                "ready": "globalThis.turingMathReady",
                "globals": ["turingMath", "tensorMath", "turingBLAS"],
                "demo": demo_path.relative_to(root).as_posix(),
            },
        },
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n",
    )
    return MathematicalLibraryProduct(
        root, manifest_path, matrix_path, python_loader, numpy_loader, javascript_path,
        wasm_path, demo_path, blas, blas_object, trigonometry_object, manifest,
    )


__all__ = [
    "MATRIX_SCHEMA",
    "MathematicalLibraryProduct",
    "MathematicalLibraryProductError",
    "PRODUCT_SCHEMA",
    "build_mathematical_library_product",
]
