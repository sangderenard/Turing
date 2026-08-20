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
            return hook(provider)
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
    return new TuringMathematicalLibrary(matrix,blas);
  }}
  constructor(matrix,blas){{this.matrix=matrix;this.blas=blas;}}
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
    return '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Turing mathematical library</title><style>body{background:#081019;color:#eaf2f8;font:15px ui-monospace,monospace;max-width:900px;margin:40px auto;padding:20px}button{padding:10px;background:#65d6e8;border:0;border-radius:6px;font-weight:bold}pre{background:#0d1924;padding:18px;border-radius:9px;white-space:pre-wrap}</style></head><body><h1>Turing mathematical library</h1><p>One semantic catalog, with BLAS packaged as a synchronized library subunit.</p><button id="run">Run first prebaked GEMM</button><pre id="out">loading…</pre><script src="./install-turing-math.js" data-turing-math-base="./"></script><script type="module">const out=document.querySelector('#out'),library=await globalThis.turingMathReady;out.textContent=JSON.stringify({libraries:library.libraries,blas_methods:library.blas.methods,deployed:library.blas.deployedMethods,shapes:library.blas.shapes},null,2);document.querySelector('#run').onclick=async()=>{const [m,n,k]=library.blas.shapes[0],a=new Float32Array(m*k).fill(.1),b=new Float32Array(k*n).fill(.2),t=performance.now(),c=await library.blas.gemm(a,b,{m,n,k}),ms=performance.now()-t;out.textContent=JSON.stringify({method:'blas.gemm',shape:[m,n,k],elapsed_ms:ms,first_value:c[0]},null,2);};</script></body></html>'''


def _readme(product_id: str, blas: BLASServerProduct) -> str:
    methods = ", ".join(blas.manifest["methods"])
    return f"""# Turing mathematical library

Product identity: `{product_id}`.

This is the outer mathematical-library product. `libraries/blas` is its first
subunit. The semantic BLAS catalog contains {methods}; its deployment coverage
is recorded per method rather than inferred from which files happen to exist.

## Python

```python
from python import load
math = load()
result = math.blas.gemm(a, b)
numpy_math = math.install(MyNumericalHost)  # standalone NumPy is the default
native_math = math.install(AnotherHost, implementation="native")
math.close()
```

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

Include `native/turing_mathematical_library.h`. It exposes the packaged BLAS
subunit ABI; library binaries and target information are listed in
`manifest.json`.

## Browser

Serve this product directory over HTTP and open `web/index.html`. The outer
JavaScript object exposes `library.blas`, backed by the matrix-bearing outer
WASM coordinator and the BLAS subunit's own verified WASM/shader assemblage.
For embedding, copy `web/embed-template.html` and adjust the two relative paths
on its installer script. The script immediately publishes `turingMathReady`;
once resolved, `turingMath`, `tensorMath`, and `turingBLAS` are page-wide.
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
    """Build the outer shell and its currently packaged BLAS subunit."""

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
                    "provider": "TuringMathematicalLibrary.blas",
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
        wasm_path, demo_path, blas, blas_object, manifest,
    )


__all__ = [
    "MATRIX_SCHEMA",
    "MathematicalLibraryProduct",
    "MathematicalLibraryProductError",
    "PRODUCT_SCHEMA",
    "build_mathematical_library_product",
]
