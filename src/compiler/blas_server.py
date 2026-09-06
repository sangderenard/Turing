"""Package prebaked BLAS deployment as native, Python and browser products.

Version one serves the fast edge the compiler can currently make durable:
specialized GEMM.  One canonical server matrix records every logical shape,
native packing/launch permutation and individual WebGPU shader plan.  The
native library embeds those bytes; the browser reads the same bytes from a
WASM data segment; the Python loader verifies them before calling the DLL.
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping

from .native_gemm_product import compile_native_gemm_product
from .ssa_webgpu_backend import emit_blas_module, emit_gemm_module
from .wasm_binary import CodeBuilder, build_module


SERVER_SCHEMA = "turing.blas-server.v2"
MATRIX_SCHEMA = "turing.blas-server-matrix.v2"


class BLASServerError(RuntimeError):
    """A requested durable product cannot be built faithfully."""


@dataclass(frozen=True, slots=True)
class BLASServerProduct:
    directory: Path
    manifest_path: Path
    matrix_path: Path
    native_library: Path
    python_loader: Path
    wasm_path: Path
    javascript_path: Path
    demo_path: Path
    manifest: Mapping[str, Any]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _shapes(values: Iterable[int | Iterable[int]]) -> tuple[tuple[int, int, int], ...]:
    normalized = []
    for value in values:
        if isinstance(value, int):
            shape = (int(value),) * 3
        else:
            shape = tuple(map(int, value))
        if len(shape) != 3 or min(shape) <= 0:
            raise ValueError(f"BLAS server shape must be positive M,N,K, got {shape!r}")
        normalized.append(shape)
    result = tuple(sorted(set(normalized)))
    if not result:
        raise ValueError("BLAS server requires at least one shape")
    return result


def _prepare_product_root(root: Path) -> Path:
    """Make a generated-product directory repeatable without touching user data."""

    marker = root / ".turing-blas-server-build"
    if root.exists() and any(root.iterdir()):
        owned = marker.is_file()
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            try:
                existing_schema = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ).get("schema")
                owned = existing_schema in {SERVER_SCHEMA, "turing.blas-server.v1"}
            except (OSError, ValueError):
                owned = False
        if not owned:
            raise BLASServerError(
                f"refusing to replace non-BLAS-server directory: {root}"
            )
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(SERVER_SCHEMA, encoding="ascii")
    return marker


def _c_string(data: bytes) -> str:
    chunks = []
    for start in range(0, len(data), 72):
        chunk = data[start:start + 72]
        chunks.append('    "' + "".join(
            chr(value) if 32 <= value < 127 and value not in {34, 92}
            else "\\\"" if value == 34
            else "\\\\" if value == 92
            else f"\\{value:03o}"
            for value in chunk
        ) + '"')
    return "\n".join(chunks)


def _generic_native_c(variants: Mapping[str, Any]) -> str:
    signatures = {
        "scal": (
            "int turing_blas_scal(const double *x, double *y, double alpha, int n)",
            "return 0;",
        ),
        "axpy": (
            "int turing_blas_axpy(const double *x, double *y, double alpha, int n)",
            "return 0;",
        ),
        "dot": (
            "double turing_blas_dot(const double *x, const double *y, int n)",
            "return result;",
        ),
        "gemv": (
            "int turing_blas_gemv(const double *A, const double *x, double *y, double alpha, double beta, int m, int n)",
            "return 0;",
        ),
        "rot": (
            "int turing_blas_rot(double *x, double *y, double c, double s, int n)",
            "return 0;",
        ),
    }
    blocks = []
    for name, record in variants.items():
        symbol = record["compiler_entry"]
        signature, result = signatures[name]
        slots = []
        for binding in record["buffer_bindings"]:
            bound = str(binding["name"])
            slots.append(
                "&result" if bound == "return"
                else f"(void *){bound}" if binding["kind"] == "buffer"
                else f"(void *)&{bound}"
            )
        local = "    double result = 0.0;\n" if name == "dot" else ""
        blocks.append(f"""void {symbol}(void **, int32_t *);
TURING_EXPORT {signature} {{
{local}    void *buffers[] = {{{', '.join(slots)}}};
    {symbol}(buffers, NULL);
    {result}
}}""")
    return "\n\n".join(blocks)


def _server_c(
    matrix: bytes,
    digest: str,
    variants: list[Mapping[str, Any]],
    generic_variants: Mapping[str, Any],
) -> str:
    declarations = []
    branches = []
    for variant in variants:
        m, n, k = variant["shape"]
        function = variant["native"]["function"]
        declarations.append(
            f"int {function}(const double*, const double*, double*, double, double);"
        )
        declarations.append(f"void {function}_shutdown(void);")
        branches.append(
            f"    if (m == {m} && n == {n} && k == {k})\n"
            f"        return {function}(a, b, c, alpha, beta);"
        )
    shutdown = f"    {variants[0]['native']['function']}_shutdown();"
    return f"""#include <stddef.h>
#include <stdint.h>

{chr(10).join(declarations)}

static const unsigned char turing_blas_matrix[] =
{_c_string(matrix)};
static const char turing_blas_matrix_digest[] = "{digest}";

#ifdef _WIN32
#define TURING_EXPORT __declspec(dllexport)
#else
#define TURING_EXPORT __attribute__((visibility("default")))
#endif

TURING_EXPORT const unsigned char *turing_blas_server_matrix(void) {{
    return turing_blas_matrix;
}}
TURING_EXPORT uint64_t turing_blas_server_matrix_size(void) {{
    return (uint64_t)(sizeof(turing_blas_matrix) - 1u);
}}
TURING_EXPORT const char *turing_blas_server_matrix_sha256(void) {{
    return turing_blas_matrix_digest;
}}
TURING_EXPORT int turing_blas_server_variant_count(void) {{
    return {len(variants)};
}}
TURING_EXPORT int turing_blas_server_gemm(
    int m, int n, int k,
    const double *a, const double *b, double *c,
    double alpha, double beta) {{
{chr(10).join(branches)}
    return -404;
}}
TURING_EXPORT void turing_blas_server_shutdown(void) {{
{shutdown}
}}

{_generic_native_c(generic_variants)}
"""


def _python_loader() -> str:
    return '''"""Portable Python loader for one generated Turing BLAS server."""
from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path
import numpy as np


class BLASServer:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        native = self.root / self.manifest["surfaces"]["native"]["library"]
        self.library = ctypes.CDLL(str(native))
        self.library.turing_blas_server_matrix_size.restype = ctypes.c_uint64
        self.library.turing_blas_server_matrix.restype = ctypes.POINTER(ctypes.c_ubyte)
        self.library.turing_blas_server_matrix_sha256.restype = ctypes.c_char_p
        size = int(self.library.turing_blas_server_matrix_size())
        embedded = ctypes.string_at(self.library.turing_blas_server_matrix(), size)
        digest = hashlib.sha256(embedded).hexdigest()
        stated = self.library.turing_blas_server_matrix_sha256().decode("ascii")
        if digest != stated or digest != self.manifest["server_matrix_sha256"]:
            raise RuntimeError("BLAS server matrix identity mismatch")
        self.matrix = json.loads(embedded)
        pointer = ctypes.POINTER(ctypes.c_double)
        self._gemm = self.library.turing_blas_server_gemm
        self._gemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               pointer, pointer, pointer,
                               ctypes.c_double, ctypes.c_double]
        self._gemm.restype = ctypes.c_int
        p = ctypes.POINTER(ctypes.c_double)
        self._scal = self.library.turing_blas_scal
        self._scal.argtypes = [p, p, ctypes.c_double, ctypes.c_int]
        self._scal.restype = ctypes.c_int
        self._axpy = self.library.turing_blas_axpy
        self._axpy.argtypes = [p, p, ctypes.c_double, ctypes.c_int]
        self._axpy.restype = ctypes.c_int
        self._dot = self.library.turing_blas_dot
        self._dot.argtypes = [p, p, ctypes.c_int]
        self._dot.restype = ctypes.c_double
        self._gemv = self.library.turing_blas_gemv
        self._gemv.argtypes = [p, p, p, ctypes.c_double, ctypes.c_double,
                               ctypes.c_int, ctypes.c_int]
        self._gemv.restype = ctypes.c_int
        self._rot = self.library.turing_blas_rot
        self._rot.argtypes = [p, p, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        self._rot.restype = ctypes.c_int
        self.library.turing_blas_server_shutdown.restype = None

    @property
    def shapes(self):
        return tuple(tuple(item["shape"]) for item in self.matrix["variants"])

    @property
    def methods(self):
        return tuple(item["name"] for item in self.matrix["library"]["methods"])

    @property
    def deployed_methods(self):
        return tuple(self.matrix["surface_methods"]["python"])

    def supports(self, method):
        return str(method) in self.deployed_methods

    @staticmethod
    def _vector(value):
        value = np.ascontiguousarray(value, dtype=np.float64)
        if value.ndim != 1:
            raise ValueError("BLAS vector input must have rank one")
        return value

    def scal(self, x, alpha, *, y=None):
        x = self._vector(x)
        y = np.zeros_like(x) if y is None else self._vector(y).copy()
        if y.shape != x.shape:
            raise ValueError("scal x and y lengths differ")
        pointer = ctypes.POINTER(ctypes.c_double)
        self._scal(x.ctypes.data_as(pointer), y.ctypes.data_as(pointer),
                   float(alpha), x.size)
        return y

    def axpy(self, x, y, alpha):
        x, y = self._vector(x), self._vector(y).copy()
        if y.shape != x.shape:
            raise ValueError("axpy x and y lengths differ")
        pointer = ctypes.POINTER(ctypes.c_double)
        self._axpy(x.ctypes.data_as(pointer), y.ctypes.data_as(pointer),
                   float(alpha), x.size)
        return y

    def dot(self, x, y):
        x, y = self._vector(x), self._vector(y)
        if y.shape != x.shape:
            raise ValueError("dot x and y lengths differ")
        pointer = ctypes.POINTER(ctypes.c_double)
        return float(self._dot(x.ctypes.data_as(pointer),
                               y.ctypes.data_as(pointer), x.size))

    def gemv(self, a, x, *, y=None, alpha=1.0, beta=0.0):
        a = np.ascontiguousarray(a, dtype=np.float64)
        x = self._vector(x)
        if a.ndim != 2 or a.shape[1] != x.size:
            raise ValueError("gemv expects matrix A and compatible vector x")
        y = np.zeros(a.shape[0], dtype=np.float64) if y is None else self._vector(y).copy()
        if y.size != a.shape[0]:
            raise ValueError("gemv y length does not match A rows")
        pointer = ctypes.POINTER(ctypes.c_double)
        self._gemv(a.ctypes.data_as(pointer), x.ctypes.data_as(pointer),
                   y.ctypes.data_as(pointer), float(alpha), float(beta),
                   a.shape[0], a.shape[1])
        return y

    def rot(self, x, y, c, s):
        x, y = self._vector(x).copy(), self._vector(y).copy()
        if y.shape != x.shape:
            raise ValueError("rot x and y lengths differ")
        pointer = ctypes.POINTER(ctypes.c_double)
        self._rot(x.ctypes.data_as(pointer), y.ctypes.data_as(pointer),
                  float(c), float(s), x.size)
        return x, y

    def gemm(self, a, b, *, c=None, alpha=1.0, beta=0.0):
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("gemm expects compatible rank-two A and B")
        m, k = a.shape; _k, n = b.shape
        if c is None:
            c = np.zeros((m, n), dtype=np.float64)
        else:
            c = np.ascontiguousarray(c, dtype=np.float64).copy()
        if c.shape != (m, n):
            raise ValueError("C shape does not match A@B")
        pointer = ctypes.POINTER(ctypes.c_double)
        status = self._gemm(
            m, n, k,
            a.ctypes.data_as(pointer), b.ctypes.data_as(pointer),
            c.ctypes.data_as(pointer), float(alpha), float(beta),
        )
        if status == -404:
            raise KeyError(f"shape {(m, n, k)} is not prebaked; available: {self.shapes}")
        if status not in {0, 1}:
            raise RuntimeError(f"native BLAS server returned {status}")
        return c

    def close(self):
        self.library.turing_blas_server_shutdown()


def load(root=None):
    return BLASServer(root)
'''


def _native_header() -> str:
    return """#ifndef TURING_BLAS_SERVER_H
#define TURING_BLAS_SERVER_H

#include <stdint.h>

#ifdef __cplusplus
extern \"C\" {
#endif

const unsigned char *turing_blas_server_matrix(void);
uint64_t turing_blas_server_matrix_size(void);
const char *turing_blas_server_matrix_sha256(void);
int turing_blas_server_variant_count(void);
int turing_blas_server_gemm(
    int m, int n, int k,
    const double *a, const double *b, double *c,
    double alpha, double beta);
int turing_blas_scal(const double *x, double *y, double alpha, int n);
int turing_blas_axpy(const double *x, double *y, double alpha, int n);
double turing_blas_dot(const double *x, const double *y, int n);
int turing_blas_gemv(
    const double *A, const double *x, double *y,
    double alpha, double beta, int m, int n);
int turing_blas_rot(double *x, double *y, double c, double s, int n);
void turing_blas_server_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif
"""


def _readme(shapes: tuple[tuple[int, int, int], ...], digest: str) -> str:
    shape_text = ", ".join("x".join(map(str, shape)) for shape in shapes)
    return f"""# Turing BLAS server

This directory is one immutable, shape-specialized BLAS product. Its product
identity is the SHA-256 of `server-matrix.json`: `{digest}`.

Prebaked GEMM shapes: {shape_text}. GEMM means `C = alpha * A @ B + beta * C`.

## Python

Add this directory to `sys.path`, then:

```python
from python import load
server = load()
result = server.gemm(a, b, c=c, alpha=1.0, beta=0.0)
server.close()
```

The loader verifies that the matrix embedded in the native library matches the
product manifest before it accepts a call. Python requires NumPy; the DLL does
not require Python.

## Native

Include `native/turing_blas_server.h` and link or dynamically load the library
named in `manifest.json`. Inputs are contiguous row-major `double` arrays. The
generic entry returns `-404` for a shape that was not prebaked. This library is
compiled for the system and CPU recorded in `manifest.json`; rebuild the
product when distributing to another native target.

## Browser

Serve the `web` directory over HTTP and open `web/index.html`, or import
`web/blas-server.js`. The JavaScript coordinator instantiates WebAssembly,
verifies its embedded matrix, selects an individual WGSL prebake, and dispatches
it through WebGPU. Browser inputs are contiguous row-major `Float32Array`s.
Opening the files directly with a `file:` URL is not supported by browser fetch.
"""


def _javascript(matrix_bytes: int, digest: str) -> str:
    return f'''const MATRIX_BYTES={matrix_bytes};
const MATRIX_SHA256="{digest}";
const hex=bytes=>[...new Uint8Array(bytes)].map(v=>v.toString(16).padStart(2,"0")).join("");
const storage=(device,data,usage)=>{{const b=device.createBuffer({{size:Math.max(4,data.byteLength),usage,mappedAtCreation:true}});new data.constructor(b.getMappedRange()).set(data);b.unmap();return b;}};

export class TuringBLASServer {{
  static async load(base=new URL("./",import.meta.url)) {{
    const wasm=await (await fetch(new URL("server-matrix.wasm",base))).arrayBuffer();
    const instance=(await WebAssembly.instantiate(wasm,{{}})).instance;
    const bytes=new Uint8Array(instance.exports.memory.buffer,0,MATRIX_BYTES).slice();
    const digest=hex(await crypto.subtle.digest("SHA-256",bytes));
    if(digest!==MATRIX_SHA256)throw new Error("BLAS server WASM matrix identity mismatch");
    const matrix=JSON.parse(new TextDecoder().decode(bytes));
    if(!navigator.gpu)throw new Error("WebGPU is unavailable in this browser");
    const adapter=await navigator.gpu.requestAdapter({{powerPreference:"high-performance"}});
    if(!adapter)throw new Error("no WebGPU adapter was granted");
    const device=await adapter.requestDevice();
    return new TuringBLASServer(base,matrix,adapter,device);
  }}
  constructor(base,matrix,adapter,device){{this.base=base;this.matrix=matrix;this.adapter=adapter;this.device=device;this.pipelines=new Map();}}
  get shapes(){{return this.matrix.variants.map(v=>v.shape);}}
  get methods(){{return this.matrix.library.methods.map(v=>v.name);}}
  get deployedMethods(){{return [...this.matrix.surface_methods.webgpu];}}
  supports(method){{return this.matrix.surface_methods.webgpu.includes(String(method));}}
  variant(m,n,k,kind="fast"){{const item=this.matrix.variants.find(v=>v.shape[0]===m&&v.shape[1]===n&&v.shape[2]===k);if(!item)throw new Error(`shape ${{m}}x${{n}}x${{k}} is not prebaked`);return item.webgpu[kind==="source"?"source":"fast"];}}
  prebake(method,dimensions){{const records=this.matrix.webgpu_prebakes[String(method)]??[];const record=records.find(item=>Object.entries(dimensions).every(([name,value])=>item.problem_shape[name]===value));if(!record)throw new Error(`${{method}} specialization ${{JSON.stringify(dimensions)}} is not prebaked`);return record;}}
  async pipeline(record){{if(this.pipelines.has(record.source_sha256))return this.pipelines.get(record.source_sha256);const source=await(await fetch(new URL(record.path,this.base))).text();const digest=hex(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(source)));if(digest!==record.source_sha256)throw new Error(`shader identity mismatch: ${{record.path}}`);const module=this.device.createShaderModule({{code:source}});const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(e=>e.message).join("\\n"));const pipeline=this.device.createComputePipeline({{layout:"auto",compute:{{module,entryPoint:"main"}}}});this.pipelines.set(record.source_sha256,pipeline);return pipeline;}}
  async dispatch(record,resources,outputs){{const pipeline=await this.pipeline(record),device=this.device,buffers=resources.map((resource,index)=>storage(device,resource.data,(resource.uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|(outputs.includes(index)?GPUBufferUsage.COPY_SRC:0))),reads=outputs.map(index=>device.createBuffer({{size:resources[index].data.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}})),bind=device.createBindGroup({{layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({{binding,resource:{{buffer}}}}))}}),encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(...record.groups);pass.end();outputs.forEach((index,slot)=>encoder.copyBufferToBuffer(buffers[index],0,reads[slot],0,resources[index].data.byteLength));device.queue.submit([encoder.finish()]);await Promise.all(reads.map(read=>read.mapAsync(GPUMapMode.READ)));const result=reads.map(read=>new Float32Array(read.getMappedRange().slice(0)));reads.forEach(read=>read.unmap());[...buffers,...reads].forEach(value=>value.destroy());return result;}}
  async scal(x,alpha,options={{}}){{x=x instanceof Float32Array?x:new Float32Array(x);const y=options.y?(options.y instanceof Float32Array?options.y:new Float32Array(options.y)):new Float32Array(x.length);if(y.length!==x.length)throw new Error("scal x and y lengths differ");return (await this.dispatch(this.prebake("scal",{{n:x.length}}),[{{data:x}},{{data:y}},{{data:new Float32Array([alpha]),uniform:true}}],[1]))[0];}}
  async axpy(x,y,alpha){{x=x instanceof Float32Array?x:new Float32Array(x);y=y instanceof Float32Array?y:new Float32Array(y);if(y.length!==x.length)throw new Error("axpy x and y lengths differ");return (await this.dispatch(this.prebake("axpy",{{n:x.length}}),[{{data:x}},{{data:y}},{{data:new Float32Array([alpha]),uniform:true}}],[1]))[0];}}
  async dot(x,y){{x=x instanceof Float32Array?x:new Float32Array(x);y=y instanceof Float32Array?y:new Float32Array(y);if(y.length!==x.length)throw new Error("dot x and y lengths differ");return (await this.dispatch(this.prebake("dot",{{n:x.length}}),[{{data:x}},{{data:y}},{{data:new Float32Array(1)}}],[2]))[0][0];}}
  async gemv(a,x,options={{}}){{a=a instanceof Float32Array?a:new Float32Array(a);x=x instanceof Float32Array?x:new Float32Array(x);const n=options.n??x.length,m=options.m??Math.round(a.length/n),alpha=options.alpha??1,beta=options.beta??0,y=options.y?(options.y instanceof Float32Array?options.y:new Float32Array(options.y)):new Float32Array(m);if(a.length!==m*n||x.length!==n||y.length!==m)throw new Error("GEMV buffer lengths disagree with M,N");return (await this.dispatch(this.prebake("gemv",{{m,n}}),[{{data:a}},{{data:x}},{{data:y}},{{data:new Float32Array([alpha,beta]),uniform:true}}],[2]))[0];}}
  async rot(x,y,c,s){{x=x instanceof Float32Array?x:new Float32Array(x);y=y instanceof Float32Array?y:new Float32Array(y);if(y.length!==x.length)throw new Error("rot x and y lengths differ");return this.dispatch(this.prebake("rot",{{n:x.length}}),[{{data:x}},{{data:y}},{{data:new Float32Array([c,s]),uniform:true}}],[0,1]);}}
  async gemm(a,b,options={{}}){{const m=options.m??Math.round(Math.sqrt(a.length)),k=options.k??Math.round(a.length/m),n=options.n??Math.round(b.length/k),alpha=options.alpha??1,beta=options.beta??0,kind=options.variant??"fast";a=a instanceof Float32Array?a:new Float32Array(a);b=b instanceof Float32Array?b:new Float32Array(b);const c=options.c?(options.c instanceof Float32Array?options.c:new Float32Array(options.c)):new Float32Array(m*n);if(a.length!==m*k||b.length!==k*n||c.length!==m*n)throw new Error("GEMM buffer lengths disagree with M,N,K");const record=this.variant(m,n,k,kind),pipeline=await this.pipeline(record),device=this.device;const ab=storage(device,a,GPUBufferUsage.STORAGE),bb=storage(device,b,GPUBufferUsage.STORAGE),cb=storage(device,c,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC),scalars=storage(device,new Float32Array([alpha,beta]),GPUBufferUsage.UNIFORM),read=device.createBuffer({{size:c.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}});const bind=device.createBindGroup({{layout:pipeline.getBindGroupLayout(0),entries:[{{binding:0,resource:{{buffer:ab}}}},{{binding:1,resource:{{buffer:bb}}}},{{binding:2,resource:{{buffer:cb}}}},{{binding:3,resource:{{buffer:scalars}}}}]}});const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(...record.groups);pass.end();encoder.copyBufferToBuffer(cb,0,read,0,c.byteLength);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const result=new Float32Array(read.getMappedRange().slice(0));read.unmap();[ab,bb,cb,scalars,read].forEach(value=>value.destroy());return result;}}
}}

export default TuringBLASServer;
'''


_DEMO = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Turing BLAS server</title><style>body{color:#eaf2f8;background:#081019;font:15px ui-monospace,monospace;max-width:900px;margin:40px auto;padding:20px}button,select{padding:10px;margin:5px;background:#142637;color:#eaf2f8;border:1px solid #3b6078;border-radius:6px}button{background:#65d6e8;color:#061016;font-weight:bold}pre{padding:18px;background:#0d1924;border-radius:9px;white-space:pre-wrap}</style></head><body><h1>Durable Turing BLAS server</h1><p>The WASM coordinator owns the baked matrix. JavaScript verifies it and launches its individual WebGPU shader products.</p><select id="shape"></select><select id="variant"><option value="fast">compiler-tiled</option><option value="source">source-order</option></select><button id="run">Run GEMM</button><pre id="out">loading…</pre><script type="module">import {TuringBLASServer} from './blas-server.js';const out=document.querySelector('#out'),shape=document.querySelector('#shape'),server=await TuringBLASServer.load();globalThis.turingBLASServer=server;shape.innerHTML=server.shapes.map(v=>`<option value="${v.join(',')}">${v.join(' × ')}</option>`).join('');out.textContent=`ready: ${server.shapes.length} prebaked shape(s)`;document.querySelector('#run').onclick=async()=>{const [m,n,k]=shape.value.split(',').map(Number),a=Float32Array.from({length:m*k},(_,i)=>(i%29-14)/17),b=Float32Array.from({length:k*n},(_,i)=>(i%31-15)/19),variant=document.querySelector('#variant').value,t0=performance.now(),c=await server.gemm(a,b,{m,n,k,variant}),ms=performance.now()-t0,gflops=2*m*n*k/(ms/1000)/1e9;let expected=0;for(let p=0;p<k;p++)expected+=a[p]*b[p*n];out.textContent=JSON.stringify({shape:[m,n,k],variant,elapsed_ms:ms,gflops,first_value:c[0],first_value_expected:expected,first_value_error:Math.abs(c[0]-expected)},null,2);};</script></body></html>'''


def build_blas_server(
    bank: Any,
    shapes: Iterable[int | Iterable[int]],
    directory: str | Path,
    *,
    contract: str | None = "fast",
    cores: int | None = None,
    candidate_sizes: tuple[int, ...] = (16, 32, 64, 128, 256),
) -> BLASServerProduct:
    """Compile and package a shape-specialized GEMM service for every surface."""

    shapes = _shapes(shapes)
    root = Path(directory).resolve()
    build_marker = _prepare_product_root(root)
    native_root, web_root, python_root = (
        root / "native", root / "web", root / "python",
    )
    shader_root = web_root / "shaders"
    for path in (native_root, shader_root, python_root):
        path.mkdir(parents=True, exist_ok=True)

    variants: list[dict[str, Any]] = []
    products = []
    from ..common.tensors.blas import blas_role
    from ..common.tensors.mathematical_library import BLAS_LIBRARY

    for m, n, k in shapes:
        key = f"gemm-{m}-{n}-{k}"
        function = f"turing_blas_gemm_{m}_{n}_{k}"
        eligible_candidates = tuple(
            value for value in candidate_sizes if int(value) < min(m, n, k)
        ) or (max(1, min(m, n, k) // 2),)
        product = compile_native_gemm_product(
            bank, {"m": m, "n": n, "k": k},
            root / ".build" / "variants", contract=contract, cores=cores,
            candidate_sizes=eligible_candidates, function_name=function,
        )
        products.append(product)
        webgpu = {}
        for name, shader_variant in (
            ("source", "source_algorithm"),
            ("fast", "webgpu_tiled_gemm"),
        ):
            module = emit_gemm_module(m, n, k, variant=shader_variant)
            relative = Path("shaders") / key / f"{name}.wgsl"
            target = web_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(module.source, encoding="utf-8", newline="\n")
            webgpu[name] = {
                "variant": shader_variant,
                "path": relative.as_posix(),
                "source_sha256": _sha(module.source.encode("utf-8")),
                "workgroup_size": list(module.launch_plan.workgroup_size),
                "groups": list(module.launch_plan.groups),
                "io_layout": module.io_layout.to_mapping(),
                "backend_intrinsic": module.api.metadata["backend_intrinsic"],
                "backend_identities": module.api.metadata["backend_identities"],
            }
        variants.append({
            "key": key,
            "role": "blas.gemm",
            "shape": [m, n, k],
            "dtype": {"native": "float64", "webgpu": "float32"},
            "native": {
                "function": product.function_name,
                "serial_control": product.function_name + "_serial",
                "launch_matrix_sha256": product.manifest["launch_matrix_sha256"],
                "launch_matrix": product.manifest["launch_matrix"],
            },
            "webgpu": webgpu,
        })

    role = blas_role("gemm")
    from .work_contract import PRESETS

    contract_name = contract or "develop"
    try:
        contract_record = dataclasses.asdict(PRESETS[contract_name])
    except KeyError as error:
        raise BLASServerError(
            f"unknown BLAS product contract {contract_name!r}; expected one of "
            f"{tuple(PRESETS)!r}"
        ) from error
    generic_products = {}
    for method in BLAS_LIBRARY.methods:
        if method.name == "gemm":
            continue
        product = bank.get(method.name, contract=contract)
        by_id = {identifier: name for name, identifier in product.id_by_name.items()}
        parameter_kind = {item.name: item.kind for item in method.parameters}
        bindings = []
        for slot, identifier in enumerate(product.native.buffer_order):
            name = by_id.get(int(identifier))
            if name is None and int(identifier) in product.ret_ids:
                name = "return"
            if name is None:
                raise BLASServerError(
                    f"{method.identity}: native ABI slot {slot} value "
                    f"{identifier} has no semantic binding"
                )
            bindings.append({
                "slot": slot,
                "value_id": int(identifier),
                "name": name,
                "kind": "scalar" if name == "return" else parameter_kind[name],
                "dtype": product.native.buffer_dtypes[slot],
            })
        llvm_path = Path(product.native.library_path).with_suffix(".ll")
        generic_products[method.name] = {
            "variant_key": product.key,
            "compiler_entry": product.native.name,
            "source_sha256": method.source_sha256,
            "llvm_sha256": _sha(llvm_path.read_bytes()),
            "buffer_bindings": bindings,
            "llvm_path": str(llvm_path),
        }
    webgpu_prebakes: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("scal", "axpy", "dot", "gemv", "rot")
    }
    vector_sizes = sorted({axis for shape in shapes for axis in shape})
    method_shapes = {
        name: ({"n": size} for size in vector_sizes)
        for name in ("scal", "axpy", "dot", "rot")
    }
    method_shapes["gemv"] = (
        {"m": m, "n": k} for m, _n, k in shapes
    )
    for method, specializations in method_shapes.items():
        seen = set()
        for dimensions in specializations:
            key = tuple(sorted(dimensions.items()))
            if key in seen:
                continue
            seen.add(key)
            module = emit_blas_module(method, **dimensions)
            suffix = "-".join(str(value) for value in dimensions.values())
            relative = Path("shaders") / method / f"{method}-{suffix}.wgsl"
            target = web_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(module.source, encoding="utf-8", newline="\n")
            metadata = module.api.metadata
            webgpu_prebakes[method].append({
                "path": relative.as_posix(),
                "source_sha256": _sha(module.source.encode("utf-8")),
                "problem_shape": metadata["problem_shape"],
                "workgroup_size": list(module.launch_plan.workgroup_size),
                "groups": list(module.launch_plan.groups),
                "io_layout": module.io_layout.to_mapping(),
                "parameter_bindings": metadata["parameter_bindings"],
                "role_source_sha256": metadata["role_source_sha256"],
                "variant": metadata["variant"],
            })
    surface_methods = {
        "native": [method.name for method in BLAS_LIBRARY.methods],
        "python": [method.name for method in BLAS_LIBRARY.methods],
        "webgpu": [method.name for method in BLAS_LIBRARY.methods],
    }
    matrix = {
        "schema": MATRIX_SCHEMA,
        "role": role.identity,
        "role_source_sha256": _sha(role.source.encode("utf-8")),
        "contract": contract_name,
        "work_contract": contract_record,
        "library": BLAS_LIBRARY.to_mapping(include_source=True),
        "surface_methods": surface_methods,
        "generic_native": {
            name: {
                key: value for key, value in record.items() if key != "llvm_path"
            }
            for name, record in generic_products.items()
        },
        "webgpu_prebakes": webgpu_prebakes,
        "variants": variants,
    }
    matrix_bytes = _canonical(matrix)
    matrix_sha = _sha(matrix_bytes)
    matrix_path = root / "server-matrix.json"
    matrix_path.write_bytes(matrix_bytes)

    index_source = native_root / "turing_blas_server.c"
    index_source.write_text(
        _server_c(matrix_bytes, matrix_sha, variants, generic_products),
        encoding="utf-8",
    )
    pool_source = (
        Path(__file__).resolve().parents[1]
        / "common" / "tensors" / "accelerator_backends" / "c_backend"
        / "turing_pool.c"
    )
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    native_library = native_root / f"turing_blas_server{suffix}"
    core_sources = sorted({
        str(Path(product.manifest["build"]["core_llvm"]))
        for product in products
    } | {
        str(record["llvm_path"]) for record in generic_products.values()
    })
    command = [
        sys.executable, "-m", "ziglang", "cc", "-shared", "-O3", "-march=native",
        str(index_source), *(str(product.source_path) for product in products),
        str(pool_source), *core_sources, "-o", str(native_library),
    ]
    if os.name != "nt":
        command.append("-pthread")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode or not native_library.is_file():
        raise BLASServerError(
            f"BLAS server native build failed ({completed.returncode}):\n"
            + (completed.stderr or completed.stdout)[-4000:]
        )
    shutil.rmtree(root / ".build")
    native_header = native_root / "turing_blas_server.h"
    native_header.write_text(_native_header(), encoding="utf-8", newline="\n")

    wasm = build_module(
        function_name="turing_blas_server_matrix",
        parameter_types=[], body=CodeBuilder("i32", 0),
        memory_pages=max(1, (len(matrix_bytes) + 65535) // 65536),
        data=matrix_bytes,
    )
    wasm_path = web_root / "server-matrix.wasm"
    wasm_path.write_bytes(wasm)
    javascript_path = web_root / "blas-server.js"
    javascript_path.write_text(
        _javascript(len(matrix_bytes), matrix_sha), encoding="utf-8", newline="\n",
    )
    demo_path = web_root / "index.html"
    demo_path.write_text(_DEMO, encoding="utf-8", newline="\n")
    python_loader = python_root / "turing_blas_server.py"
    python_loader.write_text(_python_loader(), encoding="utf-8", newline="\n")
    (python_root / "__init__.py").write_text(
        "from .turing_blas_server import BLASServer, load\n",
        encoding="utf-8", newline="\n",
    )

    readme_path = root / "README.md"
    readme_path.write_text(
        _readme(shapes, matrix_sha), encoding="utf-8", newline="\n",
    )
    build_marker.unlink()

    artifacts = {}
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda value: value.as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        artifacts[relative] = {"sha256": _sha(path.read_bytes()), "bytes": path.stat().st_size}
    manifest = {
        "schema": SERVER_SCHEMA,
        "product_id": matrix_sha,
        "server_matrix": matrix_path.relative_to(root).as_posix(),
        "server_matrix_sha256": matrix_sha,
        "contract": contract_name,
        "library": "blas",
        "methods": [method.name for method in BLAS_LIBRARY.methods],
        "roles": [method.identity for method in BLAS_LIBRARY.methods],
        "deployed_roles": [method.identity for method in BLAS_LIBRARY.methods],
        "surface_roles": {
            surface: [f"blas.{name}" for name in names]
            for surface, names in surface_methods.items()
        },
        "shapes": [list(shape) for shape in shapes],
        "surfaces": {
            "native": {
                "library": native_library.relative_to(root).as_posix(),
                "header": native_header.relative_to(root).as_posix(),
                "entry": "turing_blas_server_gemm",
                "matrix_storage": "embedded",
                "dtype": "float64",
                "python_runtime_dependency": False,
                "target": {
                    "system": platform.system(),
                    "machine": platform.machine(),
                    "cpu_tuning": "native",
                },
            },
            "python": {
                "loader": python_loader.relative_to(root).as_posix(),
                "entry": "load",
                "native_dependency": native_library.relative_to(root).as_posix(),
                "dependencies": ["Python", "NumPy"],
            },
            "web": {
                "wasm": wasm_path.relative_to(root).as_posix(),
                "javascript": javascript_path.relative_to(root).as_posix(),
                "demo": demo_path.relative_to(root).as_posix(),
                "matrix_storage": "WASM data segment",
                "shader_storage": "individual deterministic WGSL prebakes",
                "dtype": "float32",
                "python_runtime_dependency": False,
                "javascript_runtime_dependency": True,
                "browser_dependencies": ["WebAssembly", "WebGPU"],
            },
        },
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n",
    )
    return BLASServerProduct(
        root, manifest_path, matrix_path, native_library, python_loader,
        wasm_path, javascript_path, demo_path, manifest,
    )


__all__ = [
    "BLASServerError",
    "BLASServerProduct",
    "MATRIX_SCHEMA",
    "SERVER_SCHEMA",
    "build_blas_server",
]
