"""Build a self-contained browser benchmark from compiler-emitted WGSL.

The page is deliberately a deployment consumer, not a second shader author.
Every selectable numerical operation is emitted by ``ssa_webgpu_backend``;
the runtime only binds buffers, dispatches, synchronizes, verifies and reports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .ssa_webgpu_backend import (
    benchmarkable_tensor_operations,
    emit_gemm_module,
    emit_operator_module,
)


SCHEMA = "turing.webgpu-operator-benchmark.v1"
DEFAULT_COUNTS = (65_536, 262_144, 1_048_576, 4_194_304)
DEFAULT_GEMM_SIZES = (128, 256, 512, 1024)


@dataclass(frozen=True, slots=True)
class WebGPUBenchmarkBundle:
    directory: Path
    page_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _module_record(module, **extra: Any) -> dict[str, Any]:
    metadata = module.api.to_mapping()["metadata"]
    return {
        **extra,
        "name": module.name,
        "source": module.source,
        "source_sha256": hashlib.sha256(module.source.encode("utf-8")).hexdigest(),
        "workgroup_size": list(module.launch_plan.workgroup_size),
        "groups": list(module.launch_plan.groups),
        "io_layout": metadata["io_layout"],
        "backend_identities": metadata.get("backend_identities", []),
    }


def build_webgpu_benchmark_manifest(
    *,
    counts: Iterable[int] = DEFAULT_COUNTS,
    gemm_sizes: Iterable[int] = DEFAULT_GEMM_SIZES,
) -> dict[str, Any]:
    """Compile the selectable operation inventory and its deterministic ABI."""

    counts = tuple(sorted(set(map(int, counts))))
    gemm_sizes = tuple(sorted(set(map(int, gemm_sizes))))
    if not counts or min(counts) <= 0:
        raise ValueError("WebGPU benchmark counts must be positive")
    if any(count % 256 for count in counts):
        raise ValueError("WebGPU benchmark counts must be multiples of 256")
    if not gemm_sizes or min(gemm_sizes) <= 0:
        raise ValueError("WebGPU GEMM sizes must be positive")

    maximum_count = max(counts)
    kernels = []
    for operation, arity in benchmarkable_tensor_operations().items():
        module = emit_operator_module(operation, maximum_count)
        if not module.complete:
            reasons = "; ".join(item.format() for item in module.shortfalls)
            raise RuntimeError(f"WebGPU benchmark {operation} is incomplete: {reasons}")
        kernels.append(_module_record(
            module,
            id=f"elementwise:{operation}",
            kind="elementwise",
            operation=operation,
            label=operation.replace("_", " "),
            arity=arity,
            maximum_count=maximum_count,
            useful_operations_per_element=1,
        ))
    for size in gemm_sizes:
        for variant, label in (
            ("source_algorithm", "GEMM source-order (naive)"),
            ("webgpu_tiled_gemm", "GEMM compiler-tiled"),
        ):
            module = emit_gemm_module(size, size, size, variant=variant)
            kernels.append(_module_record(
                module,
                id=f"gemm:{variant}:{size}",
                kind="gemm",
                operation="gemm",
                label=label,
                variant=variant,
                size=size,
                arity=2,
                useful_operations=2 * size ** 3,
            ))
    body = {
        "schema": SCHEMA,
        "timing": {
            "preferred": "WebGPU timestamp-query",
            "fallback": "queue.onSubmittedWorkDone synchronized wall time",
            "warmups": 3,
            "iterations": 20,
        },
        "element_counts": list(counts),
        "gemm_sizes": list(gemm_sizes),
        "kernels": kernels,
    }
    return {**body, "manifest_sha256": hashlib.sha256(
        _canonical_json(body).encode("utf-8")
    ).hexdigest()}


def _page(manifest: dict[str, Any]) -> str:
    embedded = json.dumps(manifest, separators=(",", ":")).replace("</", "<\\/")
    return _HTML.replace("__TURING_BENCHMARK_MANIFEST__", embedded)


def write_webgpu_benchmark_bundle(
    directory: str | Path,
    *,
    counts: Iterable[int] = DEFAULT_COUNTS,
    gemm_sizes: Iterable[int] = DEFAULT_GEMM_SIZES,
) -> WebGPUBenchmarkBundle:
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = build_webgpu_benchmark_manifest(
        counts=counts, gemm_sizes=gemm_sizes,
    )
    manifest_path = directory / "benchmark.json"
    page_path = directory / "index.html"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n",
    )
    page_path.write_text(_page(manifest), encoding="utf-8", newline="\n")
    return WebGPUBenchmarkBundle(directory, page_path, manifest_path, manifest)


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Turing WebGPU operator benchmark</title>
<style>
:root{color-scheme:dark;--bg:#090d12;--panel:#111923;--line:#263545;--ink:#e9f1f7;--muted:#91a3b4;--cyan:#64d8e8;--green:#75e29c;--red:#ff7f8c}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#173041 0,transparent 32rem),var(--bg);color:var(--ink);font:15px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
main{max-width:1100px;margin:auto;padding:42px 22px}h1{font:700 clamp(28px,5vw,58px)/1 system-ui;margin:0 0 12px;letter-spacing:-.045em}.lede{max-width:760px;color:var(--muted);font-size:17px}.panel{background:#111923e8;border:1px solid var(--line);border-radius:14px;padding:18px;margin:22px 0;box-shadow:0 16px 45px #0005}.controls{display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:12px;align-items:end}label{display:grid;gap:6px;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}select,input,button{border:1px solid #35485a;border-radius:8px;background:#0b1219;color:var(--ink);padding:11px;font:inherit}button{background:var(--cyan);color:#061015;border:0;font-weight:800;cursor:pointer}button:disabled{filter:grayscale(1);opacity:.55}.device{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.device div,.metric{background:#0b1219;border-radius:8px;padding:11px}.device span{display:block;color:var(--muted);font-size:11px}.status{min-height:24px;color:var(--muted)}.status.good{color:var(--green)}.status.bad{color:var(--red)}table{width:100%;border-collapse:collapse}th,td{text-align:right;padding:10px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}.rate{color:var(--cyan);font-weight:800}.ok{color:var(--green)}.bad{color:var(--red)}details{color:var(--muted)}code{color:var(--cyan)}@media(max-width:760px){.controls{grid-template-columns:1fr 1fr}.device{grid-template-columns:1fr}table{font-size:12px}}
</style></head><body><main>
<h1>Your browser, through the compiler.</h1>
<p class="lede">Every kernel below is emitted from Turing's repository SSA or its <code>blas.gemm</code> role. The page performs real WebGPU dispatches, synchronizes them, reads results back for validation, and reports the browser's own device capacity.</p>
<section class="panel"><div class="device" id="device"><div><span>Adapter</span>waiting</div><div><span>Timing</span>waiting</div><div><span>Limits</span>waiting</div></div></section>
<section class="panel"><div class="controls">
<label>Operation<select id="operation"></select></label>
<label>Work size<select id="size"></select></label>
<label>Warmups<input id="warmups" type="number" min="0" max="100" value="3"></label>
<label>Iterations<input id="iterations" type="number" min="1" max="500" value="20"></label>
</div><p><button id="run">Benchmark selected operation</button></p><div id="status" class="status">Requesting a WebGPU adapter…</div></section>
<section class="panel"><table><thead><tr><th>Kernel</th><th>Work</th><th>Mean</th><th>Throughput</th><th>Validation</th><th>Timer</th></tr></thead><tbody id="results"></tbody></table></section>
<details class="panel"><summary>Deployment identity</summary><pre id="identity"></pre></details>
</main><script>
const MANIFEST=__TURING_BENCHMARK_MANIFEST__;
const $=id=>document.getElementById(id), op=$('operation'), size=$('size'), run=$('run'), status=$('status'), results=$('results');
const elementKernels=MANIFEST.kernels.filter(k=>k.kind==='elementwise');
const gemmKernels=MANIFEST.kernels.filter(k=>k.kind==='gemm');
op.innerHTML='<option value="gemm:compare">GEMM — naive versus compiler-tiled</option>'+elementKernels.map(k=>`<option value="${k.id}">${k.label}</option>`).join('');
function repopulate(){const gemm=op.value==='gemm:compare';const values=gemm?MANIFEST.gemm_sizes:MANIFEST.element_counts;size.innerHTML=values.map(v=>`<option value="${v}">${gemm?v+' × '+v+' × '+v:Number(v).toLocaleString()+' elements'}</option>`).join('');size.value=String(values[Math.max(0,values.length-2)]);}
op.addEventListener('change',repopulate);repopulate();
let adapter,device,timestamp=false;
function fail(error){status.textContent=error.message||String(error);status.className='status bad';run.disabled=false;throw error;}
async function initialize(){if(!navigator.gpu)throw new Error('WebGPU is unavailable. Use a current secure-context browser.');adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw new Error('No WebGPU adapter was granted.');timestamp=adapter.features.has('timestamp-query');device=await adapter.requestDevice({requiredFeatures:timestamp?['timestamp-query']:[]});const info=adapter.info||{};$('device').innerHTML=`<div><span>Adapter</span>${info.vendor||'WebGPU'} ${info.architecture||info.device||''}</div><div><span>Timing</span>${timestamp?'GPU timestamp queries':'synchronized queue wall time'}</div><div><span>Limits</span>${device.limits.maxComputeInvocationsPerWorkgroup} lanes · ${Math.round(device.limits.maxStorageBufferBindingSize/1048576)} MiB buffer</div>`;status.textContent='Ready. Select any compiler-supported operation.';status.className='status good';run.disabled=false;}
function inputs(count){const a=new Float32Array(count),b=new Float32Array(count);for(let i=0;i<count;i++){a[i]=Math.fround(.25+(i%97)/194);b[i]=Math.fround(.75+(i%89)/178);}return[a,b];}
function buffer(data,usage){const value=device.createBuffer({size:Math.max(4,data.byteLength),usage,mappedAtCreation:true});new data.constructor(value.getMappedRange()).set(data);value.unmap();return value;}
function f32bits(value){const f=new Float32Array([value]);return new Uint32Array(f.buffer)[0];}function bitsf32(value){const u=new Uint32Array([value>>>0]);return new Float32Array(u.buffer)[0];}
function reference(name,a,b){switch(name){case'abs':return Math.abs(a);case'acos':return Math.acos(a);case'add':return a+b;case'asin':return Math.asin(a);case'atan':return Math.atan(a);case'bitand':return bitsf32(f32bits(a)&f32bits(b));case'bitor':return bitsf32(f32bits(a)|f32bits(b));case'bitxor':return bitsf32(f32bits(a)^f32bits(b));case'ceil':return Math.ceil(a);case'copy':return a;case'cos':return Math.cos(a);case'cosh':return Math.cosh(a);case'equal':return +(a===b);case'exp':return Math.exp(a);case'floor':return Math.floor(a);case'floordiv':return Math.floor(a/b);case'greater':return +(a>b);case'greater_equal':return +(a>=b);case'invert':return bitsf32(~f32bits(a));case'less':return +(a<b);case'less_equal':return +(a<=b);case'log':return Math.log(a);case'logical_and':return +(a!==0&&b!==0);case'logical_not':return +(a===0);case'logical_or':return +(a!==0||b!==0);case'maximum':return Math.max(a,b);case'minimum':return Math.min(a,b);case'mod':return a%b;case'mul':return a*b;case'neg':return-a;case'not_equal':return +(a!==b);case'pow':return Math.pow(a,b);case'round':return Math.round(a);case'sign':return Math.sign(a);case'sin':return Math.sin(a);case'sinh':return Math.sinh(a);case'sqrt':return Math.sqrt(a);case'sub':return a-b;case'tan':return Math.tan(a);case'tanh':return Math.tanh(a);case'truediv':return a/b;case'trunc':return Math.trunc(a);default:throw new Error('missing reference '+name);}}
async function benchmark(kernel,work,warmups,iterations){const shader=device.createShaderModule({code:kernel.source,label:kernel.name});const info=await shader.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(e=>e.message).join('\n'));const pipeline=device.createComputePipeline({layout:'auto',compute:{module:shader,entryPoint:'main'}});let count,feeds,groups,expected;
if(kernel.kind==='elementwise'){count=Number(work);feeds=inputs(count);groups=[Math.ceil(count/kernel.workgroup_size[0]),1,1];expected=i=>reference(kernel.operation,feeds[0][i],feeds[1][i]);}
else{const n=Number(kernel.size);count=n*n;feeds=inputs(n*n);groups=kernel.groups;expected=index=>{const row=Math.floor(index/n),column=index%n;let sum=0;for(let p=0;p<n;p++)sum=Math.fround(sum+Math.fround(feeds[0][row*n+p]*feeds[1][p*n+column]));return sum;};}
const feedBuffers=feeds.slice(0,kernel.arity).map(v=>buffer(v,GPUBufferUsage.STORAGE));const output=device.createBuffer({size:count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[...feedBuffers.map((b,i)=>({binding:i,resource:{buffer:b}})),{binding:kernel.arity,resource:{buffer:output}}]});
function encode(repetitions,queries){const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass(queries?{timestampWrites:{querySet:queries.set,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:undefined);pass.setPipeline(pipeline);pass.setBindGroup(0,bind);for(let i=0;i<repetitions;i++)pass.dispatchWorkgroups(...groups);pass.end();if(queries){encoder.resolveQuerySet(queries.set,0,2,queries.resolve,0);encoder.copyBufferToBuffer(queries.resolve,0,queries.read,0,16);}return encoder.finish();}
if(warmups){device.queue.submit([encode(warmups,null)]);await device.queue.onSubmittedWorkDone();}let queries=null;if(timestamp){queries={set:device.createQuerySet({type:'timestamp',count:2}),resolve:device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),read:device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ})};}const t0=performance.now();device.queue.submit([encode(iterations,queries)]);await device.queue.onSubmittedWorkDone();const wall=(performance.now()-t0)/iterations;let mean=wall,timer='queue wall';if(queries){await queries.read.mapAsync(GPUMapMode.READ);const stamps=new BigUint64Array(queries.read.getMappedRange().slice(0));mean=Number(stamps[1]-stamps[0])/1e6/iterations;queries.read.unmap();timer='GPU timestamp';}
const samples=Math.min(64,count),read=device.createBuffer({size:samples*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),copy=device.createCommandEncoder();copy.copyBufferToBuffer(output,0,read,0,samples*4);device.queue.submit([copy.finish()]);await read.mapAsync(GPUMapMode.READ);const produced=new Float32Array(read.getMappedRange().slice(0));let worst=0,valid=true;for(let i=0;i<samples;i++){const wanted=expected(i),error=Math.abs(produced[i]-wanted),scale=Math.max(1,Math.abs(wanted));if((Number.isNaN(produced[i])!==Number.isNaN(wanted))||(!Number.isNaN(error)&&error>5e-3*scale))valid=false;if(Number.isFinite(error))worst=Math.max(worst,error);}read.unmap();[...feedBuffers,output,read,queries?.resolve,queries?.read].filter(Boolean).forEach(b=>b.destroy());queries?.set.destroy();const useful=kernel.kind==='gemm'?kernel.useful_operations:count*kernel.useful_operations_per_element;return{mean,rate:useful/(mean/1000)/1e9,valid,worst,timer};}
function row(kernel,work,value){const tr=document.createElement('tr');tr.innerHTML=`<td>${kernel.label}</td><td>${kernel.kind==='gemm'?work+'³':Number(work).toLocaleString()}</td><td>${value.mean.toFixed(3)} ms</td><td class="rate">${value.rate.toFixed(2)} Gop/s</td><td class="${value.valid?'ok':'bad'}">${value.valid?'pass':'FAIL'} · ${value.worst.toExponential(1)}</td><td>${value.timer}</td>`;results.prepend(tr);}
run.disabled=true;run.addEventListener('click',async()=>{run.disabled=true;status.textContent='Compiling and dispatching…';status.className='status';try{const work=Number(size.value),warmups=Number($('warmups').value),iterations=Number($('iterations').value);let kernels;if(op.value==='gemm:compare')kernels=gemmKernels.filter(k=>k.size===work);else kernels=[elementKernels.find(k=>k.id===op.value)];const measurements=[];for(const kernel of kernels){const value=await benchmark(kernel,work,warmups,iterations);measurements.push(value);row(kernel,work,value);$('identity').textContent=JSON.stringify({manifest:MANIFEST.manifest_sha256,kernel:kernel.source_sha256,identities:kernel.backend_identities,workgroup:kernel.workgroup_size,groups:kernel.groups},null,2);}if(measurements.length===2)status.textContent=`Complete. Compiler-tiled GEMM is ${(measurements[0].mean/measurements[1].mean).toFixed(2)}× source-order on this browser.`;else status.textContent='Complete. Result readback passed the CPU sample oracle.';status.className='status good';}catch(error){fail(error);}finally{run.disabled=false;}});initialize().catch(fail);
</script></body></html>'''


__all__ = [
    "DEFAULT_COUNTS",
    "DEFAULT_GEMM_SIZES",
    "SCHEMA",
    "WebGPUBenchmarkBundle",
    "build_webgpu_benchmark_manifest",
    "write_webgpu_benchmark_bundle",
]
