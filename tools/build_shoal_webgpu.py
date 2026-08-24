"""Compile Shoal's authored SymPy stencil into an interactive WebGPU page.

The numerical update is emitted from repository SSA.  This file owns only the
browser boundary: toroidal neighbour gathering, state writeback, interaction,
and presentation of the compiler-published fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler import ssa_webgpu_backend as webgpu
from src.compiler.symbolic_fluid_model import compile_symbolic_fluid_step
from src.compiler.work_contract import set_active_contract


STATE = ("height", "momentum_x", "momentum_y", "tracer")
DIRECTIONS = ("center", "east", "west", "north", "south")


GATHER_WGSL = r"""
struct Shape { wide: u32, high: u32, count: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read_write> feeds: array<f32>;
// Four fields, each ordered center/east/west/north/south.
@group(0) @binding(2) var<storage, read> offsets: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= shape.count) { return; }
  let row = i / shape.wide;
  let col = i % shape.wide;
  let east = row * shape.wide + (col + 1u) % shape.wide;
  let west = row * shape.wide + (col + shape.wide - 1u) % shape.wide;
  let north = ((row + shape.high - 1u) % shape.high) * shape.wide + col;
  let south = ((row + 1u) % shape.high) * shape.wide + col;
  for (var field = 0u; field < 4u; field = field + 1u) {
    let at = field * 5u;
    let source = offsets[at];
    feeds[offsets[at + 1u] + i] = feeds[source + east];
    feeds[offsets[at + 2u] + i] = feeds[source + west];
    feeds[offsets[at + 3u] + i] = feeds[source + north];
    feeds[offsets[at + 4u] + i] = feeds[source + south];
  }
}
"""


PERTURB_WGSL = r"""
struct Shape { wide: u32, high: u32, count: u32, _pad: u32 };
struct Interaction {
  x: f32, y: f32, radius: f32, strength: f32,
  enabled: u32, pressed: u32, _pad0: u32, _pad1: u32,
};
@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read_write> feeds: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<uniform> interaction: Interaction;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= shape.count || interaction.enabled == 0u) { return; }
  let row = f32(i / shape.wide);
  let col = f32(i % shape.wide);
  let raw_x = abs(col - interaction.x);
  let raw_y = abs(row - interaction.y);
  let dx = min(raw_x, f32(shape.wide) - raw_x);
  let dy = min(raw_y, f32(shape.high) - raw_y);
  let d2 = dx * dx + dy * dy;
  let r2 = interaction.radius * interaction.radius;
  if (d2 >= r2) { return; }
  let falloff = 1.0 - d2 / r2;
  let impulse = interaction.strength * falloff * falloff;
  let height_at = offsets[0] + i;
  let tracer_at = offsets[15] + i;
  feeds[height_at] = max(0.25, feeds[height_at] + impulse);
  let dye = select(0.0012, -0.0007, interaction.pressed != 0u);
  feeds[tracer_at] = clamp(feeds[tracer_at] + dye * falloff, 0.0, 1.0);
}
"""


PRESENT_WGSL = r"""
struct Shape { wide: u32, high: u32, count: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read> feeds: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};
@vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
  var points = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  let p = points[index];
  var out: VertexOut;
  out.position = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return out;
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4<f32> {
  let x = min(u32(in.uv.x * f32(shape.wide)), shape.wide - 1u);
  let y = min(u32(in.uv.y * f32(shape.high)), shape.high - 1u);
  let i = y * shape.wide + x;
  let h = feeds[offsets[0] + i];
  let mx = feeds[offsets[5] + i];
  let my = feeds[offsets[10] + i];
  let tracer = clamp(feeds[offsets[15] + i], 0.0, 1.0);
  let speed = length(vec2<f32>(mx, my)) / max(h, 0.05);
  let elevation = clamp((h - 1.0) * 5.0, -1.0, 1.0);
  let deep = vec3<f32>(0.012, 0.032, 0.075);
  let raised = vec3<f32>(0.05, 0.82, 0.92);
  let lowered = vec3<f32>(0.45, 0.08, 0.62);
  var color = mix(deep, select(lowered, raised, elevation >= 0.0), abs(elevation));
  color = mix(color, vec3<f32>(1.0, 0.48, 0.05), tracer * 0.82);
  color = color + min(speed * 1.8, 0.55) * vec3<f32>(0.55, 0.75, 1.0);
  return vec4<f32>(color, 1.0);
}
"""


TILE_WGSL = r"""
struct View { width: f32, height: f32, tile: f32, _pad: f32 };
@group(0) @binding(0) var field_sampler: sampler;
@group(0) @binding(1) var field_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> view: View;
struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};
@vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
  var points = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  let p = points[index];
  var out: VertexOut;
  out.position = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return out;
}
@fragment fn fs(in: VertexOut) -> @location(0) vec4<f32> {
  let repeats = vec2<f32>(view.width, view.height) / view.tile;
  return textureSample(field_texture, field_sampler, in.uv * repeats);
}
"""


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shoal · compiled viscous shallow water</title>
<style>
:root { color-scheme: dark; } html,body { width:100%;min-height:100%; }
body { margin:0;overflow:hidden;background:#050914;color:#dce8f4;
  font:14px/1.45 ui-sans-serif,system-ui,sans-serif; }
canvas { position:fixed;inset:0;width:100vw;height:100vh;image-rendering:pixelated;
  touch-action:none;cursor:crosshair; }
main { position:relative;z-index:1;width:min(680px,calc(100vw - 40px));
  margin:20px auto;padding:14px 16px;border:1px solid rgb(201 225 244/.18);
  border-radius:9px;background:rgb(5 9 20/.78);backdrop-filter:blur(10px);
  box-shadow:0 16px 56px rgb(0 0 0/.34);pointer-events:none; }
#status,#health,#pointer { font-variant-numeric:tabular-nums;min-height:1.45em; }
#health { color:#93a8bc; } #stamp { color:#60758a;font-size:12px; }
.legend { margin-top:7px;color:#a8b8c8;font-size:13px; }
.cyan{color:#3de4f2}.violet{color:#c166ef}.amber{color:#ff9b31}.bright{color:#d8efff}
b { color:#eef7ff; font-weight:600; }
</style></head><body><canvas id="view" width="512" height="512"></canvas>
<main><div id="status">loading…</div><div id="health"></div>
<div id="pointer">move for a surface hill · press for a depression</div>
<div id="stamp">build …</div><div class="legend">
This is <b>Shoal</b>, the repository’s compiled depth-averaged free-surface
Navier–Stokes model. <span class="cyan">Cyan is raised water</span>,
<span class="violet">violet is depressed water</span>,
<span class="amber">amber is transported tracer</span>, and
<span class="bright">brightness is flow speed</span>. Every tile observes the
same toroidal field at the same completed step; color is a measurement of its
four evolving state planes, not an independent animation.
</div></main><script type="module">
const MANIFEST=__MANIFEST__, KERNEL=__KERNEL__, GATHER=__GATHER__;
const PERTURB=__PERTURB__, PRESENT=__PRESENT__, TILE=__TILE__;
const canvas=document.querySelector('#view'),status=document.querySelector('#status');
const health=document.querySelector('#health'),pointerText=document.querySelector('#pointer');
const stamp=document.querySelector('#stamp'); stamp.textContent='build '+MANIFEST.build;
let failed=false;
function fail(message){failed=true;status.textContent=message;status.style.color='#ff9d9d';console.error(message);}
function splitmix32(seed){return()=>{seed|=0;seed=seed+0x9e3779b9|0;let t=seed;
  t=Math.imul(t^t>>>16,0x21f0aaad);t=Math.imul(t^t>>>15,0x735a2d97);
  return((t^t>>>15)>>>0)/4294967296;};}
async function main(){
 if(!navigator.gpu){fail('WebGPU is unavailable in this browser');return;}
 const adapter=await navigator.gpu.requestAdapter();if(!adapter){fail('no WebGPU adapter');return;}
 const device=await adapter.requestDevice();window.__gpu={adapter,device};
 device.lost.then(info=>fail('WebGPU device lost: '+(info.message||info.reason||'unknown')));
 device.addEventListener?.('uncapturederror',e=>fail('WebGPU error: '+e.error.message));
 const {wide,high,count}=MANIFEST;
 const context=canvas.getContext('webgpu'),format=navigator.gpu.getPreferredCanvasFormat();
 context.configure({device,format,alphaMode:'opaque'});
 const storage=bytes=>device.createBuffer({size:Math.max(4,bytes),usage:
   GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
 const feeds=storage(MANIFEST.inputs.length*count*4), staging=new Float32Array(MANIFEST.inputs.length*count);
 const random=splitmix32(MANIFEST.seed), slot=Object.fromEntries(MANIFEST.inputs.map((n,i)=>[n,i]));
 const center={};
 for(const field of MANIFEST.state) center[field]=slot[field+'_center'];
 for(let y=0;y<high;y++)for(let x=0;x<wide;x++){
   const i=y*wide+x, nx=(x+.5)/wide*2-1, ny=(y+.5)/high*2-1;
   const r2=nx*nx+ny*ny, ring=Math.exp(-28*(Math.sqrt(r2)-.42)**2);
   const plume=Math.exp(-22*((nx+.34)**2+(ny-.12)**2));
   staging[center.height*count+i]=1+.13*Math.exp(-9*r2)-.065*ring+.005*(random()-.5);
   staging[center.momentum_x*count+i]=-ny*.12*Math.exp(-4*r2);
   staging[center.momentum_y*count+i]= nx*.12*Math.exp(-4*r2);
   staging[center.tracer*count+i]=Math.min(1,.82*plume+.32*ring);
 }
 for(const [name,value] of Object.entries(MANIFEST.constants))
   staging.fill(value,slot[name]*count,(slot[name]+1)*count);
 device.queue.writeBuffer(feeds,0,staging);
 const offsetsArray=[];
 for(const field of MANIFEST.state)for(const direction of MANIFEST.directions)
   offsetsArray.push(slot[field+'_'+direction]*count);
 const offsets=storage(Math.max(16,offsetsArray.length*4));
 device.queue.writeBuffer(offsets,0,new Uint32Array(offsetsArray));
 const shape=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 device.queue.writeBuffer(shape,0,new Uint32Array([wide,high,count,0]));
 const outputs=MANIFEST.state.map(()=>storage(count*4));
 const modules={kernel:device.createShaderModule({code:KERNEL}),gather:device.createShaderModule({code:GATHER}),
   perturb:device.createShaderModule({code:PERTURB}),present:device.createShaderModule({code:PRESENT}),
   tile:device.createShaderModule({code:TILE})};
 for(const [name,module] of Object.entries(modules)){const info=await module.getCompilationInfo();
   const error=info.messages.find(m=>m.type==='error');if(error){fail(name+' WGSL: '+error.message);return;}}
 const kernelPipe=device.createComputePipeline({layout:'auto',compute:{module:modules.kernel,entryPoint:'main'}});
 const gatherPipe=device.createComputePipeline({layout:'auto',compute:{module:modules.gather,entryPoint:'main'}});
 const perturbPipe=device.createComputePipeline({layout:'auto',compute:{module:modules.perturb,entryPoint:'main'}});
 const presentPipe=device.createRenderPipeline({layout:'auto',vertex:{module:modules.present,entryPoint:'vs'},
   fragment:{module:modules.present,entryPoint:'fs',targets:[{format}]},primitive:{topology:'triangle-list'}});
 const tilePipe=device.createRenderPipeline({layout:'auto',vertex:{module:modules.tile,entryPoint:'vs'},
   fragment:{module:modules.tile,entryPoint:'fs',targets:[{format}]},primitive:{topology:'triangle-list'}});
 const fieldTexture=device.createTexture({size:[wide,high],format,usage:
   GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
 const fieldView=fieldTexture.createView(),sampler=device.createSampler({addressModeU:'repeat',addressModeV:'repeat',
   magFilter:'nearest',minFilter:'nearest'});
 const viewUniform=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 const interactionUniform=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 const tilePixels=256,pointer={inside:false,pressed:false,x:0,y:0};
 function point(event,inside=null){const rect=canvas.getBoundingClientRect(),px=event.clientX-rect.left,py=event.clientY-rect.top;
   pointer.x=(((px%tilePixels)+tilePixels)%tilePixels)/tilePixels*wide;
   pointer.y=(((py%tilePixels)+tilePixels)%tilePixels)/tilePixels*high;
   pointer.inside=inside??(px>=0&&py>=0&&px<rect.width&&py<rect.height);
   pointerText.textContent=`page ${px.toFixed(0)}, ${py.toFixed(0)} → fluid ${pointer.x.toFixed(1)}, ${pointer.y.toFixed(1)} · ${pointer.pressed?'depression':'hill'}`;}
 canvas.addEventListener('pointermove',point);canvas.addEventListener('pointerenter',point);
 canvas.addEventListener('pointerdown',e=>{pointer.pressed=true;canvas.setPointerCapture(e.pointerId);point(e);});
 const release=e=>{pointer.pressed=false;point(e);if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);};
 canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);
 canvas.addEventListener('pointerleave',e=>point(e,false));
 const kernelEntries=[{binding:0,resource:{buffer:feeds}},...outputs.map((buffer,i)=>({binding:i+1,resource:{buffer}}))];
 const bind=(pipe,entries)=>device.createBindGroup({layout:pipe.getBindGroupLayout(0),entries});
 const kernelBind=bind(kernelPipe,kernelEntries), gatherBind=bind(gatherPipe,[{binding:0,resource:{buffer:shape}},
   {binding:1,resource:{buffer:feeds}},{binding:2,resource:{buffer:offsets}}]);
 const perturbBind=bind(perturbPipe,[{binding:0,resource:{buffer:shape}},{binding:1,resource:{buffer:feeds}},
   {binding:2,resource:{buffer:offsets}},{binding:3,resource:{buffer:interactionUniform}}]);
 const presentBind=bind(presentPipe,[{binding:0,resource:{buffer:shape}},{binding:1,resource:{buffer:feeds}},
   {binding:2,resource:{buffer:offsets}}]);
 const tileBind=bind(tilePipe,[{binding:0,resource:sampler},{binding:1,resource:fieldView},
   {binding:2,resource:{buffer:viewUniform}}]);
 let step=0,last=performance.now(),frames=0;
 function frame(){if(failed)return;const ratio=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*ratio)),h=Math.max(1,Math.round(canvas.clientHeight*ratio));
   if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
   device.queue.writeBuffer(viewUniform,0,new Float32Array([canvas.clientWidth,canvas.clientHeight,tilePixels,0]));
   const interaction=new ArrayBuffer(32),fv=new Float32Array(interaction),uv=new Uint32Array(interaction);
   fv.set([pointer.x,pointer.y,5.5,pointer.pressed?-.0022:.0015]);uv[4]=pointer.inside?1:0;uv[5]=pointer.pressed?1:0;
   device.queue.writeBuffer(interactionUniform,0,interaction);
   const encoder=device.createCommandEncoder();let pass=encoder.beginComputePass();pass.setPipeline(perturbPipe);pass.setBindGroup(0,perturbBind);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
   pass=encoder.beginComputePass();pass.setPipeline(gatherPipe);pass.setBindGroup(0,gatherBind);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
   pass=encoder.beginComputePass();pass.setPipeline(kernelPipe);pass.setBindGroup(0,kernelBind);pass.dispatchWorkgroups(MANIFEST.groups);pass.end();
   MANIFEST.state.forEach((field,i)=>encoder.copyBufferToBuffer(outputs[i],0,feeds,center[field]*count*4,count*4));
   let render=encoder.beginRenderPass({colorAttachments:[{view:fieldView,loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1}}]});
   render.setPipeline(presentPipe);render.setBindGroup(0,presentBind);render.draw(6);render.end();
   render=encoder.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:{r:.005,g:.009,b:.02,a:1}}]});
   render.setPipeline(tilePipe);render.setBindGroup(0,tileBind);render.draw(6);render.end();
   device.queue.submit([encoder.finish()]);step++;frames++;const now=performance.now();
   if(now-last>500){const fps=frames*1000/(now-last);status.textContent=`${wide}×${high} Shoal cells · step ${step.toLocaleString()} · ${fps.toFixed(1)} fps`;
     health.textContent='four GPU-resident fields · fixed dt '+MANIFEST.constants.dt.toFixed(4)+' · compiled '+MANIFEST.kernelBytes.toLocaleString()+' byte kernel';last=now;frames=0;}
   device.queue.onSubmittedWorkDone().then(schedule).catch(e=>fail('WebGPU submission failed: '+(e.message||e)));}
 function schedule(){if(failed)return;if(document.hidden)setTimeout(frame,16);else requestAnimationFrame(frame);} schedule();
}
main().catch(e=>fail(String(e&&e.message||e)));
</script></body></html>"""


def build(width: int, height: int, local_size: int, destination: Path) -> Path:
    set_active_contract("deploy")
    compiled = compile_symbolic_fluid_step()
    output_names = tuple(compiled.function.metadata["output_names"])
    returned = compiled.function.blocks["entry"].instrs[-1].args
    state_outputs = tuple(f"{field}_next" for field in STATE)
    outputs = tuple(returned[output_names.index(name)] for name in state_outputs)
    count = width * height
    emitted = webgpu.emit_module(
        compiled.module,
        name="shoal",
        count=count,
        outputs={compiled.function.name: outputs},
        preferred_local_size=local_size,
    )
    if not emitted.complete:
        raise RuntimeError("; ".join(item.format() for item in emitted.shortfalls))
    inputs = tuple(compiled.function.metadata["argument_names"])
    constants = {
        "dt": 0.0125, "dx": 1.0, "gravity": 1.0,
        "viscosity": 0.035, "tracer_diffusivity": 0.018,
        "linear_drag": 0.008, "coriolis": 0.055,
        "minimum_height": 0.05,
    }
    manifest = {
        "wide": width, "high": height, "count": count, "seed": 19,
        "state": STATE, "directions": DIRECTIONS, "inputs": inputs,
        "constants": constants, "kernelBytes": len(emitted.source),
        "groups": int(emitted.launch_plan.groups[0]),
    }
    stamp_source = emitted.source + PAGE + json.dumps(manifest, sort_keys=True)
    manifest["build"] = hashlib.sha256(stamp_source.encode()).hexdigest()[:8]
    html = (PAGE.replace("__MANIFEST__", json.dumps(manifest))
            .replace("__KERNEL__", json.dumps(emitted.source))
            .replace("__GATHER__", json.dumps(GATHER_WGSL))
            .replace("__PERTURB__", json.dumps(PERTURB_WGSL))
            .replace("__PRESENT__", json.dumps(PRESENT_WGSL))
            .replace("__TILE__", json.dumps(TILE_WGSL)))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "shoal.wgsl").write_text(emitted.source, encoding="utf-8", newline="\n")
    page = destination / "index.html"
    page.write_text(html, encoding="utf-8", newline="\n")
    print(f"Shoal: {count:,} cells, {len(emitted.source):,} byte WGSL, build {manifest['build']}")
    print(f"wrote {page}")
    return page


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--local-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("build/shoal-webgpu"))
    args = parser.parse_args(argv)
    build(args.width, args.height, args.local_size, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
