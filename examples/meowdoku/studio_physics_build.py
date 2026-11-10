"""Explicit Studio products through the incumbent SSA/WebGPU/WASM pipeline.

No startup hook, default rebuild, endpoint, or publication is added. Callers
must supply a validated scene and an isolated output directory. Topology is
compile-time; the exact authored coordinates remain runtime uniforms.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

from meowdoku import fountain3d as F, fountain3d_build as B, studio_contacts
from meowdoku import studio_environment
from meowdoku.studio_environment import StudioEnvironment


def topology(environment):
    return [{"id": part.id, "bodyId": part.body_id, "closed": part.closed,
             "vertices": len(part.vertices), "triangles": part.triangles,
             "materialIds": part.materials} for part in environment.parts]


def digest(environment, pad, active, dt=F.DT):
    source = "\0".join((B._sources(), inspect.getsource(studio_contacts),
                         inspect.getsource(studio_environment), inspect.getsource(sys.modules[__name__]),
                         json.dumps(topology(environment), sort_keys=True), str(pad), str(active), repr(dt)))
    if hasattr(environment, "geometry_contract"):
        from meowdoku import studio_mesh_environment, studio_mesh_manifold
        source += "\0" + "\0".join((inspect.getsource(studio_mesh_environment), inspect.getsource(studio_mesh_manifold),
                                    json.dumps(environment.geometry_contract, sort_keys=True)))
    return hashlib.sha256(source.encode()).hexdigest()[:20]


def environment_for(packet):
    """Explicit source-only product choice; never selects the legacy toy."""
    if packet.get("contact_model") is None:
        return StudioEnvironment(packet)
    from meowdoku.studio_mesh_environment import StudioMeshEnvironment, CONTACT_MODEL
    if packet["contact_model"] != CONTACT_MODEL:
        raise ValueError("unsupported Studio contact model")
    return StudioMeshEnvironment(packet)


def trace(environment, pad=2, active=1, contract=B.WGSL_CONTRACT, *, dt=F.DT):
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    if type(pad) is not int or type(active) is not int or not 1 <= active <= pad:
        raise ValueError("Studio product requires 1 <= active <= pad")
    with scoped_work_contract(PRESETS[contract]):
        program = SSATensorProgram("fountain_step")
        bindings = {name: SSATensorOperations.input(program, (pad,) if name in F.STATE + F.CONSTANTS else ())
                    for name in environment.arguments}
        with program.activate():
            produced = environment.law(pad, active, dt=dt)(*(bindings[name] for name in environment.arguments))
        outputs = [item.data.value for item in produced]
    return program, outputs, bindings


def passes(environment, pad=2, active=1, *, dt=F.DT):
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract
    program, outputs, bindings = trace(environment, pad, active, dt=dt)
    ids = {name: int(binding.data.value.id) for name, binding in bindings.items()}
    with scoped_work_contract(PRESETS[B.WGSL_CONTRACT]):
        product = webgpu.emit_tensor_program_passes(
            program.module, "fountain_step", outputs, count=pad,
            name="studio_fountain", iota_args=[ids[name] for name in F.IOTA],
            binding_bytes=F.BINDING_SPLIT_BYTES)
    if not product.complete:
        raise RuntimeError("incomplete Studio native product: " + "; ".join(s.format() for s in product.shortfalls))
    return product, program, outputs, ids


def physical_step(radius, gravity=9.81, rest_overlap_fraction=.01):
    """Existing DEM with a radius-aware fixed step, in real metres/seconds.

    Target at most 1% static spring compression under gravity. Raise stiffness
    with a power-of-two smaller step so the incumbent omega*dt CFL allowance
    is unchanged. This is not a new integrator or a hard-body/CCD claim.
    """
    if not math.isfinite(radius) or not .0001 <= radius <= .2:
        raise ValueError("Studio physical radius must be between 0.1 and 200 mm")
    if not math.isfinite(gravity) or gravity <= 0 or not 0 < rest_overlap_fraction <= .01:
        raise ValueError("invalid Studio contact resolution")
    omega = max(F.OMEGA, math.sqrt(gravity / (rest_overlap_fraction * radius)))
    substeps = 2 ** max(0, math.ceil(math.log2(omega / F.OMEGA_MAX)))
    return {"dt": F.DT / substeps, "omega": omega, "substeps": substeps,
            "gravity": gravity, "rest_overlap_fraction": gravity / (omega * omega * radius)}


def build(packet, destination, *, radius, pad=2, active=1, wasm=False):
    """Write a genuine ComputeRuntime-compatible product for a validated scene.

    This is an explicit potentially expensive build, never a request-time
    substitute. Default false WASM avoids accidentally launching native
    lowering; opted-in WASM uses graph_wasm, not a handwritten integrator.
    """
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.deployment_classification import WebGPUComputeCostModel, webgpu_compute_payoff
    environment = environment_for(packet)
    resolution = physical_step(radius)
    if active > F.K_RUN_MAX:
        raise ValueError("Studio active count exceeds the exact native neighbor-walk bound")
    uniforms = F.uniforms(n=active, gravity=(0, -9.81, 0))
    uniforms.update(environment.grid_uniforms(radius, active))
    uniforms.update(environment.uniforms())
    # The single physical radius must reach the species, emitter and swelling
    # paths as well as rs. Never pass it through the toy slider's 5..20mm clamp.
    for i in range(len(F.SPECIES)):
        uniforms[f"r{i}"] = F._f32(radius)
        uniforms[f"swell{i}"] = F._f32(0)
    uniforms["emit_rate"] = F._f32(0)
    uniforms["omega"] = F._f32(resolution["omega"])
    product, program, outputs, ids = passes(environment, pad, active, dt=resolution["dt"])
    model = WebGPUComputeCostModel()
    payoff = webgpu_compute_payoff(program.module, "fountain_step", outputs=outputs,
                                  model=model, dispatches=product.dispatches)
    names = environment.arguments
    uniform_names = F.UNIFORMS + environment.input_names
    output_ids = [int(v.id) for v in outputs]
    fingerprint = digest(environment, pad, active, resolution["dt"])
    prefix = f"studio_physics_{fingerprint}"
    record = {
        "tag": "studio", "n": active, "pad": pad, "variant": "studio",
        "route": "webgpu-compute" if payoff.pays else "wasm", "verdict": payoff.as_record(), "source_digest": fingerprint,
        "wgsl": prefix + ".wgsl", "display_stride": 1,
        "wgsl_sha256": hashlib.sha256(product.source.encode("utf-8")).hexdigest(),
        "uniforms": {k: float(v) for k, v in uniforms.items()},
        "geometry": {"room": packet["room"], "topology": topology(environment)},
        "wgsl_abi": {
            "count": pad, "workgroup_size": list(product.workgroup_size), "groups": list(product.groups),
            "entries": list(product.entries), "slots": product.slots, "dispatches": product.dispatches,
            "lane_base": product.lane_base, "arena_elements": product.arena_elements,
            "bindings": product.bindings, "slots_per_binding": product.slots_per_binding,
            "storage_bindings_per_pass": product.bindings + 1, "binding_budget": webgpu.MAX_STORAGE_BINDINGS_PER_STAGE,
            "binding_bytes": min(product.slots_per_binding, product.slots) * pad * 4,
            "arguments": list(names), "uniforms": list(uniform_names), "iota": list(F.IOTA), "outputs": list(F.OUTPUTS),
            "feed_slots": {name: product.feed_slots[ids[name]] for name in names if ids[name] in product.feed_slots},
            "uniform_index": {name: product.uniform_index[ids[name]] for name in uniform_names if ids[name] in product.uniform_index},
            "output_slots": {name: product.output_slots[output_ids[k]] for k, name in enumerate(F.OUTPUTS)},
            "backend_identities": [d.as_record() for d in product.backend_identity_decisions],
        },
        "memory": {"arena_bytes": product.arena_elements * 4, "bytes_per_id": product.arena_elements * 4 / pad,
                   "display_ids": pad, "display_bytes": 3 * 20 * pad},
    }
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / record["wgsl"]).write_text(product.source, encoding="utf-8", newline="\n")
    if wasm:
        from meowdoku import graph_wasm
        shapes = {name: ((pad,) if name in F.STATE + F.CONSTANTS else ()) for name in names}
        abi = graph_wasm._compile(prefix, shapes,
            lambda b: environment.law(pad, active, dt=resolution["dt"])(*(b[name] for name in names)),
            F.OUTPUTS, B.WASM_CONTRACT, destination, {"argument_names": list(names), "count": pad})
        record.update(wasm=prefix + ".wasm", wasm_abi=prefix + ".abi.json", wasm_buffers=len(abi["order"]))
        record["route"] = "webgpu-compute" if payoff.pays else "wasm"
    manifest = {
        "schema": B.SCHEMA, "sizes": {"studio": record}, "initial": {}, "emit_span": 0,
        "cost_model": model.__dict__,
        "law": {"arguments": list(names), "state": list(F.STATE), "constants": list(F.CONSTANTS),
                "uniforms": list(uniform_names), "outputs": list(F.OUTPUTS), "diagnostics": list(F.DIAGNOSTICS),
                "dt": resolution["dt"], "runtime_inputs": list(F.RUNTIME_INPUTS + environment.input_names),
                "gravity": 9.81, "k_run_max": F.K_RUN_MAX, "depth_max": F.DEPTH_MAX},
        "points": {"x": "x", "y": "y", "z": "z", "vx": "vx", "vy": "vy", "vz": "vz",
                   "rgb": ["cr", "cg", "cb"], "live": "shown", "radius": "rad", "rest_speed": .15, "bytes": 20},
        "settle": {**F.SETTLE, "sample": F.SETTLE["sample"] * resolution["substeps"]},
        "species": [[item[0], radius, *item[2:]] for item in F.SPECIES],
        "scene_schema": packet["schema"], "source_topology": topology(environment),
        "source_frame": packet.get("frame"),
        "scene_digest": hashlib.sha256(json.dumps(packet, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "contact_resolution": resolution,
    }
    if hasattr(environment, "geometry_contract"):
        manifest["geometry_contract"] = environment.geometry_contract
    if "sourceCaptureDigest" in packet:
        manifest["source_capture"] = {"capture_digest":packet["sourceCaptureDigest"],
            "mesh_digest":packet.get("sourceMeshDigest"),"revision":packet.get("sourceRevision")}
    (destination / (prefix + ".manifest.json")).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest
