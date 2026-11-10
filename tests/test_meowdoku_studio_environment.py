"""Bounded full native-law/ABI proof, without active server or GPU work."""
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from meowdoku import fountain3d as F, studio_physics_build as B
from meowdoku.studio_environment import StudioEnvironment, SCHEMA
from src.common.tensors import AbstractTensor as T


def packet(parts=None):
    return {"schema": SCHEMA, "room": {"low": [-.5, .001, -.5], "high": [.5, 1, .5]},
            "parts": parts or []}


def sheet(y=.2):
    return {"id": "tabletop", "bodyId": "tabletop", "convex": True, "closed": False,
            "vertices": [[-.2,y,-.2],[-.2,y,.2],[.2,y,.2],[.2,y,-.2]],
            "triangles": [[0,1,2],[0,2,3]], "materialIds": [17, 23]}


def columns(environment, p=(.03,.209,0), velocity=(0,0,0), radius=.01):
    c = F.initial_columns(2, active=1)
    c.update(environment.uniforms())
    c.update(environment.grid_uniforms(radius))
    for key, value in zip(("x","y","z"), p):
        c[key][0] = value
    for key, value in zip(("vx","vy","vz"), velocity):
        c[key][0] = value
    c["age"][:] = 1
    c["rs"][0] = radius
    for key in ("grav_x","grav_y","grav_z","fl_rho","fl_mu","spin_drag","temp","emit_rate"):
        c[key] = F._f32(0)
    for i in range(4):
        c[f"r{i}"] = F._f32(radius)
        c[f"swell{i}"] = F._f32(0)
    return c


def step(environment, c):
    values = environment.law(2,1)(*(T.get_tensor(np.asarray(c[k],dtype=np.float64)) for k in environment.arguments))
    return {name: np.asarray(v.data) for name,v in zip(F.OUTPUTS, values)}


def test_full_native_law_uses_authored_table_and_not_old_chamber():
    env = StudioEnvironment(packet([sheet()]))
    c = columns(env)
    out = step(env,c)
    assert out["vy_next"][0] > 0  # penetration is resolved by native force, not a clamp
    assert out["y_next"][0] == pytest.approx(c["y"][0] + F.DT*out["vy_next"][0])
    assert out["x_next"][0] == c["x"][0]
    assert out["n_out"][0] == 0
    changed = dict(c)
    changed.update({f"px{i}": F._f32(0) for i in range(F.NV)})
    changed.update({f"py{i}": F._f32(1000) for i in range(F.NV)})
    changed["lid"] = F._f32(-1000)
    # Legacy walls, ceiling and old SDF contents cannot leak into a Studio product.
    changed["sdf_on"] = F._f32(1)
    again = step(env,changed)
    for key in F.OUTPUTS:
        np.testing.assert_array_equal(again[key],out[key],err_msg=key)


def test_runtime_vertex_uniform_moves_the_actual_contact_without_topology_rebuild():
    env = StudioEnvironment(packet([sheet()]))
    c = columns(env)
    moved = StudioEnvironment(packet([sheet(.1)]))
    c.update(moved.uniforms())
    out = step(env,c)
    assert out["vy_next"][0] == 0
    assert env.parts[0].materials == (17,23)
    assert B.digest(env,2,1) == B.digest(moved,2,1)


def test_room_corners_and_post_step_diagnostics_use_authored_bounds():
    env = StudioEnvironment(packet())
    c = columns(env,p=(.499,.006,.499))
    out = step(env,c)
    assert out["vx_next"][0] < 0 and out["vy_next"][0] > 0 and out["vz_next"][0] < 0
    far = columns(env,p=(.53,.4,0))
    out = step(env,far)
    assert out["vx_next"][0] < 0 and out["n_out"][0] == 1
    assert out["n_live"][0] + out["n_out"][0] + out["n_parked"][0] + out["n_nan"][0] == 2


def test_native_contact_is_continuous_across_the_authored_diagonal():
    env = StudioEnvironment(packet([sheet()]))
    outputs = [step(env,columns(env,p=(x,.209,-x))) for x in (-1e-5,0,1e-5)]
    for out in outputs:
        assert out["vx_next"][0] == pytest.approx(0,abs=1e-15)
        assert out["vz_next"][0] == pytest.approx(0,abs=1e-15)
    np.testing.assert_allclose([out["vy_next"][0] for out in outputs], outputs[0]["vy_next"][0],rtol=0,atol=1e-15)


def test_topology_and_grid_reject_unproven_geometry_instead_of_faking_contact():
    bad = sheet()
    bad["vertices"][0][1] += .01
    with pytest.raises(ValueError,match="planar"):
        StudioEnvironment(packet([bad]))
    bad = sheet(); bad["closed"] = True
    with pytest.raises(ValueError,match="open boundary"):
        StudioEnvironment(packet([bad]))
    second = sheet(.3); second["id"] = "second"
    with pytest.raises(ValueError,match="union"):
        StudioEnvironment(packet([sheet(),second]))
    env = StudioEnvironment(packet())
    with pytest.raises(ValueError,match="capacity"):
        env.grid_uniforms(1e-6,F.K_RUN_MAX+1)
    small = env.grid_uniforms(1e-6)
    assert float(small["cell"]) > 2.1e-6
    assert float(small["cols"]*small["rows"]*small["lays"])+1 < 2**24
    assert float(small["k_run"]) == F.K_RUN_MAX
    with pytest.raises(ValueError,match="sdf=False"):
        F.make_law(2,1,environment=env)


def test_native_topology_rejects_tiny_real_folds_and_zero_volume_closed_shells():
    bad = sheet()
    bad["vertices"][2][1] += 1e-7  # retained after float32 conversion
    with pytest.raises(ValueError,match="planar"):
        StudioEnvironment(packet([bad]))
    flat = {"id":"flat", "convex":True, "closed":True,
            "vertices":[[0,.2,0],[1,.2,0],[0,.2,1]],
            "triangles":[[0,1,2],[0,2,1]],"materialIds":[1,1]}
    with pytest.raises(ValueError,match="volume"):
        StudioEnvironment(packet([flat]))
    tiny = {"id":"tiny", "convex":True, "closed":False,
            "vertices":[[0,0,0],[1e-14,0,0],[0,1e-14,0]],
            "triangles":[[0,1,2]],"materialIds":[1]}
    with pytest.raises(ValueError,match="contact precision"):
        StudioEnvironment(packet([tiny]))


def test_coarsened_hash_keeps_all_bounded_particle_contacts():
    coarse = StudioEnvironment(packet())
    small_packet = packet()
    small_packet["room"]["low"][0] = small_packet["room"]["low"][2] = -.001
    small_packet["room"]["high"][0] = small_packet["room"]["high"][2] = .001
    fine = StudioEnvironment(small_packet)
    radius = .0001
    outputs = []
    for env in (coarse,fine):
        c = F.initial_columns(3,active=3)
        c.update(env.uniforms());c.update(env.grid_uniforms(radius,3))
        c["x"][:] = [-.00009,.00009,.0007];c["y"][:] = .5
        c["rs"][:] = radius;c["age"][:] = 1
        for key in ("grav_x","grav_y","grav_z","fl_rho","fl_mu","temp","emit_rate"):
            c[key] = F._f32(0)
        for i in range(4):c[f"r{i}"] = F._f32(radius)
        values = env.law(3,3)(*(T.get_tensor(np.asarray(c[k],dtype=np.float64)) for k in env.arguments))
        outputs.append({name:np.asarray(v.data) for name,v in zip(F.OUTPUTS,values)})
    assert float(coarse.grid_uniforms(radius,3)["cell"]) > float(fine.grid_uniforms(radius,3)["cell"])
    assert outputs[0]["vx_next"][0] < 0 and outputs[0]["vx_next"][1] > 0
    assert outputs[0]["vx_next"][2] == 0
    assert outputs[0]["n_drop"][0] == 0
    for name in F.STATE:
        np.testing.assert_array_equal(outputs[0][name+"_next"],outputs[1][name+"_next"])


def test_timestep_reaches_both_native_friction_impulse_limits():
    scalar = lambda x:T.get_tensor(np.asarray([x],dtype=np.float64))
    zero,one = scalar(0),scalar(1)
    results=[]
    for dt in (F.DT,F.DT/16):
        result = F._contact(scalar(.01),one,(zero,one,zero),
            (one,zero,zero),(zero,zero,zero),(one,zero,zero),
            scalar(1e4),one,scalar(100),zero,scalar(100),one,one,one,zero,zero,True,dt=dt)
        results.append(([np.asarray(v.data)*dt for v in result[0]],
                        [np.asarray(v.data)*dt for v in result[2]]))
    np.testing.assert_allclose(results[0][0][0],results[1][0][0],rtol=0,atol=1e-12)
    np.testing.assert_allclose(results[0][1][0],results[1][1][0],rtol=0,atol=1e-12)


@pytest.mark.parametrize("radius", [.0001, .001, .01, .2])
def test_radius_aware_native_step_holds_the_same_fractional_rest_contact(radius):
    env = StudioEnvironment(packet([sheet()]))
    resolution = B.physical_step(radius)
    assert resolution["omega"] * resolution["dt"] <= F.OMEGA_MAX * F.DT
    assert resolution["rest_overlap_fraction"] <= .01000000001
    c = columns(env,radius=radius)
    c["omega"] = F._f32(resolution["omega"])
    c["grav_y"] = F._f32(-9.81)
    c["hertz"] = F._f32(0)
    # Construct the incumbent spring's equilibrium, not a position projection.
    surface = float(env.parts[0].vertices[0][1])
    c["y"][0] = surface + radius - 9.81 / float(c["omega"])**2
    values = env.law(2,1,dt=resolution["dt"])(*(T.get_tensor(np.asarray(c[k],dtype=np.float64)) for k in env.arguments))
    out = {name:np.asarray(v.data) for name,v in zip(F.OUTPUTS,values)}
    assert abs(out["vy_next"][0]) < 1e-8
    assert out["age_next"][0] == pytest.approx(1+resolution["dt"])
    assert out["y_next"][0] == pytest.approx(c["y"][0],abs=1e-10)


def test_complete_native_product_carries_shared_scene_abi(tmp_path):
    manifest = B.build(packet([sheet()]),tmp_path,radius=.01)
    record = manifest["sizes"]["studio"]
    abi = record["wgsl_abi"]
    env = StudioEnvironment(packet([sheet()]))
    assert set(env.input_names) <= set(abi["uniform_index"])
    assert manifest["law"]["state"] == list(F.STATE)
    assert all(k+"_next" in abi["output_slots"] for k in F.STATE)
    assert record["uniforms"]["grav_y"] == pytest.approx(-9.81)
    assert record["uniforms"]["r0"] == pytest.approx(.01)
    assert record["geometry"]["topology"][0]["materialIds"] == (17,23)
    assert abi["dispatches"] >= 2  # existing neighbor sort + contact/integrator + reductions
    assert "fn " in (tmp_path / record["wgsl"]).read_text()
    assert len(list(tmp_path.glob("*.manifest.json"))) == 1
    assert not list(tmp_path.glob("*.wasm"))  # no accidental native lowering job
    assert json.loads(next(tmp_path.glob("*.manifest.json")).read_text())["law"]["dt"] == B.physical_step(.01)["dt"]
    assert manifest["cost_model"]["min_speedup"] > 0
    assert manifest["settle"]["sample"] * manifest["law"]["dt"] == pytest.approx(F.SETTLE["sample"] * F.DT)
    node = shutil.which("node")
    if node:
        check = subprocess.run([node, str(ROOT / "tests/meowdoku_studio_product_check.mjs"),
                                str(next(tmp_path.glob("*.manifest.json")))],
                               capture_output=True, text=True, timeout=15)
        assert check.returncode == 0, check.stdout + check.stderr
