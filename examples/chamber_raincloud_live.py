"""Continuously step the existing LLVM-piece chamber inside its OpenGL view."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

from chamber_dt_join import load_law_module_cached
from chamber_raincloud_record import _col, _load_demo, build_native
from chamber_raincloud_view import main as view
from src.common.chemistry import (
    GAS,
    ComponentChemistry,
    atmospheric_aqueous_nucleus,
    deployment_manifest,
    state_key,
)
from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.common.dt_system.dt_graph import (
    AdvanceNode, GraphBuilder, MetaLoopRunner, StateNode,
)
from src.common.dt_system.engine_api import EngineRegistration
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.error_channels import DT_CHANNEL_NAMES
from src.common.dt_system.state_table import StateTable
from src.common.tensors import AbstractTensor

ENGINE_TOY = HERE.parents[1] / "engine_toy"
SPECTRAL_ANALYZER = HERE.parents[1] / "spectral-analyzer"
sys.path[:0] = [str(ENGINE_TOY), str(SPECTRAL_ANALYZER)]
from air_treatment import ATM_PA, Contaminant, Stream, humidity_ratio  # noqa: E402
from air_volumes import AMBIENT_TEMP_K  # noqa: E402
from dewar_production import atmospheric_port, machine as build_dewar  # noqa: E402
from dc_power import DCBattery  # noqa: E402
from calibres import get_calibre  # noqa: E402
from dt_benchmark import CycleEngine  # noqa: E402
from drivetrain_graph import (  # noqa: E402
    DrivetrainSolver, FluidCircuitInputs, FluidVolume,
)
from engine_cycle_sim import EngineCycleSim  # noqa: E402
from engine_mesh import build_engine_mesh  # noqa: E402
from engine_rays import RayMesh  # noqa: E402
from engines import get as get_engine  # noqa: E402
from machines import MachineSim, MachineSystem  # noqa: E402
from hole_emitters import HoleEmitter  # noqa: E402
from thermal_domains import ThermalSystem, build_thermal_assembly  # noqa: E402
from voxel_ports import project_circular_port  # noqa: E402
from electrical_dt_engine import ComplexElectricalEngine  # noqa: E402
from electrical_tensor_network import (  # noqa: E402
    ElectricalDeviceRegistry, battery_port_law, impedance_port_law,
)
from machine_electrical_graph import register_machine_electrical_graph  # noqa: E402


class OpenAirPort:
    """Pressure-driven fixed and damage-created ports on chamber voxels.

    A physical port remains one identity.  Its face emitters are a derived
    area partition, so an aperture crossing several voxels does not multiply
    its area or mass flux.
    """

    def __init__(self, demo, *, cell_volume_m3: float,
                 shape: tuple[int, int, int], bounds_min_m, bounds_max_m,
                 relative_humidity: float, geometry):
        self.demo = demo
        self.cell_volume_m3 = float(cell_volume_m3)
        self.shape = tuple(int(v) for v in shape)
        self.bounds_min_m = tuple(float(v) for v in bounds_min_m)
        self.bounds_max_m = tuple(float(v) for v in bounds_max_m)
        self.relative_humidity = float(relative_humidity)
        self.mass_flow_kg_s = 0.0
        self.pressure_difference_pa = 0.0
        self.regime = "none"
        self.new_port_count = 0
        self.damage_port_count = 0
        self.damage_area_m2 = 0.0
        self.decompression = False
        self.pressure_drop_rate_pa_s = 0.0
        self._known_damage_ports: set[str] = set()
        self._damage_emitters: dict[str, tuple[HoleEmitter, ...]] = {}
        self.geometry = geometry
        water_ratio = humidity_ratio(AMBIENT_TEMP_K, ATM_PA, self.relative_humidity)
        self.supply = Stream(
            temp_k=AMBIENT_TEMP_K,
            pressure_pa=ATM_PA,
            carried=Contaminant(water_kg_per_kg=water_ratio),
        )
        self.emitters = self._face_emitters(
            self.geometry.identity, self.geometry.part,
            tuple(float(v) for v in self.geometry.position),
            tuple(float(v) for v in self.geometry.direction),
            float(self.geometry.radius_m), kind="open-port")

    def _face_emitters(self, identity, part, position, direction, radius_m, *, kind):
        shares = project_circular_port(
            position, direction, radius_m, shape=self.shape,
            bounds_min_m=self.bounds_min_m, bounds_max_m=self.bounds_max_m)
        return tuple(HoleEmitter(
            identity=f"{identity}.voxel_{share.voxel}", part=part,
            circuit="chamber-air", fluid="gas",
            position=tuple(float(v) for v in position),
            direction=tuple(float(v) for v in direction),
            radius_m=math.sqrt(share.area_m2 / math.pi), through=True,
            wets_surfaces=False, kind=kind,
        ) for share in shares)

    @property
    def face_emitters(self):
        groups = (self.emitters, *self._damage_emitters.values())
        return tuple((int(em.identity.rsplit("_", 1)[-1]), em)
                     for group in groups for em in group)

    def sync_damage_ports(self, machine_sim, chamber_identity: str) -> None:
        """Expose new through punctures as ordinary chamber boundary ports."""
        nodes = {node["identity"]: node for node in machine_sim.graph.get("nodes", ())}
        made = 0
        for part_identity, state in machine_sim.state.part_damage.items():
            if nodes.get(part_identity, {}).get("damage_opens_to") != chamber_identity:
                continue
            for index, puncture in enumerate(state.punctures):
                if not puncture.through:
                    continue
                identity = f"{part_identity}.puncture_{index + 1}.chamber_port"
                if identity in self._known_damage_ports:
                    continue
                points = [puncture.entry_position_m]
                if puncture.exit_position_m is not None:
                    points.append(puncture.exit_position_m)
                position, direction = self._inner_boundary_point(points)
                faces = self._face_emitters(
                    identity, part_identity, position, direction,
                    puncture.radius_m, kind="damage-port")
                if faces:
                    self._damage_emitters[identity] = faces
                    self._known_damage_ports.add(identity)
                    made += 1
        self.new_port_count = made
        self.damage_port_count = len(self._damage_emitters)
        self.damage_area_m2 = sum(
            math.pi * emitter.radius_m ** 2
            for emitters in self._damage_emitters.values()
            for emitter in emitters)

    def _inner_boundary_point(self, points):
        lo = np.asarray(self.bounds_min_m, dtype=float)
        hi = np.asarray(self.bounds_max_m, dtype=float)
        best = None
        for point in points:
            p = np.asarray(point, dtype=float)
            distances = np.concatenate((abs(p - lo), abs(p - hi)))
            choice = int(np.argmin(distances))
            distance = float(distances[choice])
            if best is None or distance < best[0]:
                axis = choice % 3
                normal = np.zeros(3, dtype=float)
                normal[axis] = -1.0 if choice < 3 else 1.0
                best = (distance, tuple(float(v) for v in p), tuple(normal))
        return best[1], best[2]

    def copy_shallow(self):
        return {
            "mass_flow_kg_s": float(self.mass_flow_kg_s),
            "pressure_difference_pa": float(self.pressure_difference_pa),
            "regime": self.regime,
            "new_port_count": int(self.new_port_count),
            "damage_port_count": int(self.damage_port_count),
            "damage_area_m2": float(self.damage_area_m2),
            "decompression": bool(self.decompression),
            "pressure_drop_rate_pa_s": float(self.pressure_drop_rate_pa_s),
            "known_damage_ports": set(self._known_damage_ports),
            "damage_emitters": copy.deepcopy(self._damage_emitters),
            "supply": copy.deepcopy(vars(self.supply)),
            "emitters": tuple(copy.deepcopy(vars(item)) for item in self.emitters),
        }

    def restore(self, snapshot) -> None:
        self.mass_flow_kg_s = float(snapshot["mass_flow_kg_s"])
        self.pressure_difference_pa = float(snapshot["pressure_difference_pa"])
        self.regime = snapshot["regime"]
        self.new_port_count = int(snapshot["new_port_count"])
        self.damage_port_count = int(snapshot["damage_port_count"])
        self.damage_area_m2 = float(snapshot["damage_area_m2"])
        self.decompression = bool(snapshot["decompression"])
        self.pressure_drop_rate_pa_s = float(snapshot["pressure_drop_rate_pa_s"])
        self._known_damage_ports = set(snapshot["known_damage_ports"])
        self._damage_emitters = copy.deepcopy(snapshot["damage_emitters"])
        vars(self.supply).clear()
        vars(self.supply).update(copy.deepcopy(snapshot["supply"]))
        for emitter, saved in zip(self.emitters, snapshot["emitters"]):
            vars(emitter).clear()
            vars(emitter).update(copy.deepcopy(saved))

    def pressure_pa(self, temperature: float, dry_air_kg: float, vapour_kg: float) -> float:
        return (
            dry_air_kg * float(self.demo.WATER["R_a"]) * temperature / self.cell_volume_m3
            + vapour_kg * float(self.demo.WATER["R_v"]) * temperature / self.cell_volume_m3
        )

    def exchange(self, sim, accepted_dt: float) -> None:
        if accepted_dt <= 0.0:
            return
        air = sim.air.state.columns
        water = sim.species[sim.water].state.columns
        temperatures = list(map(float, air["T"].tolist()))
        dry_air = list(map(float, air["m_a"].tolist()))
        vapour = list(map(float, water["m_v"].tolist()))
        cv_a = float(self.demo.WATER["cv_a"])
        cv_v = float(self.demo.WATER["cv_v"])
        water_ratio = self.supply.carried.water_kg_per_kg
        signed_flow = 0.0
        regimes = []
        before_pressures = []

        faces = self.face_emitters
        for voxel, emitter in faces:
            temperature = temperatures[voxel]
            pressure = self.pressure_pa(temperature, dry_air[voxel], vapour[voxel])
            before_pressures.append(pressure)
            emitter.step(
                accepted_dt, pressure_pa=pressure, head_m=0.0,
                remaining_l=1.0, gas_temp_k=temperature)
            regimes.append(emitter.regime)
            transfer_kg = emitter.mass_flow_kg_s * accepted_dt
            if emitter.regime == "ingest" and transfer_kg > 0.0:
                incoming_dry = transfer_kg / (1.0 + water_ratio)
                incoming_vapour = incoming_dry * water_ratio
                old_capacity = dry_air[voxel] * cv_a + vapour[voxel] * cv_v
                incoming_capacity = incoming_dry * cv_a + incoming_vapour * cv_v
                temperatures[voxel] = (
                    old_capacity * temperature + incoming_capacity * self.supply.temp_k
                ) / max(old_capacity + incoming_capacity, 1e-30)
                dry_air[voxel] += incoming_dry
                vapour[voxel] += incoming_vapour
                signed_flow += emitter.mass_flow_kg_s
            elif emitter.regime == "gas" and transfer_kg > 0.0:
                gas_mass = dry_air[voxel] + vapour[voxel]
                removed = min(transfer_kg, gas_mass)
                retained = max(0.0, 1.0 - removed / max(gas_mass, 1e-30))
                dry_air[voxel] *= retained
                vapour[voxel] *= retained
                signed_flow -= removed / accepted_dt

        self.supply.mass_flow_kg_s = sum(
            emitter.ingest_kg_s for _, emitter in faces
            if emitter.regime == "ingest") / (1.0 + water_ratio)
        self.mass_flow_kg_s = signed_flow
        after_pressures = [
            self.pressure_pa(temperatures[voxel], dry_air[voxel], vapour[voxel])
            for voxel, _ in faces]
        self.pressure_difference_pa = (
            float(np.mean(after_pressures)) - self.supply.pressure_pa
            if after_pressures else 0.0)
        active = {regime for regime in regimes if regime != "none"}
        self.regime = active.pop() if len(active) == 1 else ("mixed" if active else "none")
        before = float(np.mean(before_pressures)) if before_pressures else ATM_PA
        after = float(np.mean(after_pressures)) if after_pressures else before
        self.pressure_drop_rate_pa_s = max(0.0, (before - after) / accepted_dt)
        self.decompression = bool(
            (self.new_port_count > 0 and signed_flow < 0.0)
            or (self.damage_port_count > 0
                and self.pressure_drop_rate_pa_s >= 1000.0))
        self.new_port_count = 0
        air["T"] = AbstractTensor.tensor(temperatures)
        air["m_a"] = AbstractTensor.tensor(dry_air)
        water["m_v"] = AbstractTensor.tensor(vapour)


class LiveChamber:
    """The current chamber publication in the viewer's field-source form."""

    def __init__(self, frame_seconds: float,
                 port_radius_m: float, ambient_rh: float,
                 shape: tuple[int, int, int] = (1, 1, 1)):
        self.demo = _load_demo()
        self.laws = load_law_module_cached(HERE / "symbolic_chamber_solvers.py")
        shape = tuple(int(v) for v in shape)
        dx = float(self.demo.GEOMETRY["dx"])
        self.machine_identity = "lab.dewar"
        self.machine = build_dewar(
            self.machine_identity,
            chamber_shape=shape, cell_size_m=dx,
            open_port_radius_m=port_radius_m)
        self.machine_graph = self.machine.production_graph
        self.machine_nodes = {
            node["identity"]: node for node in self.machine_graph["nodes"]
        }
        self.cold_head_identity = f"{self.machine_identity}.cold_head"
        self.compressor_identity = f"{self.cold_head_identity}.compressor"
        self.cold_tip_identity = f"{self.cold_head_identity}.cold_tip"
        nx, ny, nz = shape
        cold_voxel = (nx // 2) + nx * ((ny // 2) + ny * (nz - 1))
        tip_radius = 0.045
        tip_depth = min(0.135, dx)
        cold_area = math.pi * tip_radius ** 2 + 2.0 * math.pi * tip_radius * tip_depth
        self.sim = build_native(
            self.demo, self.laws, shape,
            # Initial condition only.  Every atmosphere step below reads the
            # cold-tip body's live temperature from the thermal engine.
            plate_temperature=AMBIENT_TEMP_K,
            top_surface_areas={cold_voxel: cold_area},
        )
        self.water = self.sim.species[self.sim.water]
        chamber_node = self.machine_nodes[f"{self.machine_identity}.chamber_volume"]
        chamber_position = np.asarray(chamber_node["reference_position"], dtype=float)
        chamber_half = np.asarray(chamber_node["body_half_extent_m"], dtype=float)
        self.port = OpenAirPort(
            self.demo, cell_volume_m3=self.sim.dx ** 3, shape=shape,
            bounds_min_m=chamber_position - chamber_half,
            bounds_max_m=chamber_position + chamber_half,
            geometry=atmospheric_port(
                self.machine_identity, chamber_shape=shape, cell_size_m=dx,
                open_port_radius_m=port_radius_m),
            relative_humidity=ambient_rh,
        )
        self.sim.boundary_state = self.port
        self.targets = Targets(
            cfl=self.demo.DT["cfl"], div_max=self.demo.DT["div_max"],
            mass_max=self.demo.DT["mass_max"],
            energy_exchange_fraction=self.demo.DT["energy_exchange_fraction"],
        )
        self.controller = STController(dt_min=1e-9)
        self.frame_seconds = float(frame_seconds)
        self.time = 0.0
        self.dt = min(float(self.demo.DT["dt_initial"]), self.frame_seconds)
        self.n_frames = 1
        publications = {
            row.output: {"law": law, "semantic": row.semantic, "unit": row.unit}
            for law, rows in self.laws.LAW_PUBLICATIONS.items() for row in rows
        }
        self.by_semantic = {v["semantic"]: k for k, v in publications.items()}
        self.nx, self.ny, self.nz = self.sim.shape
        self.dx = float(self.sim.dx)
        self.machine_sim = MachineSim(machine=self.machine)
        self.machine_system = MachineSystem(self.machine_sim)
        # The service-platform prime mover is the same engine system used by
        # the game and validator.  CycleEngine is its existing dt-system ABI;
        # it carries the full engine state span through rollback.
        self.prime_mover = EngineCycleSim(
            engine=get_engine("ldt465-multifuel-deuce"))
        self.prime_mover.start()
        self.prime_mover.throttle = 0.55
        self.cycle_engine = CycleEngine(
            self.prime_mover, label="engine-cycle-system")
        self.drivetrain = DrivetrainSolver(self.machine_graph)
        self.fluid_system = self.drivetrain.fluid_system
        self.machine_sim.bind_fluid_circuits(self.fluid_system.circuits)
        self.thermal = build_thermal_assembly(
            self.machine_graph,
            initial_temperature_k=AMBIENT_TEMP_K,
            temperatures_by_group={
                f"{self.machine_identity}.warm_jacket": AMBIENT_TEMP_K,
                f"{self.cold_head_identity}.warm_jacket": AMBIENT_TEMP_K,
                f"{self.cold_head_identity}.cold_end": AMBIENT_TEMP_K,
                f"{self.compressor_identity}.compressor": AMBIENT_TEMP_K,
                f"{self.compressor_identity}.radiator": AMBIENT_TEMP_K,
            },
        )
        self.electrical_registry = ElectricalDeviceRegistry(
            [0.0, 50.0, 60.0, 1_000.0])
        electrical_service = self.machine_nodes[
            f"{self.machine_identity}.battery"]["electrical_service"]

        def bus(part: str, role: str) -> str:
            return f"{part}::{electrical_service}::{role}"

        battery_node = self.machine_nodes[f"{self.machine_identity}.battery"]
        self.site_battery = DCBattery(
            identity=battery_node["identity"],
            chemistry=battery_node["chemistry"],
            capacity_ah=float(battery_node["capacity_ah"]),
            series_packs=int(battery_node["series_packs"]),
            state_of_charge=float(battery_node["state_of_charge"]),
        )
        motor_node = self.machine_nodes[f"{self.compressor_identity}.motor"]
        fan_node = self.machine_nodes[f"{self.compressor_identity}.fan"]
        voltage_v = float(battery_node["nominal_voltage_v"])
        motor_resistance = voltage_v ** 2 / float(
            motor_node["rated_electrical_power_w"])
        fan_resistance = voltage_v ** 2 / float(fan_node["rated_w"])
        device_laws = {
            battery_node["identity"]: battery_port_law(
                battery_node["identity"],
                bus(battery_node["identity"], "positive"),
                self.site_battery,
                return_node=bus(battery_node["identity"], "return"),
            ),
            motor_node["identity"]: impedance_port_law(
                motor_node["identity"],
                bus(motor_node["identity"], "positive"),
                bus(motor_node["identity"], "return"),
                [motor_resistance] * self.electrical_registry.lanes,
            ),
            fan_node["identity"]: impedance_port_law(
                fan_node["identity"],
                bus(fan_node["identity"], "positive"),
                bus(fan_node["identity"], "return"),
                [fan_resistance] * self.electrical_registry.lanes,
            ),
        }
        register_machine_electrical_graph(
            self.electrical_registry, self.machine_graph,
            device_laws=device_laws)
        # This scene is the native integration demonstration.  A silent
        # Python fallback made ordinary runs exercise a different numerical
        # backend from the one the scene claims to demonstrate.
        self.electrical_registry.compile_llvm_solver(
            HERE.parent / "build" / "chamber_electrical_llvm")
        self.electrical_registry.thermal_target_by_branch[
            motor_node["identity"]] = motor_node["identity"]
        self.electrical_registry.thermal_fraction_by_branch[
            motor_node["identity"]] = max(
                0.0, 1.0 - float(motor_node["efficiency"]))
        self.electrical_system = ComplexElectricalEngine(
            self.electrical_registry, label="electrical-system")
        self.electrical_system.bind_thermal_assembly(self.thermal)
        self.dt_table = StateTable()
        self.sim.register(
            self.dt_table,
            lambda _system: {"pos": (0.0, 0.0, 0.0), "mass": 0.0},
            (self.sim,), group_label="atmosphere-system")
        self.machine_system.register(
            self.dt_table,
            lambda _system: {"pos": (0.0, 0.0, 0.0), "mass": 0.0},
            (self.machine_system,), group_label="machine-system")
        # The voxel chamber is one fluid residence domain owned by atmosphere.
        # Registering it does not transfer its voxel state to the circuit
        # solver; it makes the shared fluid topology explicit.
        self.fluid_system.register_volume(FluidVolume(
            identity=chamber_node["identity"],
            volume_m3=float(chamber_node["fluid_volume_l"]) / 1000.0,
            fluid="gas", pressure_pa=ATM_PA, temperature_k=AMBIENT_TEMP_K,
            state_owner="atmosphere", spatial_model="voxel"))
        self.fluid_system.register(
            self.dt_table,
            lambda _system: {"pos": (0.0, 0.0, 0.0), "mass": 0.0},
            (self.fluid_system,), group_label="fluid-system")
        self.thermal_system = ThermalSystem((self.thermal,), source_w=self._thermal_sources)
        self.thermal_system.register(
            self.dt_table,
            lambda _system: {"pos": (0.0, 0.0, 0.0), "mass": 0.0},
            (self.thermal_system,), group_label="thermal-system")
        self.cycle_engine.register(
            self.dt_table,
            lambda _system: {"pos": (0.0, 0.0, 0.0), "mass": 0.0},
            (self.cycle_engine,), group_label="engine-cycle-system")
        self.electrical_system.register(
            self.dt_table,
            lambda _system: {"pos": (0.0, 0.0, 0.0), "mass": 0.0},
            (self.electrical_system,), group_label="electrical-system")
        thermal_registration = EngineRegistration(
            name="thermal-system", engine=self.thermal_system,
            targets=self.targets, dx=self.dx, localize=False)
        fluid_registration = EngineRegistration(
            name="fluid-system", engine=self.fluid_system,
            targets=self.targets, dx=self.dx, localize=False)
        cycle_registration = EngineRegistration(
            name="engine-cycle-system", engine=self.cycle_engine,
            targets=self.targets, dx=self.dx, localize=False)
        electrical_registration = EngineRegistration(
            name="electrical-system", engine=self.electrical_system,
            targets=self.targets, dx=self.dx, localize=False)
        inner_builder = GraphBuilder(
            ctrl=STController(dt_min=1e-12), targets=self.targets, dx=self.dx)
        self.system_round = inner_builder.round(
            dt=self.frame_seconds,
            engines=[cycle_registration, fluid_registration,
                     electrical_registration, thermal_registration],
            state_table=self.dt_table)
        self.system_round.children[0:0] = [
            AdvanceNode(self._advance_atmosphere, StateNode(None),
                        label="advance:atmosphere-system", transaction_owner=self.sim),
            AdvanceNode(self._advance_machine_system, StateNode(None),
                        label="advance:machine-system", transaction_owner=self.machine_sim),
        ]
        self.dt_round = self.system_round
        # The existing realtime dt path is the slipping contract: every
        # top-level engine sees the requested interval and advances only to
        # its own causal ceiling. No child engine pins another child's clock.
        self.dt_runner = MetaLoopRunner(
            realtime=True, state_table=self.dt_table)
        self.dt_runner.set_process_graph(self.dt_round)
        self.dt_profile = {
            label: {
                "calls": 0,
                "wall_s": 0.0,
                "wall_max_s": 0.0,
                "requested_s": 0.0,
                "advanced_s": 0.0,
                "slip_s": 0.0,
            }
            for label in self.dt_runner.get_schedule_labels()
        }
        self.last_dt_profile = []
        chemistry = atmospheric_aqueous_nucleus()
        atmospheric_states = tuple(
            state_key(species, GAS)
            for species in ("H2O", "CO2", "N2", "O2", "Ar")
        )
        self.chemistry_manifest = deployment_manifest(chemistry, (
            ComponentChemistry("outside.atmosphere", atmospheric_states),
        ))
        self.chemistry_plan = self.chemistry_manifest.dewar_packing_plan(shape)
        self.chemistry_exchange = self.chemistry_manifest.engine_exchange_layout((
            "outside.atmosphere", "chamber.open_port",
            "cold_head.supply", "drain.receiver",
        ))
        self.meta = {
            "shape": [self.nx, self.ny, self.nz], "dx": self.dx,
            "publications": publications,
            "surfaces": [{"voxel": int(s.voxel), "pool": s.pool,
                          "catches_rain": bool(s.catches_rain),
                          "T_plate": float(s.params["T_plate"])}
                         for s, _ in self.sim.surfaces],
            "pools": [{"voxel": int(s.voxel)} for s, _ in self.sim.pools],
            "chemistry": {
                "abi": self.chemistry_manifest.abi_identity,
                "states": list(self.chemistry_manifest.states),
                "species": list(self.chemistry_manifest.species),
                "reactions": list(self.chemistry_manifest.reactions),
                "owners": self.chemistry_plan.owner_count,
                "boundaries": self.chemistry_plan.boundary_count,
                "precision_limbs": self.chemistry_plan.limbs,
                "inventory_bytes": self.chemistry_plan.physical_inventory_size * 8,
                "exchange_bytes": self.chemistry_exchange.physical_size * 8,
                "mode": "closure-packed; chamber phase laws active; master solve pending",
            },
            "dt_mode": "fully-slipping-realtime",
        }
        self.arrays: dict[str, np.ndarray] = {}
        self._publish()

    def field(self, name: str, frame: int):
        return self.arrays[name][0].reshape(self.nz, self.ny, self.nx)

    def frame_at(self, _time: float) -> int:
        return 0

    def _put(self, name: str, value):
        self.arrays[name] = np.asarray(value, dtype=np.float64).reshape(1, -1)

    def _publish(self):
        outputs = self.water.state.outputs
        self._put("t", self.time)
        self._put("dt", self.dt)
        self._put("T", _col(self.sim.air.state.columns["T"]))
        for name in ("m_a",):
            self._put(name, _col(self.sim.air.state.columns[name]))
        for name in ("m_v", "m_l", "m_i", "m_r"):
            self._put(name, _col(self.water.state.columns[name]))
        volume = self.sim.dx ** 3
        temperature = _col(self.sim.air.state.columns["T"])
        pressure = (_col(self.sim.air.state.columns["m_a"]) * self.demo.WATER["R_a"] * temperature / volume
                    + _col(self.water.state.columns["m_v"]) * self.demo.WATER["R_v"] * temperature / volume)
        self._put("P", pressure)
        self._put("port_mass_flow", self.port.mass_flow_kg_s)
        self._put("port_dp", self.port.pressure_difference_pa)
        self._put("port_regime", {"none": 0.0, "ingest": 1.0, "gas": -1.0, "mixed": 2.0}[self.port.regime])
        self._put("port_radius", self.port.geometry.radius_m)
        self._put("ambient_rh", self.port.relative_humidity)
        self._put("damage_port_count", self.port.damage_port_count)
        self._put("damage_port_area", self.port.damage_area_m2)
        self._put("decompression", float(self.port.decompression))
        self._put("pressure_drop_rate", self.port.pressure_drop_rate_pa_s)
        self._put("fluid_system_time", self.fluid_system.world_time)
        self._put("thermal_system_time", self.thermal_system.world_time)
        self._put("electrical_system_time", self.electrical_system.world_time)
        self._put("engine_system_time", self.cycle_engine.world_time)
        self._put("engine_rpm", self.prime_mover.rpm)
        self._put("engine_power", self.prime_mover.state.power_kw * 1000.0)
        self._put("engine_battery_soc", self.prime_mover.state.battery_soc_frac)
        self._put("engine_battery_voltage", self.prime_mover.state.battery_voltage)
        self._put("engine_alternator_power",
                  self.prime_mover.state.alternator_current_a
                  * self.prime_mover.state.battery_voltage)
        self._put("site_battery_soc", self.site_battery.state_of_charge)
        electrical_reading = self.electrical_system.reading
        motor_identity = f"{self.compressor_identity}.motor"
        motor_current = (0.0 if electrical_reading is None else float(
            np.sqrt(sum(abs(complex(value)) ** 2 for value in
                        electrical_reading.device_current_a[
                            motor_identity].tolist()))))
        self._put("compressor_motor_current", motor_current)
        motor_input_w = self.electrical_system.branch_loss_w.get(
            motor_identity, 0.0)
        self._put("electrical_heat", sum(
            self.electrical_system.joule_heat_w.values()))
        self._put("electrical_safety_events",
                  len(self.electrical_system.safety_events))
        fluid_power = self.fluid_system.component_power_w
        process_loop = next((circuit for circuit in self.fluid_system.circuits
                             if circuit.process_resolved), None)
        self._put("working_gas_mass_flow", 0.0 if process_loop is None
                  else process_loop.delivered_flow_kg_s)
        self._put("compressor_shaft_power", fluid_power.get(
            "compressor_shaft_input_w", 0.0))
        self._put("expander_shaft_power", fluid_power.get(
            "expander_shaft_output_w", 0.0))
        self._put("cold_head_heat_removed", fluid_power.get(
            "cold_tip_heat_removed_w", 0.0))
        self._put("radiator_rejection", fluid_power.get(
            "radiator_ambient_rejection_w", 0.0))
        atmosphere = self.dt_table.get(
            "coupling", self.machine_identity, "atmosphere") or {}
        vacuum = self.machine_nodes[f"{self.machine_identity}.jacket_vacuum"]
        gas = self.machine_nodes[f"{self.machine_identity}.working_gas_bottle"]
        drain = self.machine_nodes[f"{self.machine_identity}.drain_tank"]
        floor_drain = self.machine_nodes[f"{self.machine_identity}.floor_drain"]
        self._put("cold_tip_temperature", self.thermal.temperature_k(self.cold_tip_identity))
        self._put("cold_tip_heat", atmosphere.get("cold_tip_heat_w", 0.0))
        self._put("jacket_heat_leak", float("nan"))
        self._put("vacuum_pressure", vacuum["pressure_pa"])
        self._put("vacuum_intact", float(vacuum["vacuum_intact"]))
        self._put("battery_soc", self.site_battery.state_of_charge)
        compressor_parts = [
            state for state in self.thermal.states.values()
            if state.domain.group == f"{self.compressor_identity}.compressor"
        ]
        compressor_temperature = (
            sum(state.energy_j for state in compressor_parts)
            / sum(state.total_capacity_j_k for state in compressor_parts)
            if compressor_parts else AMBIENT_TEMP_K)
        self._put("compressor_temperature", compressor_temperature)
        self._put("compressor_fan", 1.0 if motor_input_w > 0.0 else 0.0)
        self._put("compressor_power", motor_input_w)
        self._put("working_gas_fill", gas.get("held_kg", 0.0)
                  / max(gas["capacity_kg"], 1e-30))
        self._put("drain_tank_fill", drain.get("held_kg", 0.0)
                  / max(drain["capacity_kg"], 1e-30))
        self._put("drain_blocked", floor_drain.get("blocked_frac", 0.0))
        self._put("maintenance_count", 0.0)
        exchange = self.dt_table.get(
            "coupling", self.machine_identity, "exchange") or {}
        self._put("machine_elapsed", self.machine_sim.elapsed_s)
        self._put("machine_requested", exchange.get("requested_s", 0.0))
        self._put("machine_accepted", exchange.get("accepted_s", 0.0))
        self._put("S", _col(outputs["S"]) if "S" in outputs else [1.0])
        for name in ("LWC", "IWC", "RWC", "rain_out", "drizzle_out", "snow_out"):
            self._put(name, _col(outputs[name]) if name in outputs else [0.0])
        for index, (_spec, engine) in enumerate(self.sim.surfaces):
            for name in ("h_film", "h_frost"):
                value = engine.state.outputs.get(name)
                self._put(f"surface{index}.{name}", _col(value) if value is not None else [0.0])
        for index, (_spec, engine) in enumerate(self.sim.pools):
            for name in ("A_wet", "h_pool"):
                value = engine.state.outputs.get(name)
                self._put(f"pool{index}.{name}", _col(value) if value is not None else [0.0])
        self.t = np.array([self.time], dtype=np.float64)

    def _advance_atmosphere(self, state, dt: float, *, realtime=False,
                            state_table=None):
        self.port.sync_damage_ports(
            self.machine_sim, f"{self.machine_identity}.chamber_volume")
        boundary = {
            "cold_tip_temperature_k": self.thermal.temperature_k(
                self.cold_tip_identity),
        }
        cold_surface = self.sim.surfaces[0][1]
        cold_surface.params["T_plate"] = AbstractTensor.tensor(
            boundary["cold_tip_temperature_k"])
        ok, metrics, _ = self.sim.step_with_state(
            self.sim.state, float(dt), realtime=bool(realtime),
            state_table=state_table)
        accepted_dt = float(metrics.advanced_dt or 0.0)
        if ok:
            self.port.exchange(self.sim, accepted_dt)
        report = {
            "requested_s": float(dt),
            "accepted_s": accepted_dt if ok else 0.0,
            "chamber_mean_k": float(np.mean(
                _col(self.sim.air.state.columns["T"]))),
            "cold_tip_heat_w": 0.0,
            "cold_tip_frost_kg": 0.0,
            "drained_water_kg": 0.0,
        }
        if ok:
            surface_temperature = float(cold_surface.state.columns["T_s"].tolist()[0])
            u_eff = float(cold_surface.state.outputs["U_eff"].tolist()[0])
            area = float(np.asarray(cold_surface.params["A_s"].tolist()).reshape(-1)[0])
            report["cold_tip_heat_w"] = u_eff * area * (
                surface_temperature - boundary["cold_tip_temperature_k"])
            report["cold_tip_frost_kg"] = float(
                np.sum(_col(cold_surface.state.columns["m_frost"])))
            for _pool_spec, pool_engine in self.sim.pools:
                rate = pool_engine.state.outputs.get("overflow_rate")
                if rate is not None:
                    report["drained_water_kg"] += (
                        max(0.0, float(np.sum(_col(rate)))) * accepted_dt)
        state_table.set("coupling", self.machine_identity, "atmosphere", report)
        return bool(ok), metrics, state

    def apply_damage_ray(self, ray, calibre_name: str):
        """Route viewer input through the common machine damage path."""
        calibre = get_calibre(calibre_name)
        projectile = calibre.projectile(ray.direction)
        ray_mesh = RayMesh(*build_engine_mesh(
            self.machine_graph, crank_angle_deg=0.0, covers_off=False))
        penetration = ray_mesh.penetrate(
            ray, energy_j=projectile.energy_j,
            calibre_m=calibre.diameter_m, projectile=projectile)
        recorded = self.machine_sim.apply_penetration(penetration)
        return recorded, penetration

    def _advance_machine_system(self, state, dt: float, *, realtime=False,
                                state_table=None):
        ok, metrics, _ = self.machine_system.step_with_state(
            self.machine_sim, float(dt), realtime=bool(realtime),
            state_table=state_table)
        accepted_dt = float(metrics.advanced_dt or 0.0)
        # The existing fluid engine resolves the graph-declared compressor,
        # pipes, recuperator, expander and cold-tip exchanger.  It receives
        # measured electrical power and live thermal boundary temperatures;
        # no aggregate cold-head object owns a second state or clock.
        self.fluid_system.stage(FluidCircuitInputs(
            electrical_power_w=dict(self.electrical_system.branch_loss_w),
            thermal_temperature_k={
                identity: thermal_state.mean_temperature_k
                for identity, thermal_state in self.thermal.states.items()
            },
        ))
        atmosphere = state_table.get(
            "coupling", self.machine_identity, "atmosphere") or {}
        state_table.set("coupling", self.machine_identity, "exchange", {
            **atmosphere,
            "requested_s": float(dt),
            "accepted_s": accepted_dt,
        })
        return bool(ok), metrics, state

    def _thermal_sources(self, _index, _assembly):
        atmosphere = self.dt_table.get(
            "coupling", self.machine_identity, "atmosphere") or {}
        sources = {self.cold_tip_identity: float(
            atmosphere.get("cold_tip_heat_w", 0.0))}
        for identity, watts in self.electrical_system.thermal_source_w(
                assembly=_assembly).items():
            sources[identity] = sources.get(identity, 0.0) + float(watts)
        for identity, watts in self.fluid_system.thermal_source_w().items():
            if identity in _assembly.states:
                sources[identity] = sources.get(identity, 0.0) + float(watts)
        return sources

    def step(self):
        self.dt_round.plan.dt_init = min(
            max(float(self.dt), 1e-30), self.frame_seconds)
        result = self.dt_runner.run_round(
            dt=self.frame_seconds, realtime=True, state_table=self.dt_table)
        self.last_dt_profile = []
        slip_index = DT_CHANNEL_NAMES.index("time_slip")
        for label, wall_s in zip(
                self.dt_runner.get_schedule_labels(),
                self.dt_runner.get_last_schedule_timings()):
            requested_s = float(
                self.dt_table.get("dt_tape", label, "dt") or 0.0)
            metrics = self.dt_table.get("dt_tape", label, "metrics")
            advanced_s = requested_s
            slip_s = 0.0
            if metrics is not None:
                if metrics.advanced_dt is not None:
                    advanced_s = float(metrics.advanced_dt)
                elif metrics.hard_failure:
                    # A refused engine step has no accepted time.  Treating
                    # an absent advanced_dt as the requested interval hid the
                    # chamber's cascade pause while every later profile row
                    # claimed lockstep progress.
                    advanced_s = 0.0
                present = metrics.error_present.tolist()
                if slip_index < len(present) and present[slip_index]:
                    slip_s = float(metrics.error_channels.tolist()[slip_index])
                else:
                    slip_s = max(0.0, requested_s - advanced_s)
            aggregate = self.dt_profile[label]
            aggregate["calls"] += 1
            aggregate["wall_s"] += float(wall_s)
            aggregate["wall_max_s"] = max(
                aggregate["wall_max_s"], float(wall_s))
            aggregate["requested_s"] += requested_s
            aggregate["advanced_s"] += advanced_s
            aggregate["slip_s"] += slip_s
            self.last_dt_profile.append({
                "name": self._profile_name(label),
                "requested_s": requested_s,
                "advanced_s": advanced_s,
                "slip_s": slip_s,
                "wall_s": float(wall_s),
                "sim_time_s": aggregate["advanced_s"],
                "accumulated_slip_s": aggregate["slip_s"],
            })
        self.dt = float(self.frame_seconds)
        self.time += float(self.frame_seconds)
        self._publish()

    @staticmethod
    def _json_number(value):
        value = float(value)
        return value if math.isfinite(value) else None

    @staticmethod
    def _profile_name(label: str) -> str:
        name = label.removeprefix("advance:").split("/")[-1]
        stem, separator, suffix = name.rpartition("_")
        return stem if separator and suffix.isdigit() else name

    def headless_record(self, step_index: int) -> dict:
        """Return reporting data from the current physical and dt state."""

        def scalar(name, reducer=lambda values: values[0]):
            values = np.asarray(self.arrays[name][0], dtype=np.float64)
            return self._json_number(reducer(values))

        return {
            "kind": "state",
            "step": int(step_index),
            "requested_time_s": float(self.time),
            "electrical_nodal_backend": self.electrical_system.solve_backend,
            "state": {
                "chamber_temperature_mean_k": scalar("T", np.mean),
                "chamber_temperature_min_k": scalar("T", np.min),
                "chamber_pressure_mean_pa": scalar("P", np.mean),
                "cold_tip_temperature_k": scalar("cold_tip_temperature"),
                "water_vapour_kg": scalar("m_v", np.sum),
                "liquid_water_kg": scalar("m_l", np.sum),
                "ice_kg": scalar("m_i", np.sum),
                "rain_kg": scalar("m_r", np.sum),
                "open_port_mass_flow_kg_s": scalar("port_mass_flow"),
                "open_port_pressure_difference_pa": scalar("port_dp"),
                "working_gas_mass_flow_kg_s": scalar("working_gas_mass_flow"),
                "compressor_shaft_power_w": scalar("compressor_shaft_power"),
                "expander_shaft_power_w": scalar("expander_shaft_power"),
                "cold_tip_heat_removed_w": scalar("cold_head_heat_removed"),
                "radiator_rejection_w": scalar("radiator_rejection"),
                "compressor_temperature_k": scalar("compressor_temperature"),
                "site_battery_soc": scalar("site_battery_soc"),
                "jacket_pressure_pa": scalar("vacuum_pressure"),
            },
            "dt_sims": list(self.last_dt_profile),
        }

    def headless_profile_summary(self, wall_s: float) -> dict:
        rows = []
        for label in self.dt_runner.get_schedule_labels():
            values = self.dt_profile[label]
            calls = int(values["calls"])
            rows.append({
                "name": self._profile_name(label),
                "calls": calls,
                "wall_total_s": values["wall_s"],
                "wall_mean_s": values["wall_s"] / max(calls, 1),
                "wall_max_s": values["wall_max_s"],
                "requested_s": values["requested_s"],
                "advanced_s": values["advanced_s"],
                "slip_s": values["slip_s"],
            })
        return {
            "kind": "profile",
            "steps": max((int(row["calls"]) for row in rows), default=0),
            "requested_time_s": float(self.time),
            "wall_s": float(wall_s),
            "realtime_factor": float(self.time) / max(float(wall_s), 1e-30),
            "electrical_nodal_backend": self.electrical_system.solve_backend,
            "dt_sims": rows,
        }


def run_headless(live: LiveChamber, *, steps: int, log_every: int,
                 log_file: Path | None = None,
                 profile_file: Path | None = None) -> dict:
    """Advance through the existing dt runner and emit structured telemetry."""

    if steps <= 0:
        raise ValueError("--steps must be positive")
    if log_every <= 0:
        raise ValueError("--log-every must be positive")
    stream = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stream = log_file.open("w", encoding="utf-8")

    def emit(record):
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        print(line, flush=True)
        if stream is not None:
            stream.write(line + "\n")
            stream.flush()

    started = time.perf_counter()
    completed_steps = 0

    def finish(completed: bool, failure: dict | None = None):
        summary = live.headless_profile_summary(time.perf_counter() - started)
        summary["completed"] = bool(completed)
        if failure is not None:
            summary["failure"] = failure
        emit(summary)
        if profile_file is not None:
            profile_file.parent.mkdir(parents=True, exist_ok=True)
            profile_file.write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
                + "\n", encoding="utf-8")
        return summary

    try:
        for index in range(1, steps + 1):
            live.step()
            completed_steps = index
            if index == 1 or index % log_every == 0 or index == steps:
                emit(live.headless_record(index))
        return finish(True)
    except Exception as exc:
        failure = {
            "step": completed_steps + 1,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        emit({"kind": "failure", **failure})
        finish(False, failure)
        raise
    finally:
        if stream is not None:
            stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-seconds", type=float, default=0.02,
                        help="simulation time advanced per displayed update")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--port-radius", type=float, default=0.003,
                        help="radius in m of the pressure-driven open top port")
    parser.add_argument("--ambient-rh", type=float, default=0.60,
                        help="relative humidity of replacement air")
    parser.add_argument("--shape", type=int, default=1,
                        help="cubic chamber voxel extent; 1 is the currently runnable live artifact")
    parser.add_argument("--headless", action="store_true",
                        help="run the registered dt systems without creating a viewer")
    parser.add_argument("--steps", type=int, default=100,
                        help="number of dt frames in headless mode")
    parser.add_argument("--log-every", type=int, default=10,
                        help="emit one structured state record every N headless frames")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="optional JSONL copy of headless state and profile records")
    parser.add_argument("--profile-file", type=Path, default=None,
                        help="optional JSON file containing the final dt profile")
    parser.add_argument("--snapshot", type=str, default=None,
                        help="write one visible frame to this image")
    parser.add_argument("--machine-snapshot", type=str, default=None,
                        help="after a headless run, render the same state with atmosphere hidden")
    parser.add_argument("--exit-after", type=float, default=None,
                        help="close the viewer after this many wall seconds")
    parser.add_argument("--hide-atmosphere", action="store_true",
                        help="advance the chamber without drawing its atmosphere")
    args = parser.parse_args(argv)
    live = LiveChamber(args.frame_seconds, args.port_radius, args.ambient_rh,
                       shape=(args.shape, args.shape, args.shape))
    if args.headless:
        run_headless(
            live, steps=args.steps, log_every=args.log_every,
            log_file=args.log_file, profile_file=args.profile_file)
        if args.snapshot is not None:
            view(["--snapshot", args.snapshot, "--exit-after", "0.5"],
                 recording=live)
        if args.machine_snapshot is not None:
            view(["--snapshot", args.machine_snapshot, "--exit-after", "0.5",
                  "--hide-atmosphere"], recording=live)
        return
    viewer_args = ["--gain", str(args.gain)]
    if args.snapshot is not None:
        viewer_args.extend(("--snapshot", args.snapshot))
    if args.exit_after is not None:
        viewer_args.extend(("--exit-after", str(args.exit_after)))
    if args.hide_atmosphere:
        viewer_args.append("--hide-atmosphere")
    view(viewer_args, recording=live, live_step=live.step)


if __name__ == "__main__":
    main()
