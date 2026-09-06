"""Native whole-topology assembly for the compiler-owned balloon tire graph.

The scalar equations remain authored in :mod:`vehicle_balloon_tire`.  This
module emits only their compile-static mesh assembly: face scatter, bead
reduction, persistent state integration, and finite hard-triangle iteration.
It introduces no alternative contact or material law.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.common.tensors.riemann.mesh_laplace import build_cotangent_geometry

from .abstract_ui_vehicles import WHEEL_NAMES, load_default_car_configuration
from .vehicle_balloon_tire import (
    balloon_tire_graph_abi,
    compile_balloon_bead_implicit_step_c,
    compile_balloon_contact_geometry_c,
    compile_balloon_cylinder_contact_geometry_c,
    compile_balloon_contact_impulse_c,
    compile_balloon_gas_c,
    compile_balloon_membrane_face_c,
)


MAX_PLANES_PER_WHEEL = 2


@dataclass(frozen=True, slots=True)
class NativeBalloonTireAssembly:
    name: str
    source: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    state_scalar_count: int
    vertex_count: int
    face_count: int

    def write(self, destination: str | Path) -> Path:
        path = Path(destination).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source, encoding="utf-8")
        return path


def _c_rows(rows: Any, cast=str) -> str:
    return ",\n".join("{" + ",".join(cast(item) for item in row) + "}" for row in rows)


def _f(value: float) -> str:
    return format(float(value), ".17g")


def compile_native_balloon_tire_assembly(
    wheel_names: tuple[str, ...] = WHEEL_NAMES,
    *,
    tire_dimensions: tuple[float, float, float, float, float, float] | None = None,
    pneumatic_mode: str | None = None,
    material_profile: str = "configured",
) -> NativeBalloonTireAssembly:
    """Emit the exact appendage boundary specialized to loaded wheel axes."""

    wheel_names = tuple(map(str, wheel_names))
    if not wheel_names:
        raise ValueError("native balloon tire assembly requires at least one wheel")
    # Keep the generated offset expressions below expressed through one local
    # wheel-axis tuple; the public module constant remains only the default.
    WHEEL_NAMES = wheel_names
    config = copy.deepcopy(load_default_car_configuration().source)
    if tire_dimensions is not None:
        config["tires"]["radius"] = float(tire_dimensions[0])
        config["tires"]["toroid_section_radius_m"] = float(tire_dimensions[1])
        config["tires"]["width"] = float(tire_dimensions[2])
        config["drivetrain"]["tire_mass_kg"] = float(tire_dimensions[3])
        config["tires"]["pressure_pa"] = float(tire_dimensions[4])
        config["wheels"]["rim_radius"] = float(tire_dimensions[5])
    if pneumatic_mode is not None:
        config["tire_skin"]["pneumatic_mode"] = str(pneumatic_mode)
    if material_profile == "cheap-commercial-retread":
        config["tire_skin"].update({
            "material_profile": material_profile,
            "circumferential_segments": 32,
            "section_segments": 24,
        })
    elif material_profile != "configured":
        raise ValueError(f"unknown tire material profile {material_profile!r}")
    abi = balloon_tire_graph_abi(config)
    topology = abi["topology"]
    rest = topology.rest_positions
    faces = topology.faces
    rest_data = topology.face_rest_data
    bending_geometry = build_cotangent_geometry(
        np.asarray(rest), np.asarray(faces),
    )
    bending_edges = tuple(map(tuple, bending_geometry.edges.tolist()))
    bending_weights = tuple(map(float, bending_geometry.cotangent_weights.tolist()))
    bending_areas = tuple(map(float, bending_geometry.lumped_vertex_areas.tolist()))
    bead = tuple(index for ring in topology.bead_rings for index in ring)
    vertex_count = len(rest)
    face_count = len(faces)
    state_stride = 6 * vertex_count
    parameters = abi["parameters"]

    wheel_input_fields = [
        *(f"hub_position_{axis}" for axis in "xyz"),
        *(f"hub_velocity_{axis}" for axis in "xyz"),
        *(f"hub_basis_{local}_{world}" for local in "xyz" for world in "xyz"),
        *(f"hub_angular_velocity_{axis}" for axis in "xyz"),
        "hub_angle_rad", "hub_angular_velocity_z", "surface_kind",
        "cylinder_radius_m",
        "plane_count",
    ]
    for plane in range(MAX_PLANES_PER_WHEEL):
        wheel_input_fields.extend(
            f"plane_{plane}_{quantity}_{axis}"
            for quantity in ("point", "normal", "velocity") for axis in "xyz"
        )
    parameter_names = (
        "vertex_mass_kg", "reference_pressure_pa", "gas_charge_fraction",
        "reference_volume_m3",
        "reference_temperature_k", "gas_polytropic_exponent",
        "gas_molar_mass_kg_per_mol", "gas_specific_heat_ratio",
        "membrane_gas_permeability_mol_m_per_m2_s_pa",
        "gas_permeability_activation_energy_j_per_mol",
        "minimum_volume_fraction", "skin_thickness_m", "lame_lambda_pa",
        "lame_mu_pa", "membrane_damping_lambda_pa_s",
        "membrane_damping_mu_pa_s", "bead_stiffness_n_per_m",
        "bending_stiffness_nm",
        "bead_damping_n_s_per_m", "contact_skin_offset_m",
        "contact_restitution", "friction_coefficient",
    )
    input_names = ("dt", "gravity_y", *parameter_names, *(
        f"{corner}.{field}" for corner in wheel_names for field in wheel_input_fields
    ))
    wheel_output_fields = (
        *(f"rim_force_{axis}_n" for axis in "xyz"),
        *(f"rim_moment_{axis}_nm" for axis in "xyz"),
        "gas_pressure_pa", "volume_ratio", "gas_temperature_k",
        "contact_count", "minimum_skin_y_m", "strain_energy_j",
        "dissipation_power_w", "bending_energy_j",
    )
    output_names = tuple(
        f"{corner}.{field}" for corner in wheel_names for field in wheel_output_fields
    )
    ii = {name: index for index, name in enumerate(input_names)}

    membrane = compile_balloon_membrane_face_c()
    gas = compile_balloon_gas_c()
    bead_kernel = compile_balloon_bead_implicit_step_c()
    geometry = compile_balloon_contact_geometry_c()
    cylinder_geometry = compile_balloon_cylinder_contact_geometry_c()
    impulse = compile_balloon_contact_impulse_c()
    mi = {name: i for i, name in enumerate(membrane.input_names)}
    mo = {name: i for i, name in enumerate(membrane.output_names)}
    gi = {name: i for i, name in enumerate(gas.input_names)}
    go = {name: i for i, name in enumerate(gas.output_names)}
    bi = {name: i for i, name in enumerate(bead_kernel.input_names)}
    bo = {name: i for i, name in enumerate(bead_kernel.output_names)}
    xi = {name: i for i, name in enumerate(geometry.input_names)}
    xo = {name: i for i, name in enumerate(geometry.output_names)}
    cxi = {name: i for i, name in enumerate(cylinder_geometry.input_names)}
    ji = {name: i for i, name in enumerate(impulse.input_names)}
    jo = {name: i for i, name in enumerate(impulse.output_names)}

    defaults = {
        name: (1.0 if name == "gas_charge_fraction" else float(parameters[name]))
        for name in parameter_names
    }
    default_lines = "".join(
        f"    in[{ii[name]}]={_f(value)};\n" for name, value in defaults.items()
    )
    default_lines += "".join(
        f"    in[{ii[f'{corner}.hub_basis_{local}_{world}']}]=1.0;\n"
        for corner in wheel_names for local in "xyz" for world in "xyz"
        if local == world
    )
    scatter_lines = "".join(
        f"            force[tire_faces[f][{local}]][{axis_index}]+="
        f"mout[{mo[f'force_{local}_{axis}_n']}]-mout[{mo[f'damping_force_{local}_{axis}_n']}];\n"
        f"            damping_force[tire_faces[f][{local}]][{axis_index}]+="
        f"mout[{mo[f'damping_force_{local}_{axis}_n']}];\n"
        for local in range(3) for axis_index, axis in enumerate("xyz")
    )
    face_vertex_lines = ""
    for local in range(3):
        for axis_index, axis in enumerate("xyz"):
            face_vertex_lines += (
                f"            minput[{mi[f'x{local}_{axis}']}]=ws[6*tire_faces[f][{local}]+{axis_index}];\n"
                f"            minput[{mi[f'v{local}_{axis}']}]=ws[6*tire_faces[f][{local}]+{3 + axis_index}];\n"
            )
        face_vertex_lines += (
            f"            {{double lx=ca*tire_rest[tire_faces[f][{local}]][0]-sa*tire_rest[tire_faces[f][{local}]][1],ly=sa*tire_rest[tire_faces[f][{local}]][0]+ca*tire_rest[tire_faces[f][{local}]][1],lz=tire_rest[tire_faces[f][{local}]][2];\n"
            f"            minput[{mi[f'r{local}_x']}]=hub_x+basis[0][0]*lx+basis[1][0]*ly+basis[2][0]*lz;\n"
            f"            minput[{mi[f'r{local}_y']}]=hub_y+basis[0][1]*lx+basis[1][1]*ly+basis[2][1]*lz;\n"
            f"            minput[{mi[f'r{local}_z']}]=hub_z+basis[0][2]*lx+basis[1][2]*ly+basis[2][2]*lz;}}\n"
        )
    bead_bind_lines = (
        f"bin[{bi['rim_center_x']}]=hub_x;bin[{bi['rim_center_y']}]=hub_y;bin[{bi['rim_center_z']}]=hub_z;"
        f"double blx=ca*tire_rest[v][0]-sa*tire_rest[v][1],bly=sa*tire_rest[v][0]+ca*tire_rest[v][1],blz=tire_rest[v][2];"
        f"double br[3]={{basis[0][0]*blx+basis[1][0]*bly+basis[2][0]*blz,basis[0][1]*blx+basis[1][1]*bly+basis[2][1]*blz,basis[0][2]*blx+basis[1][2]*bly+basis[2][2]*blz}};"
        f"bin[{bi['target_x']}]=hub_x+br[0];bin[{bi['target_y']}]=hub_y+br[1];bin[{bi['target_z']}]=hub_z+br[2];"
        f"bin[{bi['target_velocity_x']}]=hub_vx+total_omega[1]*br[2]-total_omega[2]*br[1];"
        f"bin[{bi['target_velocity_y']}]=hub_vy+total_omega[2]*br[0]-total_omega[0]*br[2];"
        f"bin[{bi['target_velocity_z']}]=hub_vz+total_omega[0]*br[1]-total_omega[1]*br[0];"
        + "".join(
            f"bin[{bi[f'vertex_{axis}']}]=previous[{axis_index}];bin[{bi[f'free_velocity_{axis}']}]=velocity[{axis_index}];"
            for axis_index, axis in enumerate("xyz")
        ) + "\n"
    )
    bead_result_lines = "".join(
        f"velocity[{axis_index}]=bout[{bo[f'velocity_{axis}_next']}];"
        f"predicted[{axis_index}]=bout[{bo[f'position_{axis}_next']}];"
        f"wo[{axis_index}]+=bout[{bo[f'rim_force_{axis}_n']}];"
        f"wo[{3 + axis_index}]+=bout[{bo[f'rim_moment_{axis}_nm']}];"
        for axis_index, axis in enumerate("xyz")
    )
    geometry_motion_lines = "".join(
        # Geometry is expressed against the current plane.  Translate the
        # previous vertex by plane_velocity*dt so its signed distance is the
        # distance to the previous plane pose; otherwise a moving roller can
        # pass completely through a stationary skin without a CCD crossing.
        f"xin[{xi[f'previous_{axis}']}]=previous[{axis_index}]+dt*pv[{axis_index}];"
        f"xin[{xi[f'current_{axis}']}]=predicted[{axis_index}];"
        for axis_index, axis in enumerate("xyz")
    )
    geometry_triangle_lines = "".join(
        f"xin[{xi[f'triangle_a_{axis}']}]=corner[0][{axis_index}];"
        f"xin[{xi[f'triangle_b_{axis}']}]=corner[tri?2:1][{axis_index}];"
        f"xin[{xi[f'triangle_c_{axis}']}]=corner[tri?3:2][{axis_index}];"
        for axis_index, axis in enumerate("xyz")
    )
    cylinder_geometry_lines = "".join(
        f"cgin[{cxi[f'previous_{axis}']}]=previous[{axis_index}]+dt*pv[{axis_index}];"
        f"cgin[{cxi[f'current_{axis}']}]=predicted[{axis_index}];"
        for axis_index, axis in enumerate("xyz")
    ) + "".join(
        f"cgin[{cxi[f'cylinder_center_{axis}']}]=point[{axis_index}];"
        for axis_index, axis in enumerate("xy")
    ) + (
        f"cgin[{cxi['cylinder_radius_m']}]=in["
        f"{ii[f'{wheel_names[0]}.cylinder_radius_m']}+w*{len(wheel_input_fields)}];"
        f"cgin[{cxi['skin_offset_m']}]=in[{ii['contact_skin_offset_m']}];"
    )
    impulse_bind_lines = "".join(
        f"jin[{ji[f'normal_{axis}']}]=xout[{xo[f'normal_{axis}']}];"
        f"jin[{ji[f'velocity_{axis}']}]=velocity[{axis_index}]-pv[{axis_index}];"
        for axis_index, axis in enumerate("xyz")
    )
    impulse_apply_lines = "".join(
        f"velocity[{axis_index}]+=jout[{jo[f'skin_impulse_{axis}_ns']}]/mass;"
        f"predicted[{axis_index}]=xout[{xo[f'contact_{axis}_m']}]+in[{ii['contact_skin_offset_m']}]*xout[{xo[f'normal_{axis}']}]+(1.0-xout[{xo['time_of_impact_fraction']}])*dt*velocity[{axis_index}];"
        for axis_index, axis in enumerate("xyz")
    )
    source = f'''#include <math.h>
#include <stddef.h>
#include <string.h>
#if defined(_WIN32)
#define TURING_EXPORT __declspec(dllexport)
#else
#define TURING_EXPORT __attribute__((visibility("default")))
#endif
#define TIRE_WHEELS {len(wheel_names)}
#define TIRE_VERTICES {vertex_count}
#define TIRE_FACES {face_count}
#define TIRE_BENDING_EDGES {len(bending_edges)}
#define TIRE_BEADS {len(bead)}
#define TIRE_STATE_STRIDE {state_stride}
#define TIRE_INPUT_COUNT {len(input_names)}
#define TIRE_OUTPUT_STRIDE {len(wheel_output_fields)}
void {membrane.name}(const double*,double*);
void {gas.name}(const double*,double*);
void {bead_kernel.name}(const double*,double*);
void {geometry.name}(const double*,double*);
void {cylinder_geometry.name}(const double*,double*);
void {impulse.name}(const double*,double*);
static const double tire_rest[TIRE_VERTICES][3]={{
{_c_rows(rest, _f)}
}};
static const int tire_faces[TIRE_FACES][3]={{
{_c_rows(faces)}
}};
static const int tire_bending_edges[TIRE_BENDING_EDGES][2]={{
{_c_rows(bending_edges)}
}};
static const double tire_bending_weights[TIRE_BENDING_EDGES]={{{','.join(_f(value) for value in bending_weights)}}};
static const double tire_bending_areas[TIRE_VERTICES]={{{','.join(_f(value) for value in bending_areas)}}};
static const double tire_face_rest[TIRE_FACES][5]={{
{_c_rows(rest_data, _f)}
}};
static const int tire_beads[TIRE_BEADS]={{{','.join(map(str, bead))}}};
static const unsigned char tire_bead_mask[TIRE_VERTICES]={{{','.join('1' if index in set(bead) else '0' for index in range(vertex_count))}}};
static double balloon_contact_debug[16];
TURING_EXPORT void balloon_tire_contact_diagnostics(double *out){{for(int i=0;i<16;++i)out[i]=balloon_contact_debug[i];}}
static double tire_signed_volume(const double *state){{
    double v=0.0; int f;
    for(f=0;f<TIRE_FACES;f++){{
        const double *a=state+6*tire_faces[f][0],*b=state+6*tire_faces[f][1],*c=state+6*tire_faces[f][2];
        v+=(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6.0;
    }} return v;
}}
TURING_EXPORT void balloon_tire_appendage_defaults(double *in){{
    memset(in,0,sizeof(double)*TIRE_INPUT_COUNT);
    in[{ii['dt']}]=1.0/4096.0; in[{ii['gravity_y']}]=-9.81;
{default_lines}}}
TURING_EXPORT void balloon_tire_appendage_initialize(const double *in,double *state){{
    int w,v,a; for(w=0;w<TIRE_WHEELS;w++){{double ca=cos(in[{ii[f'{WHEEL_NAMES[0]}.hub_angle_rad']}+w*{len(wheel_input_fields)}]),sa=sin(in[{ii[f'{WHEEL_NAMES[0]}.hub_angle_rad']}+w*{len(wheel_input_fields)}]),spin=in[{ii[f'{WHEEL_NAMES[0]}.hub_angular_velocity_z']}+w*{len(wheel_input_fields)}],basis[3][3],omega[3];for(a=0;a<3;a++){{basis[0][a]=in[{ii[f'{WHEEL_NAMES[0]}.hub_basis_x_x']}+w*{len(wheel_input_fields)}+a];basis[1][a]=in[{ii[f'{WHEEL_NAMES[0]}.hub_basis_y_x']}+w*{len(wheel_input_fields)}+a];basis[2][a]=in[{ii[f'{WHEEL_NAMES[0]}.hub_basis_z_x']}+w*{len(wheel_input_fields)}+a];omega[a]=in[{ii[f'{WHEEL_NAMES[0]}.hub_angular_velocity_x']}+w*{len(wheel_input_fields)}+a]+spin*basis[2][a];}}for(v=0;v<TIRE_VERTICES;v++){{double *s=state+w*TIRE_STATE_STRIDE+6*v,lx=ca*tire_rest[v][0]-sa*tire_rest[v][1],ly=sa*tire_rest[v][0]+ca*tire_rest[v][1],lz=tire_rest[v][2],r[3];for(a=0;a<3;a++)r[a]=basis[0][a]*lx+basis[1][a]*ly+basis[2][a]*lz;
        s[0]=in[{ii[f'{WHEEL_NAMES[0]}.hub_position_x']}+w*{len(wheel_input_fields)}]+r[0];
        s[1]=in[{ii[f'{WHEEL_NAMES[0]}.hub_position_y']}+w*{len(wheel_input_fields)}]+r[1];
        s[2]=in[{ii[f'{WHEEL_NAMES[0]}.hub_position_z']}+w*{len(wheel_input_fields)}]+r[2];
        for(a=0;a<3;a++)s[3+a]=in[{ii[f'{WHEEL_NAMES[0]}.hub_velocity_x']}+w*{len(wheel_input_fields)}+a];
        s[3]+=omega[1]*r[2]-omega[2]*r[1];s[4]+=omega[2]*r[0]-omega[0]*r[2];s[5]+=omega[0]*r[1]-omega[1]*r[0];
    }}}}
}}
TURING_EXPORT void balloon_tire_appendage_step(const double *in,double *state,double *out){{
    int w,v,f,a,p,tri; const double dt=in[{ii['dt']}],mass=in[{ii['vertex_mass_kg']}];
    memset(out,0,sizeof(double)*TIRE_WHEELS*TIRE_OUTPUT_STRIDE);
    for(w=0;w<TIRE_WHEELS;w++){{balloon_contact_debug[4*w]=1e300;balloon_contact_debug[4*w+1]=1e300;balloon_contact_debug[4*w+2]=0;balloon_contact_debug[4*w+3]=0;}}
    for(w=0;w<TIRE_WHEELS;w++){{
        double *ws=state+w*TIRE_STATE_STRIDE,*wo=out+w*TIRE_OUTPUT_STRIDE;
        const double hub_x=in[{ii[f'{WHEEL_NAMES[0]}.hub_position_x']}+w*{len(wheel_input_fields)}],hub_y=in[{ii[f'{WHEEL_NAMES[0]}.hub_position_y']}+w*{len(wheel_input_fields)}],hub_z=in[{ii[f'{WHEEL_NAMES[0]}.hub_position_z']}+w*{len(wheel_input_fields)}];
        const double hub_vx=in[{ii[f'{WHEEL_NAMES[0]}.hub_velocity_x']}+w*{len(wheel_input_fields)}],hub_vy=in[{ii[f'{WHEEL_NAMES[0]}.hub_velocity_y']}+w*{len(wheel_input_fields)}],hub_vz=in[{ii[f'{WHEEL_NAMES[0]}.hub_velocity_z']}+w*{len(wheel_input_fields)}],hub_omega=in[{ii[f'{WHEEL_NAMES[0]}.hub_angular_velocity_z']}+w*{len(wheel_input_fields)}],ca=cos(in[{ii[f'{WHEEL_NAMES[0]}.hub_angle_rad']}+w*{len(wheel_input_fields)}]),sa=sin(in[{ii[f'{WHEEL_NAMES[0]}.hub_angle_rad']}+w*{len(wheel_input_fields)}]);double basis[3][3],total_omega[3];for(a=0;a<3;a++){{basis[0][a]=in[{ii[f'{WHEEL_NAMES[0]}.hub_basis_x_x']}+w*{len(wheel_input_fields)}+a];basis[1][a]=in[{ii[f'{WHEEL_NAMES[0]}.hub_basis_y_x']}+w*{len(wheel_input_fields)}+a];basis[2][a]=in[{ii[f'{WHEEL_NAMES[0]}.hub_basis_z_x']}+w*{len(wheel_input_fields)}+a];total_omega[a]=in[{ii[f'{WHEEL_NAMES[0]}.hub_angular_velocity_x']}+w*{len(wheel_input_fields)}+a]+hub_omega*basis[2][a];}}
        double force[TIRE_VERTICES][3],damping_force[TIRE_VERTICES][3]; double gin[{len(gas.input_names)}],gout[{len(gas.output_names)}];
        double min_y=1e300,energy=0.0,dissipation=0.0,bending_energy=0.0; memset(force,0,sizeof(force));memset(damping_force,0,sizeof(damping_force));
        gin[{gi['current_volume_m3']}]=tire_signed_volume(ws);
        gin[{gi['gas_polytropic_exponent']}]=in[{ii['gas_polytropic_exponent']}];
        gin[{gi['minimum_volume_fraction']}]=in[{ii['minimum_volume_fraction']}];
        /* Gas charge is independent of the carcass construction prestress.
           This makes evacuation/inflation an actual state experiment instead
           of silently changing both opposing sides of the same equation. */
        gin[{gi['reference_pressure_pa']}]=in[{ii['reference_pressure_pa']}]*fmax(0.0,in[{ii['gas_charge_fraction']}]);
        gin[{gi['reference_temperature_k']}]=in[{ii['reference_temperature_k']}];
        gin[{gi['reference_volume_m3']}]=in[{ii['reference_volume_m3']}];
        {gas.name}(gin,gout);
        for(f=0;f<TIRE_FACES;f++){{
            double minput[{len(membrane.input_names)}],mout[{len(membrane.output_names)}]; int lv;
            memset(minput,0,sizeof(minput));
            minput[{mi['gas_pressure_pa']}]=gout[{go['gas_pressure_pa']}];
            minput[{mi['reference_pressure_pa']}]=in[{ii['reference_pressure_pa']}];
            minput[{mi['lame_lambda_pa']}]=in[{ii['lame_lambda_pa']}]; minput[{mi['lame_mu_pa']}]=in[{ii['lame_mu_pa']}];
            minput[{mi['membrane_damping_lambda_pa_s']}]=in[{ii['membrane_damping_lambda_pa_s']}];
            minput[{mi['membrane_damping_mu_pa_s']}]=in[{ii['membrane_damping_mu_pa_s']}];
            minput[{mi['skin_thickness_m']}]=in[{ii['skin_thickness_m']}];
            minput[{mi['rest_inverse_00']}]=tire_face_rest[f][0]; minput[{mi['rest_inverse_01']}]=tire_face_rest[f][1];
            minput[{mi['rest_inverse_10']}]=tire_face_rest[f][2]; minput[{mi['rest_inverse_11']}]=tire_face_rest[f][3];
            minput[{mi['rest_area_m2']}]=tire_face_rest[f][4];
{face_vertex_lines}
            {membrane.name}(minput,mout); energy+=mout[{mo['strain_energy_j']}]; dissipation+=mout[{mo['dissipation_power_w']}];
{scatter_lines}
        }}
        /* Force outputs are not contiguous by vertex, so scatter them by authored ABI below. */
'''
    source += f'''        /* Passivity limiter for the explicit Kelvin field.  It scales only
           the authored damping impulse, never the elastic/gas force, so a
           timestep cannot turn dissipation into kinetic energy. */
        {{double damping_power_step=0.0,damping_quadratic=0.0,damping_scale=1.0;
          for(v=0;v<TIRE_VERTICES;v++)for(a=0;a<3;a++){{double relative_v=ws[6*v+3+a]-in[{ii[f'{WHEEL_NAMES[0]}.hub_velocity_x']}+w*{len(wheel_input_fields)}+a];damping_power_step+=relative_v*damping_force[v][a];damping_quadratic+=damping_force[v][a]*damping_force[v][a]/mass;}}
          if(damping_quadratic>1e-24){{double passive_limit=-1.9*damping_power_step/(dt*damping_quadratic);damping_scale=fmin(1.0,fmax(0.0,passive_limit));}}
          for(v=0;v<TIRE_VERTICES;v++)for(a=0;a<3;a++)force[v][a]+=damping_scale*damping_force[v][a];
        }}
        /* Conservative thin-shell bending from the established reference
           cotangent Laplace--Beltrami operator.  This is the exact gradient
           of .5*D*integral(|Delta(x-R*r)|^2 dA), not a smoothing force.
           Rubber-only plate theory gives about 1.7 N*m for this skin; the
           live JSON default is the measured reinforced-carcass value. */
        {{double displacement[TIRE_VERTICES][3],laplace_numerator[TIRE_VERTICES][3],dual[TIRE_VERTICES][3];memset(laplace_numerator,0,sizeof(laplace_numerator));
          for(v=0;v<TIRE_VERTICES;v++){{double lx=ca*tire_rest[v][0]-sa*tire_rest[v][1],ly=sa*tire_rest[v][0]+ca*tire_rest[v][1],lz=tire_rest[v][2],reference[3]={{hub_x+basis[0][0]*lx+basis[1][0]*ly+basis[2][0]*lz,hub_y+basis[0][1]*lx+basis[1][1]*ly+basis[2][1]*lz,hub_z+basis[0][2]*lx+basis[1][2]*ly+basis[2][2]*lz}};for(a=0;a<3;a++)displacement[v][a]=ws[6*v+a]-reference[a];}}
          for(int edge=0;edge<TIRE_BENDING_EDGES;edge++){{int left=tire_bending_edges[edge][0],right=tire_bending_edges[edge][1];double weight=tire_bending_weights[edge];for(a=0;a<3;a++){{double flux=weight*(displacement[right][a]-displacement[left][a]);laplace_numerator[left][a]+=flux;laplace_numerator[right][a]-=flux;}}}}
          for(v=0;v<TIRE_VERTICES;v++)for(a=0;a<3;a++){{dual[v][a]=in[{ii['bending_stiffness_nm']}]*laplace_numerator[v][a]/tire_bending_areas[v];bending_energy+=0.5*laplace_numerator[v][a]*dual[v][a];}}
          for(int edge=0;edge<TIRE_BENDING_EDGES;edge++){{int left=tire_bending_edges[edge][0],right=tire_bending_edges[edge][1];double weight=tire_bending_weights[edge];for(a=0;a<3;a++){{double bending_force=weight*(dual[left][a]-dual[right][a]);force[left][a]+=bending_force;force[right][a]-=bending_force;}}}}
        }}
        for(v=0;v<TIRE_VERTICES;v++)force[v][1]+=mass*in[{ii['gravity_y']}];
        for(v=0;v<TIRE_VERTICES;v++){{double *s=ws+6*v,previous[3]={{s[0],s[1],s[2]}},velocity[3],predicted[3];int hit=0;
            for(a=0;a<3;a++){{velocity[a]=s[3+a]+dt*force[v][a]/mass;predicted[a]=previous[a]+dt*velocity[a];}}
            if(tire_bead_mask[v]){{double bin[{len(bead_kernel.input_names)}],bout[{len(bead_kernel.output_names)}];memset(bin,0,sizeof(bin));
                bin[{bi['dt']}]=dt;bin[{bi['vertex_mass_kg']}]=mass;bin[{bi['bead_damping_n_s_per_m']}]=in[{ii['bead_damping_n_s_per_m']}];bin[{bi['bead_stiffness_n_per_m']}]=in[{ii['bead_stiffness_n_per_m']}];
{bead_bind_lines}
                {bead_kernel.name}(bin,bout);{bead_result_lines}
            }}
            for(p=0;p<(int)in[{ii[f'{WHEEL_NAMES[0]}.plane_count']}+w*{len(wheel_input_fields)}];p++){{
                int base={ii[f'{WHEEL_NAMES[0]}.plane_0_point_x']}+w*{len(wheel_input_fields)}+p*9;double point[3]={{in[base],in[base+1],in[base+2]}},normal[3]={{in[base+3],in[base+4],in[base+5]}},pv[3]={{in[base+6],in[base+7],in[base+8]}},t1[3],t2[3],len;
                if(fabs(normal[1])<.9){{t1[0]=-normal[2];t1[1]=0;t1[2]=normal[0];}}else{{t1[0]=0;t1[1]=normal[2];t1[2]=-normal[1];}}len=sqrt(t1[0]*t1[0]+t1[1]*t1[1]+t1[2]*t1[2]);for(a=0;a<3;a++)t1[a]/=len;
                t2[0]=normal[1]*t1[2]-normal[2]*t1[1];t2[1]=normal[2]*t1[0]-normal[0]*t1[2];t2[2]=normal[0]*t1[1]-normal[1]*t1[0];
                int surface_kind=in[{ii[f'{WHEEL_NAMES[0]}.surface_kind']}+w*{len(wheel_input_fields)}]>=0.5,surface_hit=0;
                for(tri=0;tri<(surface_kind?1:2)&&!surface_hit;tri++){{double xin[{len(geometry.input_names)}],cgin[{len(cylinder_geometry.input_names)}],xout[{len(geometry.output_names)}],jin[{len(impulse.input_names)}],jout[{len(impulse.output_names)}],corner[4][3];memset(xin,0,sizeof(xin));memset(cgin,0,sizeof(cgin));
                    if(surface_kind){{{cylinder_geometry_lines}{cylinder_geometry.name}(cgin,xout);}}else{{for(a=0;a<3;a++){{corner[0][a]=point[a]-2.0*t1[a]-2.0*t2[a];corner[1][a]=point[a]+2.0*t1[a]-2.0*t2[a];corner[2][a]=point[a]+2.0*t1[a]+2.0*t2[a];corner[3][a]=point[a]-2.0*t1[a]+2.0*t2[a];}}{geometry_motion_lines}{geometry_triangle_lines}xin[{xi['skin_offset_m']}]=in[{ii['contact_skin_offset_m']}];{geometry.name}(xin,xout);}}balloon_contact_debug[4*w]=fmin(balloon_contact_debug[4*w],xout[{xo['previous_signed_distance_m']}]);balloon_contact_debug[4*w+1]=fmin(balloon_contact_debug[4*w+1],xout[{xo['current_signed_distance_m']}]);if(xout[{xo['previous_signed_distance_m']}]>=-1e-9&&xout[{xo['current_signed_distance_m']}]<=0.0)balloon_contact_debug[4*w+2]+=1.0;if(xout[{xo['barycentric_u']}]>=-1e-10&&xout[{xo['barycentric_v']}]>=-1e-10&&xout[{xo['barycentric_w']}]>=-1e-10)balloon_contact_debug[4*w+3]+=1.0;
                    /* A crossing remains a unilateral contact while the skin is
                       on the forbidden side.  Never make support eligibility
                       depend on penetration depth: one finite-step overshoot
                       must not turn the one-sided terrain off.  The impulse
                       still removes only inward relative velocity; it adds no
                       penetration spring, overlap rejection, or outward kick. */
                    if(xout[{xo['current_signed_distance_m']}]<=0.0&&xout[{xo['barycentric_u']}]>=-1e-10&&xout[{xo['barycentric_v']}]>=-1e-10&&xout[{xo['barycentric_w']}]>=-1e-10){{memset(jin,0,sizeof(jin));jin[{ji['contact_active']}]=1.0;jin[{ji['friction_coefficient']}]=in[{ii['friction_coefficient']}];jin[{ji['inverse_effective_mass_per_kg']}]=1.0/mass;jin[{ji['restitution']}]=in[{ii['contact_restitution']}];
                        {impulse_bind_lines}{impulse.name}(jin,jout);
                        {impulse_apply_lines}surface_hit=1;hit+=1;wo[9]+=1.0;
                    }}
                }}
            }}
            for(a=0;a<3;a++){{s[a]=predicted[a];s[3+a]=velocity[a];}}if(s[1]<min_y)min_y=s[1];
        }}
        wo[6]=gout[{go['gas_pressure_pa']}];wo[7]=gout[{go['volume_ratio']}];wo[8]=gout[{go['gas_temperature_k']}];wo[10]=min_y;wo[11]=energy;wo[12]=dissipation;wo[13]=bending_energy;
    }}
}}
'''
    return NativeBalloonTireAssembly(
        name="balloon_tire_appendage_step",
        source=source,
        input_names=tuple(input_names),
        output_names=output_names,
        state_scalar_count=len(wheel_names) * state_stride,
        vertex_count=vertex_count,
        face_count=face_count,
    )


__all__ = [
    "MAX_PLANES_PER_WHEEL", "NativeBalloonTireAssembly",
    "compile_native_balloon_tire_assembly",
]
