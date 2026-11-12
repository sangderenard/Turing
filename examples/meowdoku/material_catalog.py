"""Authored scene materials for the shared spectral MaterialDatabase.

The names are stable browser material IDs.  Every record carries the PBR
surface fields and all four authored Phong properties; MaterialDatabase also
derives the enamel, texture-stack, and ray records from these same entries.
"""

from __future__ import annotations


def _surface(albedo, *, roughness, metallic=0.0, transmission=0.0,
             ior=1.5, opacity=1.0, emission=(0.0, 0.0, 0.0), ambient=0.0,
             spec_strength=0.5, shininess=80.0, grain=0.0, inner_color=None):
    return {
        "albedo": albedo, "roughness": roughness, "metallic": metallic,
        "transmission": transmission, "ior": ior, "opacity": opacity,
        "emission": emission, "ambient": ambient,
        "spec_strength": spec_strength, "shininess": shininess,
        "grain": grain, "inner_color": inner_color or albedo,
    }


# Linear RGB values are artistic surface colours, not measured conductor IORs.
# Distinct polish, roughness, and grain make the metal families useful in the
# studio and as parts of composite marble meshes.
MATERIAL_CATALOG = {
    "metal_gold": _surface((1.00, 0.72, 0.29), roughness=0.12, metallic=1.0,
                           spec_strength=0.96, shininess=180, grain=0.006),
    "metal_silver": _surface((0.91, 0.93, 0.96), roughness=0.10, metallic=1.0,
                             spec_strength=0.98, shininess=210, grain=0.003),
    "metal_copper": _surface((0.91, 0.47, 0.26), roughness=0.18, metallic=1.0,
                             spec_strength=0.90, shininess=120, grain=0.012,
                             inner_color=(0.55, 0.24, 0.13)),
    "metal_brass": _surface((0.83, 0.68, 0.29), roughness=0.21, metallic=1.0,
                            spec_strength=0.88, shininess=105, grain=0.015,
                            inner_color=(0.58, 0.43, 0.15)),
    "metal_bronze": _surface((0.55, 0.36, 0.17), roughness=0.32, metallic=1.0,
                             spec_strength=0.77, shininess=62, grain=0.026),
    "metal_steel": _surface((0.59, 0.64, 0.68), roughness=0.27, metallic=1.0,
                            spec_strength=0.84, shininess=86, grain=0.020,
                            inner_color=(0.39, 0.44, 0.48)),
    "metal_chrome": _surface((0.82, 0.86, 0.89), roughness=0.055, metallic=1.0,
                             spec_strength=1.0, shininess=340, grain=0.002),
    "metal_aluminum": _surface((0.72, 0.75, 0.77), roughness=0.24, metallic=1.0,
                               spec_strength=0.83, shininess=88, grain=0.016),
    "metal_titanium": _surface((0.49, 0.52, 0.57), roughness=0.29, metallic=1.0,
                               spec_strength=0.79, shininess=70, grain=0.022),
    "metal_iron": _surface((0.34, 0.36, 0.37), roughness=0.43, metallic=0.92,
                           spec_strength=0.68, shininess=40, grain=0.040),
    "metal_oxidized_copper": _surface((0.20, 0.47, 0.42), roughness=0.72, metallic=0.42,
                                      spec_strength=0.38, shininess=18, grain=0.055),
    # Nominal artistic glass compositions, not measured dispersion curves:
    # soda-lime-like clear glass n=1.52 and slightly higher-index dyed glass
    # n=1.54. Transmission=1 / opacity=0 enables the dielectric branch; its
    # Fresnel reflection remains. Opacity is a forced surface-reflection mix
    # in the current ray shader, not pigment density or volume scattering.
    "marble_clear_glass": _surface((0.93, 0.97, 1.0), roughness=0.035,
                                   transmission=1.0, opacity=0.0, ior=1.52,
                                   spec_strength=0.96, shininess=260,
                                   grain=0.002, inner_color=(0.91, 0.95, 1.0)),
    # Retained artistic milky preset; not a calibrated scattering medium.
    "marble_milk_glass": _surface((0.96, 0.92, 0.88), roughness=0.38,
                                  transmission=0.42, opacity=0.79, ior=1.48,
                                  spec_strength=0.48, shininess=30,
                                  grain=0.045, inner_color=(0.93, 0.86, 0.79)),
    # Opaque legacy inner defaults requested by the user. Optical rod/dye
    # snapshots independently use explicit volume spectra and transparent interfaces.
    "marble_cat_eye_blue": _surface((0.10, 0.43, 0.91), roughness=0.12,
                                    transmission=0.0, opacity=1.0, ior=1.54,
                                    spec_strength=0.88, shininess=160,
                                    grain=0.012, inner_color=(0.07, 0.23, 0.77)),
    "marble_cat_eye_green": _surface((0.08, 0.73, 0.39), roughness=0.13,
                                     transmission=0.0, opacity=1.0, ior=1.54,
                                     spec_strength=0.86, shininess=150,
                                     grain=0.014, inner_color=(0.05, 0.46, 0.24)),
    "marble_cat_eye_amber": _surface((0.98, 0.49, 0.08), roughness=0.13,
                                     transmission=0.0, opacity=1.0, ior=1.54,
                                     spec_strength=0.86, shininess=150,
                                     grain=0.013, inner_color=(0.78, 0.24, 0.04)),
    "stone_onyx": _surface((0.028, 0.035, 0.045), roughness=0.16,
                           spec_strength=0.68, shininess=130, grain=0.015),
    "stone_jade": _surface((0.13, 0.48, 0.31), roughness=0.25,
                           transmission=0.17, opacity=0.95, ior=1.61,
                           spec_strength=0.65, shininess=72, grain=0.025),
    "ceramic_porcelain": _surface((0.91, 0.89, 0.84), roughness=0.16,
                                  spec_strength=0.62, shininess=120, grain=0.008),
    # A genuine photographic blackout material: dead matte black cloth, not
    # polished black stone. stone_onyx is glossy (spec_strength 0.68,
    # shininess 130) -- it still catches a visible highlight. Felt has
    # near-zero albedo AND near-zero specular/shininess, so it absorbs
    # instead of reflecting: the real studio technique of blacking out
    # everything except the lights actually being tested.
    "black_felt": _surface((0.012, 0.012, 0.013), roughness=1.0,
                           spec_strength=0.0, shininess=1.0, grain=0.08),
    # Real architectural matte plaster: a warm bone/cream wall finish, not a
    # polished studio surface -- very rough (plaster is matte, not glossy),
    # near-zero specular, no metallic/transmission.
    "stucco_plaster": _surface((0.82, 0.76, 0.63), roughness=0.92,
                               spec_strength=0.08, shininess=8, grain=0.035),
    # Colour calibration chart: real pure-reference materials (not a texture
    # swap) so the backdrop roll can hang an actual emissive colour chart on
    # the Studio wall -- each patch a real catalog material with a genuine
    # emission colour, carried through the same emission-profile pipeline
    # every other emissive material uses.
    "calibration_white":   _surface((0.02, 0.02, 0.02), roughness=1.0, emission=(30, 30, 30)),
    "calibration_red":     _surface((0.02, 0.02, 0.02), roughness=1.0, emission=(30, 0, 0)),
    "calibration_green":   _surface((0.02, 0.02, 0.02), roughness=1.0, emission=(0, 30, 0)),
    "calibration_blue":    _surface((0.02, 0.02, 0.02), roughness=1.0, emission=(0, 0, 30)),
    "calibration_yellow":  _surface((0.02, 0.02, 0.02), roughness=1.0, emission=(30, 30, 0)),
    "calibration_magenta": _surface((0.02, 0.02, 0.02), roughness=1.0, emission=(30, 0, 30)),
    "calibration_cyan":    _surface((0.02, 0.02, 0.02), roughness=1.0, emission=(0, 30, 30)),
    "calibration_gray20":  _surface((0.02, 0.02, 0.02), roughness=1.0, emission=(6, 6, 6)),
    # Studio finishes are distinct opaque dielectric materials, not aliases for
    # cream stucco, porcelain or polished onyx. Append to preserve existing IDs.
    # These are artistic paint/plaster approximations, not measured samples.
    # Gloss belongs to the solid paint's surface; no thin-film interference
    # enamel is authored (MaterialDatabase keeps coating thickness at zero).
    "white_gloss_paint": _surface((0.90, 0.90, 0.90), roughness=0.065,
                                  spec_strength=0.82, shininess=220, grain=0.002),
    "white_plaster": _surface((0.90, 0.90, 0.90), roughness=0.96,
                              spec_strength=0.025, shininess=4, grain=0.04),
    "black_gloss_paint": _surface((0.012, 0.012, 0.012), roughness=0.065,
                                  spec_strength=0.82, shininess=220, grain=0.002),
    # Illustrative opaque matte paint, not measured optical calibration or felt.
    # Append only: existing saved material IDs must not move.
    "black_matte_paint": _surface((0.025, 0.025, 0.025), roughness=0.8,
                                  spec_strength=0.25, shininess=12, grain=0.0),
    # Artistic polished reflectors: RGB conductor approximation, not measured n/k spectra.
    "polished_silver": _surface((0.91, 0.93, 0.96), roughness=0.005, metallic=1.0,
                                spec_strength=0.98, shininess=512, grain=0.0),
    "polished_gold": _surface((1.00, 0.72, 0.29), roughness=0.005, metallic=1.0,
                              spec_strength=0.96, shininess=512, grain=0.0),
    # Reflective reference paints. Registered after legacy ribbon rows by
    # web_export so adding this target cannot renumber saved Studio materials.
    "reference_gray18_matte": _surface((0.18, 0.18, 0.18), roughness=1.0,
                                        spec_strength=0.0, shininess=1, grain=0.0),
    "reference_red_matte": _surface((0.65, 0.025, 0.025), roughness=1.0,
                                     spec_strength=0.0, shininess=1, grain=0.0),
    "reference_green_matte": _surface((0.025, 0.65, 0.025), roughness=1.0,
                                       spec_strength=0.0, shininess=1, grain=0.0),
    "reference_blue_matte": _surface((0.025, 0.025, 0.65), roughness=1.0,
                                      spec_strength=0.0, shininess=1, grain=0.0),
    "reference_yellow_matte": _surface((0.65, 0.65, 0.025), roughness=1.0,
                                        spec_strength=0.0, shininess=1, grain=0.0),
}

# The spectral calibration QR shares each chart cell's visible paint. These
# rows have identical RGB/PBR/Phong values; only their native 850 nm emission
# profile distinguishes them during the camera exposure.
for _paint in ("white_plaster", "black_matte_paint", "reference_gray18_matte",
               "reference_red_matte", "reference_green_matte",
               "reference_blue_matte", "reference_yellow_matte"):
    MATERIAL_CATALOG[_paint + "__nir850"] = dict(MATERIAL_CATALOG[_paint])

# Append fabric finishes after every existing row so saved Studio material IDs
# remain stable. These are artistic surface profiles, not measured BRDFs.
MATERIAL_CATALOG["fabric_canvas_cotton"] = _surface(
    (0.68, 0.57, 0.42), roughness=0.96, spec_strength=0.035,
    shininess=4, grain=0.11)
MATERIAL_CATALOG["fabric_muslin_white"] = _surface(
    (0.78, 0.76, 0.69), roughness=0.99, spec_strength=0.025,
    shininess=3, grain=0.09)
MATERIAL_CATALOG["fabric_velvet_blue"] = _surface(
    (0.025, 0.09, 0.23), roughness=1.0, spec_strength=0.018,
    shininess=2, grain=0.13)
