"""Authored scene materials for the shared spectral MaterialDatabase.

The names are stable browser material IDs.  Every record carries the PBR
surface fields and all four authored Phong properties; MaterialDatabase also
derives the enamel, texture-stack, and ray records from these same entries.
"""

from __future__ import annotations


def _surface(albedo, *, roughness, metallic=0.0, transmission=0.0,
             ior=1.5, opacity=1.0, emission=(0.0, 0.0, 0.0), ambient=0.18,
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
    "marble_clear_glass": _surface((0.93, 0.97, 1.0), roughness=0.035,
                                   transmission=0.96, opacity=0.16, ior=1.52,
                                   ambient=0.12, spec_strength=0.96, shininess=260,
                                   grain=0.002, inner_color=(0.91, 0.95, 1.0)),
    "marble_milk_glass": _surface((0.96, 0.92, 0.88), roughness=0.38,
                                  transmission=0.42, opacity=0.79, ior=1.48,
                                  ambient=0.19, spec_strength=0.48, shininess=30,
                                  grain=0.045, inner_color=(0.93, 0.86, 0.79)),
    "marble_cat_eye_blue": _surface((0.10, 0.43, 0.91), roughness=0.12,
                                    transmission=0.35, opacity=0.90, ior=1.52,
                                    ambient=0.18, spec_strength=0.88, shininess=160,
                                    grain=0.012, inner_color=(0.07, 0.23, 0.77)),
    "marble_cat_eye_green": _surface((0.08, 0.73, 0.39), roughness=0.13,
                                     transmission=0.34, opacity=0.91, ior=1.52,
                                     ambient=0.18, spec_strength=0.86, shininess=150,
                                     grain=0.014, inner_color=(0.05, 0.46, 0.24)),
    "marble_cat_eye_amber": _surface((0.98, 0.49, 0.08), roughness=0.13,
                                     transmission=0.32, opacity=0.92, ior=1.52,
                                     ambient=0.18, spec_strength=0.86, shininess=150,
                                     grain=0.013, inner_color=(0.78, 0.24, 0.04)),
    "stone_onyx": _surface((0.028, 0.035, 0.045), roughness=0.16,
                           spec_strength=0.68, shininess=130, grain=0.015),
    "stone_jade": _surface((0.13, 0.48, 0.31), roughness=0.25,
                           transmission=0.17, opacity=0.95, ior=1.61,
                           spec_strength=0.65, shininess=72, grain=0.025),
    "ceramic_porcelain": _surface((0.91, 0.89, 0.84), roughness=0.16,
                                  spec_strength=0.62, shininess=120, grain=0.008),
}
