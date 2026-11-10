"""Authenticated exact-sphere admission, before any active job is replaced."""
from __future__ import annotations

import copy
import math
import numpy as np

EXACT_REQUIREMENT = ("Exact spheres require an uncoated spherical marble with saved analytic sphere metadata. "
                     "Save this design again or choose Saved triangle mesh.")


def saved_analytic_spheres(saved, vertices, material_ids, pbr, config):
    if config.get("sphere_coverage", "mesh") == "mesh":
        return None
    if config.get("sphere_coverage") != "exact":
        raise ValueError("sphere_coverage must be mesh or exact")
    design = saved.get("design", {})
    metadata = design.get("analytic_spheres")
    if (saved.get("generator") != "meowdoku-marble-object-v1" or
            design.get("base-shape") != "marble" or float(design.get("coverage", 0)) != 0 or
            float(design.get("band-count", 0)) != 0 or not isinstance(metadata, dict)):
        raise ValueError(EXACT_REQUIREMENT)
    if set(metadata) != {"version", "spheres"} or metadata["version"] != 1:
        raise ValueError("Unsupported saved analytic sphere metadata")
    regions = metadata["spheres"]
    bubble_count = design.get("bubble-count", 0)
    if not isinstance(regions, list) or not 1 <= len(regions) <= 65 or len(regions) != bubble_count + 1:
        raise ValueError("Saved sphere count does not match the shell and bubbles")
    verts = np.asarray(vertices, np.float64)
    ids = np.asarray(material_ids)
    occupied = np.zeros(len(verts), bool)
    shell = saved["materials"]["index"][design["shell"]]
    boundaries = design.get("boundary_media", {}).get("ranges", [])
    volume_regions = design.get("region_media", {}).get("regions")
    outer = 0
    for region in regions:
        if not isinstance(region, dict) or set(region) != {"start", "count", "center", "radius"}:
            raise ValueError("Invalid saved analytic sphere descriptor")
        start, count, center, radius = (region[k] for k in ("start", "count", "center", "radius"))
        if any(not isinstance(x, int) or isinstance(x, bool) for x in (start, count)) or start < 0 or count < 4 or start + count > len(verts):
            raise ValueError("Saved sphere membership is outside the authenticated mesh")
        if np.any(occupied[start:start+count]):
            raise ValueError("Saved analytic sphere memberships overlap")
        occupied[start:start+count] = True
        if not isinstance(center, list) or len(center) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in center):
            raise ValueError("Invalid saved sphere center")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 < radius <= 1:
            raise ValueError("Invalid saved sphere radius")
        is_outer = center == [0, 0, 0] and radius == 1
        if is_outer:
            outer += 1
            # This is the full v1 generator's declared final glass range,
            # not a guess based on matching material IDs or vertex radius.
            if count != 2 * (96-1) * 128 or start+count != len(verts):
                raise ValueError("Saved outer sphere does not match the full v1 shell membership")
        if volume_regions is not None:
            ownership = [r for r in volume_regions
                         if r["start"] == start and r["count"] == count]
            expected = ("host", shell) if is_outer else ("air_bubble", -1)
            if len(ownership) != 1 or (ownership[0]["mode"], ownership[0]["material"]) != expected:
                raise ValueError("Saved sphere ownership disagrees with its explicit volume region")
        elif bubble_count:
            pair = next((b for b in boundaries if b["start"] == start and b["count"] == count), None)
            expected = (-1, shell) if is_outer else (shell, -1)
            if pair is None or (pair["positive"], pair["negative"]) != expected:
                raise ValueError("Saved sphere ownership disagrees with its explicit medium boundary")
        elif not is_outer:
            raise ValueError("Unexpected analytic region in a bubble-free marble")
        face_ids = ids[start:start+count]
        if not np.all(face_ids == shell) or np.any(pbr[shell, 8:11] != 0):
            raise ValueError("Exact spherical boundaries require a uniform nonemissive shell material")
        tri = verts[start:start+count]
        if not np.allclose(np.linalg.norm(tri-np.asarray(center), axis=2), radius, rtol=0, atol=3e-6):
            raise ValueError("Saved sphere metadata does not match its authenticated vertices")
        cross = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
        valid = np.linalg.norm(cross, axis=1) > 1e-12
        if not np.any(valid) or np.any(np.einsum('ij,ij->i',cross[valid],tri[valid,0]-center) <= 0):
            raise ValueError("Saved analytic sphere winding must be outward")
    if outer != 1:
        raise ValueError("Exact spherical coverage requires one saved outer shell")
    return copy.deepcopy(metadata)


def preflight_triangle_budget(vertices, config, spheres=None):
    """Reject over-budget submitted scene geometry before replacing a job.

    The worker repeats this on final packed geometry, including any generic
    calibration scene. Studio requests supply the complete fixture geometry.
    """
    budget = config.get("triangle_budget", 250000)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1000 <= budget <= 1000000:
        raise ValueError("triangle_budget must be an integer from 1000 to 1000000")
    def valid_count(verts, scale, center):
        world = np.asarray(verts, np.float32) * np.float32(scale) + np.asarray(center, np.float32)
        cross = np.cross(world[:,1]-world[:,0], world[:,2]-world[:,0])
        return np.linalg.norm(cross, axis=1) > max(float(scale)**2*1e-12, 1e-20)
    keep = valid_count(vertices, config["scale_m"], config["object_center_m"])
    if spheres:
        for sphere in spheres["spheres"]:
            keep[sphere["start"]:sphere["start"]+sphere["count"]] = False
    count = int(keep.sum())
    fixtures = config.get("fixtures")
    if fixtures is not None:
        faces = np.asarray(fixtures["vertices"], np.float32).reshape(-1,3,3)
        count += int(valid_count(faces, 1., [0,0,0]).sum())
    if count > budget:
        raise ValueError(f"Scene needs {count} triangles; exposure budget is {budget}. Raise the budget or simplify the saved design.")
    return count
