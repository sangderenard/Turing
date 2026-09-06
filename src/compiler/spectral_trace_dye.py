"""Join shell trace emissions to SSA influence dye, paths, and time spectra.

The runtime emits only an integer trace site and real launch timing.  The
compile-time trace manifest resolves that site to SSA values, while an
``InfluenceField`` supplies the dye provenance and the paths that can reach
those values.  This module is the deliberate join: it does not invent a new
trace format, profiler, graph walker, or colour encoding.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from .influence_field import InfluenceField, SPECTRUM_END


@dataclass(frozen=True)
class TraceEmission:
    """One runtime emission in the shell's own time coordinates."""

    site: int
    sequence: int
    at_ns: int
    duration_ns: int
    phase: str = "enter"


def emissions_from_shell_payload(payload: Mapping[str, Any]) -> tuple[TraceEmission, ...]:
    """Normalise either shell telemetry JSON or the native trace-ring JSON.

    Telemetry records retain their actual ``at_ns``. Native ring output has
    launch durations but no separate wall-clock stamp, so its events are
    placed consecutively by their measured shell duration rather than given a
    fabricated wall time.
    """

    records = payload.get("records") or ()
    out: list[TraceEmission] = []
    for record in records:
        if str(record.get("kind")) != "trace":
            continue
        detail = dict(record.get("detail") or {})
        if "region" not in detail:
            continue
        out.append(TraceEmission(
            site=int(detail["region"]),
            sequence=int(record.get("sequence", len(out))),
            at_ns=int(record.get("at_ns", 0)),
            duration_ns=int(detail.get("nanoseconds", detail.get("shell_ns", 0)) or 0),
            phase=str(detail.get("phase", "enter")),
        ))
    if out:
        return tuple(sorted(out, key=lambda item: (item.at_ns, item.sequence)))

    trace = dict(payload.get("trace") or {})
    elapsed = 0
    for index, launch in enumerate(trace.get("launches") or ()):
        duration = int(launch.get("shell_ns", launch.get("device_ns", 0)) or 0)
        out.append(TraceEmission(
            site=int(launch["region"]), sequence=int(launch.get("seq", index)),
            at_ns=elapsed, duration_ns=duration,
        ))
        elapsed += duration
    return tuple(out)


def compare_emission_sequences(
    reference_payload: Mapping[str, Any], observed_payload: Mapping[str, Any],
    *, duration_tolerance_ns: int = 0,
) -> dict[str, Any]:
    """Align two shell traces and name their first control or timing split.

    This deliberately compares execution evidence only. A matching trace does
    not claim matching values; it says the next diagnostic should use the
    existing watch/bisection path rather than chase a changed control route.
    """

    reference = emissions_from_shell_payload(reference_payload)
    observed = emissions_from_shell_payload(observed_payload)
    def mapped(item: TraceEmission | None):
        return None if item is None else {
            "site": item.site, "sequence": item.sequence, "at_ns": item.at_ns,
            "duration_ns": item.duration_ns, "phase": item.phase,
        }
    for index in range(max(len(reference), len(observed))):
        left = reference[index] if index < len(reference) else None
        right = observed[index] if index < len(observed) else None
        if left is None or right is None:
            return {"equal": False, "index": index, "kind": "length", "reference": mapped(left), "observed": mapped(right)}
        if (left.site, left.phase) != (right.site, right.phase):
            return {"equal": False, "index": index, "kind": "control", "reference": mapped(left), "observed": mapped(right)}
        if abs(left.duration_ns - right.duration_ns) > int(duration_tolerance_ns):
            return {"equal": False, "index": index, "kind": "timing", "reference": mapped(left), "observed": mapped(right),
                    "delta_ns": right.duration_ns - left.duration_ns}
    return {"equal": True, "matched_emissions": len(reference)}


def _manifest_values(manifest: Mapping[str, Any], site: int, level: str) -> tuple[int, ...]:
    levels = dict(manifest.get("levels") or {})
    table = dict(levels.get(level) or {})
    values = table.get(site, table.get(str(site), ()))
    if not values:
        # Native shells historically name the compiled region in their record;
        # the manifest names the trace site. They are often equal but must not
        # be assumed so: resolve the region through the manifest's own table.
        for entry in manifest.get("sites") or ():
            if int(entry.get("region", -1)) == int(site):
                resolved_site = int(entry["site"])
                values = table.get(resolved_site, table.get(str(resolved_site), ()))
                break
    return tuple(int(value) for value in (values if isinstance(values, (list, tuple)) else (values,)))


def _field_keys_for_values(field: InfluenceField, values: Sequence[int]) -> tuple[Any, ...]:
    wanted = set(values)
    return tuple(
        key for key in field.node_keys()
        if key in wanted or (isinstance(key, tuple) and key and key[-1] in wanted)
    )


def _heaviest_paths(field: InfluenceField, targets: Sequence[Any], *, limit: int) -> list[dict[str, Any]]:
    """Bounded reverse walks over real relaxed transports, heaviest first."""

    incoming: dict[Any, list[Any]] = defaultdict(list)
    for transport in field.trace():
        incoming[transport.target_key].append(transport)
    paths: list[dict[str, Any]] = []
    for target in targets:
        frontier = [(target, (), float("inf"), frozenset({target}))]
        while frontier and len(paths) < limit:
            node, reversed_edges, weight, seen = frontier.pop(0)
            options = sorted(incoming.get(node, ()), key=lambda item: item.weight, reverse=True)
            if not options:
                if reversed_edges:
                    paths.append({
                        "target": repr(target),
                        "weight": weight,
                        "edges": list(reversed([{
                            "source": repr(item.source_key), "target": repr(item.target_key),
                            "role": item.role, "weight": item.weight,
                        } for item in reversed_edges])),
                    })
                continue
            for transport in options:
                if transport.source_key in seen:
                    continue
                frontier.append((
                    transport.source_key, reversed_edges + (transport,),
                    min(weight, transport.weight), seen | {transport.source_key},
                ))
    return sorted(paths, key=lambda item: item["weight"], reverse=True)[:limit]


def analyse_trace_dye(
    payload: Mapping[str, Any], manifest: Mapping[str, Any] | None = None,
    field: InfluenceField | None = None,
    *, level: str = "ssa", top: int = 12, paths_per_target: int = 4,
    target_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Rank executed targets and retain their dye, paths, timing, and phase.

    Frequency is measured from a target's real emission cadence; phase is that
    cadence's accumulated angle at each observed emission. A one-shot target
    has no measurable recurrence, so its frequency and phase are reported as
    ``None`` rather than guessed.
    """

    emissions = emissions_from_shell_payload(payload)
    by_site: dict[int, list[TraceEmission]] = defaultdict(list)
    for emission in emissions:
        by_site[emission.site].append(emission)

    names = dict((manifest or {}).get("names") or {})
    correlations = tuple((manifest or {}).get("name_correlations") or ())
    if not correlations:
        correlations = tuple(
            {"name": str(name), "occurrence": occurrence, "value": int(value)}
            for name, values in sorted(names.items())
            for occurrence, value in enumerate(values)
        )
    requested_names = tuple(dict.fromkeys(str(name) for name in target_names))
    resolved_names = {
        name: tuple(int(value) for value in (names.get(name) or ()))
        for name in requested_names if names.get(name)
    }
    requested = frozenset(value for values in resolved_names.values() for value in values)
    candidate_sites = tuple(by_site)
    if requested and manifest:
        candidate_sites = tuple(
            site for site in candidate_sites
            if requested.intersection(_manifest_values(manifest, site, level))
        )
    ranked = sorted(
        candidate_sites, key=lambda site: (
            sum(item.duration_ns for item in by_site[site]), len(by_site[site]), -site,
        ), reverse=True)[:max(0, int(top))]
    sites = tuple((manifest or {}).get("sites") or ())
    hue_by_site = {
        int(item["site"]): SPECTRUM_END * index / max(1, len(sites) - 1)
        for index, item in enumerate(sites)
    }
    targets = []
    for site in ranked:
        samples = sorted(by_site[site], key=lambda item: (item.at_ns, item.sequence))
        values = _manifest_values(manifest, site, level) if manifest else ()
        value_set = set(values)
        target_correlations = [
            dict(row) for row in correlations
            if int(row["value"]) in value_set
        ]
        keys = _field_keys_for_values(field, values) if field is not None else ()
        starts = [item.at_ns for item in samples]
        span = starts[-1] - starts[0] if len(starts) > 1 else 0
        frequency = (len(samples) - 1) * 1_000_000_000.0 / span if span > 0 else None
        timings = [
            {
                "sequence": item.sequence, "at_ns": item.at_ns,
                "duration_ns": item.duration_ns, "phase": (
                    None if frequency is None else (2.0 * math.pi * frequency *
                    (item.at_ns - starts[0]) / 1_000_000_000.0) % (2.0 * math.pi)
                ),
            }
            for item in samples
        ]
        readings = []
        for key in keys:
            reading = field.reading(key)
            readings.append({
                "key": repr(key), "value": reading.value,
                "staging": reading.staging, "recurrence": reading.recurrence,
                "categories": {
                    name: {"hue": value.hue, "saturation": value.saturation,
                           "weight": value.weight, "dispersion": value.dispersion}
                    for name, value in reading.categories.items()
                },
            })
        targets.append({
            "site": site, "dye_hue": hue_by_site.get(site, 0.0),
            "ssa_values": list(values), "field_keys": [repr(key) for key in keys],
            "authored_names": list(dict.fromkeys(
                str(row["name"]) for row in target_correlations
            )),
            "name_correlations": target_correlations,
            "emission_count": len(samples), "total_duration_ns": sum(item.duration_ns for item in samples),
            "frequency_hz": frequency, "timings": timings, "dye": readings,
            "paths": _heaviest_paths(field, keys, limit=paths_per_target)
            if field is not None else [],
            "resolution": "resolved" if manifest and field is not None else (
                "unresolved: provide --manifest and --ssa from the same trace-enabled AOT build"
            ),
        })
    return {
        "schema": "turing-spectral-dye-trace-v1", "level": level,
        "emission_count": len(emissions), "target_count": len(targets), "targets": targets,
        "target_names": list(requested_names),
        "identity_resolution": "authored name + occurrence -> deterministic SSA value",
        "unmatched_target_names": sorted(set(requested_names) - set(resolved_names)),
        "unmatched_target_values": sorted(requested - {
            value for target in targets for value in target["ssa_values"]
        }),
    }


__all__ = ["TraceEmission", "emissions_from_shell_payload", "compare_emission_sequences", "analyse_trace_dye"]
