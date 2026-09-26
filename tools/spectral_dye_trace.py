"""Render shell trace emissions as vivid SSA dye, paths, timings, and phase.

    python tools/spectral_dye_trace.py --telemetry build/fluid-trace/trace.json
    python tools/spectral_dye_trace.py --telemetry shell.json --manifest trace_manifest.json --ssa build/run/control_repository_ssa.pkl
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _names(spec: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(token for token in spec.replace(",", " ").split() if token))


def _render_timeline(path: Path, report: dict) -> None:
    from PIL import Image, ImageDraw
    from src.rendering.influence_field_image import dye_rgb

    targets = report["targets"]
    width, height, margin = 1400, max(300, 110 + 68 * len(targets)), 90
    image = Image.new("RGB", (width, height), (12, 13, 20))
    draw = ImageDraw.Draw(image)
    all_times = [sample["at_ns"] for target in targets for sample in target["timings"]]
    low, high = (min(all_times), max(all_times)) if all_times else (0, 1)
    span = max(1, high - low)
    draw.text((margin, 24), "shell trace -> SSA dye: timing, phase, and path targets", fill=(235, 235, 244))
    for row, target in enumerate(targets):
        y = 78 + 68 * row
        draw.line((margin, y, width - margin, y), fill=(48, 53, 72), width=1)
        label = (f"site {target['site']}  {target['emission_count']} emissions  "
                 f"{target['total_duration_ns']} ns  f={target['frequency_hz']!s} Hz")
        draw.text((8, y - 28), label, fill=(192, 201, 220))
        for sample in target["timings"]:
            x = margin + (sample["at_ns"] - low) / span * (width - 2 * margin)
            phase = sample["phase"]
            pulse = 0.65 if phase is None else 0.35 + 0.65 * (0.5 + 0.5 * math.cos(phase))
            radius = 5 + min(19, sample["duration_ns"] ** 0.5 / 50.0)
            colour = dye_rgb(target["dye_hue"], 0.95, pulse)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        help="trace manifest from compilation.map_ir['trace']")
    parser.add_argument("--ssa", type=Path,
                        help="matching lowered control_repository_ssa.pkl")
    parser.add_argument("--level", default="ssa")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--target", default="",
                        help="authored names to select, e.g. loss,max_vel (requires --manifest/--ssa)")
    parser.add_argument("--reference", type=Path,
                        help="a second telemetry/native-trace JSON to align against this run")
    parser.add_argument("--duration-tolerance-ns", type=int, default=0)
    parser.add_argument("--json", type=Path, default=Path("build/spectral_dye_trace.json"))
    parser.add_argument("--image", type=Path, default=Path("build/spectral_dye_trace.png"))
    args = parser.parse_args()
    if not args.telemetry.is_file():
        parser.error(f"telemetry file does not exist: {args.telemetry}")
    if bool(args.manifest) != bool(args.ssa):
        parser.error("--manifest and --ssa must be supplied together for SSA/dye path resolution")
    if args.manifest and not args.manifest.is_file():
        parser.error(
            f"trace manifest does not exist: {args.manifest}. Historical trace output "
            "can run without --manifest; to resolve SSA paths, export "
            "compilation.map_ir['trace'] from the same trace-enabled AOT build."
        )
    if args.ssa and not args.ssa.is_file():
        parser.error(f"SSA artifact does not exist: {args.ssa}")
    if args.reference and not args.reference.is_file():
        parser.error(f"reference trace does not exist: {args.reference}")
    targets = _names(args.target)
    if targets and not args.manifest:
        parser.error("--target requires --manifest and --ssa; trace-only sites have no authored identity")
    telemetry = json.loads(args.telemetry.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    from src.compiler.influence_field import InfluenceContract, field_from_ssa
    from src.compiler.spectral_trace_dye import analyse_trace_dye, compare_emission_sequences

    field = None
    if args.ssa:
        with args.ssa.open("rb") as stream:
            module, _outputs, _exports = pickle.load(stream)
        field = field_from_ssa(module, InfluenceContract(enabled=True))
        field.propagate()
    report = analyse_trace_dye(
        telemetry, manifest, field, level=args.level, top=args.top, target_names=targets,
    )
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        report["trace_comparison"] = compare_emission_sequences(
            reference, telemetry, duration_tolerance_ns=args.duration_tolerance_ns,
        )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _render_timeline(args.image, report)
    print(f"{report['emission_count']} emissions -> {report['target_count']} targets")
    if field is None:
        print("unresolved runtime sites: add --manifest and --ssa from one trace-enabled AOT build for SSA dye paths")
    elif report["unmatched_target_names"]:
        print(f"targets absent from this manifest: {report['unmatched_target_names']}")
    elif report["unmatched_target_values"]:
        print("target histories have no traced producing site in this run")
    if "trace_comparison" in report:
        comparison = report["trace_comparison"]
        print("trace sequences match" if comparison["equal"] else
              f"first trace split: {comparison['kind']} at emission {comparison['index']}")
    print(f"wrote {args.json}\nwrote {args.image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
