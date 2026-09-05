"""Static site over ONE field, coloured by colorimetry.

Pick a source text, pick a view. Every colour on the page comes from the
same diffusion over the same translation graph, so the representations
cannot disagree -- they are one spectral map seen from different sides.

Views
-----
* **spectral** -- the location's normalised spectrum integrated against
  the CIE 1931 matching functions and taken to sRGB. Composition only:
  intensity is deliberately excluded so a value deep in a fan-out is not
  reported as uncertain merely for being dim.
* **spectral x power** -- the same chromaticity with luminance carrying
  the arrived weight, for reading how much rather than what.
* **state** -- live / attenuated / unreachable. Reachability is a graph
  fact and does not move with epsilon; only the live/attenuated split
  does. This is the view that answers "is this path dead".

Hovering any highlight dissects its spectrum: which authored text reached
that location, and in what proportion. Every source is allotted a distinct
frequency, so the inversion is exact rather than an attribution guess.

What this replaced
------------------
This used to build a field per representation and correlate them by
authored name afterwards. A name lookup is one-to-one, so it could not
express a translation that spreads one thing into many or mixes many into
one, and the panes had to be patched with cone walks to fake it. Those
tables and that patch are gone: diffusion on a graph that includes the
crossings does both by construction.

    python tools/build_tracer_site.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

STEP = "symbolic_fluid_control__symbolic_fluid_step"
REGION = STEP + "__planned_region_0"
ADVANCE = "symbolic_fluid_control__symbolic_fluid_advance"
#: An unstamped token sits at the surround lightness so it recedes without
#: becoming a second colour competing with the spectrum.
UNSTAMPED = None
STATE_COLOUR = {
    "live": (0.24, 0.62, 0.40),
    "attenuated": (0.72, 0.60, 0.20),
    "unreachable": (0.62, 0.26, 0.26),
}


def to_hex(triple) -> str:
    red, green, blue = (max(0, min(255, int(round(c * 255)))) for c in triple)
    return f"#{red:02x}{green:02x}{blue:02x}"


def colour_for(field: Any, key: Any, view: str, states: dict):
    """(background, ink) for one token, both decided here.

    The ink is chosen by measured contrast against the background it will
    actually sit on, so a token stays readable whether the spectrum put it
    at saturated violet or at dim red. Deciding it in the page would mean a
    second colour model in JavaScript disagreeing with this one.
    """
    from src.compiler.spectral_colorimetry import contrasting_ink, spectrum_rgb

    if view == "state":
        for state, keys in states.items():
            if key in keys:
                rgb = STATE_COLOUR[state]
                return to_hex(rgb), to_hex(contrasting_ink(rgb))
        return None, None
    accumulator = field.moments(key).get("dynamic")
    lines = tuple(getattr(accumulator, "lines", ()) or ())
    if not lines:
        return None, None
    red, green, blue = spectrum_rgb(accumulator.normalised().lines)
    if view == "power":
        # Luminance carries arrived weight; chromaticity is untouched, so
        # dimness reads as "less got here" and never as "less certain".
        # It also drives the ink flip: the same hue at low power needs the
        # opposite ink from the same hue at full power.
        power = float(getattr(accumulator, "power", 0.0) or 0.0)
        scale = max(0.10, min(1.0, power) ** 0.5)
        red, green, blue = red * scale, green * scale, blue * scale
    rgb = (red, green, blue)
    return to_hex(rgb), to_hex(contrasting_ink(rgb))


def origins(field: Any, key: Any, limit: int = 8) -> list:
    """Dissect: which authored text reached here, and how much of it."""
    by_frequency = {
        float(source.hue): str(source.label) for source in field.sources
    }
    accumulator = field.moments(key).get("dynamic")
    lines = tuple(getattr(accumulator, "lines", ()) or ())
    total = sum(weight for _frequency, weight in lines)
    if total <= 0.0:
        return []
    rows = sorted(
        ((by_frequency.get(float(f), "?"), w / total) for f, w in lines),
        key=lambda row: -row[1],
    )
    return [[name, round(share, 4)] for name, share in rows[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="build/tracer_site.html")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    arguments = parser.parse_args()

    from translation_graph import build, classify
    from src.compiler.influence_field import InfluenceContract
    from src.compiler.symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE

    contract = InfluenceContract(enabled=True, spectral=True)
    field, occurrences, _home, module, model, _crossings, edges = build(contract)
    states = {
        state: set(keys)
        for state, keys in classify(
            edges, [source.key for source in field.sources], field,
            arguments.epsilon,
        ).items()
    }

    sources: dict[str, dict] = {
        "python": {
            "label": "authored Python -- the traversal",
            "lines": SYMBOLIC_FLUID_DT_SOURCE.splitlines(),
            "tokens": [
                (row, start, end, ("py", row, start, end))
                for _key, _name, row, start, end in occurrences
            ],
        },
    }

    equation_lines, equation_tokens = [], []
    for row, equation in enumerate(model.equations):
        equation_lines.append(f"{equation.lhs} = {equation.rhs}")
        equation_tokens.append(
            (row, 0, len(str(equation.lhs)), ("sy", equation.lhs))
        )
    sources["sympy"] = {
        "label": "authored SymPy -- the mathematics",
        "lines": equation_lines,
        "tokens": equation_tokens,
    }

    # Every function of the translated program, so the whole sim is here
    # rather than the one region that happens to hold the arithmetic.
    for function_name, function in module.functions.items():
        rows, marks = [], []
        for block_name, block in function.blocks.items():
            rows.append(f"{block_name}:")
            for instruction in block.instrs:
                result = instruction.res
                target = "" if result is None else f"t{int(result.id)}"
                operands = ", ".join(f"t{int(a.id)}" for a in instruction.args)
                callee = instruction.attributes.get("callee")
                suffix = f"    ; {str(callee).split('__')[-1]}" if callee else ""
                rows.append(
                    f"  {target:>7} = {instruction.op}({operands}){suffix}"
                )
        for row, line in enumerate(rows):
            for match in re.finditer(r"\bt(\d+)\b", line):
                marks.append((
                    row, match.start(), match.end(),
                    ("ssa", function_name, int(match.group(1))),
                ))
        short = function_name.split("__", 1)[-1]
        sources[function_name] = {
            "label": f"SSA  {short}",
            "lines": rows,
            "tokens": marks,
        }

    views = {
        "spectral": "spectral colour (composition)",
        "power": "spectral colour x arrived power",
        "state": "live / attenuated / unreachable",
    }
    payload = {
        "sources": {
            key: {"label": value["label"], "lines": value["lines"]}
            for key, value in sources.items()
        },
        "views": views,
        "colours": {
            key: {
                view: [
                    [row, start, end, *colour_for(field, node, view, states)]
                    for row, start, end, node in value["tokens"]
                    if colour_for(field, node, view, states)[0] is not None
                ]
                for view in views
            }
            for key, value in sources.items()
        },
        "dissect": {
            key: {
                f"{row},{start}": origins(field, node)
                for row, start, end, node in value["tokens"]
            }
            for key, value in sources.items()
        },
    }

    destination = ROOT / arguments.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_page(payload), encoding="utf-8")
    for state, keys in states.items():
        print(f"  {state:12} {len(keys):6}")
    coloured = sum(
        1 for value in sources.values() for _t in value["tokens"]
    )
    print(f"  {len(sources)} sources, {coloured} coloured tokens")
    print(f"wrote {destination}")
    return 0


def _page(payload: dict) -> str:
    """Render the page against a measured neutral surround.

    Every colour below is derived from the L* 60 achromatic surround, so
    the page is a viewing condition rather than a theme: a swatch is judged
    against neutral grey, which is the only background that does not bias
    the hue being read. Ink is whichever of black or white measures higher
    contrast against it, so it flips with the surround instead of being
    asserted.
    """
    from src.compiler.spectral_colorimetry import (
        contrasting_ink, lstar_to_srgb_grey,
    )

    grey = lstar_to_srgb_grey()
    surround = to_hex((grey, grey, grey))
    ink = to_hex(contrasting_ink((grey, grey, grey)))
    panel = to_hex((grey * 0.88, grey * 0.88, grey * 0.88))
    edge = to_hex((grey * 0.72, grey * 0.72, grey * 0.72))
    muted = to_hex(tuple(
        channel * 0.45 + grey * 0.55 for channel in contrasting_ink(
            (grey, grey, grey)
        )
    ))
    data = json.dumps(payload)
    return (
        "<!doctype html><meta charset='utf-8'><title>fluid sim tracer</title>"
        "<style>"
        f"body{{background:{surround};color:{ink};font:13px/1.55 ui-monospace,"
        "Consolas,monospace;margin:0;padding:16px 20px}"
        "header{display:flex;gap:14px;align-items:center;flex-wrap:wrap;"
        "margin-bottom:10px}select,label{font:12px ui-sans-serif,system-ui}"
        f"select{{background:{panel};color:{ink};border:1px solid {edge};"
        "border-radius:4px;padding:4px 6px;max-width:34em}"
        f"#note{{color:{muted};font:12px ui-sans-serif,system-ui;margin:0 0 10px}}"
        ".l{white-space:pre}"
        f".n{{color:{muted};display:inline-block;width:4em;"
        "text-align:right;padding-right:1em;user-select:none}"
        "b{font-weight:600;border-radius:2px;cursor:help}"
        f"#tip{{position:fixed;background:{panel};color:{ink};"
        f"border:1px solid {edge};"
        "border-radius:5px;padding:8px 10px;font:11px ui-sans-serif,system-ui;"
        "pointer-events:none;display:none;max-width:24em;z-index:9;"
        "box-shadow:0 6px 20px rgba(0,0,0,.35)}"
        "</style><header>"
        "<label>source <select id='source'></select></label>"
        "<label>view <select id='view'></select></label></header>"
        "<p id='note'></p><div id='code'></div><div id='tip'></div>"
        f"<script>const D={data};"
        "const S=document.getElementById('source'),V=document.getElementById('view'),"
        "C=document.getElementById('code'),N=document.getElementById('note'),"
        "T=document.getElementById('tip');"
        "for(const k in D.sources){const o=document.createElement('option');"
        "o.value=k;o.textContent=D.sources[k].label;S.append(o);}"
        "for(const k in D.views){const o=document.createElement('option');"
        "o.value=k;o.textContent=D.views[k];V.append(o);}"
        "function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;')"
        ".replace(/>/g,'&gt;');}"
        "function draw(){const s=S.value,v=V.value,L=D.sources[s].lines,"
        "sp=D.colours[s][v],by={};for(const x of sp){(by[x[0]]||(by[x[0]]=[]))"
        ".push(x);}let h='';"
        "for(let r=0;r<L.length;r++){const line=L[r];let o='',at=0;"
        "for(const x of (by[r]||[]).sort((a,b)=>a[1]-b[1])){if(x[1]<at)continue;"
        "o+=esc(line.slice(at,x[1]));o+='<b data-k=\"'+r+','+x[1]+'\" style="
        "\"background:'+x[3]+';color:'+x[4]+'\">'+esc(line.slice(x[1],x[2]))"
        "+'</b>';at=x[2];}"
        "o+=esc(line.slice(at));h+='<div class=\"l\"><span class=\"n\">'+(r+1)"
        "+'</span>'+o+'</div>';}C.innerHTML=h;"
        "N.textContent=D.sources[s].label+' \\u2014 '+D.views[v]+'. '+sp.length+"
        "' tokens. Hover a highlight to dissect its spectrum.';}"
        "C.addEventListener('mousemove',e=>{const b=e.target.closest('b');"
        "if(!b){T.style.display='none';return;}"
        "const rows=(D.dissect[S.value]||{})[b.dataset.k]||[];"
        "if(!rows.length){T.style.display='none';return;}"
        "T.innerHTML='<b style=\"background:none\">reached by</b><br>'"
        "+rows.map(r=>esc(r[0])+' <span style=\"opacity:.72\">'"
        "+(100*r[1]).toFixed(1)+'%</span>').join('<br>');"
        "T.style.display='block';T.style.left=Math.min(e.clientX+14,"
        "window.innerWidth-320)+'px';T.style.top=(e.clientY+14)+'px';});"
        "S.onchange=V.onchange=draw;draw();</script>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
