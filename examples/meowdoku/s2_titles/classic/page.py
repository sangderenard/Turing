"""One html file that draws a Meowdoku board state with the compiled shader.

The page binds the WebGL fragment module exactly as the compiler's own page
runtime does (``wasm_html_shell``): one R32F texture per feed, NEAREST,
CLAMP_TO_EDGE, sampler uniforms ``turing_feed_<id>`` from the module's
``feed_bindings``, ``turing_resolution`` for normalized sampling, and the
module's own full-screen vertex shader.  After drawing it reads the canvas
back into ``<pre id="readback">`` as JSON (rows top first) so a headless
browser can be compared pixel for pixel with the Python version.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np


def emit_board_page(shader: Any, feeds: Mapping[str, np.ndarray], *,
                    width: int, height: int, title: str = "Meowdoku") -> str:
    api = shader.webgl.api.to_mapping() if hasattr(shader.webgl.api, "to_mapping") else dict(shader.webgl.api)
    metadata = api["metadata"]
    bindings = {int(item["value_id"]): item["uniform"] for item in metadata["feed_bindings"]}
    textures = []
    for name, value_id in shader.feed_ids.items():
        table = np.asarray(feeds[name], dtype=np.float32)
        rows, cols = table.shape
        textures.append({
            "uniform": bindings[int(value_id)], "name": name,
            "width": int(cols), "height": int(rows),
            "data": [float(x) for x in table.ravel(order="C")],
        })
    payload = {
        "width": int(width), "height": int(height),
        "vertex": metadata["vertex_shader"],
        "fragment": shader.webgl.source,
        "textures": textures,
    }
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>body{{margin:0;background:#101827;color:#e8ecf4;font:14px system-ui;display:grid;place-items:center;min-height:100vh}}
canvas{{image-rendering:pixelated;width:min(90vmin,{width * 4}px);height:auto;border:1px solid #334}}
pre{{display:none}}</style></head><body>
<canvas id="board" width="{width}" height="{height}"></canvas>
<pre id="readback"></pre>
<script id="turing-board" type="application/json">{json.dumps(payload)}</script>
<script>
(function () {{
  const spec = JSON.parse(document.getElementById('turing-board').textContent);
  const canvas = document.getElementById('board');
  const gl = canvas.getContext('webgl2', {{preserveDrawingBuffer: true, antialias: false, premultipliedAlpha: false}});
  const out = document.getElementById('readback');
  if (!gl) {{ out.textContent = JSON.stringify({{error: 'WebGL 2 is required'}}); document.title = 'no webgl2'; return; }}
  const compile = (type, source) => {{
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }};
  try {{
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, spec.vertex));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, spec.fragment));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    spec.textures.forEach((feed, index) => {{
      const texture = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + index);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, feed.width, feed.height, 0, gl.RED, gl.FLOAT, new Float32Array(feed.data));
      gl.uniform1i(gl.getUniformLocation(program, feed.uniform), index);
    }});
    const resolution = gl.getUniformLocation(program, 'turing_resolution');
    if (resolution) gl.uniform2f(resolution, spec.width, spec.height);
    gl.viewport(0, 0, spec.width, spec.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.finish();
    const pixels = new Uint8Array(spec.width * spec.height * 4);
    gl.readPixels(0, 0, spec.width, spec.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const rows = [];
    for (let y = spec.height - 1; y >= 0; y--) {{
      const row = [];
      for (let x = 0; x < spec.width; x++) {{
        const at = (y * spec.width + x) * 4;
        row.push(pixels[at], pixels[at + 1], pixels[at + 2]);
      }}
      rows.push(row);
    }}
    out.textContent = JSON.stringify({{width: spec.width, height: spec.height, rows}});
    document.title = 'meowdoku rendered';
  }} catch (error) {{
    out.textContent = JSON.stringify({{error: String(error)}});
    document.title = 'meowdoku error';
  }}
}})();
</script>
</body></html>
"""


__all__ = ["emit_board_page"]
