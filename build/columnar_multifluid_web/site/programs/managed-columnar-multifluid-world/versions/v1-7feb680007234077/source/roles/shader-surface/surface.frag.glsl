#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D turing_output_texture;
uniform vec2 turing_resolution;
layout(location = 0) out vec4 turing_output_0;

void main() {
    vec2 uv = gl_FragCoord.xy / max(turing_resolution, vec2(1.0));
    uv.y = 1.0 - uv.y;
    turing_output_0 = texture(turing_output_texture, uv);
}
