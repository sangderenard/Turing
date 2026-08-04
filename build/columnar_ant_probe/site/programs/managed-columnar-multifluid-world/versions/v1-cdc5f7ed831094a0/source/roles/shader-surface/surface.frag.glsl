#version 300 es
precision highp float;
precision highp int;

uniform vec2 turing_resolution;
uniform sampler2D turing_feed_0;
uniform sampler2D turing_feed_1;
uniform sampler2D turing_feed_2;
layout(location = 0) out vec4 turing_output_0;

void main() {
    vec2 turing_uv = gl_FragCoord.xy / max(turing_resolution, vec2(1.0));
    turing_uv.y = 1.0 - turing_uv.y;
    float v_0 = texture(turing_feed_0, turing_uv).r;
    float v_1 = texture(turing_feed_1, turing_uv).r;
    float v_2 = texture(turing_feed_2, turing_uv).r;
    float v_4 = v_0 / 255.0;
    float v_6 = v_1 / 255.0;
    float v_8 = v_2 / 255.0;
    turing_output_0 = vec4(v_4, v_6, v_8, 1.0);
}
