#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 output_color;

layout(push_constant) uniform MainOutput {
    vec4 color;
} app;

void main() {
    float vignette = 0.72 + 0.28 * (1.0 - length(uv - vec2(0.5)));
    output_color = vec4(app.color.rgb * vignette, 1.0);
}
