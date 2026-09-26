#version 330 core
in vec3 vNormV;
in vec3 vPosV;
out vec4 FragColor;

uniform vec4  uColor;   // base RGBA
uniform vec3  uInnerColor;
uniform vec3  uLightV;  // light direction in view space
uniform float uAmbient;
uniform float uSpecStrength;
uniform float uShininess;
uniform float uGrain;

// Scene field seed — from SceneFieldIntegration; uploaded once per frame.
// uSceneRgb: spectral colour of the ambient environment (unit range).
// uSceneIndirectRatio: fraction of scene power that is reflected/bounced;
//   0 = all direct (hard shadows), 1 = fully diffuse (soft fill light).
uniform vec3  uSceneRgb;
uniform float uSceneIndirectRatio;

void main() {
    vec3 N    = normalize(gl_FrontFacing ? vNormV : -vNormV);
    vec3 L    = normalize(uLightV);
    vec3 V    = normalize(-vPosV);
    vec3 H    = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), max(uShininess, 1.0));
    vec3  base = gl_FrontFacing ? uColor.rgb : uInnerColor;
    float grain = 0.5 + 0.5 * sin(vPosV.x * 80.0 + vPosV.y * 31.0 + vPosV.z * 17.0);
    base *= mix(1.0, 0.82 + 0.28 * grain, uGrain);
    vec3  col  = base * (0.78 * diff)
               + mix(vec3(1.0, 0.88, 0.62), uSceneRgb, 0.30) * (uSpecStrength * spec);

    // Optional fast emissive-light proxy pass (same idea as C/base-material).
    // uSceneIndirectRatio is the strength knob; zero disables the pass.
    float emissivePass = clamp(uSceneIndirectRatio, 0.0, 1.0);
    if (emissivePass > 0.0) {
        float emissiveWrap = clamp(0.25 + 0.75 * diff, 0.0, 1.0);
        col += base * uSceneRgb * emissivePass * (0.10 + 0.30 * emissiveWrap);
    }

    FragColor  = vec4(col, uColor.a);
}
