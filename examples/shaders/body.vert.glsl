#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;

uniform mat4 uMVP;
uniform mat4 uMV;   // view * model (for normal transform)

out vec3 vNormV;
out vec3 vPosV;

void main() {
    vec4 posV   = uMV * vec4(aPos, 1.0);
    vPosV       = posV.xyz;
    vNormV      = mat3(uMV) * aNorm;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
