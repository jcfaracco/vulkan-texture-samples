#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple directional lighting
    vec3 lightDir = normalize(vec3(0.5, -1.0, 0.3));
    vec3 normal = normalize(fragNormal);

    // Ambient
    float ambient = 0.3;

    // Diffuse
    float diff = max(dot(-lightDir, normal), 0.0);

    // Final color
    vec3 result = fragColor * (ambient + diff * 0.7);

    outColor = vec4(result, 1.0);
}
