#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Check if we have valid normals
    vec3 normal = normalize(fragNormal);

    // Ambient lighting (base illumination)
    vec3 ambient = 0.35 * fragColor;

    // Key light (main directional light from top-right)
    vec3 keyLightDir = normalize(vec3(0.5, 1.0, 0.3));
    float keyDiff = max(dot(normal, keyLightDir), 0.0);
    vec3 keyDiffuse = 0.6 * keyDiff * vec3(1.0, 0.95, 0.9); // Warm light

    // Fill light (softer light from opposite side)
    vec3 fillLightDir = normalize(vec3(-1.0, 0.2, -0.5));
    float fillDiff = max(dot(normal, fillLightDir), 0.0);
    vec3 fillDiffuse = 0.25 * fillDiff * vec3(0.8, 0.85, 1.0); // Cool fill

    // Rim light (highlight edges from behind)
    vec3 rimLightDir = normalize(vec3(0.0, -0.3, -1.0));
    float rimDiff = max(dot(normal, rimLightDir), 0.0);
    vec3 rimDiffuse = 0.2 * rimDiff * vec3(1.0, 1.0, 1.0);

    // Simple fresnel-like rim effect
    vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0)); // Simple approximation
    float rim = 1.0 - max(dot(normal, viewDir), 0.0);
    rim = pow(rim, 3.0);
    vec3 rimGlow = 0.15 * rim * vec3(1.0, 1.0, 1.0);

    // Combine all lighting with the surface color
    vec3 lighting = ambient + keyDiffuse + fillDiffuse + rimDiffuse + rimGlow;
    vec3 result = fragColor * lighting;

    outColor = vec4(result, 1.0);
}
