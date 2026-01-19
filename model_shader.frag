#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(-fragPos);

    // Ambient lighting
    vec3 ambient = 0.4 * fragColor;

    // Key light (main directional light from top-right)
    vec3 keyLightDir = normalize(vec3(-0.5, -1.0, -0.3));
    float keyDiff = max(dot(normal, -keyLightDir), 0.0);
    vec3 keyDiffuse = 0.7 * keyDiff * fragColor;

    // Fill light (softer light from left to fill shadows)
    vec3 fillLightDir = normalize(vec3(1.0, 0.0, 0.5));
    float fillDiff = max(dot(normal, -fillLightDir), 0.0);
    vec3 fillDiffuse = 0.3 * fillDiff * fragColor;

    // Back/rim light (highlight edges)
    vec3 rimLightDir = normalize(vec3(0.0, 0.5, 1.0));
    float rimDiff = max(dot(normal, -rimLightDir), 0.0);
    vec3 rimDiffuse = 0.2 * rimDiff * vec3(1.0, 1.0, 1.0);

    // Specular highlight (key light only)
    vec3 reflectDir = reflect(keyLightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = 0.3 * spec * vec3(1.0, 1.0, 1.0);

    // Combine all lighting
    vec3 result = ambient + keyDiffuse + fillDiffuse + rimDiffuse + specular;

    outColor = vec4(result, 1.0);
}
