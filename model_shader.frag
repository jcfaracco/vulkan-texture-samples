/*
 * Fragment Shader for Model Viewer
 *
 * Implements three-point lighting for professional-looking 3D rendering:
 * - Key Light: Main directional light (warm white)
 * - Fill Light: Softer light from opposite side (cool blue tint)
 * - Rim Light: Backlight to highlight edges
 * - Rim Glow: Fresnel-like effect for depth perception
 *
 * This creates depth and prevents the model from looking flat
 */

#version 450

// Input from vertex shader (interpolated across triangle)
layout(location = 0) in vec3 fragColor;   // Base material color
layout(location = 1) in vec3 fragNormal;  // Surface normal (world space)
layout(location = 2) in vec3 fragPos;     // Fragment position (world space)

// Output color to framebuffer
layout(location = 0) out vec4 outColor;

void main() {
    // Normalize the interpolated normal (may not be unit length after interpolation)
    vec3 normal = normalize(fragNormal);

    // === Ambient Lighting (base illumination) ===
    // Ensures all surfaces have minimum visibility, even in shadow
    // 35% of base color prevents completely black areas
    vec3 ambient = 0.35 * fragColor;

    // === Key Light (main directional light from top-right) ===
    // Primary light source - creates main highlights and shadows
    // Direction: From top-right-front
    // Color: Slightly warm white (1.0, 0.95, 0.9)
    vec3 keyLightDir = normalize(vec3(0.5, 1.0, 0.3));
    float keyDiff = max(dot(normal, keyLightDir), 0.0);  // Lambertian diffuse
    vec3 keyDiffuse = 0.6 * keyDiff * vec3(1.0, 0.95, 0.9); // Warm light

    // === Fill Light (softer light from opposite side) ===
    // Secondary light to reduce harsh shadows from key light
    // Direction: From left-front
    // Color: Cool blue tint (0.8, 0.85, 1.0) for color contrast
    vec3 fillLightDir = normalize(vec3(-1.0, 0.2, -0.5));
    float fillDiff = max(dot(normal, fillLightDir), 0.0);
    vec3 fillDiffuse = 0.25 * fillDiff * vec3(0.8, 0.85, 1.0); // Cool fill

    // === Rim Light (highlight edges from behind) ===
    // Backlight that creates a bright outline around the model
    // Helps separate model from background and adds depth
    // Direction: From behind and slightly below
    vec3 rimLightDir = normalize(vec3(0.0, -0.3, -1.0));
    float rimDiff = max(dot(normal, rimLightDir), 0.0);
    vec3 rimDiffuse = 0.2 * rimDiff * vec3(1.0, 1.0, 1.0);

    // === Fresnel-like Rim Glow ===
    // Adds a subtle glow to edges facing away from camera
    // Creates a "halo" effect that enhances 3D perception
    // Based on Fresnel effect: surfaces are more reflective at grazing angles
    vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0)); // Simplified view direction
    float rim = 1.0 - max(dot(normal, viewDir), 0.0);  // Stronger at edges
    rim = pow(rim, 3.0);  // Sharpen the falloff
    vec3 rimGlow = 0.15 * rim * vec3(1.0, 1.0, 1.0);

    // === Combine All Lighting ===
    // Additive blending of all light contributions
    vec3 lighting = ambient + keyDiffuse + fillDiffuse + rimDiffuse + rimGlow;

    // Multiply lighting by material color (albedo)
    vec3 result = fragColor * lighting;

    // Output final color with full opacity
    outColor = vec4(result, 1.0);
}
