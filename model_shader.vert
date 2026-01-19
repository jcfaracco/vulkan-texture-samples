/*
 * Vertex Shader for Model Viewer
 *
 * Transforms vertex positions from model space to clip space
 * and prepares data for the fragment shader
 *
 * Pipeline: Model Space -> World Space -> View Space -> Clip Space
 */

#version 450

// Uniform buffer containing MVP matrices (updated per frame)
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;  // Model-to-world transformation
    mat4 view;   // World-to-camera transformation
    mat4 proj;   // Camera-to-clip space (perspective projection)
} ubo;

// Input vertex attributes (from vertex buffer)
layout(location = 0) in vec3 inPosition;  // Vertex position in model space
layout(location = 1) in vec3 inNormal;    // Surface normal in model space
layout(location = 2) in vec3 inColor;     // Vertex color (material)

// Output to fragment shader
layout(location = 0) out vec3 fragColor;   // Pass-through color
layout(location = 1) out vec3 fragNormal;  // Transformed normal (world space)
layout(location = 2) out vec3 fragPos;     // Vertex position (world space)

void main() {
    // Transform position through MVP matrices
    // Result is in clip space [-1,1] for x,y and [0,1] for z (Vulkan)
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    // Pass color unchanged to fragment shader
    fragColor = inColor;

    // Transform normal to world space
    // Use normal matrix (inverse transpose) to handle non-uniform scaling
    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;

    // Transform position to world space (for lighting calculations)
    fragPos = vec3(ubo.model * vec4(inPosition, 1.0));
}
