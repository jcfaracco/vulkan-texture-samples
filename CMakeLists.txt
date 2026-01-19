cmake_minimum_required(VERSION 3.10)
project(VulkanTextureBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find GLFW
find_package(glfw3 REQUIRED)

# Find GLM
find_package(glm REQUIRED)

# Add executables
add_executable(texture_benchmark texture_benchmark.cpp)
add_executable(model_viewer model_viewer.cpp)

# Link libraries for texture_benchmark
target_link_libraries(texture_benchmark
    Vulkan::Vulkan
    glfw
    pthread
)

# Link libraries for model_viewer
target_link_libraries(model_viewer
    Vulkan::Vulkan
    glfw
    pthread
)

# Compile shaders
find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Texture benchmark shaders
set(VERT_SHADER ${SHADER_DIR}/shader.vert)
set(FRAG_SHADER ${SHADER_DIR}/shader.frag)
set(BC1_DECOMP_SHADER ${SHADER_DIR}/bc1_decompressor.comp)

# Model viewer shaders
set(MODEL_VERT_SHADER ${SHADER_DIR}/model_shader.vert)
set(MODEL_FRAG_SHADER ${SHADER_DIR}/model_shader.frag)

add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/shader.vert.spv
    COMMAND ${GLSL_VALIDATOR} -V ${VERT_SHADER} -o ${SHADER_OUTPUT_DIR}/shader.vert.spv
    DEPENDS ${VERT_SHADER}
    COMMENT "Compiling vertex shader"
)

add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/shader.frag.spv
    COMMAND ${GLSL_VALIDATOR} -V ${FRAG_SHADER} -o ${SHADER_OUTPUT_DIR}/shader.frag.spv
    DEPENDS ${FRAG_SHADER}
    COMMENT "Compiling fragment shader"
)

add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/bc1_decompressor.comp.spv
    COMMAND ${GLSL_VALIDATOR} -V ${BC1_DECOMP_SHADER} -o ${SHADER_OUTPUT_DIR}/bc1_decompressor.comp.spv
    DEPENDS ${BC1_DECOMP_SHADER}
    COMMENT "Compiling BC1 decompression compute shader (DirectStorage)"
)

add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/model_shader.vert.spv
    COMMAND ${GLSL_VALIDATOR} -V ${MODEL_VERT_SHADER} -o ${SHADER_OUTPUT_DIR}/model_shader.vert.spv
    DEPENDS ${MODEL_VERT_SHADER}
    COMMENT "Compiling model viewer vertex shader"
)

add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}/model_shader.frag.spv
    COMMAND ${GLSL_VALIDATOR} -V ${MODEL_FRAG_SHADER} -o ${SHADER_OUTPUT_DIR}/model_shader.frag.spv
    DEPENDS ${MODEL_FRAG_SHADER}
    COMMENT "Compiling model viewer fragment shader"
)

add_custom_target(shaders DEPENDS
    ${SHADER_OUTPUT_DIR}/shader.vert.spv
    ${SHADER_OUTPUT_DIR}/shader.frag.spv
    ${SHADER_OUTPUT_DIR}/bc1_decompressor.comp.spv
    ${SHADER_OUTPUT_DIR}/model_shader.vert.spv
    ${SHADER_OUTPUT_DIR}/model_shader.frag.spv
)

add_dependencies(texture_benchmark shaders)
add_dependencies(model_viewer shaders)
