cmake_minimum_required(VERSION 3.10)
project(VulkanTextureBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find GLFW
find_package(glfw3 REQUIRED)

# Add executable
add_executable(texture_benchmark texture_benchmark.cpp)

# Link libraries
target_link_libraries(texture_benchmark
    Vulkan::Vulkan
    glfw
    pthread
)

# Compile shaders
find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(VERT_SHADER ${SHADER_DIR}/shader.vert)
set(FRAG_SHADER ${SHADER_DIR}/shader.frag)

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

add_custom_target(shaders DEPENDS
    ${SHADER_OUTPUT_DIR}/shader.vert.spv
    ${SHADER_OUTPUT_DIR}/shader.frag.spv
)

add_dependencies(texture_benchmark shaders)
