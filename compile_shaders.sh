#!/bin/bash

# Compile Vulkan shaders to SPIR-V

echo "Compiling shaders..."

glslangValidator -V shader.vert -o shader.vert.spv
if [ $? -eq 0 ]; then
    echo "✓ Compiled shader.vert -> shader.vert.spv"
else
    echo "✗ Failed to compile shader.vert"
    exit 1
fi

glslangValidator -V shader.frag -o shader.frag.spv
if [ $? -eq 0 ]; then
    echo "✓ Compiled shader.frag -> shader.frag.spv"
else
    echo "✗ Failed to compile shader.frag"
    exit 1
fi

echo "All shaders compiled successfully!"
