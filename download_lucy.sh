#!/bin/bash

# Download Stanford Lucy model
# This is a high-resolution 3D scan with ~28 million triangles

echo "Downloading Stanford Lucy model..."

# Create models directory
mkdir -p models

# Download Lucy from Stanford 3D Scanning Repository
# Note: The full Lucy model is very large (>100MB)
# Using the medium resolution version (~14M triangles)

cd models

# Download using wget or curl
if command -v wget &> /dev/null; then
    echo "Using wget to download..."
    wget http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz
elif command -v curl &> /dev/null; then
    echo "Using curl to download..."
    curl -O http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Extract
echo "Extracting..."
tar -xzf lucy.tar.gz

# Convert PLY to OBJ (requires meshlabserver or similar tool)
# Note: The Stanford models come in PLY format
# You may need to convert them to OBJ format

echo ""
echo "Download complete!"
echo "Note: Stanford models are in PLY format."
echo "You may need to convert to OBJ format using:"
echo "  - MeshLab (meshlabserver)"
echo "  - Blender (import PLY, export OBJ)"
echo "  - Online converter"
echo ""
echo "Alternative: Use a smaller test model:"
echo "  wget https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj -O bunny.obj"
