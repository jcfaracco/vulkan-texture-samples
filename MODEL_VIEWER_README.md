# Vulkan Model Viewer

A Vulkan-based 3D model viewer that loads and displays large OBJ models like Stanford Lucy.

## Features

- Loads OBJ models using tinyobjloader
- Real-time 3D rendering with Vulkan
- Automatic camera rotation
- Simple directional lighting
- Vertex deduplication for memory efficiency
- Depth buffering for correct rendering

## Requirements

- Vulkan SDK
- GLFW3
- GLM (OpenGL Mathematics library)
- tinyobjloader (header-only library)
- CMake 3.10+
- C++17 compatible compiler
- glslangValidator (for shader compilation)

## Installation

### Install Dependencies

**Fedora/RHEL:**
```bash
sudo dnf install vulkan-headers vulkan-loader-devel glfw-devel glm-devel cmake gcc-c++
```

**Ubuntu/Debian:**
```bash
sudo apt install libvulkan-dev libglfw3-dev libglm-dev cmake g++
```

**Arch Linux:**
```bash
sudo pacman -S vulkan-devel glfw-wayland glm cmake base-devel
```

### Install tinyobjloader

tinyobjloader is header-only. Download it:

```bash
cd /usr/local/include
sudo wget https://raw.githubusercontent.com/tinyobjloader/tinyobjloader/release/tiny_obj_loader.h
```

Or clone the repository and copy the header:

```bash
git clone https://github.com/tinyobjloader/tinyobjloader.git
sudo cp tinyobjloader/tiny_obj_loader.h /usr/local/include/
```

## Building

```bash
mkdir build
cd build
cmake ..
make
```

This will compile:
- `texture_benchmark` - The original texture benchmark
- `model_viewer` - The new 3D model viewer

## Downloading Test Models

### Stanford Lucy (Large Model - ~100MB+)

Stanford Lucy is a high-resolution 3D scan with approximately 28 million triangles.

**Note:** Stanford models come in PLY format and need to be converted to OBJ.

Using Blender (recommended):
```bash
# Install Blender
sudo dnf install blender  # or apt install blender

# Download Lucy PLY
wget http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz
tar -xzf lucy.tar.gz

# Convert to OBJ using Blender
blender --background --python - <<EOF
import bpy
bpy.ops.import_mesh.ply(filepath="lucy.ply")
bpy.ops.export_scene.obj(filepath="lucy.obj")
EOF
```

### Alternative: Smaller Test Models

**Stanford Bunny (easier to start with):**
```bash
mkdir -p models
cd models

# Download bunny OBJ (already in OBJ format)
wget https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj
```

**Or use the provided script:**
```bash
./download_lucy.sh
```

## Usage

```bash
# Run from the build directory
cd build

# View Stanford Bunny
./model_viewer ../models/stanford-bunny.obj

# View Stanford Lucy (after conversion)
./model_viewer ../models/lucy.obj

# View any OBJ model
./model_viewer /path/to/your/model.obj
```

## Controls

- The camera automatically rotates around the model
- Close the window to exit

## Performance Notes

### Model Loading
- Uses vertex deduplication to reduce memory usage
- Average vertex reuse ratio: ~6 triangles per vertex
- Loading times vary based on model size:
  - Stanford Bunny (~70K triangles): <100ms
  - Stanford Dragon (~870K triangles): ~500ms
  - Stanford Lucy (~28M triangles): several seconds

### GPU Upload
- Uses staging buffers for efficient GPU transfer
- Separate vertex and index buffers
- Device-local memory for optimal performance

## Troubleshooting

### "Failed to find GPUs with Vulkan support"
- Ensure Vulkan drivers are installed
- Check: `vulkaninfo`

### "Failed to open file: model_shader.vert.spv"
- Run `make` from the build directory
- Shaders must be compiled before running

### "Failed to load model"
- Ensure the OBJ file exists and path is correct
- Check that the model has proper vertex data

### Model appears too large/small
- Edit `cameraDistance` in model_viewer.cpp (line 124)
- Default: 3.0 units

### Model appears dark
- Model uses simple directional lighting
- Edit light direction in model_shader.frag (line 10)

## Architecture

The viewer consists of:

1. **Model Loading** (`loadModel`)
   - Parses OBJ with tinyobjloader
   - Deduplicates vertices
   - Tracks loading performance

2. **Vulkan Pipeline**
   - Vertex/Index buffers
   - Uniform buffers for transforms
   - Depth testing
   - Simple lighting shader

3. **Rendering Loop**
   - Automatic camera rotation
   - Per-frame uniform buffer updates
   - Double buffering (2 frames in flight)

## Files

- `model_viewer.cpp` - Main viewer implementation
- `model_shader.vert` - Vertex shader with MVP transforms
- `model_shader.frag` - Fragment shader with simple lighting
- `download_lucy.sh` - Helper script to download models
- `CMakeLists.txt` - Build configuration

## Future Enhancements

- [ ] Camera controls (mouse/keyboard)
- [ ] Material support
- [ ] Texture loading
- [ ] Multiple light sources
- [ ] Wireframe mode
- [ ] Model statistics display
- [ ] FPS counter

## License

See LICENSE file for details.
