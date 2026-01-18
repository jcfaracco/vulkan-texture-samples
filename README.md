# Vulkan Texture Loading Benchmark

A comprehensive Vulkan benchmark tool to measure and demonstrate texture loading performance from filesystem to GPU memory. Ideal for benchmarking high-performance filesystems with AMD Radeon GPUs.

## Features

- **Precise Timing Instrumentation**: Separates file I/O time from GPU upload time
- **Multiple Texture Sizes**: Tests with 2K, 4K, and 8K textures
- **Visual Display**: Renders loaded textures for visual verification
- **Detailed Metrics**: Reports throughput (MB/s), individual timings, and aggregated results
- **Real-world Workflow**: Uses Vulkan staging buffers for optimal CPU→GPU transfer

## Requirements

### System Requirements
- Vulkan-capable GPU (AMD Radeon recommended)
- Vulkan SDK installed
- Linux system with X11 or Wayland

### Build Dependencies
```bash
# Ubuntu/Debian
sudo apt install build-essential cmake libvulkan-dev glslang-tools libglfw3-dev

# Fedora
sudo dnf install cmake vulkan-devel glslang glfw-devel

# Arch Linux
sudo pacman -S cmake vulkan-devel glslang glfw-x11
```

## Building

```bash
mkdir build
cd build
cmake ..
make
```

The build system automatically compiles GLSL shaders to SPIR-V during the build process.

### Manual Shader Compilation (Optional)
If you need to compile shaders manually:
```bash
glslangValidator -V shader.vert -o shader.vert.spv
glslangValidator -V shader.frag -o shader.frag.spv
```

## Usage

### Basic Execution
```bash
cd build
./texture_benchmark
```

### What It Does
1. Initializes Vulkan rendering context
2. Generates 10 test textures (2K, 4K, 8K sizes)
3. Loads each texture with precise timing:
   - **Read Time**: File I/O from disk to RAM
   - **Upload Time**: CPU RAM to GPU VRAM transfer
4. Displays timing table with throughput metrics
5. Opens window showing loaded textures (press SPACE to cycle, ESC to exit)

### Sample Output
```
=== Starting Texture Load Benchmark ===
Using GPU: AMD Radeon RX 7900 XTX

Loaded texture_2048_0.raw: 12.34 ms (1245.67 MB/s)
Loaded texture_4096_1.raw: 45.67 ms (1456.78 MB/s)
...

=== Benchmark Results ===
Texture                    | Read (ms) | Upload (ms) | Total (ms) | Size (MB) | Throughput (MB/s)
---------------------------|-----------|-------------|------------|-----------|------------------
texture_2048_0.raw         |      4.23 |        8.11 |      12.34 |     16.00 |          1296.10
texture_4096_1.raw         |     15.34 |       30.33 |      45.67 |     64.00 |          1401.58
...
---------------------------|-----------|-------------|------------|-----------|------------------
TOTAL                      |           |             |     234.56 |    320.00 |          1364.21
```

## Benchmarking Your Filesystem

### Testing Standard Filesystem
```bash
cd build
./texture_benchmark
# Note the throughput values
```

### Testing Your Fast Filesystem
Mount your custom filesystem and modify the texture generation location in the code, or:
```bash
# Create symbolic link to your fast filesystem
cd build
ln -s /path/to/your/fast/fs ./fast_storage
# Modify texture paths in code to use ./fast_storage/
```

### Comparing Results
Compare the "Throughput (MB/s)" column, specifically:
- **Read (ms)**: Shows filesystem read performance
- **Upload (ms)**: Shows PCIe/GPU memory bandwidth (constant across filesystems)
- **Total Throughput**: Overall data pipeline efficiency

## Customization

### Modify Texture Count and Sizes
Edit `texture_benchmark.cpp`:
```cpp
// In loadTestTextures() function
const int numTextures = 20;  // Change number of textures
const std::vector<uint32_t> sizes = {2048, 4096, 8192, 16384};  // Add 16K textures
```

### Use Real Texture Files
Replace `generateTestTexture()` calls with actual image loading:
```cpp
// Use stb_image or similar library to load PNG/JPG files
// Ensure format is RGBA 8-bit
```

### Adjust Window Resolution
```cpp
// In initWindow()
window = glfwCreateWindow(3840, 2160, "...", nullptr, nullptr);  // 4K display
```

## Technical Details

### Texture Format
- **Format**: R8G8B8A8_UNORM (32-bit RGBA)
- **Channel Layout**: Red, Green, Blue, Alpha (8 bits each)
- **Memory Layout**: Tightly packed, row-major

### Vulkan Pipeline
1. **Staging Buffer**: Host-visible memory for CPU writes
2. **Device-Local Image**: GPU-optimal memory for rendering
3. **Transfer Operations**: Vulkan command buffer transfers
4. **Sampler**: Linear filtering, anisotropic filtering enabled

### Timing Methodology
- Uses `std::chrono::high_resolution_clock`
- Measures wall-clock time (includes OS overhead)
- File reads use standard C++ `ifstream` (modify for `read()` syscalls if needed)
- GPU uploads include all Vulkan synchronization overhead

## Troubleshooting

### "Failed to find GPUs with Vulkan support"
- Install Vulkan drivers: `sudo apt install mesa-vulkan-drivers` (Mesa) or AMD proprietary drivers
- Verify: `vulkaninfo | grep deviceName`

### "Failed to open file: shader.vert.spv"
- Run from build directory: `cd build && ./texture_benchmark`
- Or compile shaders manually to current directory

### Window doesn't appear
- Check display server: `echo $DISPLAY`
- For Wayland: may need `GLFW_PLATFORM=wayland` environment variable

### Low throughput numbers
- Check CPU frequency scaling: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- Set performance mode: `sudo cpupower frequency-set -g performance`
- Monitor GPU clocks: `sudo radeontop` or `watch -n1 cat /sys/class/drm/card0/device/pp_dpm_sclk`

## License

Public domain / MIT - use freely for benchmarking and testing.

## Contributing

This is a standalone benchmark tool. Modify as needed for your specific use case.
