# Vulkan Texture Benchmark - Advanced

A DirectStorage-inspired Vulkan texture benchmark that measures GPU texture upload performance with various configurations. Supports both headless and windowed modes.

## Features

- **DirectStorage-inspired optimizations**: Parallel/bulk texture loading
- **Multiple staging buffer sizes**: Tests with 1MB to 64MB staging buffers
- **Queue depth testing**: Measures parallel upload performance (1, 4, 8, 16 concurrent operations)
- **Texture size scaling**: Benchmarks 2K, 4K, and 8K textures
- **CPU usage tracking**: Monitors CPU time during uploads
- **Bandwidth measurement**: Reports transfer speeds in GB/s
- **Dual mode operation**: Run with or without a window (--headless flag)

## Requirements

- Vulkan SDK
- GLFW3
- CMake 3.10+
- C++17 compatible compiler
- glslangValidator (for shader compilation)

## Building

```bash
mkdir build
cd build
cmake ..
make
```

This will compile the shaders and build the `texture_benchmark` executable.

## Usage

### Windowed Mode (Default)
```bash
./texture_benchmark
```

Creates a window and runs the benchmark. The window shows "Vulkan Texture Benchmark" while the tests run.

### Headless Mode
```bash
./texture_benchmark --headless
```

Runs without creating a window. Ideal for:
- Server environments
- Automated testing
- CI/CD pipelines
- Systems without display

## Benchmark Tests

The benchmark runs three test suites:

### Test 1: Staging Buffer Size Impact
- Tests a single 4K texture with varying staging buffer sizes (1-64MB)
- Measures how staging buffer size affects upload performance
- Helps identify optimal buffer size for your GPU

### Test 2: Queue Depth Impact
- Tests multiple 2K textures loaded in parallel (16MB staging buffer)
- Varies queue depth from 1 to 16 concurrent operations
- Demonstrates parallel upload scaling

### Test 3: Texture Size Scaling
- Tests different texture resolutions (2K, 4K, 8K)
- Uses 32MB staging buffer and single queue depth
- Shows how performance scales with texture size

## Output

The benchmark displays:

**GPU Information:**
```
=== GPU Information ===
Mode: Headless/Windowed
Device: [GPU Name]
API Version: [Vulkan Version]
VRAM: [Memory in MB]
Transfer Queue Family: [Queue Index]
Graphics Queue Family: [Queue Index]
```

**Benchmark Results:**
```
Test Name              | Staging | Queue | Total (ms) | Bandwidth (GB/s) | CPU (ms) | Size (MB)
-----------------------|---------|-------|------------|------------------|----------|----------
4096x4096              |     1MB |     1 |      XX.XX |            X.XXX |    XX.XX |    XX.XX
...
```

**Summary:**
```
=== Key Findings ===
Peak Bandwidth: X.XXX GB/s
Minimum CPU Time: XX.XX ms
```

## Architecture

The benchmark uses:
- **Dedicated transfer queue**: Optimized GPU upload path
- **Staging buffers**: Host-visible memory for CPU-to-GPU transfers
- **Optimal image tiling**: GPU-optimized texture layout
- **Parallel uploads**: Multi-threaded texture loading
- **CPU profiling**: Resource usage tracking via getrusage()

## Performance Tips

1. **Staging buffer size**: Larger buffers (16-64MB) typically perform better
2. **Queue depth**: Higher parallelism improves throughput on modern GPUs
3. **Headless mode**: Slightly better performance without window overhead
4. **Transfer queue**: Dedicated transfer queue reduces graphics queue contention

## Files

- `texture_benchmark.cpp`: Main benchmark implementation
- `shader.vert`: Vertex shader (for windowed mode)
- `shader.frag`: Fragment shader (for windowed mode)
- `CMakeLists.txt`: Build configuration
- `compile_shaders.sh`: Manual shader compilation script

## Notes

- The benchmark generates random texture data to simulate real-world loading
- Textures are uploaded and immediately freed (no rendering in current version)
- Results may vary based on GPU, driver version, and system load
- Run the benchmark multiple times and average results for consistency

## License

See LICENSE file for details.
