# Advanced Vulkan Texture Benchmark
## DirectStorage-Inspired Performance Testing

This advanced benchmark incorporates techniques from Microsoft's DirectStorage samples, adapted for Vulkan on Linux. It provides comprehensive testing of texture loading performance with various configurations.

## Key Improvements Over Basic Benchmark

### 1. **Parallel/Bulk Loading** (DirectStorage EnqueueRequests)
- Tests multiple textures loaded simultaneously
- Configurable queue depth (1, 4, 8, 16 concurrent operations)
- Uses multi-threading for parallel file I/O and GPU uploads
- Demonstrates PCIe bandwidth saturation

### 2. **Staging Buffer Size Analysis** (DirectStorage Buffer Management)
- Tests 1MB, 2MB, 4MB, 8MB, 16MB, 32MB, 64MB staging buffers
- Identifies optimal buffer size for your hardware
- Shows memory bandwidth vs. buffer size relationship
- Mimics DirectStorage's configurable staging memory

### 3. **CPU Usage Tracking** (DirectStorage CPU Metrics)
- Measures CPU cycles using `getrusage()` syscall
- Reports user + system CPU time per operation
- Identifies CPU bottlenecks in the loading pipeline
- Useful for comparing filesystem overhead

### 4. **Dedicated Transfer Queue**
- Uses separate Vulkan transfer queue when available
- Offloads transfers from graphics queue
- Reduces contention and improves parallelism
- Similar to DirectStorage's hardware queue management

### 5. **Comprehensive Metrics** (DirectStorage Benchmarking)
- **Bandwidth in GB/s** (not just MB/s)
- **CPU time per transfer**
- **Queue depth impact**
- **Staging buffer size impact**
- **Per-texture and aggregate statistics**

## What It Tests

### Test Suite 1: Staging Buffer Size Impact
```
Measures: How staging buffer size affects single 4K texture load
Tests: 1MB, 2MB, 4MB, 8MB, 16MB, 32MB, 64MB
Metric: GB/s bandwidth, CPU milliseconds
Purpose: Find optimal staging buffer for your GPU
```

### Test Suite 2: Queue Depth Impact (Parallel Loading)
```
Measures: How many concurrent loads can saturate the bus
Tests: 1, 4, 8, 16 textures loaded in parallel
Texture Size: 2K (manageable for parallel ops)
Staging: Fixed at 16MB
Purpose: Determine optimal parallelism for bulk loading
```

### Test Suite 3: Texture Size Scaling
```
Measures: How performance scales with texture size
Tests: 2K, 4K, 8K textures
Staging: Fixed at 32MB
Queue Depth: 1 (isolate size impact)
Purpose: Understand per-texture overhead vs. throughput
```

## Building

```bash
cd /kmt
mkdir -p build
cd build
cmake ..
make texture_benchmark_advanced
```

## Running

### Basic Run
```bash
cd build
./texture_benchmark_advanced
```

### Expected Output
```
=== GPU Information ===
Device: AMD Radeon RX 7900 XTX
API Version: 1.3.0
VRAM: 24576 MB
Transfer Queue Family: 1
Graphics Queue Family: 0

=== DirectStorage-Inspired Vulkan Benchmark Suite ===
Testing staging buffer sizes: 1MB 2MB 4MB 8MB 16MB 32MB 64MB
Testing queue depths: 1 4 8 16

[Test 1] Staging Buffer Size Impact (4K texture)
  1MB: 2.345 GB/s, 12.3ms CPU
  2MB: 3.456 GB/s, 10.1ms CPU
  4MB: 4.567 GB/s, 9.2ms CPU
  8MB: 5.234 GB/s, 8.5ms CPU
  16MB: 5.891 GB/s, 8.1ms CPU
  32MB: 6.123 GB/s, 8.0ms CPU
  64MB: 6.145 GB/s, 8.0ms CPU

[Test 2] Queue Depth Impact (Multiple 2K textures, 16MB staging)
  Depth 1: 3.234 GB/s, 15.2ms CPU
  Depth 4: 8.456 GB/s, 45.6ms CPU
  Depth 8: 12.123 GB/s, 78.9ms CPU
  Depth 16: 14.567 GB/s, 125.3ms CPU

[Test 3] Texture Size Scaling (32MB staging, depth=1)
  2048x2048: 4.123 GB/s
  4096x4096: 6.234 GB/s
  8192x8192: 7.891 GB/s

=== Comprehensive Benchmark Results ===
Test Name              | Staging | Queue | Total (ms) | Bandwidth (GB/s) | CPU (ms) | Size (MB)
-----------------------|---------|-------|------------|------------------|----------|----------
...

=== Key Findings ===
Peak Bandwidth: 14.567 GB/s
Minimum CPU Time: 8.0 ms
```

## Interpreting Results

### Staging Buffer Size
- **Too small**: Frequent allocations, CPU overhead
- **Too large**: Memory waste, no performance gain
- **Sweet spot**: Bandwidth plateaus, minimal CPU time
- **Recommendation**: Use 2-4x typical texture size

### Queue Depth
- **Depth = 1**: Single-threaded performance baseline
- **Depth = 4-8**: Good parallelism without thrashing
- **Depth = 16+**: Maximum throughput, but high CPU usage
- **Look for**: Where bandwidth stops scaling linearly

### Filesystem Comparison
To compare your fast filesystem vs. standard filesystem:

1. **Run baseline** on standard filesystem:
   ```bash
   cd /standard/mount && ./texture_benchmark_advanced
   # Note "Read Time" and "CPU Time"
   ```

2. **Run test** on your fast filesystem:
   ```bash
   cd /your/fast/fs && ./texture_benchmark_advanced
   # Compare metrics
   ```

3. **Key metrics** to compare:
   - **File Read Time**: Direct filesystem performance
   - **CPU Time**: Filesystem driver overhead
   - **Total Bandwidth**: End-to-end efficiency

### Example Speedup Calculation
```
Standard FS:  Read=50ms, CPU=25ms, BW=3.2 GB/s
Fast FS:      Read=15ms, CPU=8ms,  BW=10.5 GB/s

Speedup: 3.3x faster, 3.1x less CPU overhead
```

## Advanced Configuration

### Modify Test Parameters

Edit `texture_benchmark_advanced.cpp`:

```cpp
// Line ~30: Adjust staging buffer sizes (MB)
const std::vector<uint32_t> stagingBufferSizes = {1, 2, 4, 8, 16, 32, 64, 128, 256};

// Line ~31: Adjust queue depths
const std::vector<uint32_t> queueDepths = {1, 2, 4, 8, 16, 32};

// Line ~32: Adjust texture sizes
const std::vector<uint32_t> textureSizes = {1024, 2048, 4096, 8192, 16384};
```

### Real File I/O Testing

Replace the benchmark texture generation with actual file reads:

```cpp
// In benchmarkSingleTexture(), replace:
// std::vector<uint8_t> pixels(imageSize);
// for (size_t i = 0; i < imageSize; i++) pixels[i] = rand() % 256;

// With:
std::string filename = "/path/to/textures/texture_" + std::to_string(width) + ".raw";
std::vector<uint8_t> pixels(imageSize);
std::ifstream file(filename, std::ios::binary);
file.read(reinterpret_cast<char*>(pixels.data()), imageSize);
file.close();
```

### Add Compressed Texture Support

To test with BC (Block Compression) formats:

1. Generate BC-compressed textures using `texconv` or similar
2. Modify `createImage()` to use `VK_FORMAT_BC1_RGB_UNORM_BLOCK` (or BC3, BC7)
3. Adjust size calculations (BC1 = 8 bytes per 4x4 block)

## DirectStorage Equivalents

| DirectStorage Feature | Vulkan Implementation | This Benchmark |
|-----------------------|----------------------|----------------|
| `EnqueueRequest()` | Multi-threaded uploads | ✅ Parallel loading test |
| GPU Decompression | BC/DXT compressed formats | ⚠️ TODO (see roadmap) |
| Staging Buffer Config | `VkBuffer` size tuning | ✅ Multiple buffer sizes |
| CPU Usage Tracking | `getrusage()` | ✅ Per-operation metrics |
| Bandwidth Measurement | GB/s calculation | ✅ All tests |
| Queue Management | Transfer queue separation | ✅ Dedicated transfer queue |
| Bulk Operations | Parallel texture loads | ✅ Queue depth testing |

## Roadmap / Future Improvements

### 1. GPU Decompression (High Priority)
- Add BC1/BC3/BC7 compressed texture support
- Implement CPU decompression baseline
- Compare GPU vs CPU decompression bandwidth
- Mirrors DirectStorage's `GpuDecompressionBenchmark`

### 2. Memory-Mapped I/O
- Test `mmap()` vs `read()` for file loading
- Measure page fault overhead
- Compare with DirectStorage's zero-copy approach

### 3. Direct Storage Integration (Linux)
- Test with `io_uring` for async I/O
- Use `O_DIRECT` flag for unbuffered I/O
- Bypass page cache like DirectStorage

### 4. Compression Ratio Analysis
- Test various compression formats
- Measure decompression throughput
- Calculate effective bandwidth gains

### 5. Multi-Subresource Requests
- Load texture arrays and mipmaps
- Test `VkBufferImageCopy` with multiple regions
- Equivalent to DirectStorage's `DSTORAGE_REQUEST_DESTINATION_MULTIPLE_SUBRESOURCES`

## Troubleshooting

### "Failed to find transfer queue family"
- Some GPUs share graphics and transfer queues
- Benchmark will still work but may show lower parallelism
- This is expected on integrated GPUs

### Low bandwidth numbers
- Check GPU frequency: `cat /sys/class/drm/card0/device/pp_dpm_sclk`
- Ensure PCIe 4.0/5.0 is active: `lspci -vv | grep "LnkSta:"`
- Disable CPU frequency scaling: `sudo cpupower frequency-set -g performance`

### High CPU usage
- Expected for high queue depths
- Indicates CPU-bound operations (filesystem overhead)
- Compare with baseline to measure your FS driver efficiency

## Comparison with Original Benchmark

| Feature | Basic Benchmark | Advanced Benchmark |
|---------|----------------|-------------------|
| Textures tested | 10 fixed | 30+ (3 test suites) |
| Buffer sizes | Fixed | 7 variants (1-64MB) |
| Queue depth | 1 (sequential) | 1, 4, 8, 16 (parallel) |
| CPU tracking | ❌ | ✅ (getrusage) |
| Transfer queue | ❌ | ✅ (dedicated) |
| Metrics | MB/s, ms | GB/s, CPU cycles, queue impact |
| Visual output | ✅ (window) | ❌ (headless for precision) |
| DirectStorage-inspired | ❌ | ✅ |

## References

- [DirectStorage Samples](https://github.com/microsoft/DirectStorage/tree/main/Samples)
- [Vulkan Transfer Operations](https://www.khronos.org/registry/vulkan/specs/1.3/html/chap7.html#synchronization-pipeline-stages-transfer)
- [Linux `io_uring` for Storage](https://kernel.dk/io_uring.pdf)
- [AMD GPU Memory Architecture](https://gpuopen.com/learn/rdna-performance-guide/)

## License

Public domain / MIT - modify freely for your benchmarking needs.
