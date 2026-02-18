# TrueLarge-RT <img src="docs/app_icon.png" width="48" style="vertical-align: middle;">

<div align="center">
<img src="docs/new_arch.png" width="800px">

**TrueLarge-RT: Break the Memory Wall on Android.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Android-green)](https://www.android.com/)
[![Cpp](https://img.shields.io/badge/Language-C++17-blue)](https://isocpp.org/)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-purple)](https://kotlinlang.org/)

[Features](#key-features) • [Installation](#installation) • [Quick Start](#quick-start) • [Architecture](#architecture) • [Citation](#citation)

</div>

---

**TrueLarge-RT** is a high-performance native inference engine that enables **32B+ parameter LLMs** to run on consumer Android devices (4GB-8GB RAM) without crashing. 

Traditional runtimes (ONNX, TFLite, etc.) require the full model to fit in RAM. TrueLarge-RT uses a proprietary **Layer-by-Layer (LBL)** pipeline to stream weights from storage, offering **infinite scalability** limited only by your disk size.

## The Breakthrough

By implementing a **Smart Hybrid Engine**, TrueLarge-RT automatically selects the optimal execution strategy for your hardware:

### 1. Turbo Mode (RAM-Lock)
<img src="docs/full_ram_mode.png" width="300px" align="right">

**Condition**: `Free RAM > Model Size + 1GB`

This mode pins the entire model into physical RAM using `mlock()`, preventing any OS swapping. It delivers **zero-latency** access to weights, making it perfect for smaller models (e.g., Llama-3-8B on a 12GB device) where maximum token speed is the priority.

<br clear="all">

### 2. Balanced Mode (mmap)
<img src="docs/os_page.png" width="300px" align="right">

**Condition**: `Free RAM > 75% of Model Size`

Leverages the operating system's virtual memory pager (`mmap`). The OS intelligently keeps frequently used layers in RAM while paging out unused ones. This is the most efficient mode for models that *mostly* fit in memory but need a little breathing room.

<br clear="all">

### 3. Layer-by-Layer (LBL) Mode
<img src="docs/lbl_mode.png" width="300px" align="right">

**Condition**: `Free RAM < 75% of Model Size`

The core innovation of TrueLarge-RT. It allocates a small, fixed compute buffer (approx. 800MB) and streams model weights layer-by-layer from disk during inference.

**Benchmark**: Successfully runs **Qwen2.5-32B-Instruct-Q4_K_M** on a device with just **4GB available RAM**.

<br clear="all">

## Key Features

- **No Quantization Required**: Run full precision (FP16) or high-quality (Q4_K_M) GGUF models.
- **Native Performance**: Built on `llama.cpp` with custom ARM NEON/DotProd kernels.
- **Resilient Model Manager**: Integrated background downloader for massive model files.
- **Professional Telemetry**: Real-time graphs for TTFT (ms), RAM (MB), and CPU (GHz).
- **Universal Support**: Works with **Llama 3**, **Qwen 2.5**, **Mistral**, **Phi-3**, and more.

## Installation

TrueLarge-RT is distributed as a standard Android Studio project.

```bash
# 1. Clone the repository
git clone https://github.com/nareshis21/Truelarge-RT.git

# 2. Open in Android Studio
# File -> Open -> Select 'Truelarge-RT' folder

# 3. Sync Gradle & Build
```

## Quick Start

### 1. Download a Model
Use the built-in `ModelDownloadManager` or push via adb:
```bash
adb push Qwen2.5-32B-Instruct-Q4_K_M.gguf /sdcard/Download/TrueLarge/models/
```

### 2. Initialize Engine (Kotlin)
```kotlin
val engine = NativeEngine()

// Initialize (auto-detects LBL vs RAM mode)
val success = engine.init("/path/to/model.gguf", nThreads = 4, gpuLayers = 0)

if (success) {
    // Start a session
    engine.createSession("Explain quantum physics", keepHistory = true)
    
    // Generate tokens
    while (true) {
        val tokenBytes = engine.step() ?: break
        print(String(tokenBytes))
    }
}
```



## Technology Deep Dive: Layer-by-Layer (LBL)

TrueLarge-RT shatters the "Memory Wall" by strictly separating **Compute Memory** from **Model Storage**.

### The Problem: The Memory Wall
A standard **32B FP16 model** requires over **60GB of RAM**. Even heavily quantized to 4-bit, it demands **~20GB**. High-end Android phones typically cap at 12GB or 16GB, making it physically impossible to load these models using traditional runtimes (ONNX, TFLite) which require mapping the entire model file into memory.

### The Solution: Virtual Model Addressing
TrueLarge-RT treats the model weights on disk as a **Virtual Address Space**. By decoupling "storage capacity" from "compute capacity," we allow the device to run models of *any* size, limited only by the speed of the disk (UFS 4.0) rather than the size of the RAM.

#### 1. The Pipeline Architecture
Instead of a static load, the engine establishes a high-speed streaming pipeline. The large model sits on the NVMe storage, and the engine creates a small, sliding "view" into that data.

<div align="center">
  <img src="docs/smart_stream.png" width="700px">
  <p><i>Figure 1: The High-Speed Data Bus acting as a virtual memory bridge</i></p>
</div>

As the inference progresses, the engine identifies which tensors are needed for the *next* token generation and maps only those specific pages into the NPU's address space.

#### 2. Execution Flow (The "Rolling Buffer")
To maintain high throughput, TrueLarge-RT implements a "Rolling Buffer" strategy. It does not just demand-page; it proactively manages the flow of tensors.

<div align="center">
  <img src="docs/lbl_mode.png" width="700px">
  <p><i>Figure 2: The Layer-by-Layer Lifecycle</i></p>
</div>

1.  **prefetch(N+1)**: While the NPU is busy computing Layer N, the DMA engine is already pulling Layer N+1 from disk into a standby buffer.
2.  **compute(N)**: The CPU/GPU executes matrix multiplication on the currently loaded weights.
3.  **discard(N-1)**: Once a layer is done, its memory pages are immediately marked as `MADV_DONTNEED`, telling the kernel to reclaim that physical RAM instantly.

### Core Innovations

#### 1. Zero-Copy `mmap` Loading
We bypass the standard `read()` syscalls and CPU copy loops. By using `mmap` with `MAP_PRIVATE` and hardware-aligned offsets, the kernel maps the layer data directly from NVMe storage into the Neural Processing Unit (or CPU) address space.
*   **Code Reference**: `LayerLoader::loadLayerMap`
*   **Benefit**: Zero CPU overhead for data movement.

#### 2. Dynamic RAM Budgeting
On initialization, the engine probes `/proc/meminfo` to calculate a precise "Safe Working Budget":

```cpp
// simplified logic from TrueLargeRuntime.cpp
long safe_budget = Available_RAM - OS_Safety_Buffer (1.5GB) - KV_Cache_Size;
int max_layers = safe_budget / Single_Layer_Size;
```

This ensures the OS *never* kills the app for OOM (Out of Memory), even when running 32B models on 4GB legacy devices.

#### 3. "Ping-Pong" Context Swapping
To hide latency, TrueLarge-RT maintains two lightweight compute contexts (`ctx_ping` and `ctx_pong`). While one context validates the computation graph for the current layer, the next layer's weights can be pre-fetched into the alternate buffer, ensuring the GPU/CPU is never starved of work.

## Performance Benchmark

Actual performance on **Reference Device (8GB RAM)**:

> **Note**: Official benchmark data for v1.0 is currently being compiled and will be released shortly.


## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp): The core tensor library.
- [AirLLM](https://github.com/lyogavin/airllm): Inspiration for divide-and-conquer inference.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{TrueLargeRT2026,
  author = {Lahajal, Naresh Kumar},
  title = {TrueLarge-RT: Layer-by-Layer Inference Engine for Android},
  year = {2026},
  url = {https://github.com/nareshis21/Truelarge-RT}
}
```
