# TrueLarge-RT: High-Performance GGUF Runtime for Android

TrueLarge-RT is a native Android inference engine built on top of `llama.cpp`. It is designed for maximum efficiency, providing real-time telemetry and advanced benchmarking for Large Language Models (LLMs) on mobile hardware.

![Architecture](docs/architecture.png)

## Key Features

- **Native llama.cpp Integration**: Pure C++ core with **ARM DotProd** acceleration for 2x-4x faster math on modern SoCs.
- **Professional-Grade Benchmarking**: 5-question standardized suite with millisecond-precision TTFT (Time To First Token), and statistical TPS (Avg/Peak/Low/Median) tracking.
- **Real-Time Telemetry Graphs**: Live visualization of inference speed, memory footprint (RSS MB), and CPU clock frequency (GHz) with visual per-question dividers.
- **Surgical CPU Pinning**: Intelligent affinity mapping that targets ultra-performance cores for generation to maximize sustained TPS.
- **Auto-Discovery**: Automatically recognizes manually added GGUF models in `/Downloads/TrueLarge/models`.
- **Smart Memory Management**: Dynamic `mlock` support and RAM-aware loading to prevent OOM crashes.
- **Multi-Turn Persistence**: Optimized KV cache management for fast, conversational multi-turn inference.
- **Developer-First Profiling**: Detailed breakdowns of CPU core ID (#ID) and instantaneous hardware stats.

## Low-RAM Optimization (4GB Devices)

TrueLarge-RT is optimized to run large models (7B - 13B) even on constrained hardware:
- **Smart Paging (Mmap)**: Automatically disables `mlock` when low RAM is detected, leveraging high-speed storage as virtual memory.
- **KV Cache Capping**: Caps context buffers to 2048 tokens to maintain a stable memory footprint.
- **CPU Affinity**: Intelligent thread mapping to "Big" performance cores to minimize latency during memory swaps.

## Architecture: Hybrid Loading Strategy

TrueLarge-RT employs a unique 3-tier loading strategy to enable large models (up to 13B+) on mobile devices with limited RAM (4GB-8GB):

![Architecture Diagram](docs/architecture_hybrid.png)

1.  **Full RAM (mlock)**:
    - **Trigger**: `Free RAM > Model Size + 1GB`.
    - **Behavior**: Locks the entire model in memory to prevent swapping. Delivers maximum speed.
2.  **OS Paging (mmap)**:
    - **Trigger**: `Free RAM > 75% of Model Size`.
    - **Behavior**: Uses standard memory mapping. The OS handles paging pages in/out as needed. This is efficient for models that *mostly* fit in RAM.
3.  **Layer-by-Layer (LBL)**:
    - **Trigger**: `Free RAM < 75% of Model Size`.
    - **Behavior**: Loads one layer at a time from storage, computes, and unloads.
    - **Benefit**: Runs huge models (e.g., 7B on 3GB RAM) that would otherwise crash. Slower, but enables inference on constrained hardware.

## Advanced Telemetry & Benchmarking

The benchmark screen provides high-fidelity hardware profiling for LLM inference:

- **Precision TTFT**: Captured in milliseconds (ms) for high-resolution timing of the initial response delay.
- **Statistical TPS**: Detailed breakdown of generation speed including **Peak**, **Median**, and **Lowest** rates per question.
- **Hardware Monitoring**: Real-time tracking of RAM usage (MB) and CPU Frequency (GHz).
- **Visual Analysis**: Synchronized telemetry graphs with vertical dividers to correlate hardware spikes with model outputs across the 5-question suite.

## Comparison with Other Runtimes

| Feature | TrueLarge-RT | SmolChat | ONNX Runtime | Google AICore |
| :--- | :--- | :--- | :--- | :--- |
| **Core** | `llama.cpp` (Native) | Web/High-level | General Purpose | Proprietary |
| **Model Format** | GGUF (Optimized) | Various | ONNX | Proprietary |
| **Openness** | Any GGUF model | Limited | Broad | Restricted (Gemini) |
| **Telemetry** | High-Res (ms/TPS/RSS) | Minimal | Profiling Tools | Opaque |
| **Persistence** | Native KV Cache | Session-based | Execution Provider | System-level |

### Why TrueLarge-RT?
Unlike general-purpose runtimes like **ONNX**, TrueLarge-RT is laser-focused on the GGUF ecosystem, leveraging `llama.cpp`'s highly optimized ARM NEON and DotProd kernels. Compared to **AICore**, it offers complete freedom—allowing researchers and developers to run any model (Qwen, Llama, Phi, Mistral) without proprietary restrictions.

## Getting Started

1. **Clone the Repo**: `git clone https://github.com/nareshis21/Truelarge-RT.git`
2. **Setup Models**: Paste your `.gguf` files into `/sdcard/Download/TrueLarge/models/`.
3. **Run**: Build via Android Studio and use the **Benchmark** icon to trigger the High-Resolution Hardware Profiler.

## License
MIT License. Built for the open-source LLM community.

## Citation

If you use TrueLarge-RT in your research or project, please cite:

```bibtex
@software{TrueLargeRT2026,
  author = {Lahajal, Naresh Kumar},
  title = {TrueLarge-RT: High-Performance GGUF Runtime for Android},
  year = {2026},
  url = {https://github.com/nareshis21/Truelarge-RT}
}
```
