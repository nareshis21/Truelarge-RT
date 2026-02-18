# Mixture-of-Experts (MoE) Inference Plan

## 🎯 Objective
Enable efficient inference for MoE models (e.g., Mixtral 8x7B, Qwen1.5-32B-MoE) on Android devices by extending the Layer-by-Layer (LBL) engine to support **Expert-by-Expert (EBE)** loading.

## 🧠 The Problem
MoE models have a massive parameter count (e.g., 47B) but only use a fraction of them per token (e.g., 13B active).
*   **Standard LBL**: Loads *all* experts for every layer, wasting I/O bandwidth and RAM.
*   **Result**: Extremely slow performance as we load 8 experts but only use 2.

## 💡 The Solution: Dynamic Expert Loading
Instead of loading the entire "Feed Forward Network" (FFN) layer, we parse the "Gating Network" output first to identify the top-k experts needed for the current token.

### Pipeline Changes

#### 1. Gating Pre-Computation
*   **Current**: Load Layer N -> Compute.
*   **New**: Load Layer N Gate -> Compute Top-K Indices -> Load Only Expert[A] and Expert[B] -> Compute.

#### 2. Sparse GGUF Mapping
We need to modify `LayerLoader.cpp` to map non-contiguous memory chunks.
*   GGUF stores experts sequentially: `[Expert 1][Expert 2]...[Expert 8]`
*   We need `mmap` to support "Sparse Views" or just map the specific offsets for Expert A and B.

#### 3. RAM Management
*   **LRU Cache for Experts**: Experts are often reused across tokens. We should keep "hot" experts in RAM and evict "cold" ones.
*   **Budget**: `Available_RAM = Base_Layers + Hot_Experts_Cache`.

## 🛠️ Implementation Steps

### Phase 1: GGUF & Metadata Support
- [ ] Parse MoE-specific tensor keys in `llama.cpp` / GGUF (e.g., `blk.0.ffn_gate.weight`).
- [ ] Extract expert count and active/top_k count from model metadata.

### Phase 2: The "Expert Scheduler"
- [ ] Create `ExpertScheduler` class in C++.
- [ ] Implement `predict_experts(layer_idx, hidden_states)`: Returns list of required expert IDs.
- [ ] Modify `LayerLoader` to accept `List<ExpertID>` and load specific byte ranges.

### Phase 3: Kernel Optimization
- [ ] Optimize matrix multiplication for smaller, fragmented expert tensors (batch size = 1 is tricky for GPU).
- [ ] Explore "Expert Parallelism" if using CPU + GPU (load Expert A on CPU, Expert B on GPU).

## 📱 Android Integration
*   **UI**: Add a badge "MoE" in the catalog.
*   **Settings**: Add "Expert Cache Size" slider (e.g., keep 2GB of experts in RAM).

## ⚠️ Challenges & Risks
1.  **I/O Latency**: Seeking on flash storage for random expert reads might be slower than sequential read. **Mitigation**: Prefetch next token's likely experts.
2.  **Memory Fragmentation**: Mapping many small expert chunks might exhaust virtual memory areas (VMA) limit. **Mitigation**: Use a pre-allocated pool and `read()` into it instead of `mmap()` for small chunks.
