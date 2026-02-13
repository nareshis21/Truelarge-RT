flowchart TD
    classDef ui fill:#4a90e2,stroke:#333,stroke-width:2px,color:white
    classDef engine fill:#50c878,stroke:#333,stroke-width:2px,color:white
    classDef native fill:#e67e22,stroke:#333,stroke-width:2px,color:white
    classDef hw fill:#7f8c8d,stroke:#333,stroke-width:2px,color:white
    classDef telemetry fill:#9b59b6,stroke:#333,stroke-width:1px,color:white,stroke-dasharray: 5 5

    subgraph UILayer [UI Layer Kotlin]
        Main["MainActivity.kt (Benchmark Mode)"]:::ui
        UI["User Interface - Benchmark Button"]:::ui
    end

    subgraph OrchestrationLayer [Orchestration Layer Kotlin]
        Engine["LlamaEngine.kt"]:::engine
        Config["Config: nThreads=3, nCtx=1024"]:::engine
        Affinity["Hybrid Affinity Logic"]:::engine
        Telemetry["Freq Monitor (/sys/class/cpu)"]:::telemetry
    end

    subgraph NativeBridge [Native Bridge JNI]
        JNI["NativeBridge.kt / jni_bridge.cpp"]:::native
        API["llama_init, llama_eval, llama_sample"]:::native
        KVC["llama_memory_seq_rm (KV Clear)"]:::native
    end

    subgraph NativeCore [Native Core C++]
        Llama["llama.cpp Engine (v3.x)"]:::native
        GGML["GGML Backend (CPU Optimized)"]:::native
        NEON["ARM NEON / DotProd Instructions"]:::native
    end

    subgraph Hardware [Hardware Dimensity 7200]
        Big1["Core 6 (A715) - Thread 1"]:::hw
        Big2["Core 7 (A715) - Thread 2"]:::hw
        Little["Core 0 (A510) - Thread 3"]:::hw
        Memory["6GB RAM Shared"]:::hw
    end

    %% Flow
    UI -->|Click Benchmark| Main
    Main -->|Initialize| Engine
    Main -->|Reset Context| Engine
    Engine -->|Detect Cores| Affinity
    Affinity -->|Pin Threads| JNI

    Engine -->|Generate| JNI
    JNI -->|Eval/Sample| Llama
    Llama -->|Compute| GGML
    GGML -->|Execute| NEON
    
    NEON -->|Workload| Big1
    NEON -->|Workload| Big2
    NEON -->|Workload| Little
    
    %% Telemetry Loop
    Engine -.->|Read Freq Every 10 Tokens| Telemetry
    Telemetry -.->|Log Status| Main
    
    %% Data Flow
    Llama -->|Stream Tokens| JNI
    JNI -->|Callback| Engine
    Engine -->|Flow String| Main
    Main -->|Update UI| UI

    %% Reset Loop
    Main -->|Next Question| Engine
    Engine -->|Call Clear| KVC
    KVC -->|Wipe Context| Llama
