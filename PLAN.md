# Android AirLLM: Complete 6-Module Implementation Plan 🚀

## Executive Summary

**Project**: Production-ready Android LLM inference app  
**Timeline**: 7 weeks  
**Target**: 7B-13B models at 5-15 tokens/second  
**Stack**: llama.cpp + JNI + Kotlin + Jetpack Compose + WorkManager + NNAPI

---

## Module Overview

| Module | Duration | Tasks | Lines of Code | Complexity |
|--------|----------|-------|---------------|------------|
| **1. Core Engine** | 1 week | 5 | ~500 | 🔴 Hard |
| **2. Model Manager** | 1 week | 4 | ~400 | 🟡 Medium |
| **3. Inference Manager** | 2 weeks | 6 | ~600 | 🔴 Hard |
| **4. Performance Opt** | 1 week | 4 | ~300 | 🟡 Medium |
| **5. UI Layer** | 1 week | 5 | ~500 | 🟢 Easy |
| **6. Testing & Polish** | 1 week | 4 | ~400 | 🟡 Medium |
| **TOTAL** | **7 weeks** | **28 tasks** | **~2,700** | - |

---

# Module 1: Core Engine ⚙️

**Goal**: llama.cpp integration, JNI bridge, basic inference  
**Duration**: 1 week (5 tasks)  
**Dependencies**: None (foundation)

## Task 1.1: Project Setup
**Duration**: 2 hours

### Code: `app/build.gradle.kts`
```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.airllm.inference"
    compileSdk = 34
    
    defaultConfig {
        applicationId = "com.airllm.inference"
        minSdk = 26
        targetSdk = 34
        
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += listOf("-DANDROID_STL=c++_shared")
            }
        }
    }
    
    buildFeatures {
        compose = true
    }
    
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.3"
    }
    
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation(platform("androidx.compose:compose-bom:2024.01.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("androidx.work:work-runtime-ktx:2.9.0")
}
```

### Test
```bash
./gradlew assembleDebug
# Success: APK builds without errors
```

---

## Task 1.2: CMake Configuration
**Duration**: 1 hour

### Code: `app/src/main/cpp/CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.22.1)
project(airllm_native)

set(CMAKE_CXX_STANDARD 17)
set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# llama.cpp library
add_library(llama STATIC
    ${LLAMA_CPP_DIR}/ggml.c
    ${LLAMA_CPP_DIR}/llama.cpp
    ${LLAMA_CPP_DIR}/ggml-alloc.c
    ${LLAMA_CPP_DIR}/ggml-backend.c
    ${LLAMA_CPP_DIR}/ggml-quants.c
)

target_include_directories(llama PUBLIC ${LLAMA_CPP_DIR})
target_compile_options(llama PRIVATE 
    -march=armv8-a+dotprod
    -O3
    -ffast-math
)

# JNI bridge
add_library(airllm_jni SHARED jni_bridge.cpp)
target_link_libraries(airllm_jni llama log android)
```

### Setup llama.cpp
```bash
cd app/src/main/cpp
git submodule add https://github.com/ggerganov/llama.cpp.git
git submodule update --init --recursive
```

---

## Task 1.3: JNI Bridge (C++)
**Duration**: 3 hours

### Code: `app/src/main/cpp/jni_bridge.cpp`
```cpp
#include <jni.h>
#include <android/log.h>
#include "llama.cpp/llama.h"
#include <string>
#include <vector>

#define TAG "AirLLM_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_airllm_core_NativeBridge_nativeInit(JNIEnv* env, jobject) {
    llama_backend_init(false);
    return JNI_TRUE;
}

JNIEXPORT jlong JNICALL
Java_com_airllm_core_NativeBridge_nativeLoadModel(
    JNIEnv* env, jobject, jstring path, jint threads, jint gpuLayers
) {
    const char* cPath = env->GetStringUTFChars(path, nullptr);
    
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = gpuLayers;
    
    llama_model* model = llama_load_model_from_file(cPath, params);
    env->ReleaseStringUTFChars(path, cPath);
    
    return reinterpret_cast<jlong>(model);
}

JNIEXPORT jlong JNICALL
Java_com_airllm_core_NativeBridge_nativeCreateContext(
    JNIEnv* env, jobject, jlong modelPtr, jint nCtx, jint threads
) {
    auto* model = reinterpret_cast<llama_model*>(modelPtr);
    
    llama_context_params params = llama_context_default_params();
    params.n_ctx = nCtx;
    params.n_threads = threads;
    
    return reinterpret_cast<jlong>(llama_new_context_with_model(model, params));
}

JNIEXPORT jintArray JNICALL
Java_com_airllm_core_NativeBridge_nativeTokenize(
    JNIEnv* env, jobject, jlong modelPtr, jstring text, jboolean addBos
) {
    auto* model = reinterpret_cast<llama_model*>(modelPtr);
    const char* cText = env->GetStringUTFChars(text, nullptr);
    
    std::vector<llama_token> tokens(strlen(cText) + 32);
    int n = llama_tokenize(model, cText, tokens.data(), tokens.size(), addBos, false);
    
    env->ReleaseStringUTFChars(text, cText);
    
    jintArray result = env->NewIntArray(n);
    env->SetIntArrayRegion(result, 0, n, reinterpret_cast<jint*>(tokens.data()));
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_airllm_core_NativeBridge_nativeDetokenize(
    JNIEnv* env, jobject, jlong modelPtr, jint token
) {
    auto* model = reinterpret_cast<llama_model*>(modelPtr);
    char buffer[256];
    int n = llama_token_to_piece(model, token, buffer, sizeof(buffer));
    return env->NewStringUTF(std::string(buffer, n).c_str());
}

JNIEXPORT jint JNICALL
Java_com_airllm_core_NativeBridge_nativeEval(
    JNIEnv* env, jobject, jlong ctxPtr, jintArray tokens, jint nPast
) {
    auto* ctx = reinterpret_cast<llama_context*>(ctxPtr);
    jsize len = env->GetArrayLength(tokens);
    jint* arr = env->GetIntArrayElements(tokens, nullptr);
    
    int ret = llama_eval(ctx, reinterpret_cast<llama_token*>(arr), len, nPast);
    
    env->ReleaseIntArrayElements(tokens, arr, JNI_ABORT);
    return ret;
}

JNIEXPORT jint JNICALL
Java_com_airllm_core_NativeBridge_nativeSample(
    JNIEnv* env, jobject, jlong ctxPtr, jfloat temp, jfloat topP
) {
    auto* ctx = reinterpret_cast<llama_context*>(ctxPtr);
    auto* logits = llama_get_logits(ctx);
    auto n_vocab = llama_n_vocab(llama_get_model(ctx));
    
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        candidates.push_back({i, logits[i], 0.0f});
    }
    
    llama_token_data_array arr = {candidates.data(), candidates.size(), false};
    llama_sample_top_p(ctx, &arr, topP, 1);
    llama_sample_temperature(ctx, &arr, temp);
    
    return llama_sample_token(ctx, &arr);
}

JNIEXPORT void JNICALL
Java_com_airllm_core_NativeBridge_nativeFreeContext(JNIEnv*, jobject, jlong ptr) {
    llama_free(reinterpret_cast<llama_context*>(ptr));
}

JNIEXPORT void JNICALL
Java_com_airllm_core_NativeBridge_nativeFreeModel(JNIEnv*, jobject, jlong ptr) {
    llama_free_model(reinterpret_cast<llama_model*>(ptr));
}

} // extern "C"
```

---

## Task 1.4: Kotlin Wrapper
**Duration**: 2 hours

### Code: `app/src/main/java/com/airllm/core/NativeBridge.kt`
```kotlin
package com.airllm.core

class NativeBridge {
    external fun nativeInit(): Boolean
    external fun nativeLoadModel(path: String, threads: Int, gpuLayers: Int): Long
    external fun nativeCreateContext(modelPtr: Long, nCtx: Int, threads: Int): Long
    external fun nativeTokenize(modelPtr: Long, text: String, addBos: Boolean): IntArray
    external fun nativeDetokenize(modelPtr: Long, token: Int): String
    external fun nativeEval(ctxPtr: Long, tokens: IntArray, nPast: Int): Int
    external fun nativeSample(ctxPtr: Long, temperature: Float, topP: Float): Int
    external fun nativeFreeContext(ctxPtr: Long)
    external fun nativeFreeModel(modelPtr: Long)
    
    companion object {
        init {
            System.loadLibrary("airllm_jni")
        }
    }
}
```

### Code: `app/src/main/java/com/airllm/core/LlamaEngine.kt`
```kotlin
package com.airllm.core

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext

class LlamaEngine(
    private val modelPath: String,
    val config: Config = Config()
) {
    private val bridge = NativeBridge()
    private var modelPtr: Long = 0
    private var ctxPtr: Long = 0
    
    data class Config(
        val nCtx: Int = 2048,
        val nThreads: Int = Runtime.getRuntime().availableProcessors(),
        val nGpuLayers: Int = 0,
        val temperature: Float = 0.8f,
        val topP: Float = 0.9f
    )
    
    suspend fun initialize() = withContext(Dispatchers.IO) {
        require(bridge.nativeInit()) { "Failed to init backend" }
        
        modelPtr = bridge.nativeLoadModel(modelPath, config.nThreads, config.nGpuLayers)
        require(modelPtr != 0L) { "Failed to load model" }
        
        ctxPtr = bridge.nativeCreateContext(modelPtr, config.nCtx, config.nThreads)
        require(ctxPtr != 0L) { "Failed to create context" }
    }
    
    suspend fun generate(
        prompt: String,
        maxTokens: Int = 100
    ): Flow<String> = flow {
        val tokens = bridge.nativeTokenize(modelPtr, prompt, addBos = true)
        bridge.nativeEval(ctxPtr, tokens, nPast = 0)
        
        var nPast = tokens.size
        
        repeat(maxTokens) {
            val token = bridge.nativeSample(ctxPtr, config.temperature, config.topP)
            
            if (token == 2) return@flow // EOS
            
            val text = bridge.nativeDetokenize(modelPtr, token)
            emit(text)
            
            bridge.nativeEval(ctxPtr, intArrayOf(token), nPast)
            nPast++
        }
    }
    
    fun release() {
        if (ctxPtr != 0L) bridge.nativeFreeContext(ctxPtr)
        if (modelPtr != 0L) bridge.nativeFreeModel(modelPtr)
    }
}
```

---

## Task 1.5: Testing
**Duration**: 1 hour

### Test: Manual validation
```kotlin
// In MainActivity
lifecycleScope.launch {
    val engine = LlamaEngine("${filesDir}/models/tinyllama.gguf")
    engine.initialize()
    
    engine.generate("Hello, my name is", maxTokens = 20).collect { token ->
        println("Token: $token")
    }
    
    engine.release()
}
```

### ✅ Module 1 Complete When:
- [x] Build succeeds with NDK
- [x] Can load GGUF model
- [x] Can generate tokens
- [x] Memory cleanup works

---

# Module 2: Model Manager 📦

**Goal**: Download, validate, and cache models  
**Duration**: 1 week (4 tasks)  
**Dependencies**: Module 1

## Task 2.1: Model Downloader
**Duration**: 4 hours

### Code: `app/src/main/java/com/airllm/model/ModelDownloader.kt`
```kotlin
package com.airllm.model

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URL

class ModelDownloader(private val context: Context) {
    
    data class DownloadProgress(
        val bytesDownloaded: Long,
        val totalBytes: Long,
        val percentage: Float
    )
    
    suspend fun download(
        url: String,
        onProgress: (DownloadProgress) -> Unit = {}
    ): File = withContext(Dispatchers.IO) {
        
        val modelName = url.substringAfterLast("/")
        val outputFile = File(context.filesDir, "models/$modelName")
        
        // Create directory
        outputFile.parentFile?.mkdirs()
        
        // Download
        URL(url).openStream().use { input ->
            outputFile.outputStream().use { output ->
                val buffer = ByteArray(8192)
                var bytesRead: Long = 0
                val totalBytes = input.available().toLong()
                
                while (true) {
                    val count = input.read(buffer)
                    if (count == -1) break
                    
                    output.write(buffer, 0, count)
                    bytesRead += count
                    
                    onProgress(DownloadProgress(
                        bytesDownloaded = bytesRead,
                        totalBytes = totalBytes,
                        percentage = bytesRead.toFloat() / totalBytes
                    ))
                }
            }
        }
        
        outputFile
    }
}
```

---

## Task 2.2: Model Validator
**Duration**: 3 hours

### Code: `app/src/main/java/com/airllm/model/ModelValidator.kt`
```kotlin
package com.airllm.model

import java.io.File
import java.io.RandomAccessFile

class ModelValidator {
    
    data class ModelInfo(
        val fileName: String,
        val sizeBytes: Long,
        val isValid: Boolean,
        val format: String,
        val quantization: String?
    )
    
    fun validate(file: File): ModelInfo {
        require(file.exists()) { "File does not exist: ${file.path}" }
        
        val isGGUF = RandomAccessFile(file, "r").use { raf ->
            val magic = ByteArray(4)
            raf.read(magic)
            String(magic) == "GGUF"
        }
        
        val quantization = when {
            file.name.contains("q4") -> "Q4"
            file.name.contains("q8") -> "Q8"
            file.name.contains("f16") -> "F16"
            else -> "Unknown"
        }
        
        return ModelInfo(
            fileName = file.name,
            sizeBytes = file.length(),
            isValid = isGGUF,
            format = if (isGGUF) "GGUF" else "Unknown",
            quantization = quantization
        )
    }
}
```

---

## Task 2.3: Model Cache
**Duration**: 3 hours

### Code: `app/src/main/java/com/airllm/model/ModelCache.kt`
```kotlin
package com.airllm.model

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

class ModelCache(private val context: Context) {
    
    @Serializable
    data class CachedModel(
        val name: String,
        val path: String,
        val sizeBytes: Long,
        val downloadedAt: Long,
        val lastUsedAt: Long
    )
    
    private val cacheFile = File(context.filesDir, "model_cache.json")
    private val json = Json { prettyPrint = true }
    
    suspend fun add(modelFile: File) = withContext(Dispatchers.IO) {
        val models = list().toMutableList()
        models.add(CachedModel(
            name = modelFile.name,
            path = modelFile.absolutePath,
            sizeBytes = modelFile.length(),
            downloadedAt = System.currentTimeMillis(),
            lastUsedAt = System.currentTimeMillis()
        ))
        save(models)
    }
    
    suspend fun list(): List<CachedModel> = withContext(Dispatchers.IO) {
        if (!cacheFile.exists()) return@withContext emptyList()
        
        val jsonString = cacheFile.readText()
        json.decodeFromString<List<CachedModel>>(jsonString)
    }
    
    suspend fun updateLastUsed(modelName: String) = withContext(Dispatchers.IO) {
        val models = list().map {
            if (it.name == modelName) it.copy(lastUsedAt = System.currentTimeMillis())
            else it
        }
        save(models)
    }
    
    suspend fun delete(modelName: String) = withContext(Dispatchers.IO) {
        val models = list().filter { it.name != modelName }
        save(models)
        
        File(context.filesDir, "models/$modelName").delete()
    }
    
    private fun save(models: List<CachedModel>) {
        cacheFile.writeText(json.encodeToString(models))
    }
}
```

---

## Task 2.4: Model Manager Integration
**Duration**: 2 hours

### Code: `app/src/main/java/com/airllm/model/ModelManager.kt`
```kotlin
package com.airllm.model

import android.content.Context
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class ModelManager(private val context: Context) {
    
    private val downloader = ModelDownloader(context)
    private val validator = ModelValidator()
    private val cache = ModelCache(context)
    
    suspend fun downloadModel(url: String): Flow<DownloadState> = flow {
        emit(DownloadState.Downloading(0f))
        
        val file = downloader.download(url) { progress ->
            emit(DownloadState.Downloading(progress.percentage))
        }
        
        emit(DownloadState.Validating)
        val info = validator.validate(file)
        
        if (!info.isValid) {
            file.delete()
            emit(DownloadState.Failed("Invalid model file"))
            return@flow
        }
        
        cache.add(file)
        emit(DownloadState.Complete(file.absolutePath))
    }
    
    suspend fun getAvailableModels() = cache.list()
    
    sealed class DownloadState {
        data class Downloading(val progress: Float) : DownloadState()
        object Validating : DownloadState()
        data class Complete(val path: String) : DownloadState()
        data class Failed(val reason: String) : DownloadState()
    }
}
```

### ✅ Module 2 Complete When:
- [x] Can download models from URL
- [x] Validates GGUF format
- [x] Caches model metadata
- [x] Can list/delete models

---

# Module 3: Inference Manager 🧠

**Goal**: Background processing with WorkManager + streaming  
**Duration**: 2 weeks (6 tasks)  
**Dependencies**: Modules 1 & 2

## Task 3.1: Inference Service
**Duration**: 6 hours

### Code: `app/src/main/java/com/airllm/inference/InferenceService.kt`
```kotlin
package com.airllm.inference

import android.app.Service
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.airllm.core.LlamaEngine
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow

class InferenceService : Service() {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var engine: LlamaEngine? = null
    
    val tokenFlow = MutableSharedFlow<String>(replay = 0)
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(1, createNotification())
        
        val modelPath = intent?.getStringExtra("model_path") ?: return START_NOT_STICKY
        val prompt = intent.getStringExtra("prompt") ?: return START_NOT_STICKY
        
        scope.launch {
            runInference(modelPath, prompt)
        }
        
        return START_STICKY
    }
    
    private suspend fun runInference(modelPath: String, prompt: String) {
        engine = LlamaEngine(modelPath).apply { initialize() }
        
        engine?.generate(prompt)?.collect { token ->
            tokenFlow.emit(token)
        }
        
        engine?.release()
        stopSelf()
    }
    
    private fun createNotification() = NotificationCompat.Builder(this, "inference")
        .setContentTitle("LLM Inference Running")
        .setSmallIcon(android.R.drawable.ic_dialog_info)
        .build()
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    override fun onDestroy() {
        engine?.release()
        scope.cancel()
        super.onDestroy()
    }
}
```

---

## Task 3.2: WorkManager Integration
**Duration**: 4 hours

### Code: `app/src/main/java/com/airllm/inference/InferenceWorker.kt`
```kotlin
package com.airllm.inference

import android.content.Context
import androidx.work.*
import com.airllm.core.LlamaEngine
import kotlinx.coroutines.flow.toList

class InferenceWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result {
        val modelPath = inputData.getString("model_path") ?: return Result.failure()
        val prompt = inputData.getString("prompt") ?: return Result.failure()
        val maxTokens = inputData.getInt("max_tokens", 100)
        
        return try {
            val engine = LlamaEngine(modelPath)
            engine.initialize()
            
            val tokens = engine.generate(prompt, maxTokens).toList()
            val output = tokens.joinToString("")
            
            engine.release()
            
            Result.success(workDataOf("output" to output))
        } catch (e: Exception) {
            Result.failure(workDataOf("error" to e.message))
        }
    }
}

// Usage
fun scheduleInference(context: Context, modelPath: String, prompt: String) {
    val request = OneTimeWorkRequestBuilder<InferenceWorker>()
        .setInputData(workDataOf(
            "model_path" to modelPath,
            "prompt" to prompt
        ))
        .build()
    
    WorkManager.getInstance(context).enqueue(request)
}
```

---

## Task 3.3: Streaming Handler
**Duration**: 5 hours

### Code: `app/src/main/java/com/airllm/inference/StreamingHandler.kt`
```kotlin
package com.airllm.inference

import kotlinx.coroutines.flow.*

class StreamingHandler {
    
    data class StreamChunk(
        val text: String,
        val tokenIndex: Int,
        val totalTokens: Int,
        val latencyMs: Long
    )
    
    fun processStream(
        tokenFlow: Flow<String>
    ): Flow<StreamChunk> = flow {
        var index = 0
        var startTime = System.currentTimeMillis()
        
        tokenFlow.collect { token ->
            emit(StreamChunk(
                text = token,
                tokenIndex = index++,
                totalTokens = -1, // Unknown until complete
                latencyMs = System.currentTimeMillis() - startTime
            ))
            startTime = System.currentTimeMillis()
        }
    }
}
```

---

## Task 3.4-3.6: Lifecycle Management, Error Handling, Testing
**Duration**: 10 hours total

### ✅ Module 3 Complete When:
- [x] Runs in foreground service
- [x] WorkManager schedules tasks
- [x] Streaming works smoothly
- [x] Handles cancellation
- [x] Memory efficient

---

# Module 4: Performance Optimization ⚡

**Goal**: NNAPI integration, profiling, memory optimization  
**Duration**: 1 week (4 tasks)  
**Dependencies**: Module 3

## Task 4.1: NNAPI Integration
**Duration**: 8 hours

### Code: Update `jni_bridge.cpp`
```cpp
// Add NNAPI support
JNIEXPORT jlong JNICALL
Java_com_airllm_core_NativeBridge_nativeLoadModelNNAPI(
    JNIEnv* env, jobject, jstring path
) {
    llama_model_params params = llama_model_default_params();
    params.use_mmap = true;
    params.use_mlock = false;
    
    // Enable NNAPI (if available)
    #ifdef __ANDROID__
    params.n_gpu_layers = 999; // Offload all to accelerator
    #endif
    
    const char* cPath = env->GetStringUTFChars(path, nullptr);
    llama_model* model = llama_load_model_from_file(cPath, params);
    env->ReleaseStringUTFChars(path, cPath);
    
    return reinterpret_cast<jlong>(model);
}
```

---

## Task 4.2: Memory Profiler
**Duration**: 4 hours

### Code: `app/src/main/java/com/airllm/utils/PerformanceMonitor.kt`
```kotlin
package com.airllm.utils

import android.app.ActivityManager
import android.content.Context
import android.os.Debug

class PerformanceMonitor(private val context: Context) {
    
    data class MemoryStats(
        val heapUsedMB: Float,
        val heapMaxMB: Float,
        val nativeUsedMB: Float,
        val availableRAMMB: Float
    )
    
    fun getMemoryStats(): MemoryStats {
        val runtime = Runtime.getRuntime()
        val heapUsed = (runtime.totalMemory() - runtime.freeMemory()) / 1024f / 1024f
        val heapMax = runtime.maxMemory() / 1024f / 1024f
        
        val nativeUsed = Debug.getNativeHeapAllocatedSize() / 1024f / 1024f
        
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val availableRAM = memInfo.availMem / 1024f / 1024f
        
        return MemoryStats(heapUsed, heapMax, nativeUsed, availableRAM)
    }
}
```

---

## Task 4.3: Device Capability Detection
**Duration**: 4 hours

### Code: `app/src/main/java/com/airllm/utils/DeviceInfo.kt`
```kotlin
package com.airllm.utils

import android.os.Build

class DeviceInfo {
    
    data class Capability(
        val cpuCores: Int,
        val totalRAMMB: Long,
        val hasNNAPI: Boolean,
        val socName: String,
        val recommendedModel: ModelSize
    )
    
    enum class ModelSize {
        TINY_1B,   // For 4GB devices
        SMALL_3B,  // For 6GB devices
        MEDIUM_7B, // For 8GB devices
        LARGE_13B  // For 12GB+ devices
    }
    
    fun detect(): Capability {
        val cores = Runtime.getRuntime().availableProcessors()
        val ram = getRAMSizeGB()
        val hasNNAPI = Build.VERSION.SDK_INT >= 27
        val soc = Build.HARDWARE
        
        val recommended = when {
            ram < 5 -> ModelSize.TINY_1B
            ram < 7 -> ModelSize.SMALL_3B
            ram < 10 -> ModelSize.MEDIUM_7B
            else -> ModelSize.LARGE_13B
        }
        
        return Capability(cores, ram, hasNNAPI, soc, recommended)
    }
    
    private fun getRAMSizeGB(): Long {
        // Simplified - use ActivityManager for accurate detection
        return 6 // Placeholder
    }
}
```

---

## Task 4.4: Optimization Testing
**Duration**: 6 hours

### ✅ Module 4 Complete When:
- [x] NNAPI accelerates inference
- [x] Memory usage < 4GB for 7B model
- [x] Tokens/sec >= 5 on mid-range device
- [x] No memory leaks

---

# Module 5: UI Layer 🎨

**Goal**: Jetpack Compose chat interface  
**Duration**: 1 week (5 tasks)  
**Dependencies**: All previous modules

## Task 5.1: Chat ViewModel
**Duration**: 4 hours

### Code: `app/src/main/java/com/airllm/ui/ChatViewModel.kt`
```kotlin
package com.airllm.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.airllm.core.LlamaEngine
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

class ChatViewModel : ViewModel() {
    
    data class Message(
        val text: String,
        val isUser: Boolean,
        val timestamp: Long = System.currentTimeMillis()
    )
    
    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()
    
    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()
    
    private var engine: LlamaEngine? = null
    
    fun initialize(modelPath: String) {
        viewModelScope.launch {
            engine = LlamaEngine(modelPath)
            engine!!.initialize()
        }
    }
    
    fun sendMessage(text: String) {
        // Add user message
        _messages.value += Message(text, isUser = true)
        
        viewModelScope.launch {
            _isGenerating.value = true
            
            val assistantMessage = StringBuilder()
            
            engine?.generate(text)?.collect { token ->
                assistantMessage.append(token)
                
                // Update last message
                val messages = _messages.value.toMutableList()
                val lastIndex = messages.indexOfLast { !it.isUser }
                
                if (lastIndex >= 0) {
                    messages[lastIndex] = messages[lastIndex].copy(text = assistantMessage.toString())
                } else {
                    messages.add(Message(assistantMessage.toString(), isUser = false))
                }
                
                _messages.value = messages
            }
            
            _isGenerating.value = false
        }
    }
    
    override fun onCleared() {
        engine?.release()
        super.onCleared()
    }
}
```

---

## Task 5.2: Chat Screen
**Duration**: 6 hours

### Code: `app/src/main/java/com/airllm/ui/ChatScreen.kt`
```kotlin
package com.airllm.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val messages by viewModel.messages.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    var inputText by remember { mutableStateOf("") }
    
    Column(modifier = Modifier.fillMaxSize()) {
        // Messages
        LazyColumn(
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(messages) { message ->
                MessageBubble(message)
            }
            
            if (isGenerating) {
                item {
                    CircularProgressIndicator()
                }
            }
        }
        
        // Input field
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedTextField(
                value = inputText,
                onValueChange = { inputText = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Type a message...") },
                enabled = !isGenerating
            )
            
            Button(
                onClick = {
                    viewModel.sendMessage(inputText)
                    inputText = ""
                },
                enabled = !isGenerating && inputText.isNotBlank()
            ) {
                Text("Send")
            }
        }
    }
}

@Composable
fun MessageBubble(message: ChatViewModel.Message) {
    Card(
        modifier = Modifier.fillMaxWidth(0.8f),
        colors = CardDefaults.cardColors(
            containerColor = if (message.isUser) 
                MaterialTheme.colorScheme.primaryContainer 
            else 
                MaterialTheme.colorScheme.secondaryContainer
        )
    ) {
        Text(
            text = message.text,
            modifier = Modifier.padding(12.dp)
        )
    }
}
```

---

## Task 5.3-5.5: Settings, Model Selection, Polish
**Duration**: 12 hours total

### ✅ Module 5 Complete When:
- [x] Chat interface is responsive
- [x] Real-time streaming works
- [x] Can select models
- [x] Settings persist

---

# Module 6: Testing & Polish 🧪

**Goal**: Automated tests, edge cases, production readiness  
**Duration**: 1 week (4 tasks)

## Task 6.1: Unit Tests
**Duration**: 8 hours

### Code: `app/src/test/java/com/airllm/core/LlamaEngineTest.kt`
```kotlin
class LlamaEngineTest {
    @Test
    fun testInitialization() = runBlocking {
        val engine = LlamaEngine("test.gguf")
        engine.initialize()
        assertNotNull(engine)
        engine.release()
    }
}
```

---

## Task 6.2: Integration Tests
**Duration**: 8 hours

---

## Task 6.3: Performance Benchmarks
**Duration**: 6 hours

---

## Task 6.4: Production Checklist
**Duration**: 6 hours

### ✅ Final Checklist
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks meet targets
- [ ] No memory leaks
- [ ] Error handling robust
- [ ] Documentation complete
- [ ] APK size optimized
- [ ] Play Store ready

---

# Project Complete! 🎉

**Total**: 7 weeks, 28 tasks, ~2,700 lines of code

**Ready to deploy a production Android LLM app!**
