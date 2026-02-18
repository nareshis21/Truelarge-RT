#include <jni.h>
#include <string>
#include "TrueLargeRuntime.h"
#include <android/log.h>
#include <mutex>

// Global instance (Simple singleton for POC)
// In production, manage this via a Handle passed to Java
static std::unique_ptr<TrueLargeRuntime> engine;
static std::mutex engine_mutex; // Serialize all native calls

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_truelarge_runtime_NativeEngine_init(JNIEnv* env, jobject, jstring modelPath, jint threads, jint gpuLayers) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    const char* cPath = env->GetStringUTFChars(modelPath, nullptr);
    std::string path(cPath);
    env->ReleaseStringUTFChars(modelPath, cPath);

    engine = std::make_unique<TrueLargeRuntime>();
    engine->configure(threads, gpuLayers);
    
    bool success = engine->loadModel(path);
    if (!success) {
        engine.reset();
        return JNI_FALSE;
    }
    
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_truelarge_runtime_NativeEngine_createSession(JNIEnv* env, jobject, jstring prompt, jboolean keepHistory) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    if (!engine) return JNI_FALSE;

    const char* cPrompt = env->GetStringUTFChars(prompt, nullptr);
    std::string sPrompt(cPrompt);
    env->ReleaseStringUTFChars(prompt, cPrompt);

    return engine->createSession(sPrompt, keepHistory == JNI_TRUE) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_truelarge_runtime_NativeEngine_configureSampler(JNIEnv* env, jobject, jfloat temp, jint k, jfloat p) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    if (engine) {
        engine->configureSampler(temp, k, p);
    }
}

JNIEXPORT jbyteArray JNICALL
Java_com_truelarge_runtime_NativeEngine_step(JNIEnv* env, jobject) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    if (!engine) return nullptr;
    std::string piece = engine->step();
    if (piece.empty()) return nullptr;

    jbyteArray bytes = env->NewByteArray(piece.size());
    env->SetByteArrayRegion(bytes, 0, piece.size(), (jbyte*)piece.c_str());
    return bytes;
}

JNIEXPORT jstring JNICALL
Java_com_truelarge_runtime_NativeEngine_getBenchmarkData(JNIEnv* env, jobject) {
    if (!engine) return env->NewStringUTF("0,0,0,0");
    
    // Format: "TTFT,TPS,RAM,CPU,TotalTime,Mode"
    char buf[256];
    snprintf(buf, sizeof(buf), "%.2f,%.2f,%ld,%.2f,%.2f,%s", 
             engine->lastTTFT, engine->lastTPS, engine->lastRAM, engine->lastCPUFreq, engine->lastTotalTime,
             engine->getInferenceMode().c_str());
    
    return env->NewStringUTF(buf);
}

JNIEXPORT void JNICALL
Java_com_truelarge_runtime_NativeEngine_release(JNIEnv* env, jobject) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    engine.reset();
}

JNIEXPORT jint JNICALL
Java_com_truelarge_runtime_NativeEngine_getContextTrain(JNIEnv* env, jobject) {
    if (!engine) return 0;
    return engine->getContextTrain();
}

JNIEXPORT jint JNICALL
Java_com_truelarge_runtime_NativeEngine_getContextCurrent(JNIEnv* env, jobject) {
    if (!engine) return 0;
    return engine->getContextCurrent();
}

JNIEXPORT jstring JNICALL
Java_com_truelarge_runtime_NativeEngine_getInferenceMode(JNIEnv* env, jobject) {
    if (!engine) return env->NewStringUTF("Not Initialized");
    return env->NewStringUTF(engine->getInferenceMode().c_str());
}

} // extern "C"
