#include <jni.h>
#include <string>
#include "TrueLargeRuntime.h"
#include <android/log.h>

// Global instance (Simple singleton for POC)
// In production, manage this via a Handle passed to Java
static std::unique_ptr<TrueLargeRuntime> engine;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_truelarge_runtime_NativeEngine_init(JNIEnv* env, jobject, jstring modelPath, jint threads, jint gpuLayers) {
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
    if (!engine) return JNI_FALSE;

    const char* cPrompt = env->GetStringUTFChars(prompt, nullptr);
    std::string sPrompt(cPrompt);
    env->ReleaseStringUTFChars(prompt, cPrompt);

    return engine->createSession(sPrompt, keepHistory == JNI_TRUE) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_truelarge_runtime_NativeEngine_configureSampler(JNIEnv* env, jobject, jfloat temp, jint k, jfloat p) {
    if (engine) {
        engine->configureSampler(temp, k, p);
    }
}

JNIEXPORT jstring JNICALL
Java_com_truelarge_runtime_NativeEngine_step(JNIEnv* env, jobject) {
    if (!engine) return env->NewStringUTF("");
    std::string piece = engine->step();
    return env->NewStringUTF(piece.c_str());
}

JNIEXPORT void JNICALL
Java_com_truelarge_runtime_NativeEngine_release(JNIEnv* env, jobject) {
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

} // extern "C"
