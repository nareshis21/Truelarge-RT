package com.truelarge.runtime

class NativeEngine {
    
    // Load native library
    companion object {
        init {
            System.loadLibrary("truelarge-rt")
        }
    }

    // Native methods
    external fun init(modelPath: String, threads: Int, gpuLayers: Int): Boolean
    external fun configureSampler(temp: Float, k: Int, p: Float)
    external fun createSession(prompt: String, keepHistory: Boolean): Boolean
    external fun step(): ByteArray?
    external fun release()
    external fun getContextTrain(): Int
    external fun getContextCurrent(): Int
    external fun getBenchmarkData(): String
    external fun getInferenceMode(): String
}
