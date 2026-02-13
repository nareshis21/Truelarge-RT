package com.truelarge.runtime.data

/**
 * Represents a model from HuggingFace API search results.
 */
data class ModelInfo(
    val id: String,           // e.g. "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    val author: String,
    val modelName: String,    // display name
    val downloads: Int,
    val likes: Int,
    val tags: List<String>,
    val lastModified: String
)

/**
 * A GGUF file within a HuggingFace model repo.
 */
data class ModelFile(
    val filename: String,     // e.g. "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    val sizeBytes: Long,
    val quantization: String, // extracted from filename: Q4_K_M, Q5_K_S, etc.
    val downloadUrl: String   // direct download URL
) {
    val sizeFormatted: String
        get() {
            val gb = sizeBytes / (1024.0 * 1024.0 * 1024.0)
            val mb = sizeBytes / (1024.0 * 1024.0)
            return if (gb >= 1.0) String.format("%.1f GB", gb)
                   else String.format("%.0f MB", mb)
        }
}

/**
 * A model that has been downloaded to local storage.
 */
data class LocalModel(
    val name: String,
    val filename: String,
    val path: String,
    val sizeBytes: Long,
    val quantization: String,
    val downloadDate: Long
) {
    val sizeFormatted: String
        get() {
            val gb = sizeBytes / (1024.0 * 1024.0 * 1024.0)
            val mb = sizeBytes / (1024.0 * 1024.0)
            return if (gb >= 1.0) String.format("%.1f GB", gb)
                   else String.format("%.0f MB", mb)
        }
}

/**
 * Represents a curated/recommended model entry.
 */
data class RecommendedModel(
    val repoId: String,
    val displayName: String,
    val description: String,
    val parameterSize: String,  // "0.5B", "1.1B", etc.
    val author: String
)
