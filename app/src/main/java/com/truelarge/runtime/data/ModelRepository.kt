package com.truelarge.runtime.data

import android.content.Context
import android.os.Environment
import java.io.File

/**
 * Manages local model storage on external storage under /models/ folder.
 */
class ModelRepository(private val context: Context) {

    /**
     * Returns the models directory on external storage.
     * Creates it if it doesn't exist.
     */
    fun getModelsDir(): File {
        val dir = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
            "TrueLarge/models"
        )
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /**
     * List all locally downloaded GGUF models.
     */
    fun getLocalModels(): List<LocalModel> {
        val dir = getModelsDir()
        if (!dir.exists()) return emptyList()

        return dir.listFiles()
            ?.filter { it.extension.equals("gguf", ignoreCase = true) }
            ?.map { file ->
                LocalModel(
                    name = file.nameWithoutExtension,
                    filename = file.name,
                    path = file.absolutePath,
                    sizeBytes = file.length(),
                    quantization = extractQuantization(file.name),
                    downloadDate = file.lastModified()
                )
            }
            ?.sortedByDescending { it.downloadDate }
            ?: emptyList()
    }

    /**
     * Delete a local model.
     */
    fun deleteModel(path: String): Boolean {
        val file = File(path)
        return if (file.exists()) file.delete() else false
    }

    /**
     * Get the target file path for a download.
     */
    fun getDownloadPath(filename: String): File {
        return File(getModelsDir(), filename)
    }

    /**
     * Get curated list of recommended small GGUF models.
     */
    fun getRecommendedModels(): List<RecommendedModel> = listOf(
        RecommendedModel(
            repoId = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            displayName = "Qwen 2.5 0.5B Instruct",
            description = "Tiny but capable instruction-tuned model",
            parameterSize = "0.5B",
            author = "Qwen"
        ),
        RecommendedModel(
            repoId = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            displayName = "TinyLlama 1.1B Chat",
            description = "Fast and lightweight chat model",
            parameterSize = "1.1B",
            author = "TheBloke"
        ),
        RecommendedModel(
            repoId = "microsoft/Phi-3-mini-4k-instruct-gguf",
            displayName = "Phi-3 Mini 4K",
            description = "Microsoft's compact reasoning model",
            parameterSize = "3.8B",
            author = "Microsoft"
        ),
        RecommendedModel(
            repoId = "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF",
            displayName = "SmolLM2 135M",
            description = "Ultra-small model for testing",
            parameterSize = "135M",
            author = "HuggingFace"
        ),
        RecommendedModel(
            repoId = "bartowski/gemma-2-2b-it-GGUF",
            displayName = "Gemma 2 2B IT",
            description = "Google's efficient instruction model",
            parameterSize = "2B",
            author = "bartowski"
        )
    )

    private fun extractQuantization(filename: String): String {
        val regex = Regex("""[._-]((?:IQ|Q|F|BF)\d[^\s.]*)\.""", RegexOption.IGNORE_CASE)
        val match = regex.find(filename)
        return match?.groupValues?.get(1) ?: "Unknown"
    }
}
