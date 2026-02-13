package com.truelarge.runtime.data

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * Lightweight HuggingFace API client — no external JSON lib needed.
 */
object HuggingFaceApi {

    private const val TAG = "HuggingFaceApi"
    private const val BASE_URL = "https://huggingface.co/api/models"

    /**
     * Search for GGUF models on HuggingFace.
     */
    suspend fun searchModels(query: String, limit: Int = 20): List<ModelInfo> = withContext(Dispatchers.IO) {
        try {
            val encoded = URLEncoder.encode(query, "UTF-8")
            val url = URL("$BASE_URL?search=$encoded&filter=gguf&sort=downloads&direction=-1&limit=$limit")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.connectTimeout = 10_000
            conn.readTimeout = 15_000

            val response = conn.inputStream.bufferedReader().readText()
            conn.disconnect()

            val arr = JSONArray(response)
            val models = mutableListOf<ModelInfo>()

            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                val id = obj.optString("id", "")
                val parts = id.split("/")
                val author = if (parts.size > 1) parts[0] else ""
                val name = if (parts.size > 1) parts[1] else id

                val tags = mutableListOf<String>()
                val tagsArr = obj.optJSONArray("tags")
                if (tagsArr != null) {
                    for (j in 0 until tagsArr.length()) {
                        tags.add(tagsArr.getString(j))
                    }
                }

                models.add(
                    ModelInfo(
                        id = id,
                        author = author,
                        modelName = name,
                        downloads = obj.optInt("downloads", 0),
                        likes = obj.optInt("likes", 0),
                        tags = tags,
                        lastModified = obj.optString("lastModified", "")
                    )
                )
            }
            models
        } catch (e: Exception) {
            Log.e(TAG, "Search failed", e)
            emptyList()
        }
    }

    /**
     * Get list of GGUF files in a model repo.
     */
    suspend fun getModelFiles(repoId: String): List<ModelFile> = withContext(Dispatchers.IO) {
        try {
            val url = URL("$BASE_URL/$repoId")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.connectTimeout = 10_000
            conn.readTimeout = 15_000

            val response = conn.inputStream.bufferedReader().readText()
            conn.disconnect()

            val obj = JSONObject(response)
            val siblings = obj.optJSONArray("siblings") ?: return@withContext emptyList()

            val files = mutableListOf<ModelFile>()
            for (i in 0 until siblings.length()) {
                val file = siblings.getJSONObject(i)
                val filename = file.optString("rfilename", "")

                if (filename.endsWith(".gguf", ignoreCase = true)) {
                    val size = file.optLong("size", 0L)
                    val quant = extractQuantization(filename)
                    val downloadUrl = "https://huggingface.co/$repoId/resolve/main/$filename"

                    files.add(
                        ModelFile(
                            filename = filename,
                            sizeBytes = size,
                            quantization = quant,
                            downloadUrl = downloadUrl
                        )
                    )
                }
            }

            // Sort by size ascending so smallest quants appear first
            files.sortBy { it.sizeBytes }
            files
        } catch (e: Exception) {
            Log.e(TAG, "Get files failed for $repoId", e)
            emptyList()
        }
    }

    /**
     * Extract quantization type from filename like "model.Q4_K_M.gguf" -> "Q4_K_M"
     */
    private fun extractQuantization(filename: String): String {
        val regex = Regex("""[._-]((?:IQ|Q|F|BF)\d[^\s.]*)\.""", RegexOption.IGNORE_CASE)
        val match = regex.find(filename)
        return match?.groupValues?.get(1) ?: "Unknown"
    }
}
