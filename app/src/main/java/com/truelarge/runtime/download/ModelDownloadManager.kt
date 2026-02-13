package com.truelarge.runtime.download

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Download state for tracking progress.
 */
sealed class DownloadState {
    object Idle : DownloadState()
    data class Downloading(val progress: Float, val downloadedBytes: Long, val totalBytes: Long) : DownloadState()
    data class Completed(val filePath: String) : DownloadState()
    data class Error(val message: String) : DownloadState()
    object Cancelled : DownloadState()
}

/**
 * Manages model downloads from HuggingFace with progress tracking.
 */
class ModelDownloadManager {

    private val TAG = "ModelDownloadManager"

    private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
    val downloadState: StateFlow<DownloadState> = _downloadState

    private val _activeDownloads = MutableStateFlow<Map<String, DownloadState>>(emptyMap())
    val activeDownloads: StateFlow<Map<String, DownloadState>> = _activeDownloads

    @Volatile
    private var cancelRequested = false

    /**
     * Download a file from URL to the target path.
     */
    suspend fun download(url: String, targetFile: File, fileKey: String) = withContext(Dispatchers.IO) {
        cancelRequested = false
        updateDownloadState(fileKey, DownloadState.Downloading(0f, 0, 0))

        try {
            val connection = URL(url).openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 30_000
            connection.readTimeout = 30_000
            connection.setRequestProperty("User-Agent", "TrueLarge-Android/1.0")

            // Handle redirects (HuggingFace uses them)
            connection.instanceFollowRedirects = true

            val responseCode = connection.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                updateDownloadState(fileKey, DownloadState.Error("HTTP $responseCode"))
                return@withContext
            }

            val totalBytes = connection.contentLengthLong
            var downloadedBytes = 0L

            // Use temp file to avoid partial downloads
            val tempFile = File(targetFile.parent, "${targetFile.name}.tmp")

            connection.inputStream.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Int

                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        if (cancelRequested) {
                            tempFile.delete()
                            updateDownloadState(fileKey, DownloadState.Cancelled)
                            return@withContext
                        }

                        output.write(buffer, 0, bytesRead)
                        downloadedBytes += bytesRead

                        val progress = if (totalBytes > 0) {
                            downloadedBytes.toFloat() / totalBytes.toFloat()
                        } else 0f

                        updateDownloadState(fileKey, DownloadState.Downloading(progress, downloadedBytes, totalBytes))
                    }
                }
            }

            // Rename temp file to final
            tempFile.renameTo(targetFile)
            updateDownloadState(fileKey, DownloadState.Completed(targetFile.absolutePath))
            Log.i(TAG, "Download complete: ${targetFile.absolutePath}")

        } catch (e: Exception) {
            Log.e(TAG, "Download failed", e)
            updateDownloadState(fileKey, DownloadState.Error(e.message ?: "Unknown error"))
        }
    }

    fun cancelDownload() {
        cancelRequested = true
    }

    fun clearState(fileKey: String) {
        val current = _activeDownloads.value.toMutableMap()
        current.remove(fileKey)
        _activeDownloads.value = current
    }

    private fun updateDownloadState(fileKey: String, state: DownloadState) {
        val current = _activeDownloads.value.toMutableMap()
        current[fileKey] = state
        _activeDownloads.value = current
        _downloadState.value = state
    }
}
