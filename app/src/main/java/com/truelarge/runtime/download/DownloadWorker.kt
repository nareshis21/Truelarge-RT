package com.truelarge.runtime.download

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.File
import java.io.RandomAccessFile
import java.io.InputStream

class DownloadWorker(
    appContext: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(appContext, workerParams) {

    private val TAG = "DownloadWorker"
    private val client = OkHttpClient()

    override suspend fun doWork(): Result {
        val url = inputData.getString("url") ?: return Result.failure()
        val targetPath = inputData.getString("targetPath") ?: return Result.failure()
        val fileKey = inputData.getString("fileKey") ?: "unknown"

        val targetFile = File(targetPath)
        val tempFile = File(targetFile.parent, "${targetFile.name}.tmp")
        
        // Ensure parent directories exist
        tempFile.parentFile?.mkdirs()

        // Check current progress for resume
        val existingSize = if (tempFile.exists()) tempFile.length() else 0L
        Log.i(TAG, "Starting/Resuming download for $fileKey. Existing size: $existingSize bytes")

        try {
            val request = Request.Builder()
                .url(url)
                .addHeader("Range", "bytes=$existingSize-")
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful && response.code != 206) {
                    if (response.code == 416) {
                        // Range Not Satisfiable - maybe it's already finished?
                        Log.w(TAG, "Range not satisfiable for $fileKey. Might be complete.")
                        return finalizeDownload(tempFile, targetFile)
                    }
                    return Result.retry()
                }

                val body = response.body ?: return Result.failure()
                val totalLength = (response.header("Content-Length")?.toLong() ?: 0L) + existingSize
                
                // Save metadata for persistence (so UI doesn't show "Discovering...")
                saveMetadata(targetFile, url, fileKey, totalLength)

                // Write to file using RandomAccessFile to seek to the end
                RandomAccessFile(tempFile, "rw").use { raf ->
                    raf.seek(existingSize)
                    
                    val input: InputStream = body.byteStream()
                    val buffer = ByteArray(64 * 1024)
                    var bytesRead: Int
                    var currentDownloaded = existingSize
                    var lastUpdate = 0L

                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        if (isStopped) {
                            Log.i(TAG, "Download stopped for $fileKey")
                            return Result.retry()
                        }

                        raf.write(buffer, 0, bytesRead)
                        currentDownloaded += bytesRead

                        // Throttled progress update (every 500ms or 1MB)
                        val now = System.currentTimeMillis()
                        if (now - lastUpdate > 500) {
                            val progress = if (totalLength > 0) (currentDownloaded.toFloat() / totalLength.toFloat()) else 0f
                            setProgress(workDataOf(
                                "progress" to progress,
                                "downloaded" to currentDownloaded,
                                "total" to totalLength,
                                "fileKey" to fileKey,
                                "url" to url,
                                "targetPath" to targetPath
                            ))
                            lastUpdate = now
                        }
                    }
                }
                
                return finalizeDownload(tempFile, targetFile)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Download error for $fileKey", e)
            return Result.retry()
        }
    }

    private fun saveMetadata(targetFile: File, url: String, fileKey: String, totalBytes: Long) {
        try {
            val metaFile = File(targetFile.parent, "${targetFile.name}.json")
            val json = org.json.JSONObject().apply {
                put("url", url)
                put("fileKey", fileKey)
                put("totalBytes", totalBytes)
            }
            metaFile.writeText(json.toString())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save metadata", e)
        }
    }

    private fun finalizeDownload(tempFile: File, targetFile: File): Result {
        if (tempFile.exists()) {
            if (targetFile.exists()) targetFile.delete()
            if (tempFile.renameTo(targetFile)) {
                Log.i(TAG, "Download complete: ${targetFile.name}")
                // Clean up metadata
                val metaFile = File(targetFile.parent, "${targetFile.name}.json")
                if (metaFile.exists()) metaFile.delete()
                return Result.success()
            }
        }
        return Result.failure()
    }
}
