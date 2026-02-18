package com.truelarge.runtime.download

import android.content.Context
import android.util.Log
import androidx.lifecycle.asFlow
import androidx.work.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.io.File

/**
 * Download state for tracking progress.
 */
sealed class DownloadState {
    object Idle : DownloadState()
    object Pending : DownloadState()
    data class Downloading(val progress: Float, val downloadedBytes: Long, val totalBytes: Long) : DownloadState()
    data class Completed(val filePath: String) : DownloadState()
    data class Error(val message: String) : DownloadState()
    object Paused : DownloadState()
}

class ModelDownloadManager(val context: Context) {

    private val TAG = "ModelDownloadManager"
    private val workManager = WorkManager.getInstance(context)

    // Unified flow to observe all model downloads
    val activeDownloads: StateFlow<Map<String, DownloadState>> = 
        workManager.getWorkInfosByTagLiveData("model_download").asFlow()
            .map { infos ->
                infos.associate { info ->
                    val fileKey = info.tags.firstOrNull { 
                        it != "model_download" && it != "com.truelarge.runtime.download.DownloadWorker" 
                    } ?: "unknown"
                    fileKey to when (info.state) {
                        WorkInfo.State.RUNNING -> {
                            val progress = info.progress.getFloat("progress", 0f)
                            val downloaded = info.progress.getLong("downloaded", 0)
                            val total = info.progress.getLong("total", 0)
                            DownloadState.Downloading(progress, downloaded, total)
                        }
                        WorkInfo.State.ENQUEUED -> DownloadState.Pending
                        WorkInfo.State.SUCCEEDED -> DownloadState.Completed("")
                        WorkInfo.State.FAILED -> DownloadState.Error("Failed")
                        WorkInfo.State.CANCELLED -> DownloadState.Paused
                        else -> DownloadState.Idle
                    }
                }
            }
            .stateIn(CoroutineScope(Dispatchers.Main), SharingStarted.Lazily, emptyMap())

    fun download(url: String, targetFile: File, fileKey: String, force: Boolean = false) {
        val data = workDataOf(
            "url" to url,
            "targetPath" to targetFile.absolutePath,
            "fileKey" to fileKey
        )

        val request = OneTimeWorkRequestBuilder<DownloadWorker>()
            .setInputData(data)
            .setConstraints(Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build())
            .addTag(fileKey)
            .addTag("model_download")
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, WorkRequest.MIN_BACKOFF_MILLIS, java.util.concurrent.TimeUnit.MILLISECONDS)
            .build()

        val policy = if (force) ExistingWorkPolicy.REPLACE else ExistingWorkPolicy.KEEP
        workManager.enqueueUniqueWork(fileKey, policy, request)
    }

    fun pauseDownload(fileKey: String) {
        workManager.cancelUniqueWork(fileKey)
    }

    fun deleteDownload(fileKey: String, targetFile: File) {
        workManager.cancelUniqueWork(fileKey)
        workManager.pruneWork()
        val tempFile = File(targetFile.parent, "${targetFile.name}.tmp")
        if (tempFile.exists()) tempFile.delete()
        val metaFile = File(targetFile.parent, "${targetFile.name}.json")
        if (metaFile.exists()) metaFile.delete()
    }

    fun cancelDownload() {
        // Universal cancel for legacy support
        workManager.cancelAllWork()
    }

    fun resumeDownload(url: String, targetFile: File, fileKey: String) {
        download(url, targetFile, fileKey, force = true)
    }

    fun getWorkInfo(fileKey: String): Flow<WorkInfo?> {
        return workManager.getWorkInfosForUniqueWorkLiveData(fileKey).asFlow().map { it.firstOrNull() }
    }

    fun getAllDownloads(): Flow<List<WorkInfo>> {
        return workManager.getWorkInfosByTagLiveData("model_download").asFlow()
    }
}
