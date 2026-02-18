package com.truelarge.runtime.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.work.WorkInfo
import com.truelarge.runtime.download.ModelDownloadManager
import kotlinx.coroutines.launch
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DownloadScreen(
    downloadManager: ModelDownloadManager,
    onBack: () -> Unit
) {
    // List all model downloads
    val allWorkInfos: List<WorkInfo> by downloadManager.getAllDownloads().collectAsState(initial = emptyList())
    
    // Scan for orphan .tmp files (not tracked by WorkManager)
    val modelRepository = remember { com.truelarge.runtime.data.ModelRepository(downloadManager.context) }
    val partialFiles = remember(allWorkInfos) { modelRepository.getPartialModels() }
    val recommendations = remember { modelRepository.getRecommendedModels() }
    
    val scope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Downloads") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        val managersKeys = allWorkInfos.map { 
            it.tags.firstOrNull { t -> 
                t != "model_download" && t != "com.truelarge.runtime.download.DownloadWorker" 
            } ?: "" 
        }
        val orphans = partialFiles.filter { file ->
            val expectedKey = file.name.removeSuffix(".tmp")
            managersKeys.none { key -> key == expectedKey }
        }

        if (allWorkInfos.isEmpty() && orphans.isEmpty()) {
            Box(modifier = Modifier.padding(padding).fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No active or partial downloads", style = MaterialTheme.typography.bodyLarge)
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .padding(padding)
                    .fillMaxSize()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Managed Downloads
                items(allWorkInfos) { info ->
                    val key = info.tags.firstOrNull { 
                        it != "model_download" && it != "com.truelarge.runtime.download.DownloadWorker" 
                    } ?: "unknown"
                    DownloadItem(
                        fileKey = key,
                        workInfo = info,
                        onPause = { downloadManager.pauseDownload(key) },
                        onResume = { 
                            var url = info.progress.getString("url")
                            val path = modelRepository.getDownloadPath(key)
                            
                            // Fallback to metadata if progress is empty (when paused)
                            if (url == null) {
                                try {
                                    val metaFile = File(path.parent, "${path.name}.json")
                                    if (metaFile.exists()) {
                                        val json = org.json.JSONObject(metaFile.readText())
                                        url = json.optString("url")
                                    }
                                } catch (e: Exception) { /* ignore */ }
                            }

                            if (url != null) {
                                downloadManager.resumeDownload(url, path, key)
                            }
                        },
                        onDelete = {
                            val path = modelRepository.getDownloadPath(key)
                            downloadManager.deleteDownload(key, path)
                        }
                    )
                }

                // Discovered Partial Files (Orphans)
                if (orphans.isNotEmpty()) {
                    item {
                        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                        Text(
                            "Discovered Partial Downloads",
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.secondary
                        )
                    }
                    
                    items(orphans) { file ->
                        OrphanDownloadItem(
                            file = file,
                            onScanAndResume = {
                                val fileName = file.name.removeSuffix(".tmp")
                                val match = recommendations.find { r -> 
                                    fileName.contains(r.repoId.substringAfter("/"), ignoreCase = true) ||
                                    fileName.contains(r.displayName.replace(" ", ""), ignoreCase = true)
                                }
                                
                                if (match != null) {
                                    scope.launch {
                                        val files = com.truelarge.runtime.data.HuggingFaceApi.getModelFiles(match.repoId)
                                        val actualFile = files.find { it.filename == fileName }
                                        if (actualFile != null) {
                                            downloadManager.resumeDownload(
                                                url = actualFile.downloadUrl,
                                                targetFile = file.parentFile?.let { java.io.File(it, actualFile.filename) } ?: file,
                                                fileKey = actualFile.filename
                                            )
                                        }
                                    }
                                }
                            },
                            onDelete = {
                                file.delete()
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun OrphanDownloadItem(
    file: java.io.File,
    onScanAndResume: () -> Unit,
    onDelete: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
    ) {
        Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(modifier = Modifier.weight(1f)) {
                Text(file.name, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold)
                Text("${file.length() / 1024 / 1024} MB on disk", style = MaterialTheme.typography.labelSmall)
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Default.Delete, contentDescription = "Delete", tint = MaterialTheme.colorScheme.error)
            }
            Button(onClick = onScanAndResume) {
                Text("Identify & Resume")
            }
        }
    }
}

@Composable
fun DownloadItem(
    fileKey: String,
    workInfo: WorkInfo,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onDelete: () -> Unit
) {
    var progress by remember { mutableFloatStateOf(workInfo.progress.getFloat("progress", 0f)) }
    var downloaded by remember { mutableLongStateOf(workInfo.progress.getLong("downloaded", 0L)) }
    var total by remember { mutableLongStateOf(workInfo.progress.getLong("total", 0L)) }
    val state = workInfo.state
    val context = androidx.compose.ui.platform.LocalContext.current

    // Update from WorkInfo if active
    LaunchedEffect(workInfo.progress) {
        val p = workInfo.progress.getFloat("progress", -1f)
        if (p >= 0) {
            progress = p
            downloaded = workInfo.progress.getLong("downloaded", 0L)
            total = workInfo.progress.getLong("total", 0L)
        }
    }

    // Fallback to Metadata and local file size if total is 0
    LaunchedEffect(total, workInfo.state) {
        if (total == 0L) {
            val modelRepo = com.truelarge.runtime.data.ModelRepository(context)
            val targetFile = modelRepo.getDownloadPath(fileKey)
            val metaFile = File(targetFile.parent, "${targetFile.name}.json")
            val tempFile = File(targetFile.parent, "${targetFile.name}.tmp")
            
            if (metaFile.exists()) {
                try {
                    val json = org.json.JSONObject(metaFile.readText())
                    total = json.optLong("totalBytes", 0L)
                    downloaded = if (tempFile.exists()) tempFile.length() else 0L
                    if (total > 0) progress = downloaded.toFloat() / total.toFloat()
                } catch (e: Exception) { /* ignore */ }
            } else if (tempFile.exists()) {
                 downloaded = tempFile.length()
            }
        }
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = fileKey,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )
                IconButton(onClick = onDelete) {
                    Icon(Icons.Default.Delete, contentDescription = "Cancel/Delete", tint = MaterialTheme.colorScheme.error)
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            if (state == WorkInfo.State.ENQUEUED) {
                Text("Waiting to start...", style = MaterialTheme.typography.bodySmall)
            }
            
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.fillMaxWidth()
            )
            
            Spacer(modifier = Modifier.height(4.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "${(progress * 100).toInt()}%",
                    style = MaterialTheme.typography.labelSmall
                )
                Text(
                    text = if (total > 0) "${downloaded / 1024 / 1024}MB / ${total / 1024 / 1024}MB" else "Discovering...",
                    style = MaterialTheme.typography.labelSmall
                )
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(verticalAlignment = Alignment.CenterVertically) {
                val displayState = if (state == WorkInfo.State.CANCELLED) "PAUSED" else state.name
                Text(
                    text = "Status: $displayState",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.weight(1f)
                )
                
                if (state == WorkInfo.State.RUNNING || state == WorkInfo.State.ENQUEUED) {
                    IconButton(onClick = onPause) {
                        Icon(Icons.Default.Pause, contentDescription = "Pause")
                    }
                } else if (state == WorkInfo.State.CANCELLED || state == WorkInfo.State.FAILED) {
                    IconButton(onClick = onResume) {
                        Icon(Icons.Default.PlayArrow, contentDescription = "Resume")
                    }
                }
            }
        }
    }
}
