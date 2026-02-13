@file:OptIn(ExperimentalMaterial3Api::class)

package com.truelarge.runtime.ui

import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.truelarge.runtime.data.HuggingFaceApi
import com.truelarge.runtime.data.ModelFile
import com.truelarge.runtime.data.ModelInfo
import com.truelarge.runtime.data.ModelRepository
import com.truelarge.runtime.download.DownloadState
import com.truelarge.runtime.download.ModelDownloadManager
import kotlinx.coroutines.launch
import java.text.NumberFormat

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SearchScreen(
    modelRepository: ModelRepository,
    downloadManager: ModelDownloadManager,
    initialRepoId: String? = null,
    onBack: () -> Unit
) {
    var searchQuery by remember { mutableStateOf("") }
    var searchResults by remember { mutableStateOf<List<ModelInfo>>(emptyList()) }
    var isSearching by remember { mutableStateOf(false) }
    var selectedModelId by remember { mutableStateOf<String?>(initialRepoId) }
    var modelFiles by remember { mutableStateOf<List<ModelFile>>(emptyList()) }
    var isLoadingFiles by remember { mutableStateOf(false) }
    val downloadStates by downloadManager.activeDownloads.collectAsState()
    val scope = rememberCoroutineScope()

    // If we got an initial repo ID, load its files immediately
    LaunchedEffect(initialRepoId) {
        if (initialRepoId != null) {
            isLoadingFiles = true
            modelFiles = HuggingFaceApi.getModelFiles(initialRepoId)
            isLoadingFiles = false
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Search Models") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp)
        ) {
            // Search bar
            OutlinedTextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("Search GGUF models...") },
                leadingIcon = { Icon(Icons.Default.Search, null) },
                trailingIcon = {
                    if (searchQuery.isNotEmpty()) {
                        IconButton(onClick = { searchQuery = "" }) {
                            Icon(Icons.Default.Close, "Clear")
                        }
                    }
                },
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                keyboardActions = KeyboardActions(
                    onSearch = {
                        scope.launch {
                            isSearching = true
                            selectedModelId = null
                            searchResults = HuggingFaceApi.searchModels(searchQuery)
                            isSearching = false
                        }
                    }
                ),
                singleLine = true,
                shape = RoundedCornerShape(12.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                    unfocusedBorderColor = MaterialTheme.colorScheme.outline,
                    focusedContainerColor = MaterialTheme.colorScheme.surfaceVariant,
                    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            )

            Spacer(modifier = Modifier.height(12.dp))

            if (isSearching) {
                Box(
                    modifier = Modifier.fillMaxWidth(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.padding(32.dp),
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }

            // Show file list for selected model (or initial repo)
            if (selectedModelId != null) {
                FileListView(
                    repoId = selectedModelId!!,
                    files = modelFiles,
                    isLoading = isLoadingFiles,
                    downloadStates = downloadStates,
                    onDownload = { file ->
                        val target = modelRepository.getDownloadPath(file.filename)
                        scope.launch {
                            downloadManager.download(file.downloadUrl, target, file.filename)
                        }
                    },
                    onCancel = { downloadManager.cancelDownload() },
                    onClose = {
                        selectedModelId = null
                        modelFiles = emptyList()
                    }
                )
            }

            // Search results
            if (selectedModelId == null) {
                LazyColumn(
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(searchResults) { model ->
                        SearchResultCard(
                            model = model,
                            onClick = {
                                selectedModelId = model.id
                                scope.launch {
                                    isLoadingFiles = true
                                    modelFiles = HuggingFaceApi.getModelFiles(model.id)
                                    isLoadingFiles = false
                                }
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun SearchResultCard(
    model: ModelInfo,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        onClick = onClick,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                model.modelName,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Spacer(modifier = Modifier.height(4.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "by ${model.author}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    "⬇ ${NumberFormat.getInstance().format(model.downloads)}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    "♥ ${model.likes}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
fun FileListView(
    repoId: String,
    files: List<ModelFile>,
    isLoading: Boolean,
    downloadStates: Map<String, DownloadState>,
    onDownload: (ModelFile) -> Unit,
    onCancel: () -> Unit,
    onClose: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize()
    ) {
        // Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    "GGUF Files",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.primary
                )
                Text(
                    repoId,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            TextButton(onClick = onClose) {
                Text("Close")
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        if (isLoading) {
            CircularProgressIndicator(
                modifier = Modifier
                    .align(Alignment.CenterHorizontally)
                    .padding(16.dp)
            )
        }

        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(files) { file ->
                FileCard(
                    file = file,
                    downloadState = downloadStates[file.filename] ?: DownloadState.Idle,
                    onDownload = { onDownload(file) },
                    onCancel = onCancel
                )
            }

            if (files.isEmpty() && !isLoading) {
                item {
                    Text(
                        "No GGUF files found in this repo",
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
fun FileCard(
    file: ModelFile,
    downloadState: DownloadState,
    onDownload: () -> Unit,
    onCancel: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            // Filename and size
            Text(
                file.filename,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Spacer(modifier = Modifier.height(6.dp))

            // Badges row
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                QuantBadge(file.quantization)
                SizeBadge(file.sizeFormatted)
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Download button / progress
            when (downloadState) {
                is DownloadState.Idle -> {
                    Button(
                        onClick = onDownload,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary
                        )
                    ) {
                        Text("Download")
                    }
                }
                is DownloadState.Downloading -> {
                    Column {
                        LinearProgressIndicator(
                            progress = downloadState.progress,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(6.dp),
                            color = MaterialTheme.colorScheme.primary,
                            trackColor = MaterialTheme.colorScheme.outline,
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            val downloadedMB = downloadState.downloadedBytes / (1024.0 * 1024.0)
                            val totalMB = downloadState.totalBytes / (1024.0 * 1024.0)
                            Text(
                                String.format("%.1f / %.1f MB", downloadedMB, totalMB),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                "${(downloadState.progress * 100).toInt()}%",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        OutlinedButton(
                            onClick = onCancel,
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text("Cancel")
                        }
                    }
                }
                is DownloadState.Completed -> {
                    Surface(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp),
                        color = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.15f)
                    ) {
                        Text(
                            "✓ Downloaded",
                            modifier = Modifier.padding(12.dp),
                            color = MaterialTheme.colorScheme.tertiary,
                            fontWeight = FontWeight.SemiBold,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
                is DownloadState.Error -> {
                    Column {
                        Text(
                            "Error: ${downloadState.message}",
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Button(
                            onClick = onDownload,
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text("Retry")
                        }
                    }
                }
                is DownloadState.Cancelled -> {
                    Button(
                        onClick = onDownload,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Download")
                    }
                }
            }
        }
    }
}
