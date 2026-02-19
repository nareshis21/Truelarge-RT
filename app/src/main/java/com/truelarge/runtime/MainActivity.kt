package com.truelarge.runtime

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.truelarge.runtime.ui.BenchmarkScreen
import java.net.URLDecoder
import java.net.URLEncoder
import com.truelarge.runtime.data.ModelRepository
import com.truelarge.runtime.download.ModelDownloadManager
import com.truelarge.runtime.ui.AirTheme
import com.truelarge.runtime.ui.CatalogScreen
import com.truelarge.runtime.ui.SearchScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.lazy.items

import java.util.UUID

data class ChatMessage(
    val role: String,
    var content: String,
    val id: String = UUID.randomUUID().toString()
)
class MainActivity : ComponentActivity() {

    private val engine = NativeEngine()
    private lateinit var modelRepository: ModelRepository
    private lateinit var downloadManager: ModelDownloadManager

    private val storagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* permissions handled */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        modelRepository = ModelRepository(this)
        downloadManager = ModelDownloadManager(this)
        requestStoragePermissions()

        setContent {
            AirTheme(darkTheme = true) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    AppNavigation(
                        engine = engine,
                        modelRepository = modelRepository,
                        downloadManager = downloadManager
                    )
                }
            }
        }
    }

    private fun requestStoragePermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                try {
                    val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                    intent.addCategory("android.intent.category.DEFAULT")
                    intent.data = Uri.parse("package:${packageName}")
                    startActivity(intent)
                } catch (e: Exception) {
                    val intent = Intent()
                    intent.action = Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION
                    startActivity(intent)
                }
            }
        } else {
            val perms = arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
            val needed = perms.filter {
                ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
            }
            if (needed.isNotEmpty()) {
                storagePermissionLauncher.launch(needed.toTypedArray())
            }
        }
    }
}

@Composable
fun AppNavigation(
    engine: NativeEngine,
    modelRepository: ModelRepository,
    downloadManager: ModelDownloadManager
) {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = "catalog") {

        composable("catalog") {
            CatalogScreen(
                modelRepository = modelRepository,
                onSearchClick = { navController.navigate("search") },
                onDownloadsClick = { navController.navigate("downloads") },
                onModelSelect = { targetPath, draftPath ->
                    val encodedTarget = URLEncoder.encode(targetPath, "UTF-8")
                    val route = if (draftPath != null) {
                        val encodedDraft = URLEncoder.encode(draftPath, "UTF-8")
                        "inference/$encodedTarget?draftPath=$encodedDraft"
                    } else {
                        "inference/$encodedTarget"
                    }
                    navController.navigate(route)
                },
                onRecommendedClick = { repoId ->
                    val encoded = URLEncoder.encode(repoId, "UTF-8")
                    navController.navigate("search?repoId=$encoded")
                },
                onBenchmark = { path ->
                    val encoded = URLEncoder.encode(path, "UTF-8")
                    navController.navigate("benchmark/$encoded")
                }
            )
        }

        composable(
            route = "benchmark/{modelPath}",
            arguments = listOf(
                navArgument("modelPath") { type = NavType.StringType }
            )
        ) { backStackEntry ->
            val encodedPath = backStackEntry.arguments?.getString("modelPath") ?: ""
            val modelPath = URLDecoder.decode(encodedPath, "UTF-8")
            
            BenchmarkScreen(
                engine = engine,
                modelPath = modelPath,
                onBack = { navController.popBackStack() }
            )
        }

        composable(
            route = "search?repoId={repoId}",
            arguments = listOf(navArgument("repoId") {
                type = NavType.StringType
                defaultValue = ""
            })
        ) { backStackEntry ->
            val repoId = backStackEntry.arguments?.getString("repoId") ?: ""
            val decodedRepo = if (repoId.isNotEmpty()) URLDecoder.decode(repoId, "UTF-8") else null
            SearchScreen(
                modelRepository = modelRepository,
                downloadManager = downloadManager,
                initialRepoId = decodedRepo,
                onBack = { navController.popBackStack() }
            )
        }

        composable("search") {
            SearchScreen(
                modelRepository = modelRepository,
                downloadManager = downloadManager,
                onBack = { navController.popBackStack() }
            )
        }

        composable(
            route = "inference/{modelPath}",
            arguments = listOf(
                navArgument("modelPath") { type = NavType.StringType }
            )
        ) { backStackEntry ->
            val encodedPath = backStackEntry.arguments?.getString("modelPath") ?: ""
            val modelPath = URLDecoder.decode(encodedPath, "UTF-8")
            
            InferenceScreen(
                engine = engine,
                modelPath = modelPath,
                onBack = { navController.popBackStack() }
            )
        }

        composable("downloads") {
            com.truelarge.runtime.ui.DownloadScreen(
                downloadManager = downloadManager,
                onBack = { navController.popBackStack() }
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InferenceScreen(
    engine: NativeEngine,
    modelPath: String,
    onBack: () -> Unit
) {
    val scope = rememberCoroutineScope()
    var status by remember { mutableStateOf("Ready") }
    
    // Unified Message List - Single Source of Truth
    val messages = remember { mutableStateListOf<ChatMessage>() }
    val listState = rememberLazyListState()
    
    var isRunning by remember { mutableStateOf(false) }
    var isGenerating by remember { mutableStateOf(false) }
    var prompt by remember { mutableStateOf("") }
    var temperature by remember { mutableStateOf(0.7f) }
    var topP by remember { mutableStateOf(0.9f) }
    var maxTokens by remember { mutableStateOf(256f) }
    var contextTrain by remember { mutableIntStateOf(0) }
    var tps by remember { mutableStateOf(0.0) }
    var inferenceMode by remember { mutableStateOf("Unknown") }

    // Load model on start
    LaunchedEffect(modelPath) {
        withContext(Dispatchers.IO) {
            status = "Loading model..."
            val ok = engine.init(modelPath, 4, 0)
            if (ok) {
                contextTrain = engine.getContextTrain()
                inferenceMode = engine.getInferenceMode()
                withContext(Dispatchers.Main) {
                    status = "Model loaded"
                    isRunning = true
                }
            } else {
                withContext(Dispatchers.Main) { status = "Load failed" }
            }
        }
    }

    // Extract model name from path for display
    val modelName = remember(modelPath) {
        modelPath.substringAfterLast("/").substringAfterLast("\\")
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Inference", fontWeight = FontWeight.Bold)
                        Text(
                            modelName,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back")
                    }
                },
                actions = {
                    if (messages.isNotEmpty()) {
                        TextButton(onClick = {
                            isGenerating = false // Stop generation immediately
                            messages.clear()
                            status = "Session cleared"
                            scope.launch {
                                withContext(Dispatchers.IO) {
                                    engine.createSession("", false) // Reset engine context
                                }
                            }
                        }) {
                            Text("Clear Session", color = MaterialTheme.colorScheme.error)
                        }
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
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Status card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Status: $status",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    if (contextTrain > 0) {
                        Text(
                            text = "Context Window: $contextTrain tokens",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    if (isRunning) {
                        Text(
                            text = "Engine: $inferenceMode",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Chat History Area
            if (messages.isNotEmpty()) {
                Card(
                    modifier = Modifier.fillMaxWidth().weight(1f).padding(vertical = 8.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                ) {
                    androidx.compose.foundation.lazy.LazyColumn(
                        state = listState,
                        modifier = Modifier.padding(8.dp).fillMaxWidth(),
                        reverseLayout = false
                    ) {
                        items(
                            items = messages,
                            key = { it.id }
                        ) { msg ->
                            Column(modifier = Modifier.padding(vertical = 4.dp)) {
                                Text(
                                    text = "${msg.role}:", 
                                    fontWeight = FontWeight.Bold, 
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (msg.role == "User") MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary
                                )
                                Text(
                                    text = msg.content, 
                                    style = MaterialTheme.typography.bodyMedium
                                )
                                HorizontalDivider(
                                    modifier = Modifier.padding(top = 8.dp),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.1f)
                                )
                            }
                        }
                    }
                    // Auto-scroll to bottom on new token
                    LaunchedEffect(messages.lastOrNull()?.content?.length) {
                        if (messages.isNotEmpty()) {
                            listState.animateScrollToItem(messages.size - 1)
                        }
                    }
                }
            } else {
                Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.Center) {
                    Text("Start a conversation...", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }

            // Prompt input
            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Prompt") },
                minLines = 2,
                maxLines = 4
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Run/Stop button
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                if (isGenerating) {
                    Button(
                        onClick = { isGenerating = false },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.error
                        )
                    ) {
                        Text("Stop Generation")
                    }
                } else {
                    Button(
                        onClick = {
                            val currentPrompt = prompt
                            prompt = "" // Clear input immediately
                            
                            scope.launch {
                                // Add messages to UI
                                messages.add(ChatMessage("User", currentPrompt))
                                val aiMsg = ChatMessage("AI", "")
                                messages.add(aiMsg)
                                
                                isGenerating = true
                                status = "Running inference..."
                                tps = 0.0
                                
                                val keepHistory = messages.size > 2
                                // System Prompt Injection
                                val promptToSend = if (!keepHistory) {
                                    "<|im_start|>system\nYou are a helpful AI assistant. Answer concisely.<|im_end|>\n" +
                                    "<|im_start|>user\n$currentPrompt<|im_end|>\n" + 
                                    "<|im_start|>assistant\n"
                                } else {
                                    "<|im_start|>user\n$currentPrompt<|im_end|>\n" +
                                    "<|im_start|>assistant\n"
                                }

                                withContext(Dispatchers.IO) {
                                    engine.configureSampler(temperature, 40, topP)
                                    val ok = engine.createSession(promptToSend, keepHistory) // Use history if continuing
                                    
                                    if (ok) {
                                        var tokenCount = 0
                                        val startTime = System.currentTimeMillis()
                                        val responseBytes = mutableListOf<Byte>()
                                        
                                        for (i in 0 until maxTokens.toInt()) {
                                            if (!isGenerating) break
                                            val pieceBytes = engine.step()
                                            if (pieceBytes == null || pieceBytes.isEmpty()) break
                                            
                                            tokenCount++
                                            responseBytes.addAll(pieceBytes.toList())
                                            val currentString = String(responseBytes.toByteArray(), Charsets.UTF_8)
                                            val filteredString = currentString
                                                .replace("<|im_end|>", "")
                                                .replace("<|im_start|>", "")
                                                .trimEnd()
                                                
                                            val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                                            
                                            withContext(Dispatchers.Main) {
                                                if (messages.isNotEmpty()) {
                                                    messages[messages.lastIndex] = aiMsg.copy(content = filteredString)
                                                }
                                                if (elapsed > 0) tps = tokenCount / elapsed
                                            }
                                        }
                                    }
                                }
                                withContext(Dispatchers.Main) {
                                    status = "Done"
                                    isGenerating = false
                                }
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = isRunning && status != "Loading model..." && prompt.isNotBlank()
                    ) {
                        Text("Run Inference")
                    }
                }

                if (tps > 0) {
                    Text(
                        text = "Speed: ${"%.1f".format(tps)} t/s",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Sampler Settings
            Column(modifier = Modifier.fillMaxWidth()) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Temp: ${"%.1f".format(temperature)}", modifier = Modifier.width(80.dp), style = MaterialTheme.typography.bodySmall)
                    Slider(value = temperature, onValueChange = { temperature = it }, valueRange = 0f..2f, modifier = Modifier.weight(1f))
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Top-P: ${"%.2f".format(topP)}", modifier = Modifier.width(80.dp), style = MaterialTheme.typography.bodySmall)
                    Slider(value = topP, onValueChange = { topP = it }, valueRange = 0f..1f, modifier = Modifier.weight(1f))
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Max T: ${maxTokens.toInt()}", modifier = Modifier.width(80.dp), style = MaterialTheme.typography.bodySmall)
                    Slider(
                        value = maxTokens,
                        onValueChange = { maxTokens = it },
                        valueRange = 128f..(if (contextTrain > 0) contextTrain.toFloat() else 4096f),
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}
