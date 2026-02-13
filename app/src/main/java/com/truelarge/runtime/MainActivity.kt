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
import com.truelarge.runtime.data.ModelRepository
import com.truelarge.runtime.download.ModelDownloadManager
import com.truelarge.runtime.ui.AirTheme
import com.truelarge.runtime.ui.CatalogScreen
import com.truelarge.runtime.ui.SearchScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.net.URLDecoder
import java.net.URLEncoder

class MainActivity : ComponentActivity() {

    private val engine = NativeEngine()
    private lateinit var modelRepository: ModelRepository
    private val downloadManager = ModelDownloadManager()

    private val storagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* permissions handled */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        modelRepository = ModelRepository(this)
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
                // For Android 11+, we need MANAGE_EXTERNAL_STORAGE
                // For simplicity, we'll use Downloads directory which doesn't need it
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
                onModelSelect = { path ->
                    val encoded = URLEncoder.encode(path, "UTF-8")
                    navController.navigate("inference/$encoded")
                },
                onRecommendedClick = { repoId ->
                    val encoded = URLEncoder.encode(repoId, "UTF-8")
                    navController.navigate("search?repoId=$encoded")
                }
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
            arguments = listOf(navArgument("modelPath") { type = NavType.StringType })
        ) { backStackEntry ->
            val encodedPath = backStackEntry.arguments?.getString("modelPath") ?: ""
            val modelPath = URLDecoder.decode(encodedPath, "UTF-8")
            InferenceScreen(
                engine = engine,
                modelPath = modelPath,
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
    var output by remember { mutableStateOf("") }
    var isRunning by remember { mutableStateOf(false) }
    var isGenerating by remember { mutableStateOf(false) }
    var prompt by remember { mutableStateOf("Hello, ") }
    var temperature by remember { mutableStateOf(0.7f) }
    var topP by remember { mutableStateOf(0.9f) }
    var maxTokens by remember { mutableStateOf(256f) }
    var contextTrain by remember { mutableIntStateOf(0) }
    var tps by remember { mutableStateOf(0.0) }

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
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Prompt input
            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Prompt") },
                minLines = 3,
                maxLines = 5
            )

            Spacer(modifier = Modifier.height(16.dp))

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

            Spacer(modifier = Modifier.height(16.dp))

            // Load button
            Button(
                onClick = {
                    if (!isRunning) {
                        isRunning = true
                        status = "Loading model..."
                        scope.launch {
                            withContext(Dispatchers.IO) {
                                val nThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
                                val ok = engine.init(modelPath, nThreads, 0)
                                withContext(Dispatchers.Main) {
                                    status = if (ok) "Model loaded" else "Failed to load"
                                    if (ok) {
                                        contextTrain = engine.getContextTrain()
                                        if (maxTokens > contextTrain && contextTrain > 0) {
                                            maxTokens = contextTrain.toFloat()
                                        }
                                    }
                                    if (!ok) isRunning = false
                                }
                            }
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isRunning
            ) {
                Text("Load Model")
            }

            Spacer(modifier = Modifier.height(8.dp))

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
                            scope.launch {
                                isGenerating = true
                                status = "Running inference..."
                                tps = 0.0
                                withContext(Dispatchers.IO) {
                                    engine.configureSampler(temperature, 40, topP)
                                    val ok = engine.createSession(prompt)
                                    if (ok) {
                                        val sb = StringBuilder()
                                        var tokenCount = 0
                                        val startTime = System.currentTimeMillis()
                                        repeat(maxTokens.toInt()) {
                                            if (!isGenerating) return@repeat
                                            val piece = engine.step()
                                            if (piece.isEmpty()) return@repeat
                                            tokenCount++
                                            sb.append(piece)
                                            val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                                            withContext(Dispatchers.Main) {
                                                output = sb.toString()
                                                if (elapsed > 0) tps = tokenCount / elapsed
                                            }
                                        }
                                    }
                                    withContext(Dispatchers.Main) {
                                        status = "Done"
                                        isGenerating = false
                                    }
                                }
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = isRunning && status == "Model loaded"
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

            Spacer(modifier = Modifier.height(16.dp))

            // Output
            if (output.isNotEmpty()) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Text(
                        text = output,
                        modifier = Modifier.padding(16.dp),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }
    }
}
