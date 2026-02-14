package com.truelarge.runtime.ui

import android.util.Log
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.truelarge.runtime.NativeEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class BenchmarkResult(
    val tokenIndex: Int,
    val tokenText: String, // Added this
    val ttft: Double,
    val tps: Double,
    val ramMB: Long,
    val cpuGHz: Double
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BenchmarkScreen(
    engine: NativeEngine,
    modelPath: String,
    onBack: () -> Unit
) {
    val scope = rememberCoroutineScope()
    var isRunning by remember { mutableStateOf(false) }
    var results by remember { mutableStateOf(listOf<BenchmarkResult>()) }
    var status by remember { mutableStateOf("Ready to Benchmark") }
    
    val modelName = remember(modelPath) {
        modelPath.substringAfterLast("/").substringAfterLast("\\")
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Performance Benchmark") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            // Model Info & Action
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(text = "Target Model", style = MaterialTheme.typography.labelSmall)
                    Text(text = modelName, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(text = "Status: $status", style = MaterialTheme.typography.bodySmall)
                    
                    if (!isRunning) {
                        Button(
                            onClick = {
                                scope.launch {
                                    runBenchmark(engine, modelPath, { status = it }, { 
                                        results = results + it 
                                    }, { isRunning = it })
                                }
                            },
                            modifier = Modifier.padding(top = 8.dp).fillMaxWidth()
                        ) {
                            Icon(Icons.Default.PlayArrow, null)
                            Spacer(Modifier.width(8.dp))
                            Text("Start 5-Question Benchmark")
                        }
                    } else {
                        CircularProgressIndicator(modifier = Modifier.align(Alignment.CenterHorizontally))
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            if (results.isNotEmpty()) {
                // Real-time Data Visualization
                Card(
                    modifier = Modifier.fillMaxWidth().height(250.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.Black.copy(alpha = 0.05f))
                ) {
                    Column(Modifier.padding(8.dp)) {
                        Text("TPS & RAM Tracking", style = MaterialTheme.typography.labelSmall)
                        PerformanceGraph(results)
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Detailed Log
                Text("Token Step Log", style = MaterialTheme.typography.titleSmall)
                LazyColumn(modifier = Modifier.fillMaxWidth().weight(1f)) {
                    items(results.reversed()) { res ->
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("T${res.tokenIndex}", fontWeight = FontWeight.Bold, modifier = Modifier.width(40.dp))
                            Text("'${res.tokenText}'", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f).padding(horizontal = 4.dp), maxLines = 1)
                            Text("${"%.2f".format(res.tps)} TPS ", style = MaterialTheme.typography.labelSmall)
                            Text("${res.ramMB}MB", style = MaterialTheme.typography.labelSmall)
                        }
                        HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp), color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f))
                    }
                }
            } else {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No data yet. Start benchmark to see results.", color = Color.Gray)
                }
            }
        }
    }
}

@Composable
fun PerformanceGraph(results: List<BenchmarkResult>) {
    Box(modifier = Modifier.fillMaxSize()) {
        // Legend Overlay
        Row(
            modifier = Modifier
                .padding(8.dp)
                .align(Alignment.TopEnd)
                .background(Color.Black.copy(alpha = 0.4f), RoundedCornerShape(4.dp))
                .padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(Modifier.size(8.dp).background(Color(0xFF4CAF50), RoundedCornerShape(2.dp)))
            Text(" TPS ", style = MaterialTheme.typography.labelSmall, color = Color.White)
            Spacer(Modifier.width(8.dp))
            Box(Modifier.size(8.dp).background(Color(0xFF2196F3), RoundedCornerShape(2.dp)))
            Text(" RAM", style = MaterialTheme.typography.labelSmall, color = Color.White)
        }

        // Axis Scale Labels
        Column(
            modifier = Modifier.fillMaxHeight().padding(start = 4.dp, top = 24.dp, bottom = 24.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Text("10", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
            Text("5", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
            Text("0", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
        }

        Canvas(modifier = Modifier.fillMaxSize().padding(start = 24.dp, end = 16.dp, top = 24.dp, bottom = 24.dp)) {
            if (results.isEmpty()) return@Canvas

            val width = size.width
            val height = size.height
            val maxTokens = 50f
            val maxTps = 10f
            val maxRam = 4000f // Scaling for 4GB

            fun xPos(index: Int) = (index / maxTokens) * width
            fun yPosTps(tps: Double) = height - (tps.coerceIn(0.0, 10.0).toFloat() / maxTps) * height
            fun yPosRam(ram: Long) = height - (ram.coerceIn(0, 4000).toFloat() / maxRam) * height

            // 1. Draw Grid
            val gridColor = Color.White.copy(alpha = 0.1f)
            for (i in 0..4) {
                val y = height - (i / 4f) * height
                drawLine(gridColor, Offset(0f, y), Offset(width, y), strokeWidth = 1f)
            }
            for (i in 0..5) {
                val x = (i / 5f) * width
                drawLine(gridColor, Offset(x, 0f), Offset(x, height), strokeWidth = 1f)
            }

            // 2. Draw Paths
            val tpsPath = Path().apply {
                results.forEachIndexed { i, res ->
                    val x = xPos(res.tokenIndex)
                    val y = yPosTps(res.tps)
                    if (i == 0) moveTo(x, y) else lineTo(x, y)
                }
            }
            drawPath(tpsPath, Color(0xFF4CAF50), style = Stroke(width = 4f, cap = StrokeCap.Round, join = StrokeJoin.Round))

            val ramPath = Path().apply {
                results.forEachIndexed { i, res ->
                    val x = xPos(res.tokenIndex)
                    val y = yPosRam(res.ramMB)
                    if (i == 0) moveTo(x, y) else lineTo(x, y)
                }
            }
            drawPath(ramPath, Color(0xFF2196F3), style = Stroke(width = 4f, cap = StrokeCap.Round, join = StrokeJoin.Round))
            
            // 3. Draw Axis Lines
            drawLine(Color.Gray, Offset(0f, height), Offset(width, height), strokeWidth = 2f)
            drawLine(Color.Gray, Offset(0f, 0f), Offset(0f, height), strokeWidth = 2f)
        }
    }
}

suspend fun runBenchmark(
    engine: NativeEngine,
    modelPath: String,
    onStatus: (String) -> Unit,
    onResult: (BenchmarkResult) -> Unit,
    onToggle: (Boolean) -> Unit
) {
    val questions = listOf(
        "What is the capital of France?",
        "Write a 4-line poem about the moon.",
        "If 2x + 10 = 20, what is x?",
        "List the 3 largest planets in our solar system.",
        "Summarize what a black hole is in ten words."
    )

    onToggle(true)
    onStatus("Initializing model...")
    
    val success = withContext(Dispatchers.IO) {
        engine.init(modelPath, 4, 0)
    }
    
    if (!success) {
        onStatus("Load Failed")
        onToggle(false)
        return
    }
    
    var globalTokenCount = 0
    
    questions.forEachIndexed { qIdx, question ->
        val qNum = qIdx + 1
        onStatus("Running Question $qNum/5...")
        Log.i("TRUELARGEPERF", "--------------------------------------")
        Log.i("TRUELARGEPERF", "BENCHMARK QUESTION $qNum: $question")
        
        val finalAnswer = StringBuilder()
        withContext(Dispatchers.IO) {
            engine.createSession(question, false) // Fresh session for each question
            
            // Run 10 tokens per question to get a total of 50 data points
            for (i in 1..10) {
                globalTokenCount++
                val pieceBytes = engine.step()
                if (pieceBytes == null) break
                val piece = String(pieceBytes, Charsets.UTF_8)
                finalAnswer.append(piece)
                
                val rawCsv = engine.getBenchmarkData()
                val parts = rawCsv.split(",")
                if (parts.size == 4) {
                    val res = BenchmarkResult(
                        tokenIndex = globalTokenCount,
                        tokenText = piece,
                        ttft = parts[0].toDoubleOrNull() ?: 0.0,
                        tps = parts[1].toDoubleOrNull() ?: 0.0,
                        ramMB = parts[2].toLongOrNull() ?: 0,
                        cpuGHz = parts[3].toDoubleOrNull() ?: 0.0
                    )
                    withContext(Dispatchers.Main) {
                        onResult(res)
                        onStatus("Q$qNum/5: Token $i/10... ('$piece')")
                    }
                }
                delay(5) 
            }
        }
        Log.i("TRUELARGEPERF", "BENCHMARK ANSWER $qNum: ${finalAnswer.toString()}")
    }
    
    onStatus("Benchmark Complete (5/5 Questions)")
    onToggle(false)
}
