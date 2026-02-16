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
    val questionIndex: Int,
    val tokenText: String,
    val ttft: Double,
    val tps: Double,
    val ramMB: Long,
    val cpuGHz: Double,
    val totalTime: Double
)

data class QuestionSummary(
    val questionIndex: Int,
    val questionText: String,
    val avgTps: Double,
    val maxTps: Double,
    val minTps: Double,
    val medianTps: Double,
    val ttft: Double,
    val peakRamMB: Long,
    val avgCpuGHz: Double,
    val totalTime: Double
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
    
    var questionSummaries by remember { mutableStateOf(listOf<QuestionSummary>()) }
    
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
                                // Reset logic
                                results = emptyList()
                                questionSummaries = emptyList()
                                scope.launch {
                                    runBenchmark(
                                        engine, 
                                        modelPath, 
                                        { status = it }, 
                                        { results = results + it },
                                        { questionSummaries = questionSummaries + it },
                                        { isRunning = it }
                                    )
                                }
                            },
                            modifier = Modifier.padding(top = 8.dp).fillMaxWidth()
                        ) {
                            Icon(Icons.Default.PlayArrow, null)
                            Spacer(Modifier.width(8.dp))
                            Text("Start 5-Question Benchmark")
                        }
                    } else {
                        CircularProgressIndicator(modifier = Modifier.align(Alignment.CenterHorizontally).padding(top = 8.dp))
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            if (results.isNotEmpty() || questionSummaries.isNotEmpty()) {
                // Multi-Graph Visualization
                LazyColumn(modifier = Modifier.fillMaxWidth().weight(1f)) {
                    if (questionSummaries.isNotEmpty()) {
                        item {
                            Card(
                                modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp),
                                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
                            ) {
                                Column(modifier = Modifier.padding(16.dp)) {
                                    Text("Overall Performance", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                                    Spacer(Modifier.height(8.dp))
                                    val avgTps = questionSummaries.map { it.avgTps }.average()
                                    val avgTtft = questionSummaries.map { it.ttft }.average()
                                    val peakRam = questionSummaries.maxOf { it.peakRamMB }
                                    val avgCpu = questionSummaries.map { it.avgCpuGHz }.average()
                                    
                                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                        Column {
                                            Text("Avg TPS", style = MaterialTheme.typography.labelSmall)
                                            Text("%.2f t/s".format(avgTps), fontWeight = FontWeight.Bold)
                                        }
                                        Column {
                                            Text("Avg TTFT", style = MaterialTheme.typography.labelSmall)
                                            Text("%.1f ms".format(avgTtft), fontWeight = FontWeight.Bold)
                                        }
                                        Column {
                                            Text("Peak RAM", style = MaterialTheme.typography.labelSmall)
                                            Text("${peakRam}MB", fontWeight = FontWeight.Bold)
                                        }
                                        Column {
                                            Text("Avg CPU", style = MaterialTheme.typography.labelSmall)
                                            Text("%.2f GHz".format(avgCpu), fontWeight = FontWeight.Bold)
                                        }
                                    }
                                }
                            }
                        }
                        
                        item {
                            Text("Per-Question Summary", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(bottom = 8.dp))
                            questionSummaries.forEach { summary ->
                                Card(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                                    Column(Modifier.padding(12.dp)) {
                                        Text("Q${summary.questionIndex + 1}: ${summary.questionText}", style = MaterialTheme.typography.bodySmall, maxLines = 1)
                                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                            Text("Avg: %.1f".format(summary.avgTps), style = MaterialTheme.typography.labelSmall)
                                            Text("Max: %.1f".format(summary.maxTps), style = MaterialTheme.typography.labelSmall)
                                            Text("Min: %.1f".format(summary.minTps), style = MaterialTheme.typography.labelSmall)
                                            Text("Med: %.1f".format(summary.medianTps), style = MaterialTheme.typography.labelSmall)
                                        }
                                        Row(modifier = Modifier.fillMaxWidth().padding(top = 4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                                            Text("TTFT: %.1fms".format(summary.ttft), style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                                            Text("RAM: ${summary.peakRamMB}MB", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                                            Text("CPU: %.1fG".format(summary.avgCpuGHz), style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                                        }
                                    }
                                }
                            }
                            Spacer(Modifier.height(16.dp))
                        }
                    }
                    item {
                        TelemetryGraph("TTFT (milliseconds)", results, { it.ttft.toFloat() }, Color(0xFFFF5722), 2000f)
                        Spacer(Modifier.height(16.dp))
                    }
                    item {
                        TelemetryGraph("TPS (Tokens/Sec)", results, { it.tps.toFloat() }, Color(0xFF4CAF50), 100f)
                        Spacer(Modifier.height(16.dp))
                    }
                    item {
                        TelemetryGraph("Total Question Time (Sec)", results, { it.totalTime.toFloat() }, Color(0xFFFFC107), 30f)
                        Spacer(Modifier.height(16.dp))
                    }
                    item {
                        TelemetryGraph("CPU Frequency (GHz)", results, { it.cpuGHz.toFloat() }, Color(0xFF9C27B0), 3.5f)
                        Spacer(Modifier.height(16.dp))
                    }
                    item {
                        TelemetryGraph("RAM Usage (MB)", results, { it.ramMB.toFloat() }, Color(0xFF2196F3), 6000f)
                        Spacer(Modifier.height(16.dp))
                    }
                    
                    item {
                        Text("Token Step Log", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(top = 8.dp))
                    }
                    
                    items(results.reversed()) { res ->
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("T${res.tokenIndex}", fontWeight = FontWeight.Bold, modifier = Modifier.width(40.dp))
                            Text("'${res.tokenText}'", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f).padding(horizontal = 4.dp), maxLines = 1)
                            Text("${"%.2f".format(res.tps)}t/s", style = MaterialTheme.typography.labelSmall)
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
fun TelemetryGraph(
    label: String,
    results: List<BenchmarkResult>,
    valueSelector: (BenchmarkResult) -> Float,
    lineColor: Color,
    maxValue: Float
) {
    Card(
        modifier = Modifier.fillMaxWidth().height(160.dp),
        colors = CardDefaults.cardColors(containerColor = Color.Black.copy(alpha = 0.05f))
    ) {
        Column(Modifier.padding(8.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(label, style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
                val lastVal = results.lastOrNull()?.let { valueSelector(it) } ?: 0f
                Text(String.format("%.2f", lastVal), style = MaterialTheme.typography.labelSmall, color = lineColor)
            }
            
            Box(Modifier.fillMaxSize()) {
                Canvas(modifier = Modifier.fillMaxSize().padding(top = 16.dp, bottom = 8.dp)) {
                    val width = size.width
                    val height = size.height
                    val maxTokens = 50f 

                    fun xPos(index: Int) = (index / maxTokens) * width
                    fun yPos(value: Float) = height - (value.coerceIn(0f, maxValue) / maxValue) * height

                    // Grid
                    val gridColor = Color.White.copy(alpha = 0.05f)
                    drawLine(gridColor, Offset(0f, height/2), Offset(width, height/2), strokeWidth = 1f)

                    // Question Dividers & Labels
                    val dividerColor = Color.White.copy(alpha = 0.15f)
                    for (q in 1..4) {
                        val x = xPos(q * 10)
                        drawLine(
                            dividerColor, 
                            Offset(x, 0f), 
                            Offset(x, height), 
                            strokeWidth = 2f
                        )
                    }

                    // Path
                    if (results.isNotEmpty()) {
                        val path = Path().apply {
                            results.forEachIndexed { i, res ->
                                val x = xPos(res.tokenIndex)
                                val y = yPos(valueSelector(res))
                                if (i == 0) moveTo(x, y) else lineTo(x, y)
                            }
                        }
                        drawPath(
                            path, 
                            lineColor, 
                            style = Stroke(width = 3f, cap = StrokeCap.Round, join = StrokeJoin.Round)
                        )
                    }
                    
                    // Axis
                    drawLine(Color.Gray.copy(alpha = 0.3f), Offset(0f, height), Offset(width, height), strokeWidth = 1f)
                    drawLine(Color.Gray.copy(alpha = 0.3f), Offset(0f, 0f), Offset(0f, height), strokeWidth = 1f)
                }
            }
        }
    }
}

suspend fun runBenchmark(
    engine: NativeEngine,
    modelPath: String,
    onStatus: (String) -> Unit,
    onResult: (BenchmarkResult) -> Unit,
    onSummary: (QuestionSummary) -> Unit,
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
        val questionResults = mutableListOf<BenchmarkResult>()
        
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
                if (parts.size >= 5) {
                    val res = BenchmarkResult(
                        tokenIndex = globalTokenCount,
                        questionIndex = qIdx,
                        tokenText = piece,
                        ttft = parts[0].toDoubleOrNull() ?: 0.0,
                        tps = parts[1].toDoubleOrNull() ?: 0.0,
                        ramMB = parts[2].toLongOrNull() ?: 0,
                        cpuGHz = parts[3].toDoubleOrNull() ?: 0.0,
                        totalTime = parts[4].toDoubleOrNull() ?: 0.0
                    )
                    questionResults.add(res)
                    withContext(Dispatchers.Main) {
                        onResult(res)
                        onStatus("Q$qNum/5: Token $i/10... ('$piece')")
                    }
                }
                delay(5) 
            }
        }
        
        // Calculate Question Summary
        if (questionResults.isNotEmpty()) {
            val tpsList = questionResults.map { it.tps }.sorted()
            val medianTps = if (tpsList.size % 2 == 0) {
                (tpsList[tpsList.size / 2 - 1] + tpsList[tpsList.size / 2]) / 2.0
            } else {
                tpsList[tpsList.size / 2]
            }

            val summary = QuestionSummary(
                questionIndex = qIdx,
                questionText = question,
                avgTps = tpsList.average(),
                maxTps = tpsList.maxOrNull() ?: 0.0,
                minTps = tpsList.minOrNull() ?: 0.0,
                medianTps = medianTps,
                ttft = questionResults.first().ttft,
                peakRamMB = questionResults.maxOf { it.ramMB },
                avgCpuGHz = questionResults.map { it.cpuGHz }.average(),
                totalTime = questionResults.last().totalTime
            )
            onSummary(summary)
        }
        
        Log.i("TRUELARGEPERF", "BENCHMARK ANSWER $qNum: ${finalAnswer.toString()}")
    }
    
    onStatus("Benchmark Complete (5/5 Questions)")
    onToggle(false)
}
