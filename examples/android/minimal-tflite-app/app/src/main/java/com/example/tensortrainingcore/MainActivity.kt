package com.example.tensortrainingcore

import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val statusView = TextView(this).apply {
            text = "Loading TFLite bundle from assets..."
            textSize = 16f
            setPadding(48, 64, 48, 64)
        }
        setContentView(statusView)

        lifecycleScope.launch {
            val message = withContext(Dispatchers.Default) {
                runCatching {
                    val detector = AndroidTfliteDetector.fromAssets(this@MainActivity)
                    val bitmap = assets.open("sample_input.jpg").use { input ->
                        BitmapFactory.decodeStream(input) ?: error("Unable to decode sample_input.jpg")
                    }
                    val detections = detector.detect(bitmap)
                    buildString {
                        appendLine("Top detections")
                        appendLine()
                        if (detections.isEmpty()) {
                            appendLine("No detections above the configured score threshold.")
                        } else {
                            detections.forEachIndexed { index, detection ->
                                appendLine(
                                    "${index + 1}. ${detection.label} " +
                                        "(score=${"%.3f".format(detection.score)}) " +
                                        "bbox=${detection.bboxXywhNorm}"
                                )
                            }
                        }
                    }
                }.getOrElse { error ->
                    buildString {
                        appendLine("Bundle load or inference failed.")
                        appendLine()
                        appendLine(error.message ?: error.javaClass.simpleName)
                        appendLine()
                        appendLine("Make sure app/src/main/assets contains:")
                        appendLine("- model.tflite")
                        appendLine("- label.txt")
                        appendLine("- export_metadata.json")
                        appendLine("- sample_input.jpg")
                    }
                }
            }
            statusView.text = message
        }
    }
}
