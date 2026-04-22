package com.example.tensortrainingcore

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

data class DetectionResult(
    val labelId: Int,
    val label: String,
    val score: Float,
    val bboxXywhNorm: List<Float>,
)

private data class Anchor(val cx: Float, val cy: Float, val w: Float, val h: Float)

private data class DetectorMetadata(
    val imageWidth: Int,
    val imageHeight: Int,
    val anchors: List<Anchor>,
    val scoreThreshold: Float,
    val iouThreshold: Float,
)

class AndroidTfliteDetector private constructor(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val metadata: DetectorMetadata,
) {
    companion object {
        fun fromAssets(context: Context): AndroidTfliteDetector {
            val modelBuffer = loadAssetBuffer(context, "model.tflite")
            val labelLines = context.assets.open("label.txt").use { input ->
                BufferedReader(InputStreamReader(input)).readLines().filter { it.isNotBlank() }
            }
            val metadataJson = context.assets.open("export_metadata.json").use { input ->
                BufferedReader(InputStreamReader(input)).readText()
            }
            val metadata = parseMetadata(metadataJson)
            val options = Interpreter.Options().apply {
                setNumThreads(2)
            }
            return AndroidTfliteDetector(Interpreter(modelBuffer, options), labelLines, metadata)
        }

        private fun loadAssetBuffer(context: Context, assetName: String): ByteBuffer {
            val bytes = context.assets.open(assetName).use { it.readBytes() }
            return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
                put(bytes)
                rewind()
            }
        }

        private fun parseMetadata(json: String): DetectorMetadata {
            val payload = JSONObject(json)
            val imageSize = payload.getJSONArray("image_size")
            val anchorsJson = payload.getJSONArray("anchors")
            val nms = payload.getJSONObject("postprocessing").getJSONObject("nms")
            val anchors = buildList {
                for (index in 0 until anchorsJson.length()) {
                    val item = anchorsJson.getJSONObject(index)
                    add(
                        Anchor(
                            cx = item.getDouble("cx").toFloat(),
                            cy = item.getDouble("cy").toFloat(),
                            w = item.getDouble("w").toFloat(),
                            h = item.getDouble("h").toFloat(),
                        )
                    )
                }
            }
            return DetectorMetadata(
                imageWidth = imageSize.getInt(0),
                imageHeight = imageSize.getInt(1),
                anchors = anchors,
                scoreThreshold = nms.getDouble("score_threshold").toFloat(),
                iouThreshold = nms.getDouble("iou_threshold").toFloat(),
            )
        }
    }

    fun detect(bitmap: Bitmap): List<DetectionResult> {
        val inputTensor = interpreter.getInputTensor(0)
        val inputBuffer = buildInputBuffer(bitmap, inputTensor)

        val outputs = mutableMapOf<Int, Any>()
        val descriptors = mutableListOf<Pair<Tensor, ByteBuffer>>()
        for (index in 0 until interpreter.outputTensorCount) {
            val tensor = interpreter.getOutputTensor(index)
            val buffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
            outputs[index] = buffer
            descriptors += tensor to buffer
        }

        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

        var classScores: Array<FloatArray>? = null
        var boxOffsets: Array<FloatArray>? = null
        descriptors.forEach { (tensor, buffer) ->
            buffer.rewind()
            val values = readTensorAsFloat(buffer, tensor)
            val shape = tensor.shape()
            if (shape.last() == 4) {
                boxOffsets = reshape2d(values, shape[1], shape[2])
            } else {
                classScores = reshape2d(values, shape[1], shape[2])
            }
        }

        requireNotNull(classScores) { "Missing class output tensor" }
        requireNotNull(boxOffsets) { "Missing bbox output tensor" }
        return runNms(classScores!!, boxOffsets!!)
    }

    private fun buildInputBuffer(bitmap: Bitmap, inputTensor: Tensor): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, metadata.imageWidth, metadata.imageHeight, true)
        val channels = 3
        val count = metadata.imageWidth * metadata.imageHeight * channels
        val buffer = ByteBuffer.allocateDirect(count * bytesPerElement(inputTensor.dataType()))
            .order(ByteOrder.nativeOrder())
        val quantization = inputTensor.quantizationParams()
        val pixels = IntArray(metadata.imageWidth * metadata.imageHeight)
        resized.getPixels(pixels, 0, metadata.imageWidth, 0, 0, metadata.imageWidth, metadata.imageHeight)

        pixels.forEach { pixel ->
            val rgb = floatArrayOf(
                ((pixel shr 16) and 0xFF) / 255f,
                ((pixel shr 8) and 0xFF) / 255f,
                (pixel and 0xFF) / 255f,
            )
            rgb.forEach { channel ->
                when (inputTensor.dataType()) {
                    DataType.FLOAT32 -> buffer.putFloat(channel)
                    DataType.UINT8 -> {
                        val value = if (quantization.scale > 0f) {
                            ((channel / quantization.scale) + quantization.zeroPoint).toInt()
                        } else {
                            (channel * 255f).toInt()
                        }
                        buffer.put(value.coerceIn(0, 255).toByte())
                    }
                    DataType.INT8 -> {
                        val value = if (quantization.scale > 0f) {
                            ((channel / quantization.scale) + quantization.zeroPoint).toInt()
                        } else {
                            (channel * 127f).toInt()
                        }
                        buffer.put(value.coerceIn(-128, 127).toByte())
                    }
                    else -> error("Unsupported input dtype: ${inputTensor.dataType()}")
                }
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun readTensorAsFloat(buffer: ByteBuffer, tensor: Tensor): FloatArray {
        val count = tensor.numBytes() / bytesPerElement(tensor.dataType())
        val values = FloatArray(count)
        val quantization = tensor.quantizationParams()
        when (tensor.dataType()) {
            DataType.FLOAT32 -> {
                for (index in 0 until count) {
                    values[index] = buffer.float
                }
            }
            DataType.UINT8 -> {
                for (index in 0 until count) {
                    val raw = buffer.get().toInt() and 0xFF
                    values[index] = if (quantization.scale > 0f) {
                        (raw - quantization.zeroPoint) * quantization.scale
                    } else {
                        raw.toFloat()
                    }
                }
            }
            DataType.INT8 -> {
                for (index in 0 until count) {
                    val raw = buffer.get().toInt()
                    values[index] = if (quantization.scale > 0f) {
                        (raw - quantization.zeroPoint) * quantization.scale
                    } else {
                        raw.toFloat()
                    }
                }
            }
            else -> error("Unsupported output dtype: ${tensor.dataType()}")
        }
        return values
    }

    private fun reshape2d(values: FloatArray, rows: Int, cols: Int): Array<FloatArray> {
        val output = Array(rows) { FloatArray(cols) }
        for (row in 0 until rows) {
            for (col in 0 until cols) {
                output[row][col] = values[(row * cols) + col]
            }
        }
        return output
    }

    private fun runNms(classScores: Array<FloatArray>, boxOffsets: Array<FloatArray>): List<DetectionResult> {
        val candidates = mutableListOf<DetectionResult>()
        classScores.forEachIndexed { anchorIndex, scores ->
            if (scores.size <= 1) {
                return@forEachIndexed
            }
            var bestForegroundIndex = 1
            var bestScore = scores[1]
            for (index in 2 until scores.size) {
                if (scores[index] > bestScore) {
                    bestForegroundIndex = index
                    bestScore = scores[index]
                }
            }
            if (bestScore < metadata.scoreThreshold) {
                return@forEachIndexed
            }
            val labelId = bestForegroundIndex
            val label = labels.getOrNull(labelId - 1) ?: "class_$labelId"
            candidates += DetectionResult(
                labelId = labelId,
                label = label,
                score = bestScore,
                bboxXywhNorm = decodeBoxFromAnchor(boxOffsets[anchorIndex], metadata.anchors[anchorIndex]),
            )
        }

        return candidates.sortedByDescending { it.score }.fold(mutableListOf()) { selected, candidate ->
            if (selected.none { current -> iou(candidate.bboxXywhNorm, current.bboxXywhNorm) >= metadata.iouThreshold }) {
                selected += candidate
            }
            selected
        }
    }

    private fun decodeBoxFromAnchor(offsets: FloatArray, anchor: Anchor): List<Float> {
        val boxCx = (offsets[0] * anchor.w) + anchor.cx
        val boxCy = (offsets[1] * anchor.h) + anchor.cy
        val boxW = exp(offsets[2]) * anchor.w
        val boxH = exp(offsets[3]) * anchor.h
        val x = (boxCx - (boxW / 2f)).coerceIn(0f, 1f)
        val y = (boxCy - (boxH / 2f)).coerceIn(0f, 1f)
        return listOf(x, y, boxW.coerceIn(0f, 1f), boxH.coerceIn(0f, 1f))
    }

    private fun iou(a: List<Float>, b: List<Float>): Float {
        val ax0 = a[0]
        val ay0 = a[1]
        val ax1 = a[0] + a[2]
        val ay1 = a[1] + a[3]
        val bx0 = b[0]
        val by0 = b[1]
        val bx1 = b[0] + b[2]
        val by1 = b[1] + b[3]
        val interX0 = max(ax0, bx0)
        val interY0 = max(ay0, by0)
        val interX1 = min(ax1, bx1)
        val interY1 = min(ay1, by1)
        val interW = max(0f, interX1 - interX0)
        val interH = max(0f, interY1 - interY0)
        val intersection = interW * interH
        val union = (a[2] * a[3]) + (b[2] * b[3]) - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun bytesPerElement(dataType: DataType): Int =
        when (dataType) {
            DataType.FLOAT32 -> 4
            DataType.UINT8, DataType.INT8 -> 1
            else -> error("Unsupported tensor dtype: $dataType")
        }
}
