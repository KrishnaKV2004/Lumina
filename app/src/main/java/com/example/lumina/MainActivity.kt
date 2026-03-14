package com.example.lumina

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.example.lumina.ui.theme.LuminaTheme
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import android.util.Log
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Color
import android.view.View
import android.widget.FrameLayout
import java.util.concurrent.Executors
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp

class MainActivity : ComponentActivity() {

    var interpreter: Interpreter? = null

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        val modelBuffer = FileUtil.loadMappedFile(this, "model.tflite")
        interpreter = Interpreter(modelBuffer)

        setContent {
            LuminaTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { _ ->
                    EmptyScreen()
                }
            }
        }
    }
}

@Composable
fun EmptyScreen() {
    val context = LocalContext.current

    AndroidView(
        modifier = Modifier.fillMaxSize(),
        factory = { ctx ->
            val container = FrameLayout(ctx)

            val previewView = PreviewView(ctx)
            val overlay = BoxOverlayView(ctx)

            container.addView(previewView)
            container.addView(overlay)

            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build()
                preview.setSurfaceProvider(previewView.surfaceProvider)

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                val executor = Executors.newSingleThreadExecutor()

                imageAnalysis.setAnalyzer(executor) { image: ImageProxy ->
                    val bitmap = imageProxyToBitmap(image)
                    if (bitmap != null) {
                        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

                        val tensorImage = TensorImage(DataType.FLOAT32)
                        tensorImage.load(resizedBitmap)

                        val imageProcessor = ImageProcessor.Builder()
                            .add(NormalizeOp(0f, 255f))
                            .build()

                        val processedImage = imageProcessor.process(tensorImage)

                        Log.d("Lumina", "Tensor ready: ${processedImage.width}x${processedImage.height}")

                        val activity = context as MainActivity
                        val interpreter = activity.interpreter

                        if (interpreter != null) {

                            val inputBuffer = processedImage.buffer

                            val outputBuffer = Array(1) { Array(84) { FloatArray(8400) } }

                            try {
                                interpreter.run(inputBuffer, outputBuffer)

                                // Simple YOLO decoding
                                val detections = outputBuffer[0]
                                for (i in 0 until 8400) {

                                    var bestClass = -1
                                    var bestScore = 0f

                                    // Class scores start from index 4
                                    for (c in 4 until 84) {
                                        val score = detections[c][i]
                                        if (score > bestScore) {
                                            bestScore = score
                                            bestClass = c - 4
                                        }
                                    }

                                    if (bestScore > 0.5f) {

                                        val x = detections[0][i]
                                        val y = detections[1][i]
                                        val w = detections[2][i]
                                        val h = detections[3][i]

                                        overlay.setBox(x, y, w, h)

                                        Log.d(
                                            "Lumina",
                                            "Object detected: class=$bestClass confidence=$bestScore box=($x,$y,$w,$h)"
                                        )
                                    }
                                }
                            } catch (e: Exception) {
                                Log.e("Lumina", "Inference error: ${e.message}")
                            }
                        }
                    }
                    image.close()
                }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    context as ComponentActivity,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            }, ContextCompat.getMainExecutor(ctx))

            container
        }
    )
}

fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
    val planes = image.planes
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(
        android.graphics.Rect(0, 0, image.width, image.height),
        100,
        out
    )
    val imageBytes = out.toByteArray()

    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}
class BoxOverlayView(context: android.content.Context) : View(context) {

    private val paint = Paint().apply {
        color = Color.RED
        strokeWidth = 8f
        style = Paint.Style.STROKE
    }

    private var box: FloatArray? = null

    fun setBox(x: Float, y: Float, w: Float, h: Float) {
        box = floatArrayOf(x, y, w, h)
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val b = box ?: return

        val cx = b[0] * width
        val cy = b[1] * height
        val bw = b[2] * width
        val bh = b[3] * height

        val left = cx - bw / 2
        val top = cy - bh / 2
        val right = cx + bw / 2
        val bottom = cy + bh / 2

        canvas.drawRect(left, top, right, bottom, paint)
    }
}