package com.example.lumina

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Size
import android.view.View
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import android.speech.tts.TextToSpeech
import java.util.Locale

class MainActivity : ComponentActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView

    private val inputSize = 640
    private val executor = Executors.newSingleThreadExecutor()
    private var lastProcessTime = 0L

    private lateinit var tts: TextToSpeech
    private val lastSpokenTimes = mutableMapOf<String, Long>()
    private val cooldown = 5000L

    private val priority = listOf(
        "truck",
        "bus",
        "motorcycle",
        "car",
        "person",
        "bicycle"
    )

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }

        interpreter = Interpreter(FileUtil.loadMappedFile(this,"model.tflite"))

        previewView = PreviewView(this)
        overlay = OverlayView(this)

        val container = FrameLayout(this)
        container.addView(previewView)
        container.addView(overlay)

        setContentView(container)

        tts = TextToSpeech(this){
            if(it == TextToSpeech.SUCCESS){
                tts.language = Locale.US
            }
        }

        startCamera()
    }

    private fun startCamera() {

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build()
            preview.setSurfaceProvider(previewView.surfaceProvider)

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(inputSize,inputSize))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(executor) { image ->

                val now = System.currentTimeMillis()
                if(now - lastProcessTime < 1000){
                    image.close()
                    return@setAnalyzer
                }
                lastProcessTime = now

                val bitmap = imageToBitmap(image)

                if(bitmap!=null){

                    val resized = Bitmap.createScaledBitmap(bitmap,inputSize,inputSize,true)
                    val input = bitmapToBuffer(resized)

                    val outputShape = interpreter.getOutputTensor(0).shape()

                    val output =
                        if (outputShape[1] == 8400) {
                            Array(1){ Array(8400){ FloatArray(outputShape[2]) } }
                        } else {
                            Array(1){ Array(outputShape[1]){ FloatArray(8400) } }
                        }

                    interpreter.run(input,output)

                    val detections = output[0]

                    val transposed = outputShape[1] != 8400

                    val boxes = mutableListOf<FloatArray>()
                    val scores = mutableListOf<Float>()
                    val classes = mutableListOf<Int>()

                    for(i in 0 until 8400){

                        val x = if(!transposed) detections[i][0] else detections[0][i]
                        val y = if(!transposed) detections[i][1] else detections[1][i]
                        val w = if(!transposed) detections[i][2] else detections[2][i]
                        val h = if(!transposed) detections[i][3] else detections[3][i]

                        var bestScore = 0f
                        var bestClass = -1

                        val featureCount = if(!transposed) detections[i].size else detections.size

                        for(c in 4 until featureCount){

                            val score =
                                if(!transposed) detections[i][c]
                                else detections[c][i]

                            if(score > bestScore){
                                bestScore = score
                                bestClass = c-4
                            }
                        }

                        if(bestScore > 0.4f){

                            val nx = if(x>1) x/inputSize else x
                            val ny = if(y>1) y/inputSize else y
                            val nw = if(w>1) w/inputSize else w
                            val nh = if(h>1) h/inputSize else h

                            boxes.add(floatArrayOf(nx,ny,nw,nh))
                            scores.add(bestScore)
                            classes.add(bestClass)
                        }
                    }

                    val finalBoxes = mutableListOf<FloatArray>()
                    val finalClasses = mutableListOf<Int>()
                    val finalScores = mutableListOf<Float>()

                    for(i in boxes.indices){

                        val a = boxes[i]
                        var keep = true

                        val ax1 = a[0] - a[2]/2
                        val ay1 = a[1] - a[3]/2
                        val ax2 = a[0] + a[2]/2
                        val ay2 = a[1] + a[3]/2

                        for(j in finalBoxes.indices){

                            val b = finalBoxes[j]

                            val bx1 = b[0] - b[2]/2
                            val by1 = b[1] - b[3]/2
                            val bx2 = b[0] + b[2]/2
                            val by2 = b[1] + b[3]/2

                            val interX1 = maxOf(ax1,bx1)
                            val interY1 = maxOf(ay1,by1)
                            val interX2 = minOf(ax2,bx2)
                            val interY2 = minOf(ay2,by2)

                            val interArea =
                                maxOf(0f,interX2-interX1) *
                                maxOf(0f,interY2-interY1)

                            val areaA = a[2]*a[3]
                            val areaB = b[2]*b[3]

                            val iou = interArea/(areaA + areaB - interArea)

                            if(iou > 0.45f){
                                keep = false
                                break
                            }
                        }

                        if(keep){
                            finalBoxes.add(a)
                            finalClasses.add(classes[i])
                            finalScores.add(scores[i])
                        }
                    }

                    overlay.update(finalBoxes,finalClasses,finalScores)

                    val detectedLabels = finalClasses.mapNotNull {
                        if(it in LABELS.indices) LABELS[it] else null
                    }.toSet().toMutableList()

                    detectedLabels.sortBy {
                        val idx = priority.indexOf(it)
                        if(idx >= 0) idx else 999
                    }

                    val nowSpeak = System.currentTimeMillis()

                    for(label in detectedLabels){

                        val lastTime = lastSpokenTimes[label] ?: 0L

                        if(nowSpeak - lastTime > cooldown){

                            val message = "$label ahead"

                            tts.speak(
                                message,
                                TextToSpeech.QUEUE_ADD,
                                null,
                                label
                            )

                            lastSpokenTimes[label] = nowSpeak
                        }
                    }
                }

                image.close()
            }

            cameraProvider.unbindAll()

            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )

        },ContextCompat.getMainExecutor(this))
    }

    private fun bitmapToBuffer(bitmap: Bitmap): ByteBuffer {

        val buffer = ByteBuffer.allocateDirect(1*inputSize*inputSize*3*4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize*inputSize)

        bitmap.getPixels(pixels,0,inputSize,0,0,inputSize,inputSize)

        for(p in pixels){

            buffer.putFloat(((p shr 16 and 0xFF)/255f))
            buffer.putFloat(((p shr 8 and 0xFF)/255f))
            buffer.putFloat(((p and 0xFF)/255f))
        }

        buffer.rewind()
        return buffer
    }

    private fun imageToBitmap(image: ImageProxy): Bitmap? {

        val planes = image.planes
        val y = planes[0].buffer
        val u = planes[1].buffer
        val v = planes[2].buffer

        val ySize = y.remaining()
        val uSize = u.remaining()
        val vSize = v.remaining()

        val nv21 = ByteArray(ySize+uSize+vSize)

        y.get(nv21,0,ySize)
        v.get(nv21,ySize,vSize)
        u.get(nv21,ySize+vSize,uSize)

        val yuv = YuvImage(nv21,ImageFormat.NV21,image.width,image.height,null)

        val out = java.io.ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0,0,image.width,image.height),100,out)

        val bytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(bytes,0,bytes.size)
    }
}

val LABELS = arrayOf(
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair"
)

class OverlayView(context: android.content.Context) : View(context){

    private val boxPaint = Paint().apply{
        color = Color.GREEN
        strokeWidth = 6f
        style = Paint.Style.STROKE
    }

    private val textPaint = Paint().apply{
        color = Color.GREEN
        textSize = 48f
    }

    private var boxes:List<FloatArray> = emptyList()
    private var classes:List<Int> = emptyList()
    private var scores:List<Float> = emptyList()

    fun update(b:List<FloatArray>,c:List<Int>,s:List<Float>){
        boxes=b
        classes=c
        scores=s
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas){

        for(i in boxes.indices){

            val d = boxes[i]

            val cx = d[0]*width
            val cy = d[1]*height
            val bw = d[2]*width
            val bh = d[3]*height

            val left = cx-bw/2
            val top = cy-bh/2
            val right = cx+bw/2
            val bottom = cy+bh/2

            canvas.drawRect(left,top,right,bottom,boxPaint)

            val label = LABELS.getOrElse(classes[i]){"obj"}
            val score = (scores[i]*100).toInt()

            canvas.drawText("$label $score%",left,top-10,textPaint)
        }
    }
}