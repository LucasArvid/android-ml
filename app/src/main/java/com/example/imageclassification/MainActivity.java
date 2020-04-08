package com.example.imageclassification;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.media.Image;
import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.stream.Collectors;


public class MainActivity extends AppCompatActivity {
    private static final int EPOCHS = 10;

    static final int IMG_SIZE_X = 224;
    static final int IMG_SIZE_Y = 224;

    private static final String MODEL_PATH = "mobilenet_v1_1.0_224.tflite";
    private static final String LABELS_PATH = "mobilenet_v1_1.0_224.txt";

    private TensorBuffer outputProbabilityBuffer;
    private List<String> associatedAxisLabels = null;
    private TensorProcessor probabilityProcessor;
    Map<String, Float> floatMap;

    private  int imageSizeX;
    private  int imageSizeY;
    private Bitmap bitmap;

    private List<Recognition> recognitions;

    private TensorImage inputImageBuffer;

    private Interpreter.Options tfliteOptions = new Interpreter.Options();;
    private Interpreter tflite;

    private List<String> labels;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        setupInterpreter();
        loadBitmap();
        loadLabels();

        //recognitions = recognizeImage();

        runInferenceOnImage(EPOCHS);

        setupLabels();


        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
            }
        });
    }

    private void loadBitmap(){
        try {
            bitmap = BitmapFactory.decodeStream(this.getAssets().open("skata.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("bitmapError",  "failed big time boi", e);
        }

        inputImageBuffer = new TensorImage(tflite.getInputTensor(0).dataType()); // or Data.Type.FLOAT32
        inputImageBuffer.load(bitmap);

        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize,cropSize))
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(getProcessNormalizeOp())
                .build();
        inputImageBuffer = imageProcessor.process(inputImageBuffer);

    }
    private void loadLabels() {
        try {
            associatedAxisLabels = FileUtil.loadLabels(this, LABELS_PATH);
            labels = FileUtil.loadLabels(this, LABELS_PATH);
        } catch (IOException e) {
            Log.e("tfliteError", "error reading labels", e);
        }
        outputProbabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.FLOAT32);

        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
    }

    private void setupLabels() {


        if (associatedAxisLabels != null) {
            TensorLabel labels = new TensorLabel(associatedAxisLabels, probabilityProcessor.process(outputProbabilityBuffer));

            floatMap = labels.getMapWithFloatValue();
        }
    }

    private void setupInterpreter() {
        tfliteOptions.setNumThreads(2);
        try {
            tflite = new Interpreter(loadModelFile(), tfliteOptions);
        } catch (IOException e) {
            Log.e("tfliteError","error loding model", e);
        }
    }

    private MappedByteBuffer loadModelFile( ) throws IOException {

        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private TensorOperator getProcessNormalizeOp() {
        return new NormalizeOp(127.5f, 127.5f);
    }

    private TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(0.0f, 1.0f);
    }

    private void runInferenceOnImage(int epochs) {

        Trace.beginSection("runInference");
        long startTimeForReference = SystemClock.uptimeMillis();
        for (int i = 0; i < epochs; i++){
            tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        }
        long endTimeForReference = SystemClock.uptimeMillis();
        Trace.endSection();
        System.out.println("Timecost to run model inference: " + (endTimeForReference - startTimeForReference));


    }
    /** An immutable result returned by a Classifier describing what was recognized. */
    public static class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /** Display name for the recognition. */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        public Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }
    /** Gets the top-k results. */
    private static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), 3);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    /** Runs inference and returns the classification results. */
    public List<Recognition> recognizeImage() {
        // Logs this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        // Runs the inference call.
        Trace.beginSection("runInference");
        long startTimeForReference = SystemClock.uptimeMillis();
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        long endTimeForReference = SystemClock.uptimeMillis();
        Trace.endSection();
        System.out.println("Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
        // Gets the map of label and probability.
        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        Trace.endSection();

        // Gets top-k results.
        return getTopKProbability(labeledProbability);
    }
    private void closeTflite() {
        tflite.close();
        tflite = null;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }




}
