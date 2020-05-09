package com.example.imageclassification;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

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
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


import android.view.Menu;
import android.view.MenuItem;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;


public class MainActivity extends AppCompatActivity {
    private static final int EPOCHS = 100;

    private static final String MODEL_PATH = "mobilenet_v1_1.0_224.tflite";
    private static final String LABELS_PATH = "mobilenet_v1_1.0_224.txt";

    private TensorBuffer outputProbabilityBuffer;
    private List<String> associatedAxisLabels = null;
    private TensorProcessor probabilityProcessor;
    Map<String, Float> floatMap;

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

        setupInterpreter();
        loadLabels();

        for (int i = 1; i <= EPOCHS; i++) {
            loadBitmap(String.format("dataset/image_00%d.jpg", i));
            recognitions = recognizeImage();
            //runInferenceOnImage();
        }

        closeTflite();
        setupLabels();
        this.finish();
        System.exit(0);

    }

    private void loadBitmap(String filepath){
        try {
            bitmap = BitmapFactory.decodeStream(this.getAssets().open(filepath));
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(filepath);
        inputImageBuffer = new TensorImage(tflite.getInputTensor(0).dataType());
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

    private static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
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

    public List<Recognition> recognizeImage() {
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();

        return getTopKProbability(labeledProbability);
    }
    private void closeTflite() {
        tflite.close();
        tflite = null;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}