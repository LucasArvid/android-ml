package com.example.imageclassification;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.RectF;
import android.media.Image;
import android.os.Bundle;
import android.graphics.Matrix;
import android.os.AsyncTask;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
import java.util.Vector;
import java.util.stream.Collectors;

import javax.xml.transform.Result;


public class MainActivity extends AppCompatActivity {
     static final int EPOCHS = 10;
     private AssetManager assetManager;
    final float IMAGE_MEAN = 127.5f;
    final float IMAGE_STD = 127.5f;
    private boolean tfLiteBusy = false;
     int inputSize;
     Vector<String> labels;
    float[][] labelProb;
    private static final String MODEL_PATH = "mobilenet_v1_1.0_224.tflite";
    private static final String LABEL_PATH = "mobilenet_v1_1.0_224.txt";

    List<Map<String, Object>> results;
     Bitmap bitmap;

     ByteBuffer inputByteBuffer;
     TensorBuffer outputByteBuffer;

     Interpreter.Options tfliteOptions = new Interpreter.Options();;
     Interpreter tflite;
     static final int BYTES_PER_CHANNEL = 4;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);


        // setupInterpreter();

        // load image + inference
        try {
            loadModel();
            new RunModelOnImage().executeTfliteTask();
        } catch(Exception e) {
            e.printStackTrace();
        }
        //loadImageAsByteBuffer();

        //runInferenceOnImage(EPOCHS);

    }

     ByteBuffer feedInputTensorImage(String path, float mean, float std) throws IOException{

        InputStream inputStream = this.getAssets().open(path);
        Bitmap  bitmapRaw = BitmapFactory.decodeStream(inputStream);
        return feedInputTensor(bitmapRaw, mean, std);

    }

     ByteBuffer feedInputTensor(Bitmap bitmapRaw, float mean, float std) throws IOException {
        Tensor tensor = tflite.getInputTensor(0);
        int[] shape = tensor.shape();
        inputSize = shape[1];
        int inputChannels = shape[3];

        int bytePerChannel = tensor.dataType() == DataType.UINT8 ? 1 : BYTES_PER_CHANNEL;
        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
        imgData.order(ByteOrder.nativeOrder());

        Bitmap bitmap = bitmapRaw;
        if (bitmapRaw.getWidth() != inputSize || bitmapRaw.getHeight() != inputSize) {
            Matrix matrix = getTransformationMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(),
                    inputSize, inputSize, false);
            bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
            final Canvas canvas = new Canvas(bitmap);
            canvas.drawBitmap(bitmapRaw, matrix, null);
        }

        if (tensor.dataType() == DataType.FLOAT32) {
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    int pixelValue = bitmap.getPixel(j, i);
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);
                    imgData.putFloat(((pixelValue & 0xFF) - mean) / std);
                }
            }
        } else {
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    int pixelValue = bitmap.getPixel(j, i);
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                }
            }
        }

        return imgData;
    }

    private static Matrix getTransformationMatrix(final int srcWidth,
                                                  final int srcHeight,
                                                  final int dstWidth,
                                                  final int dstHeight,
                                                  final boolean maintainAspectRatio) {
        final Matrix matrix = new Matrix();

        if (srcWidth != dstWidth || srcHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) srcWidth;
            final float scaleFactorY = dstHeight / (float) srcHeight;

            if (maintainAspectRatio) {
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        matrix.invert(new Matrix());
        return matrix;
    }

    private void close() {
        if (tflite != null)
            tflite.close();
        labels = null;
        labelProb = null;
    }

    private abstract class TfliteTask extends AsyncTask<Void, Void, Void> {

        TfliteTask() {
            if (tfLiteBusy) throw new RuntimeException("Interpreter busy");
            else tfLiteBusy = true;

        }

        abstract void runTflite();

        abstract void onRunTfliteDone();

        public void executeTfliteTask() {
                runTflite();
                tfLiteBusy = false;
                onRunTfliteDone();

        }

        protected Void doInBackground(Void... backgroundArguments) {
            runTflite();
            return null;
        }

        protected void onPostExecute(Void backgroundResult) {
            tfLiteBusy = false;
            onRunTfliteDone();
        }
    }

    private class RunModelOnImage extends TfliteTask {
        int NUM_RESULTS;
        float THRESHOLD;
        ByteBuffer input;
        long startTime;

        RunModelOnImage() throws IOException {
            float IMAGE_MEAN = 127.5f;
            float IMAGE_STD = 127.5f;
            String path = "skata.jpg";
             NUM_RESULTS = 6;
             THRESHOLD = 0.05f;

            startTime = SystemClock.uptimeMillis();
            input = feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD);
        }

        protected void runTflite() {
            tflite.run(input, labelProb);
        }

        protected void onRunTfliteDone() {
            Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));
            results = GetTopN(NUM_RESULTS, THRESHOLD);
        }
    }

    private void loadImageAsByteBuffer() {
        try {
            bitmap = BitmapFactory.decodeStream(this.getAssets().open("skata.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("bitmapError",  "failed big time boi", e);
        }

        Tensor tensor = tflite.getInputTensor(0);
        int[] shape = tensor.shape();
        int inputSize = shape[1];
        int inputChannels = shape[3];
        int bytePerChannel = 4;

        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);
        inputByteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
        inputByteBuffer.order(ByteOrder.nativeOrder());

        // Subtract and divide with mean and std for float point models. Values can be found on tensorflows model hub.
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = resizedBitmap.getPixel(j, i);
                inputByteBuffer.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                inputByteBuffer.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                inputByteBuffer.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }

    }

    private List<Map<String, Object>> GetTopN(int numResults, float threshold) {
        PriorityQueue<Map<String, Object>> pq =
                new PriorityQueue<>(
                        1,
                        new Comparator<Map<String, Object>>() {
                            @Override
                            public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
                                return Float.compare((float) rhs.get("confidence"), (float) lhs.get("confidence"));
                            }
                        });

        for (int i = 0; i < labels.size(); ++i) {
            float confidence = labelProb[0][i];
            if (confidence > threshold) {
                Map<String, Object> res = new HashMap<>();
                res.put("index", i);
                res.put("label", labels.size() > i ? labels.get(i) : "unknown");
                res.put("confidence", confidence);
                pq.add(res);
            }
        }

        final ArrayList<Map<String, Object>> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), numResults);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
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

    private String loadModel() throws IOException {
        String model = MODEL_PATH;
        assetManager = this.getAssets();
        AssetFileDescriptor fileDescriptor = assetManager.openFd(model);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

        int numThreads = 1;
        final Interpreter.Options tfliteOptions = new Interpreter.Options();
        tfliteOptions.setNumThreads(numThreads);
        tflite = new Interpreter(buffer, tfliteOptions);

        String labels = LABEL_PATH;

        if (labels.length() > 0) {
            loadLabels(assetManager, labels);
        }

        return "success";
    }

    private void loadLabels(AssetManager assetManager, String path) {
        BufferedReader br;
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(path)));
            String line;
            labels = new Vector<>();
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
            labelProb = new float[1][labels.size()];
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Failed to read label file", e);
        }
    }

    private void runInferenceOnImage(int epochs) {

        Trace.beginSection("runInference");
        long startTimeForReference = SystemClock.uptimeMillis();
        for (int i = 0; i < epochs; i++){
            tflite.run(inputByteBuffer, outputByteBuffer.getBuffer().rewind());
        }
        long endTimeForReference = SystemClock.uptimeMillis();
        Trace.endSection();
        System.out.println("Timecost to run model inference: " + (endTimeForReference - startTimeForReference));
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
