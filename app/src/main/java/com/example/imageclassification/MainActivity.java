package com.example.imageclassification;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.media.Image;
import android.opengl.Matrix;
import android.os.Bundle;

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

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
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
import java.util.stream.Collectors;


public class MainActivity extends AppCompatActivity {
    private static final int EPOCHS = 10;

    final float IMAGE_MEAN = 127.5f;
    final float IMAGE_STD = 127.5f;

    private static final String MODEL_PATH = "mobilenet_v1_1.0_224.tflite";

    private Bitmap bitmap;

    private ByteBuffer inputByteBuffer;
    private TensorBuffer outputByteBuffer;

    private Interpreter.Options tfliteOptions = new Interpreter.Options();;
    private Interpreter tflite;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        loadLabels();
        setupInterpreter();
        loadImageAsByteBuffer();
        runInferenceOnImage(EPOCHS);

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

    private void loadLabels() {
        outputByteBuffer = TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.FLOAT32);
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
