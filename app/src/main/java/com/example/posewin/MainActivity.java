package com.example.posewin;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

public class MainActivity extends AppCompatActivity {
    private static final int CAMERA_REQUEST = 1888;
    private ImageView imageView;
    private int inputSize = 0;
    private static final int MY_CAMERA_PERMISSION_CODE = 100;
    Interpreter tflite = null;
    private String TAG = "Prajwal";
    String[] partNames = {
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    };

    String[][] poseChain = {
            {"nose", "leftEye"}, {"leftEye", "leftEar"}, {"nose", "rightEye"},
            {"rightEye", "rightEar"}, {"nose", "leftShoulder"},
            {"leftShoulder", "leftElbow"}, {"leftElbow", "leftWrist"},
            {"leftShoulder", "leftHip"}, {"leftHip", "leftKnee"},
            {"leftKnee", "leftAnkle"}, {"nose", "rightShoulder"},
            {"rightShoulder", "rightElbow"}, {"rightElbow", "rightWrist"},
            {"rightShoulder", "rightHip"}, {"rightHip", "rightKnee"},
            {"rightKnee", "rightAnkle"}
    };

    Map<Integer, Object> outputMap = new HashMap<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        String modelFile="posenet_mv1_075_float_from_checkpoints.tflite";
        try {
            tflite=new Interpreter(loadModelFile(MainActivity.this,modelFile));
        } catch (IOException e) {
            e.printStackTrace();
        }
        final Tensor no = tflite.getInputTensor(0);
        Log.d(TAG, "onCreate: Input shape"+ Arrays.toString(no.shape()));

        int c = tflite.getOutputTensorCount();
        Log.d(TAG, "onCreate: Output Count" +c );
        for (int i = 0; i <4 ; i++) {
            final Tensor output = tflite.getOutputTensor(i);
            Log.d(TAG, "onCreate: Output shape" + Arrays.toString(output.shape()));
        }
        this.imageView =  this.findViewById(R.id.imageView1);
        Button photoButton = this.findViewById(R.id.button1);
        photoButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                if (checkSelfPermission(Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.CAMERA},
                            MY_CAMERA_PERMISSION_CODE);
                } else {
                    Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, CAMERA_REQUEST);
                }
            }
        });
    }

    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull
            int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_PERMISSION_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
                Intent cameraIntent = new
                        Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST);
            } else {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }

    protected void onActivityResult ( int requestCode, int resultCode, Intent data){
        if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK) {




            Bitmap photo = (Bitmap) data.getExtras().get("data");


            try {
                ByteBuffer imgData=feedInputTensor(photo, 125.0f, 125.0f);
                Object[] input = new Object[]{imgData};
                Map<Integer, Object> outputMap = new HashMap<>();

                for (int i = 0; i < tflite.getOutputTensorCount(); i++) {
                    int[] shape = tflite.getOutputTensor(i).shape();
                    float[][][][] output = new float[shape[0]][shape[1]][shape[2]][shape[3]];
                    outputMap.put(i, output);
                }

                tflite.runForMultipleInputsOutputs(input, outputMap);


                Map<String, Integer> partsIds = new HashMap<>();
                List<Integer> parentToChildEdges = new ArrayList<>();
                List<Integer> childToParentEdges = new ArrayList<>();

                int localMaximumRadius = 1;
                int outputStride = 16;
                for (int i = 0; i < partNames.length; ++i)
                    partsIds.put(partNames[i], i);

                for (int i = 0; i < poseChain.length; ++i) {
                    parentToChildEdges.add(partsIds.get(poseChain[i][1]));
                    childToParentEdges.add(partsIds.get(poseChain[i][0]));
                }

                float[][][] scores = ((float[][][][]) outputMap.get(0))[0];
                float[][][] offsets = ((float[][][][]) outputMap.get(1))[0];
                float[][][] displacementsFwd = ((float[][][][]) outputMap.get(2))[0];
                float[][][] displacementsBwd = ((float[][][][]) outputMap.get(3))[0];

                PriorityQueue<Map<String, Object>> pq = buildPartWithScoreQueue(scores, 0.5, localMaximumRadius);

                int numParts = scores[0][0].length;
                int numEdges = parentToChildEdges.size();
                int sqaredNmsRadius = 20 * 20;

                List<Map<String, Object>> results = new ArrayList<>();

                while (results.size() < 17 && pq.size() > 0) {
                    Map<String, Object> root = pq.poll();
                    float[] rootPoint = getImageCoords(root, outputStride, numParts, offsets);

                    if (withinNmsRadiusOfCorrespondingPoint(
                            results, sqaredNmsRadius, rootPoint[0], rootPoint[1], (int) root.get("partId")))
                        continue;

                    Map<String, Object> keypoint = new HashMap<>();
                    keypoint.put("score", root.get("score"));
                    keypoint.put("part", partNames[(int) root.get("partId")]);
                    keypoint.put("y", rootPoint[0] / inputSize);
                    keypoint.put("x", rootPoint[1] / inputSize);
                    Log.d(TAG, String.valueOf(root.get("score")));
                    Log.d(TAG   , partNames[(int) root.get("partId")]);

                    Map<Integer, Map<String, Object>> keypoints = new HashMap<>();
                    keypoints.put((int) root.get("partId"), keypoint);

                    for (int edge = numEdges - 1; edge >= 0; --edge) {
                        int sourceKeypointId = parentToChildEdges.get(edge);
                        int targetKeypointId = childToParentEdges.get(edge);
                        if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(targetKeypointId)) {
                            keypoint = traverseToTargetKeypoint(edge, keypoints.get(sourceKeypointId),
                                    targetKeypointId, scores, offsets, outputStride, displacementsBwd);
                            keypoints.put(targetKeypointId, keypoint);
                        }
                    }

                    for (int edge = 0; edge < numEdges; ++edge) {
                        int sourceKeypointId = childToParentEdges.get(edge);
                        int targetKeypointId = parentToChildEdges.get(edge);
                        if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(targetKeypointId)) {
                            keypoint = traverseToTargetKeypoint(edge, keypoints.get(sourceKeypointId),
                                    targetKeypointId, scores, offsets, outputStride, displacementsFwd);
                            keypoints.put(targetKeypointId, keypoint);
                        }
                    }

                    Map<String, Object> result = new HashMap<>();
                    result.put("keypoints", keypoints);
                    result.put("score", getInstanceScore(keypoints, numParts));
                    results.add(result);
                }



            } catch (IOException e) {
                e.printStackTrace();
            }

/*
            Log.d(TAG,"bhai:"+photo.getWidth()+":"+photo.getHeight());
            photo = Bitmap.createScaledBitmap(photo, 337, 337, false);
            photo = photo.copy(Bitmap.Config.ARGB_8888,true);
            Log.d(TAG, "onActivityResult: Bitmap resized");

            int width =photo.getWidth();
            int height = photo.getHeight();
            float[][][][] result = new float[1][width][height][3];
            int[] pixels = new int[width*height];
            photo.getPixels(pixels, 0, width, 0, 0, width, height);
            int pixelsIndex = 0;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    // result[i][j] =  pixels[pixelsIndex];
                    int p = pixels[pixelsIndex];
                    result[0][i][j][0]  = (p >> 16) & 0xff;
                    result[0][i][j][1]  = (p >> 8) & 0xff;
                    result[0][i][j][2]  = p & 0xff;
                    pixelsIndex++;
                }
            }
            Object [] inputs = {result};
            //inputs[0] = inp;

            outputMap.put(0, out1);
            outputMap.put(1, out2);
            outputMap.put(2, out3);
            outputMap.put(3, out4);

            tflite.runForMultipleInputsOutputs(inputs,outputMap);
            out1 = (float[][][][]) outputMap.get(0);
            out2 = (float[][][][]) outputMap.get(1);
            out3 = (float[][][][]) outputMap.get(2);
            out4 = (float[][][][]) outputMap.get(3);

            Canvas canvas = new Canvas(photo);
            Paint p = new Paint();
            p.setColor(Color.RED);



            float[][][] scores = new float[out1[0].length][out1[0][0].length][17];
            int[][] heatmap_pos = new int[17][2];

            for(int i=0;i<17;i++)
            {
                float max = -1;

                for(int j=0;j<out1[0].length;j++)
                {
                    for(int k=0;k<out1[0][0].length;k++)
                    {
                        //  Log.d("mylog", "onActivityResult: "+out1[0][j][k][i]);
                        scores[j][k][i]  = sigmoid(out1[0][j][k][i]);
                        if(max<scores[j][k][i])
                        {
                            max = scores[j][k][i];
                            heatmap_pos[i][0] = j;
                            heatmap_pos[i][1] = k;
                        }
                    }

                }
                Log.d(TAG, "onActivityResult: "+max+"    "+heatmap_pos[i][0]+"    "+heatmap_pos[i][1]);
            }

            for(int i=0;i<17;i++)
            {
                Log.d("heatlog", "onActivityResult: "+heatmap_pos[i][0]+"    "+heatmap_pos[i][1]);
            }
            float[][] offset_vector = new float[17][2];
            float[][] keypoint_pos = new float[17][2];
            for(int i=0;i<17;i++)
            {
                offset_vector[i][0] = out2[0][heatmap_pos[i][0]][heatmap_pos[i][1]][i];
                offset_vector[i][1] = out2[0][heatmap_pos[i][0]][heatmap_pos[i][1]][i+17];
                Log.d("myoff",offset_vector[i][0]+":"+offset_vector[i][1]);
                keypoint_pos[i][0] = heatmap_pos[i][0]*16+offset_vector[i][0];
                keypoint_pos[i][1] = heatmap_pos[i][1]*16+offset_vector[i][1];
                Log.d(TAG, "onActivityResult: "+keypoint_pos[i][0]+"    "+keypoint_pos[i][1]);
                canvas.drawCircle(keypoint_pos[i][0],keypoint_pos[i][1],5,p);      }
*/
            imageView.setImageBitmap(photo);
        }
    }






    private void decodePose(){


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

    ByteBuffer feedInputTensor(Bitmap bitmapRaw, float mean, float std) throws IOException {
        Tensor tensor = tflite.getInputTensor(0);
        int[] shape = tensor.shape();
        int inputSize = shape[1];
        int inputChannels = shape[3];

        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * 4);
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
    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    PriorityQueue<Map<String, Object>> buildPartWithScoreQueue(float[][][] scores,
                                                               double threshold,
                                                               int localMaximumRadius) {
        PriorityQueue<Map<String, Object>> pq =
                new PriorityQueue<>(
                        1,
                        new Comparator<Map<String, Object>>() {
                            @Override
                            public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
                                return Float.compare((float) rhs.get("score"), (float) lhs.get("score"));
                            }
                        });

        for (int heatmapY = 0; heatmapY < scores.length; ++heatmapY) {
            for (int heatmapX = 0; heatmapX < scores[0].length; ++heatmapX) {
                for (int keypointId = 0; keypointId < scores[0][0].length; ++keypointId) {
                    float score = sigmoid(scores[heatmapY][heatmapX][keypointId]);
                    if (score < threshold) continue;

                    if (scoreIsMaximumInLocalWindow(
                            keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores)) {
                        Map<String, Object> res = new HashMap<>();
                        res.put("score", score);
                        res.put("y", heatmapY);
                        res.put("x", heatmapX);
                        res.put("partId", keypointId);
                        pq.add(res);
                    }
                }
            }
        }

        return pq;
    }

    boolean scoreIsMaximumInLocalWindow(int keypointId,
                                        float score,
                                        int heatmapY,
                                        int heatmapX,
                                        int localMaximumRadius,
                                        float[][][] scores) {
        boolean localMaximum = true;
        int height = scores.length;
        int width = scores[0].length;

        int yStart = Math.max(heatmapY - localMaximumRadius, 0);
        int yEnd = Math.min(heatmapY + localMaximumRadius + 1, height);
        for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
            int xStart = Math.max(heatmapX - localMaximumRadius, 0);
            int xEnd = Math.min(heatmapX + localMaximumRadius + 1, width);
            for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
                if (sigmoid(scores[yCurrent][xCurrent][keypointId]) > score) {
                    localMaximum = false;
                    break;
                }
            }
            if (!localMaximum) {
                break;
            }
        }

        return localMaximum;
    }

    float[] getImageCoords(Map<String, Object> keypoint,
                           int outputStride,
                           int numParts,
                           float[][][] offsets) {
        int heatmapY = (int) keypoint.get("y");
        int heatmapX = (int) keypoint.get("x");
        int keypointId = (int) keypoint.get("partId");
        float offsetY = offsets[heatmapY][heatmapX][keypointId];
        float offsetX = offsets[heatmapY][heatmapX][keypointId + numParts];

        float y = heatmapY * outputStride + offsetY;
        float x = heatmapX * outputStride + offsetX;

        return new float[]{y, x};
    }

    boolean withinNmsRadiusOfCorrespondingPoint(List<Map<String, Object>> poses,
                                                float squaredNmsRadius,
                                                float y,
                                                float x,
                                                int keypointId) {
        for (Map<String, Object> pose : poses) {
            Map<Integer, Object> keypoints = (Map<Integer, Object>) pose.get("keypoints");
            Map<String, Object> correspondingKeypoint = (Map<String, Object>) keypoints.get(keypointId);
            float _x = (float) correspondingKeypoint.get("x") * inputSize - x;
            float _y = (float) correspondingKeypoint.get("y") * inputSize - y;
            float squaredDistance = _x * _x + _y * _y;
            if (squaredDistance <= squaredNmsRadius)
                return true;
        }

        return false;
    }

    Map<String, Object> traverseToTargetKeypoint(int edgeId,
                                                 Map<String, Object> sourceKeypoint,
                                                 int targetKeypointId,
                                                 float[][][] scores,
                                                 float[][][] offsets,
                                                 int outputStride,
                                                 float[][][] displacements) {
        int height = scores.length;
        int width = scores[0].length;
        int numKeypoints = scores[0][0].length;
        float sourceKeypointY = (float) sourceKeypoint.get("y") * inputSize;
        float sourceKeypointX = (float) sourceKeypoint.get("x") * inputSize;

        int[] sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypointY, sourceKeypointX,
                outputStride, height, width);

        float[] displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements);

        float[] displacedPoint = new float[]{
                sourceKeypointY + displacement[0],
                sourceKeypointX + displacement[1]
        };

        float[] targetKeypoint = displacedPoint;

        final int offsetRefineStep = 2;
        for (int i = 0; i < offsetRefineStep; i++) {
            int[] targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1],
                    outputStride, height, width);

            int targetKeypointY = targetKeypointIndices[0];
            int targetKeypointX = targetKeypointIndices[1];

            float offsetY = offsets[targetKeypointY][targetKeypointX][targetKeypointId];
            float offsetX = offsets[targetKeypointY][targetKeypointX][targetKeypointId + numKeypoints];

            targetKeypoint = new float[]{
                    targetKeypointY * outputStride + offsetY,
                    targetKeypointX * outputStride + offsetX
            };
        }

        int[] targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1],
                outputStride, height, width);

        float score = sigmoid(scores[targetKeypointIndices[0]][targetKeypointIndices[1]][targetKeypointId]);

        Map<String, Object> keypoint = new HashMap<>();
        keypoint.put("score", score);
        keypoint.put("part", partNames[targetKeypointId]);
        keypoint.put("y", targetKeypoint[0] / inputSize);
        keypoint.put("x", targetKeypoint[1] / inputSize);

        return keypoint;
    }

    int[] getStridedIndexNearPoint(float _y, float _x, int outputStride, int height, int width) {
        int y_ = Math.round(_y / outputStride);
        int x_ = Math.round(_x / outputStride);
        int y = y_ < 0 ? 0 : y_ > height - 1 ? height - 1 : y_;
        int x = x_ < 0 ? 0 : x_ > width - 1 ? width - 1 : x_;
        return new int[]{y, x};
    }


    float[] getDisplacement(int edgeId, int[] keypoint, float[][][] displacements) {
        int numEdges = displacements[0][0].length / 2;
        int y = keypoint[0];
        int x = keypoint[1];
        return new float[]{displacements[y][x][edgeId], displacements[y][x][edgeId + numEdges]};
    }

    float getInstanceScore(Map<Integer, Map<String, Object>> keypoints, int numKeypoints) {
        float scores = 0;
        for (Map.Entry<Integer, Map<String, Object>> keypoint : keypoints.entrySet())
            scores += (float) keypoint.getValue().get("score");
        return scores / numKeypoints;
    }

    public float sigmoid(float value) {
        float p =  (float)(1.0 / (1 + Math.exp(-value)));
        return p;
    }
}