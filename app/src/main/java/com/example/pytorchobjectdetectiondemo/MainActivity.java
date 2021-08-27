package com.example.pytorchobjectdetectiondemo;

import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {
    private static int RESULT_LOAD_IMAGE = 1;
    private String url = "http://" + "192.168.29.6" + ":" + "5000" + "/predict";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button buttonLoadImage = (Button) findViewById(R.id.button);
        Button detectButton = (Button) findViewById(R.id.detect);

        requestPermissions(new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        requestPermissions(new String[]{android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);

        buttonLoadImage.setOnClickListener(arg0 -> {
            TextView textView = findViewById(R.id.result_text);
            textView.setText("");

            Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            i.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
            startActivityForResult(i, RESULT_LOAD_IMAGE);
        });

        detectButton.setOnClickListener(arg0 -> {
            Bitmap bitmap = null;
            Module module = null;

            try {
                String filePath = fetchModelFile(MainActivity.this, "resnet18_traced.ptl");
                Log.d("pathTag", "obtained file path successfully");
                module = LiteModuleLoader.load(filePath);
            }catch (IOException e) {
                finish();
            }


            //Getting the image from the image view
            ImageView imageView = (ImageView) findViewById(R.id.image);

            for(int i = 0; i < 1; i++) {
                Log.d("noteTime", "starting time for " + String.valueOf(i+1) + " is " + String.valueOf(System.currentTimeMillis()/1000));
                try {
                    //Read the image as Bitmap
                    bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();

                    //Here we reshape the image into 400*400
                    bitmap = Bitmap.createScaledBitmap(bitmap, 400, 400, true);
                    Log.d("beforeTag", "reached here");
                    //Loading the model file.
                } catch (Exception e) {
                    Log.d("exception tag", "exception found");
                    e.printStackTrace();
                    finish();
                }

                Log.d("imageTag", "loaded successfully");

                //Input Tensor
                final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
                        bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB
                );

                //Calling the forward of the model to run our input
                assert module != null;
                final Tensor output = module.forward(IValue.from(input)).toTensor();
//                    Log.d("tensorShape", String.valueOf(output.dtype()) + " " + String.valueOf(output.numel()) + " " + String.valueOf(output.memoryFormat()));
//                    int[] map_shape = Arrays.stream(output.shape()).mapToInt(itx -> (int) itx).toArray();
//                    Log.d("map shape", String.valueOf(map_shape[0]) + " " + String.valueOf(map_shape[1]) + " " + String.valueOf(map_shape[2]) + " " + String.valueOf(map_shape[3]));

//                    final String outputString = module.forward(IValue.from(input)).toString();

                //get tensor in array form
                final float[] score_arr = output.getDataAsFloatArray();
                Log.d("array shape", String.valueOf(score_arr.length));

                // JSON array to send to server
//                    JSONArray mJSONActivationMap = new JSONArray(Arrays.asList(score_arr));
//                    JSONArray mJSONMapShape = new JSONArray(Arrays.asList(map_shape));

//                     Fetch the index of the value with maximum score
                float max_score = -Float.MAX_VALUE;
                int ms_ix = -1;
                for (int ix = 0; ix < score_arr.length; ix++) {
                    if (score_arr[ix] > max_score) {
                        max_score = score_arr[ix];
                        ms_ix = ix;
                    }
                }

//              Fetching the name from the list based on the index
                String detected_class = ModelClass.MODEL_CLASSES[ms_ix];
                TextView textView = findViewById(R.id.result_text);
                textView.setText(detected_class);

                addResponseToFile();
            }
        });

    }

    private void addResponseToFile(){
        // WIRINT TO .CSV FILE

        //Squash all your data here at once or you can format it accordingly.
        String csv_data = Arrays.toString(new String[]{"sample_category", "sample_score"});/// your csv data as string;

        File root = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);

        //Sub Folder
        root = new File(root, "My Results");
        root.mkdir();

        //Select the name for your file
        root = new File(root , "response.csv");

        try {
            FileOutputStream fout = new FileOutputStream(root);
            fout.write(csv_data.getBytes());

            fout.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();

            boolean bool = false;
            try {
                // try to create the file
                bool = root.createNewFile();
            } catch (IOException ex) {
                ex.printStackTrace();
            }

            if (bool){
                // call the method again
                addResponseToFile();
            }else {
                throw new IllegalStateException("Failed to create file");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };

            Cursor cursor = getContentResolver().query(selectedImage, filePathColumn, null, null, null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView imageView = (ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //Setting the URI so we can read the Bitmap from the image
            imageView.setImageURI(null);
            imageView.setImageURI(selectedImage);
        }
    }

    public static String fetchModelFile(Context context, String modelName) throws IOException {
        File file = new File(context.getFilesDir(), modelName);
        Log.d("fileTag","Referenced file successfully");
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        Log.d("parseTag","Need to parse");

        try (InputStream is = context.getAssets().open(modelName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

}