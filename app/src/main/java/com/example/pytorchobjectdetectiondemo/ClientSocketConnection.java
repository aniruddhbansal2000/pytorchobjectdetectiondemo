package com.example.pytorchobjectdetectiondemo;

import android.os.Environment;

import org.pytorch.Module;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;



public class ClientSocketConnection {

    PrintWriter out;
    BufferedReader in;
    Socket socket;

    public void connectToServer(){
        //Create socket connection
        try {
            this.socket = new Socket("192.168.29.6", 12345);
            // connect to server
            this.out = new PrintWriter(socket.getOutputStream(),
                    true);
            this.in = new BufferedReader(new InputStreamReader(
                    socket.getInputStream()));
//        } catch (UnknownHostException e) {
//            System.out.println("Unknown host: kq6py");
//            System.exit(1);
//        } catch (IOException e) {
//            System.out.println("No I/O");
//            System.exit(1);
        }catch (Exception e){
            System.out.println(e);
            e.printStackTrace();
            System.exit(1);
        }
    }

    String sendReceive(String textToSend) {
        //Send data over socket
//        String text = textField.getText();
        out.println(textToSend);
//        textField.setText(new String(""));
        out.println("");

        //Receive text from server

        String textReceived = null;
        try {
            textReceived = in.readLine();
            System.out.println("Text received: " + textReceived);
        } catch (IOException e) {
            System.out.println("Read failed");
            System.exit(1);
        }
        return textReceived;
    }

}

//    Module module = null;
//    File folder = new File(Environment.getExternalStorageDirectory(), "/sample/");
//                assert(folder.exists());

//    File[] files = folder.listFiles();
//                for (int i = 0; i < files.length; ++i) {
//        File file = files[i];
//        if (file.isDirectory()) {
////                        traverse(file);
//        } else {
//        // do something here with the file
//        Bitmap bitmap = null;
//
//        //Getting the image from the image view
//        TextView textView = findViewById(R.id.result_text);
//        textView.setText("running for " + String.valueOf(i+1));
//
//        try {
//        //Read the image as Bitmap
////                                bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
//        String pathName = file.getAbsolutePath();
//        Drawable d = Drawable.createFromPath(pathName);
//        bitmap = ((BitmapDrawable)d).getBitmap();
//        //Here we reshape the image into 400*400
//        bitmap = Bitmap.createScaledBitmap(bitmap, 400, 400, true);
//        Log.d("beforeTag","reached here");
//        //Loading the model file.
//
//        }  catch (Exception e){
//        Log.d("exception tag", "exception found");
//        e.printStackTrace();
//        finish();
//        }
//
//        Log.d("fileTag","loaded successfully");
//
////Input Tensor
//final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
//        bitmap,
//        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
//        TensorImageUtils.TORCHVISION_NORM_STD_RGB
//        );
//
//        //Calling the forward of the model to run our input
//        assert module != null;
//        final Tensor output = module.forward(IValue.from(input)).toTensor();
//
//        textView = findViewById(R.id.result_text);
//        textView.setText("detected_class");
//        }
//        }
