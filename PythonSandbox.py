import libjevois as jevois
import cv2
import numpy as np
import os
import time
class PythonSandbox:
    # recognizer
    
    
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)
        self.frame = 0
        self.facenames = []
        self.facenames = os.listdir('modules/JeVois/PythonSandbox/training-data/')
        self.facename = ""
                    
        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        Id=0
        Recognizer = cv2.face.LBPHFaceRecognizer_create();
        Recognizer.read('share/facedetector/trainingData.yml')
        confidence = 0.0
        
        color_img = inframe.getCvBGR()
        gray = inframe.getCvGRAY()
        
        
        haar = cv2.CascadeClassifier('share/facedetector/haarcascade_frontalface_alt.xml')
        faces = haar.detectMultiScale(gray, 1.1, 5);
        for (x,y,w,h) in faces:
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0,255, 0), 2)
            Id, confidence = Recognizer.predict(gray[y:y+h,x:x+w])
            
            
        
        if Id == 0:
            self.facename = "Unknown"
        else:
            self.facename = self.facenames[Id-1]
            jevois.sendSerial("Face:"+str(self.facename))
        
        outimg = color_img
        
        
        # Write a title:
        cv2.putText(outimg, str(self.facename), (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        height, width, channels = outimg.shape # if outimg is grayscale, change to: height, width = outimg.shape
        cv2.putText(outimg, str(confidence), (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Convert our BGR output image to video output format and send to host over USB. If your output image is not
        # BGR, you can use sendCvGRAY(), sendCvRGB(), or sendCvRGBA() as appropriate:
        
               
        outframe.sendCvBGR(outimg)
        
        
        