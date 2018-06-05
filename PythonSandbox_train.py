import libjevois as jevois
import cv2
import numpy as np
import os
import sys
import time
class PythonSandbox:
    # trainer for face recognition
    
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)
        self.frame = 0
        self.msg = "vide"
        self.counter = 1
     
    def detect_face(self, gray_img, scaleFactor = 1.1):
         
         gray = gray_img
         haar = cv2.CascadeClassifier('share/facedetector/haarcascade_frontalface_alt.xml')
         faces = haar.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors = 5);
         for (x,y,w,h) in faces:
             jevois.sendSerial("face detected")
             self.msg ="face for training detected"
             gray_img = gray[y:y+h, x:x+w]
             
         return (gray_img, self.msg)
        
    def train_face(self):
        facenames = []
        facenames=os.listdir('modules/JeVois/PythonSandbox/training-data/')
        
        
        
        
        
        
        
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        #list to hold all subject faces
        faces = []
        #list to hold labels for all subjects
        labels = []
        facename = ""
        for facename in facenames:
            for y in range(1, 13):
                image_path = 'modules/JeVois/PythonSandbox/training-data/'+str(facename)+'/'+str(y)+'.jpg'
                image = cv2.imread(image_path, 0)
                #jevois.sendSerial(image_path)
                face, self.msg = self.detect_face(image)
                #jevois.sendSerial("msg " + self.msg)
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
                if self.msg == "face for training detected":
                #add face to list of faces
                    faces.append(face)
                #add label for this face
                    labels.append(facename)
                    jevois.sendSerial("start storing detected face in label " + str(facename) + " pic " + str(y))
        
        recognizer.train(faces, np.array(labels))
        recognizer.write('share/facedetector/trainingData.yml')
        jevois.sendSerial("yml saved")
        return (faces, labels)            
        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
                
        if self.counter == 1:
            self.train_face()
            self.counter = 2
            jevois.sendSerial("complete load face - counter" + str(self.counter))
            self.msg = "Training done. Ctrl-c to Quit"
        
               
        outimg = inframe.getCvBGR()   
        # Write a title:
        cv2.putText(outimg, self.msg, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),
                    1, cv2.LINE_AA)
        
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        height, width, channels = outimg.shape # if outimg is grayscale, change to: height, width = outimg.shape
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Convert our BGR output image to video output format and send to host over USB. If your output image is not
        # BGR, you can use sendCvGRAY(), sendCvRGB(), or sendCvRGBA() as appropriate:
        
        cv2.putText(outimg, "Face trainer", (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        
        outframe.sendCvBGR(outimg)
        
        
        