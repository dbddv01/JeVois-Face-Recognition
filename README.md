# JeVois-Face-Recognition
Custom script for Jevois smart camera implementing basic face recognition
Organize the Jevois SD card with following structure
Replace the existing PythonSandbox.py with this one into the respective jevois module
Within the PythonSandbox directory, create a subdirectory with example of faces. ex. \train-img\Angela\01.jpg
Place the xyz_cascade xml file into a /share/facedetector directory
train the recognizer to get the model via the train.py file. It will create a yaml into the /share/facedetector.
This Sandbox works only with UsB output at the moment.
Whenever a face is detected, it will create boxes into the cam stream and mention the name associated.
Also a msg via serial is sent in format face:<namerecognized>
Takephoto is basic script who will take picture of face detected in front of him, allowing you to record automtically some faces. 
  Still a lot to improve false friend, clean code, and find a way to let it run without USB and finally integrate the device with Snips.ai
  See jevois.org for information related to the device.
