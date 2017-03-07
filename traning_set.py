import os                                                                       # import for taking the imagePaths
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()                                     #Local Binary Patterns Histograms
# This is the common interface to train all of the available cv::FaceRecognizer implementations
path = './dataset'                                                              # Folder where faces are saved

def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]              # Joining './dataset' and '<image names>'
    faces = []                                                                  # Empty array for faces
    Ids = []                                                                    # Empty array for Person Ids
    for imagePath in imagePaths:
        # Converting colored and GrayScale images into bilevel images using Floyd-Steinberg dither
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')                                     # Converting face array into numpy array
        ID = int(os.path.split(imagePath)[-1].split('.')[1])                    # Check this again
        faces.append(faceNp)
        Ids.append(ID)
        cv2.imshow("Training", faceNp)                                          # Showing the faces which are getting trained
        cv2.waitKey(10)                                                         # Waiting time id 10 milisecond
    return Ids, faces

Ids, faces = getImagesWithID(path)                                              # Calling the function
recognizer.train(faces, np.array(Ids))                                          # Training the faces
recognizer.save('./recognizer/trainingData.yml')                                # Saving the yml file
cv2.destroyAllWindows()                                                         # Closing all the opened windows
