import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
path = 'dataset'

def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        ID = os.path.split(imagePath)[-1].split('.')[1]
        faces.append(faceNp)
        Ids.append(ID)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)
    return Ids, faces

Ids, faces = getImagesWithID(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()
