import os                                                                       # import for taking the imagePaths
import cv2                                                                      # openCV
import numpy as np                                                              # for numpy arrays
from PIL import Image                                                           # for Image.open(imagePath).convert('L')

PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
recognizer = cv2.createLBPHFaceRecognizer()                                     # Local Binary Patterns Histograms
path = './dataset'                                                              # Folder where faces are saved

def get_landmarks(im):
    rects = detector(im, 1)
    rect=rects[0]
    print type(rect.width())
    fwd=int(rect.width())
    if len(rects) == 0:
        return None,None
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]),fwd

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def align(image, eye_left, eye_right):


def getImagesWithID(path):
    imageFolders = [os.path.join(path, f) for f in os.listdir(path)]    # Joining './dataset' and '<image names>'
    faces = []                                                                  # Empty array for faces
    Ids = []
    for imageFolder in imageFolders:
        imagePaths = [os.path.join(imageFolder, f) for f in os.listdir(imageFolder)]
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')        # Converting colored and GrayScale images into bilevel images using Floyd-Steinberg dither
            faceNp = np.array(faceImg, 'uint8')                                     # Converting face array into numpy array
            ID = int(os.path.split(imagePath)[-1].split('.')[1])                    # Check this again
            faces.append(faceNp)                                                    # adding the dilevel face into faces array
            Ids.append(ID)                                                          # index of ID and faceNp is same in both arrays
            cv2.imshow("Training", faceNp)                                          # Showing the faces which are getting trained
            cv2.waitKey(10)
                                                                        # Empty array for Person Ids
                                                             # Waiting time id 10 milisecond
    return Ids, faces

Ids, faces = getImagesWithID(path)                                              # Calling the function
recognizer.train(faces, np.array(Ids))                                          # Training the faces
recognizer.save('./recognizer/trainingData.yml')                                # Saving the yml file
cv2.destroyAllWindows()                                                         # Closing all the opened windows
