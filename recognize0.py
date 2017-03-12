import cv2                                                                      # openCV
import numpy as np                                                              # for numpy arrays
import sqlite3
import dlib
from skimage import io

detector = dlib.get_frontal_face_detector()
# cap = cv2.VideoCapture('test_video.mp4')
cap = cv2.VideoCapture(0)                                                       # defining which camera to take input from

def getProfile(id):
    connect = sqlite3.connect("Face-DataBase")
    cmd = "SELECT * FROM Students WHERE ID=" + str(id)
    cursor = connect.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    connect.close()
    return profile

rec = cv2.createLBPHFaceRecognizer()                                            # Local Binary Patterns Histograms
rec.load('./recognizer/trainingData.yml')                                       # loading the trained data
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_PLAIN, 2, 1, 0, 1)                # the font of text on face recognition
while(True):
    ret, img = cap.read()                                                       # reading the camera input
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                # conveting the camera input into GrayScale
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255), 2)
        if dets == None:
            break
        id, conf = rec.predict(gray[d.top():d.bottom(), d.left():d.right()])    # Comparing from the trained data
        if conf < 150:
            profile = getProfile(id)
            if profile != None:
                cv2.cv.PutText(cv2.cv.fromarray(img),
                                profile[1] + str("(%.2f)" % conf),
                                (d.left(), d.bottom()),
                                font,
                                (0, 0, 0))                                      # Writing the name of the face recognized
        else :
            cv2.cv.PutText(cv2.cv.fromarray(img),
                            "Unknown" + str(conf),
                            (d.left(), d.bottom()),
                            font,
                            255)                                            # Writing the name of the face recognized


    cv2.imshow('frame',img)                                                     # Showing each frame on the window
    k = cv2.waitKey(30) & 0xff                                                  # Turn off the recognizer using Esc Key
    if k == 27:
        break

cap.release()                                                                   # turning the webcam off
cv2.destroyAllWindows()                                                         # Closing all the opened windows
