import cv2
import dlib
import os
import time
import sqlite3

face_cascade = cv2.CascadeClassifier('./HaarCascade/haarcascade_frontalface_alt_tree.xml')
date = time.strftime("%d.%m.%Y")
path = './pics_taken/' + date
if not os.path.exists(path):
    os.makedirs(path)

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

# make an array of all the students in the database initialied as zero

picNum = 2
img = cv2.imread('IMG_20170316_171158.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                # conveting the camera input into GrayScale
faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=4,
                        minSize=(25, 25)
                        )
# folderName = path + '/pic' + str(picNum)
# if not os.path.exists(folderName):
#     os.makedirs(folderName)
totalConf = 0.0
faceRec = 0
for (x, y, w, h) in faces:
    # picName = str(i + 1) + '.jpg'
    # picFolderName = folderName + '/' + picName
    id, conf = rec.predict(gray[y:y+h, x:x+w])    # Comparing from the trained data
    if conf < 30:
        totalConf += conf
        faceRec += 1
        profile = getProfile(id)
        if profile != None:
            cv2.cv.PutText(cv2.cv.fromarray(img),
                            profile[1] + str("(%.2f)" % conf),
                            (x, y+h),
                            font,
                            (0, 0, 0))                                      # Writing the name of the face recognized
    else :
        cv2.cv.PutText(cv2.cv.fromarray(img),
                        "Unknown" + str(conf),
                        (x, y+h),
                        font,
                        255)                                                # Writing the name of the face recognized
    # cv2.imwrite(picFolderName, img[d.top():d.bottom(), d.left():d.right()])
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
cv2.imshow('frame', img)                                                     # Showing each frame on the window
cv2.imwrite(path + '/pic' + str(picNum) + '.jpg', img)
detectPrint = 'Frame' + str(picNum) + ". %d face detected" % len(faces)
if faceRec != 0:
    print detectPrint + " and ", faceRec, " face recognized with confidence %.2f"%(totalConf / faceRec)
else:
    print detectPrint + " and 0 faces recognized"

cv2.destroyAllWindows()                                                         # Closing all the opened windows
