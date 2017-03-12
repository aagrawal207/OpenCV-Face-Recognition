import cv2
import dlib
import os
import time

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
date = time.strftime("%d.%m.%Y")
path = './pics_taken/' + date
if not os.path.exists(path):
    os.makedirs(path)

secs = 30

picNum = 1
while(True):
    ret, img = cam.read()
    dets = detector(img, 1)
    folderName = path + '/pic' + str(picNum)
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    for i, d in enumerate(dets):
        picName = str(i + 1) + '.jpg'
        picFolderName = folderName + '/' + picName
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255), 2)
        cv2.imwrite(picFolderName, img[d.top():d.bottom(), d.left():d.right()])
    picNum += 1
    # cv2.imshow('frame',img)                                                   # Showing each frame on the window
    k = cv2.waitKey(5000) & 0xff                                                # Turn off the recognizer using Esc Key
    if k == 27:
        break

cam.release()                                                                   # turning the webcam off
cv2.destroyAllWindows()                                                         # Closing all the opened windows
