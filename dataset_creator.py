import cv2
import numpy as np

detector= cv2.CascadeClassifier('./HaarCascade/haarcascade_frontalface_alt_tree.xml')
#cap = cv2.VideoCapture('video_for_training.mp4')
cap = cv2.VideoCapture(0)

id = raw_input('Enter user id : ')
id = raw_input('Enter user name : ')
sampleNum = 0
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        sampleNum += 1
        cv2.imwrite("./dataset/User."+id+"."+str(sampleNum)+".jpg", img[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)
    cv2.imshow('frame',img)
    cv2.waitKey(1)
    if(sampleNum >= 20):
        break

cap.release()
cv2.destroyAllWindows()
