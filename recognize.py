import cv2
import numpy as np

detector= cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
cap = cv2.VideoCapture(0)
rec = cv2.createLBPHFaceRecognizer()
rec.load('./recognizer/trainingData.yml')
id = 0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL, 5, 1, 0, 2)
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    if(len(faces)!=0):
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            id, conf = rec.predict(gray[y:y+h, x:x+w])
            if id == 1:
                id = 'Abhishek'
            elif id == 2:
                id = 'Vaibhav'
            elif id == 3:
                id = 'Ayush'
            elif id == 4:
                id = 'Raghav'
            else:
                id = 'Unknown'
            cv2.cv.PutText(cv2.cv.fromarray(img), str(id), (x, y+h), font, 255)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
