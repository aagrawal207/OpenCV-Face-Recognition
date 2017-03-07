import cv2
import numpy as np
import sqlite3

detector = cv2.CascadeClassifier('./HaarCascade/haarcascade_frontalface_alt_tree.xml')
# cap = cv2.VideoCapture('video_for_training.mp4')
cap = cv2.VideoCapture(0)

def insertOrUpdate(Id, Name) :
    connect = sqlite3.connect("Face-DataBase")
    cmd = "SELECT * FROM Students WHERE ID=" + Id
    cursor = connect.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if isRecordExist == 1:
        cmd = "UPDATE Students SET Name = " + Name + " WHERE ID = " + Id
    else:
        cmd = "INSERT INTO People(ID,Name) Values(" + Id + "," + Name + ")"

id = raw_input('Enter user id : ')
# id = raw_input('Enter user name : ')
sampleNum = 0
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                # Converting to GrayScale
    faces = detector.detectMultiScale(
                            gray,
                            scaleFactor=1.2,
                            minNeighbors=4,
                            minSize=(25, 25)
                            )                                                   # Detecting faces
    for (x,y,w,h) in faces:
        sampleNum += 1
        cv2.imwrite("./dataset/User."+id+"."+str(sampleNum)+".jpg",
                    img[y:y+h, x:x+w])                                          # Saving the faces
        cv2.rectangle(img, (x, y),(x+w, y+h),(0,255,0) ,2)                      # Forming the rectangle
        cv2.waitKey(200)
    cv2.imshow('frame', img)
    cv2.waitKey(1)
    if(sampleNum >= 20):
        break

cap.release()
cv2.destroyAllWindows()
