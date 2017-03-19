import cv2                                                                      # openCV
import numpy as np                                                              # for numpy arrays
import sqlite3

detector = cv2.CascadeClassifier('./HaarCascade/haarcascade_frontalface_alt_tree.xml')

# cap = cv2.VideoCapture('video_for_training.mp4')
cap = cv2.VideoCapture(1)

def insertOrUpdate(Id, Name, roll) :                                            # this function is for database
    connect = sqlite3.connect("Face-DataBase")                                  # connecting to the database
    cmd = "SELECT * FROM Students WHERE ID = " + Id                             # selecting the row of an id into consideration
    cursor = connect.execute(cmd)
    isRecordExist = 0
    for row in cursor:                                                          # checking wheather the id exist or not
        isRecordExist = 1
    if isRecordExist == 1:                                                      # updating name and roll no
        connect.execute("UPDATE Students SET Name = ? WHERE ID = ?",(Name, Id))
        connect.execute("UPDATE Students SET Roll = ? WHERE ID = ?",(roll, Id))
    else:
    	params = (Id, Name, roll)                                               # insering a new student data
    	connect.execute("INSERT INTO Students VALUES(?, ?, ?)", params)
    connect.commit()                                                            # commiting into the database
    connect.close()                                                             # closing the connection

Id = raw_input('Enter user id : ')
name = raw_input("Enter student's name : ")
roll = raw_input("Enter student's roll no. : ")
insertOrUpdate(Id, name, roll)                                                  # calling the sqlite3 database
sampleNum = 0
while(True):
    ret, img = cap.read()                                                       # reading the camera input
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                # Converting to GrayScale
    faces = detector.detectMultiScale(
                            gray,
                            scaleFactor=1.2,        # trained xml cannot be changed so the input image is changed with this scale
                            minNeighbors=4,         # less then this number of faces are detected then face is ignored
                            minSize=(25, 25)        # minimum possible object size. Objects smaller than that are ignored
                            )                                                   # Detecting faces
    for (x,y,w,h) in faces:                                                     # loop will run for each face detected
        sampleNum += 1
        cv2.imwrite("./dataset/User."+Id+"."+str(sampleNum)+".jpg",
                    img[y:y+h, x:x+w])                                          # Saving the faces
        cv2.rectangle(img, (x, y),(x+w, y+h),(0,255,0) ,2)                      # Forming the rectangle
        cv2.waitKey(200)                                                        # waiting time of 200 milisecond
    cv2.imshow('frame', img)                                                    # showing the video input from camera on window
    cv2.waitKey(1)
    if(sampleNum >= 20):                                                        # will take 20 faces
        break

cap.release()                                                                   # turning the webcam off
cv2.destroyAllWindows()                                                         # Closing all the opened windows
