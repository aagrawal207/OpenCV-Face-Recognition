import cv2                                                                      # openCV
import numpy as np                                                              # for numpy arrays

detector = cv2.CascadeClassifier('./HaarCascade/haarcascade_frontalface_alt_tree.xml')

# cap = cv2.VideoCapture('test_video.mp4')
cap = cv2.VideoCapture(0)                                                       # defining which camera to take input from

rec = cv2.createLBPHFaceRecognizer()                                            # Local Binary Patterns Histograms
rec.load('./recognizer/trainingData.yml')                                       # loading the trained data
id = 0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_PLAIN, 3, 1, 0, 2)        # the font of text on face recognition
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    if(len(faces)!=0):
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x, y), (x+w,y+h), (255, 255, 255), 2)           # Drawing the rectangle on the face
            id, conf = rec.predict(gray[y:y+h, x:x+w])                          # Comparing from the trained data
            if id == 1 or id == 9:
                id = 'Abhishek'
            elif id == 2:
                id = 'Vaibhav'
            elif id == 3:
                id = 'Ayush'
            elif id == 4:
                id = 'Raghav'
            elif id == 5:
                id = 'Dhanush'
            elif id == 6:
                id = 'Sahil'
            elif id == 7:
                id = 'Malhar'
            cv2.cv.PutText(cv2.cv.fromarray(img), str(id), (x, y+h), font, 255) # Writing the name of the face recognized

    cv2.imshow('frame',img)                                                     # Showing each frame on the window
    k = cv2.waitKey(30) & 0xff                                                  # Turn off the recognizer using Esc Key
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
