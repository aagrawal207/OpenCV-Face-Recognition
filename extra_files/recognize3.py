import cv2
import dlib
import os
import time
import sqlite3

cam = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()

picNum = 1
img = cv2.imread('test1.jpg')
dets = detector(img, 1)
for i, d in enumerate(dets):
    cv2.imwrite('./Croped_faces/face' + str(picNum) + '.jpg', img[d.top():d.bottom(), d.left():d.right()])
    picNum += 1
