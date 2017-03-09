import dlib
import cv2
from skimage import io
from matplotlib import pyplot as plt

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)

    for i, d in enumerate(dets):
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255), 2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
