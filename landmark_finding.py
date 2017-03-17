import cv2
import dlib
import numpy
from matplotlib import pyplot as plt

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='./HaarCascade/haarcascade_frontalface_alt_tree.xml'
cascade = cv2.CascadeClassifier(cascade_path)

# #This is using the Dlib Face Detector . Better result more time taking
# def get_landmarks(im):
#     rects = detector(im, 1)
#     rect=rects[0]
#     print type(rect.width())
#     fwd=int(rect.width())
#     if len(rects) == 0:
#         return None,None
#
#     return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]),fwd

def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3,5)
    x,y,w,h = rects[0].astype(long)
    rect=dlib.rectangle(x,y,x+w,y+h)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=2,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

im=cv2.imread('IMG_20170316_233102_01.jpg')
image = annotate_landmarks(im,get_landmarks(im))
image2 = cv2.resize(image, (1777, 1000))
cv2.imshow("Result", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
