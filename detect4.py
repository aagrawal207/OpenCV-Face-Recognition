import cv2
import numpy as np
from matplotlib import pyplot as plt
import dlib

detector = dlib.get_frontal_face_detector()

img = cv2.imread('IMG_20170316_163918.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dets = detector(img, 1)
for i, d in enumerate(dets):
    cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255), 2)
    cv2.imwrite('malhar.jpg', img[d.top():d.bottom(), d.left():d.right()])
    # cv2.circle(img, (x + w/2, y+h/2),w, (255, 0, 0), 2)
    # roi_gray = gray[y:y+h, x:x+w]
    # roi_color = img[y:y+h, x:x+w]
    # eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 2)
    # for(ex, ey, ew, eh) in eyes:
    #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

# title = str(len(faces)) + ' faces found.'
# cv2.imshow(title, img)
# matplotlib do not show the bgr color, instead it shows rbg color
b,g,r = cv2.split(img)
img2 = cv2.merge([r, g, b])
plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y
plt.show()
# when everything is done
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.destroyAllWindows()
