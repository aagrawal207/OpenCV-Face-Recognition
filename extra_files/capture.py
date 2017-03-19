import cv2
from matplotlib import pyplot as plt

cam = cv2.VideoCapture(1)
s, img = cam.read()


plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y
plt.show()

cv2.imwrite("test.jpg", img)
