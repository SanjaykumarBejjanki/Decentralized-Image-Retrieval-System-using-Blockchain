import cv2
from PIL import Image
import numpy as np

img = Image.open("testImages/1.ppm") #load image from given path
img = np.array(img)
img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("model/FSRCNN_x3.pb") 
sr.setModel("fsrcnn",3)
result = sr.upsample(img)

cv2.imshow("original", img)
cv2.imshow("super", result)
cv2.waitKey(0)
