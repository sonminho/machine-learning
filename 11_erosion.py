# cv2.erod(image, kernel, iterations = 1)

import cv2 
import numpy as np 

img = cv2.imread("irin.jpg")
ker = np.ones((5,5), np.uint8)
new_img = cv2.erode(img, ker, iterations = 1)

cv2.imshow("image", new_img)
cv2.waitKey()
cv2.destroyAllWindows()