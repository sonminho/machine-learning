import cv2 
import numpy as np 

img = cv2.imread("irin.jpg")

# ker = np.ones(3,3)
ker = np.random.rand(3,3)

new_img = cv2.filter2D(img, -1, ker)

cv2.imshow("image", new_img)
cv2.waitKey()
cv2.destroyAllWindows()