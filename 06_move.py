import cv2
import numpy as np 

img = cv2.imread("irin.jpg")
r,c = img.shape[0:2]

M = np.float32([[1, 0, 100], [0, 1, 100]])

new_img = cv2.warpAffine(img, M, (c, r))
cv2.imshow("image", new_img)
cv2.waitKey()
cv2.destroyAllWindows()