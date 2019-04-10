import numpy as np 
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("irin.jpg")
print(img.dtype)
r, c, n = img.shape

print(r, c, n)
b_img = np.zeros((r,c,n), np.uint8)
g_img = np.zeros((r,c,n), np.uint8)
r_img = np.zeros((r,c,n), np.uint8)

b_img[:,:,0] = img[:,:,0]
g_img[:,:,1] = img[:,:,1]
r_img[:,:,2] = img[:,:,2]

# cv2.imshow("b_image",b_img)
# cv2.imshow("g_image",g_img)
# cv2.imshow("r_image",r_img)

# cv2.waitKey() 
# cv2.destroyAllWindows()

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()