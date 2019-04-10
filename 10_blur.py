import cv2 

img = cv2.imread("irin.jpg")

new_img = cv2.GaussianBlur(img, (5,5), 0)
#new_img = cv2.medianBlur(img, (5,5), 0)
#new_img = cv2.blur(img, (5,5))

cv2.imshow("image",new_img)
cv2.waitKey()
cv2.destroyAllWindows()