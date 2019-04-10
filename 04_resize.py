import cv2 

img = cv2.imread("irin.jpg")

r, c = img.shape[:2]

new_img = cv2.resize(img, ( 28, 28 ), interpolation = cv2.INTER_CUBIC)

cv2.imshow("resize", new_img)
cv2.waitKey()
cv2.destroyAllWindows()