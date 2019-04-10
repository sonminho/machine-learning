import cv2 

img = cv2.imread("irin.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray_image.jpg", gray)
cv2.imshow("image", gray)

cv2.waitKey()
cv2.destroyAllWindows()