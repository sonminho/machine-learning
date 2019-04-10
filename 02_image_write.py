import cv2

img = cv2.imread("image.jpg")
cv2.imwrite("save_image.jpg", img)