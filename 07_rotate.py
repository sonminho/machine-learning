import cv2 
import numpy as np 
from PIL import Image

def randImage(path):
    img = cv2.imread(path)
    r,c = img.shape[0:2] 

    M = cv2.getRotationMatrix2D((c/2, r/2), np.random.randint(0, 20, 1, dtype=np.int32), 1)  # center, rotate radian, scale
    new_img = cv2.warpAffine(img, M, (c,r), borderValue=(255,255,255))

    return new_img

def saveImage(fileName):
    img = cv2.imread(fileName)
    cv2.imwrite(fileName+str(".jpg"), img)

nImg = randImage("irin.jpg")

for i in range(10):
    myImg = randImage("irin.jpg")
    saveImage("irin_img/irin"+str(i))

cv2.imshow("image", nImg)
cv2.waitKey()
cv2.destroyAllWindows()