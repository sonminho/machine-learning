import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

def get_face_crop(face, area) :
    x, y, w, h = area
    ex, ey = abs(x - w), abs(y - h)

    if ex > ey : 
        gap = int(abs(y-h)/2)
        y -= gap
        h += gap
    else:
        gap = int(abs(w-h)/2)
        x -= gap
        w += gap

    print(x,y,w,h)
    return face

def get_face(img, area):
    face = get_face_crop(img, area)
    face = cv2.resize(face, (28, 28), interpolation = cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    return face

if __name__ == "__main__":
    img = cv2.imread('irin.jpg')
    face_area = [430, 450, 250, 400]

    face = get_face(img, face_area)
    cv2.imshow("face", face)

    cv2.waitKey()
    cv2.destroyAllWindows()