import numpy as np 
import cv2 

# http://blog.naver.com/samsjang/220503082434

def convertHSV(path):
    frame = cv2.imread(path)
    # BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([90, 100, 100])  
    upper_blue = np.array([130, 255, 255])
    
    lower_green = np.array([30, 100, 100])        
    upper_green = np.array([70, 255, 255])
    
    lower_red = np.array([-10, 100, 100])
    upper_red = np.array([10, 255, 255])

    # HSV 이미지에서 청색, 초록색, 빨간색 추출하기 위한 임계값
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # mask와 원본 이미지 비트를 연산함
    res1 = cv2.bitwise_and(frame, frame, mask=mask_blue)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_green)
    res3 = cv2.bitwise_and(frame, frame, mask=mask_red)

    # cv2.imshow('original', frame)
    # cv2.imshow('blue', res1)
    # cv2.imshow('green', res2)
    # cv2.imshow('red', res3)

    return res1, res2, res3
def morph():
    img = cv2.imread('../images/text.png')

    kernel = np.ones((3,3), np.uint8)
    print(kernel)
    print(kernel.shape)
    
    erosion = cv2.erode(img, kernel, iterations = 1)
    dilateion = cv2.dilate(img, kernel, iterations = 1)

    cv2.imshow('origin', img)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilateion', dilateion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def morph2():
    img1 = cv2.imread('../images/a.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../images/B.png', cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow('a', img1)
    cv2.imshow('b', img2)
    kernel = np.ones((5,5), np.uint8)
    print(kernel)
    opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

res1, res2, res3 = convertHSV('../images/traffic_light3.png')
kernel = np.array([[0,1,1,0],[1,1,1,1],[1,1,1,1],[0,1,1,0]], np.uint8)

cv2.imshow("res2", res2)
gray = cv2.imread('../images/traffic_light3.png', cv2.IMREAD_GRAYSCALE)
print(res2[0].shape)
cv2.imshow('gary', gray)

opening = cv2.morphologyEx(res2, cv2.MORPH_OPEN, kernel)

dilateion = cv2.dilate(res2, kernel, iterations = 1)
erosion = cv2.erode(dilateion, kernel, iterations = 1)

cv2.imshow('opening', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()


print(kernel)