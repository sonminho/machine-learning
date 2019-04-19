import numpy as np 
import cv2 

# http://blog.naver.com/samsjang/220504633218

def drawing():
    img = np.zeros((512, 512, 3), np.uint8)

    cv2.line(img, (0,0), (255,511), (255,0,0),5)

    cv2.imshow('drawing', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hsv():
    blue = np.uint8([[[255, 0, 0]]])
    green = np.uint8([[[0,255, 0]]])
    red = np.uint8([[[0,0,255]]])

    hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

    print(hsv_blue)
    print(hsv_green)
    print(hsv_red)

def tracking():
    try:
        print('카메라를 구동합니다')
        cap = cv2.VideoCapture('../videos/sekyung.mp4')
    except:
        print('카메라 구동 실패')
        return
    
    while True:
        ret, frame = cap.read()
        #frame = cv2.imread('../images/traffic_light3.png')
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

        cv2.imshow('original', frame)
        cv2.imshow('blue', res1)
        cv2.imshow('green', res2)
        cv2.imshow('red', res3)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllwindows()

def transform():
    img = cv2.imread('../images/joy.jpeg')
    h, w = img.shape[0:2]

    print(h, w)

    # 리사이징할 이미지 원본, dsize를 나타내는 튜플, 배율인자 x와y, 리사이징할 방법 
    img2 = cv2.resize(img, None, fx=0.5, fy=1, interpolation=cv2.INTER_AREA)
    img3 = cv2.resize(img, None, fx=1, fy=0.5, interpolation=cv2.INTER_AREA)
    img4 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    cv2.imshow('joy', img)
    cv2.imshow('joy2', img2)
    cv2.imshow('joy3', img3)
    cv2.imshow('joy4', img4)

    cv2.waitKey(0)    
    cv2.destroyAllwindows()

def img_move():
    img = cv2.imread('../images/joy.jpeg')
    h, w, c = img.shape[0:3]
    print(h, w, c)
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    print(M.shape)
    img2 = cv2.warpAffine(img, M, (w, h))
    cv2.imshow('joy ^^', img)
    cv2.imshow('moved_joy ^^', img2)
    cv2.waitKey(0)    
    cv2.destroyAllwindows()

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

def makeKernel():
    M1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    M2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    M3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))

    print('사각형\n', M1)
    print('원\n', M2)
    print('십자\n', M3)

#hsv()
#tracking()
#transform()
#img_move()
#morph()
#morph2()
makeKernel()