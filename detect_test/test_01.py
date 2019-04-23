import numpy as np
from PIL import Image
from detecter import Detecter
from detecter_image import get_detect_image
import cv2

def convertHSV(frame):
    # BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([11, 100, 100])  
    upper_yellow = np.array([65, 255, 255])
    
    lower_green = np.array([55, 100, 100])        
    upper_green = np.array([165, 255, 255])
    
    lower_red = np.array([-10, 100, 100])
    upper_red = np.array([9, 255, 255])

    # HSV 이미지에서 청색, 초록색, 빨간색 추출하기 위한 임계값
    mask_blue = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # mask와 원본 이미지 비트를 연산함
    res1 = cv2.bitwise_and(frame, frame, mask=mask_blue)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_green)
    res3 = cv2.bitwise_and(frame, frame, mask=mask_red)

    kernel2 = np.array([[0,1,1,0],[1,1,1,1],[1,1,1,1],[0,1,1,0]], np.uint8)
    
    # morphological closing, 영역의 구멍 메우기 
    res1 = cv2.morphologyEx(res1, cv2.MORPH_CLOSE, kernel2, iterations = 7)
    res2 = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel2, iterations = 7)
    res3 = cv2.morphologyEx(res3, cv2.MORPH_CLOSE, kernel2, iterations = 7)
    
    # cv2.imshow('res1', res1)
    # cv2.imshow('res2', res2)
    # cv2.imshow('res3', res3)

    # cv2.imshow('original', frame)
    # cv2.imshow('blue', res1)
    # cv2.imshow('green', res2)
    # cv2.imshow('red', res3)

    return res1, res2, res3

# 테스트 이미지 파일 리스트
TEST_IMAGE_PATHS = [ 'images/traffic_light2.png']
THRESHOLD = 0.5

detecter = Detecter()
detecter.setup('./frozen_inference_graph.pb','./mscoco_label_map.pbtxt')

for image_path in TEST_IMAGE_PATHS:
    img = cv2.imread(image_path)
    #cv2.imshow('origin', img)
    (height, width, _) = img.shape
    image,image_ex = get_detect_image(image_path)
    (boxes, scores, classes, num) = detecter.detect(image_ex)

    for idx, box in enumerate(boxes):
        # 확률이 50% 이상인 신호등
        if classes[idx] == 10 and scores[idx] > 0.5 :
            sy, sx, ey, ex = (int(box[0] *height)), (int(box[1] * width)), (int(box[2] * height)), (int(box[3] * width))
            #print(sy, sx, ey, ex, classes[idx], scores[idx])
            
            # 관심영역(ROI) 검출
            sub_img = img[sy:ey, sx:ex]

            # 그레이스케일링 후 ROI내 원 검출
            gray_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray_img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=70, param2=15, minRadius=40, maxRadius=50)
            
            print(str(idx+1) +"번째 이미지 " + str(circles.shape[1]) + "개의 원 검출")
            
            # HSV 채널로 변환
            r1,r2,r3 = convertHSV(sub_img)
            # cv2.imshow('r1' + str(idx), r1)
            # cv2.imshow('r2' + str(idx), r2)
            # cv2.imshow('r3' + str(idx), r3)

            sub_img = r1+r2+r3

            dic = dict()
            for circle in circles[0,:]:
                cx, cy, r = circle
                x = sub_img[int(cy),int(cx)][0]
                #print(sub_img[int(cy-2): int(cy+3), int(cx-2):int(cx+3)])

                sum = 0
                for a in range(-2, 3, 1):
                    for b in range(-2, 3, 1):
                        #print(sub_img[int(cy+a),int(cx+b)][0])
                        sum = sum + sub_img[int(cy+a),int(cx+b)][0]

                x = sum/25.0
                print(x)

                if x >= 55 and x < 165:
                    cv2.circle(sub_img, (cx,cy), r, (0, 255, 0), 1)
                    dic['green'] = True
                elif x > 0 and x < 11:
                    cv2.circle(sub_img, (cx,cy), r, (0, 0, 255), 1)
                    dic['red'] = True
                elif x > 11 and x < 55:
                    cv2.circle(sub_img, (cx,cy), r, (102, 255, 255), -1)
                    dic['yellow'] = True
            print(str(idx+1) +"번째 이미지 결과 ")
            print(dic)
            print()
            cv2.imshow('sub'+str(idx+1), sub_img)

    detecter.viaulize(img, boxes, classes, scores, THRESHOLD)
    cv2.imshow(image_path, img)

cv2.waitKey()
cv2.destroyAllWindows()
