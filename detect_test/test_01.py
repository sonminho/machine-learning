import numpy as np
from PIL import Image
from detecter import Detecter
from detecter_image import get_detect_image
import cv2

# 테스트 이미지 파일 리스트
TEST_IMAGE_PATHS = [ 'images/traffic_light3.png']
THRESHOLD = 0.3

detecter = Detecter()
detecter.setup('./frozen_inference_graph.pb','./mscoco_label_map.pbtxt')

for image_path in TEST_IMAGE_PATHS:
    img = cv2.imread(image_path,0)
    
    image,image_ex = get_detect_image(image_path)
    (boxes, scores, classes, num) = detecter.detect(image_ex)
        
    img = cv2.medianBlur(img,5)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50,param2=35,minRadius=30,maxRadius=50)
    circles = np.uint16(np.around(circles))

    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        
    detecter.viaulize(cimg, boxes, classes, scores, THRESHOLD)
    cv2.imshow(image_path, cimg)
cv2.waitKey()
cv2.destroyAllWindows()