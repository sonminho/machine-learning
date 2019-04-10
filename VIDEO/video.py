import numpy as np
import cv2

cap = cv2.VideoCapture('video2.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

color = (244, 255, 40)

 # 몸 인식 특징 파일 읽어 들이기 
cascade_file = "cars.xml" 
cascade = cv2.CascadeClassifier(cascade_file) 

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 몸 인식 실행하기 
    car_list = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20))
    
    for car in car_list:
        x,y,w,h = car
        cv2.rectangle(gray, (x,y), (x+w, y+h), color, thickness=2)

    
    cv2.imshow('frame', gray)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()