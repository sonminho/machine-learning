import numpy as np
import cv2

cap = cv2.VideoCapture('video1.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)

while(cap.isOpened()):
    ret, frame = cap.read()
    
    # jpg 인코딩
    ret, buffer = cv2.imencode('.jpg', frame)
    
    # 네트워크 전송, 대신 파일로 저장하기
    with open('c:/temp/output.jpg', 'wb') as f:
        f.write(buffer)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()