import numpy as np
from PIL import Image
from detecter import Detecter
from detecter_image import get_detect_image
import cv2
from threading import Thread

def detect1(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_ex = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = detecter.detect(image_ex)

    detecter.viaulize(frame, boxes, classes, scores, THRESHOLD)
    cv2.imshow('people', frame)

def play1():
    print(path)
    
    cap = cv2.VideoCapture(path)

    while(cap.isOpened()):
        ret, frame = cap.read()        
        detect1(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detecter = Detecter()
    detecter.setup('./frozen_inference_graph.pb','./mscoco_label_map.pbtxt')
    
    THRESHOLD = 0.3
    path = "vtest.avi"    
    
    t = Thread(target=play1, args=())
    t.start()