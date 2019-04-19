import cv2
import requests

url = 'http://localhost:8080/monitor/camera/1'


while(True):
    cap = cv2.VideoCapture('vtest.avi')

    try :
        while(cap.isOpened()):
            ret, frame = cap.read()
            ret, jpgImage = cv2.imencode('.jpg', frame)
            
            file = { 'image' : jpgImage }    
            res = requests.post(url, files=file)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except:
        cap.release()

cap.release()
cv2.destroyAllWindows()