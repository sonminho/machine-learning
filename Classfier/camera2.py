import requests
import io
import cv2 

url = 'http://localhost:8080/monitor/camera/2'

stream = io.BytesIO()

while(True):
    cap = cv2.VideoCapture('video2.avi')
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)

    try:
        while(cap.isOpened()):
            ret, frame = cap.read()
            ret, jpgImage = cv2.imencode('.jpg', frame)

            stream.write(jpgImage.tobytes())
            stream.seek(0)
            file = {'image':stream}
            res = requests.post(url, files=file)
            stream.seek(0)
            stream.truncate()

            if cv2.waitKey(delay) & 0xFF == ord('q') : 
                break
    except Exception as e:
        print(e)
        cap.release()

cv2.destroyAllWindows()