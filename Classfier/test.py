import os
import cv2
import sys

basedir = "./img/irin/"

cascade_file = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

list = os.listdir(basedir)
color = (0, 0, 255)
print(len(list))
def crop(face):
    (x,y,w,h) = face

    # 얼굴 부분 자르기 
    face_img = face[y:y+h, x:x+w] 
    
    # 자른 이미지를 지정한 배율로 축소하기
    face_img = cv2.resize(face_img, (28, 28)) 

    # 원래 이미지에 붙이기        
    face[:28, :28] = face_img
    
    return face

for file in list:
    img = cv2.imread(basedir+file)
    
    print(file)

    # 얼굴 인식 실행하기 
    face_list = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(150,150))

    if len(face_list) == 1:
        for (x, y, w, h) in face_list:            
            # 얼굴 부분 자르기 
            face_img = img[y:y+h, x:x+w]

            # 자른 이미지를 지정한 배율로 축소하기
            face_img = cv2.resize(face_img, (128, 128)) 
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            #cv2.rectangle(image_gs, (x,y), (x+w, y+h), color, thickness=8)
        #cv2.imshow("image", face_img)
        cv2.imwrite('./img/after/1/' + file, face_img)

    #cv2.waitKey()
    #cv2.destroyAllWindows()