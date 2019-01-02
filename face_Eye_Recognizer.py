import cv2
import numpy as np

cam = cv2.VideoCapture(0)
faces = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
eyes = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

while True:
    ret, img = cam.read()

    face = faces.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
        eye = eyes.detectMultiScale(img, 1.3, 5)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)

    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
