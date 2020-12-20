import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = 'haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
path = r'/Users/trietnguyen/Downloads/IMG_4089.MOV'
# iniciate id counter
id = 0

names = ['Triet', 'Vu', 'Deo bit']

cam = cv2.VideoCapture(path)

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
    )
    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # If confidence is less them 90 ==> "0" : perfect match
        if (confidence < 90):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id = names[id]
            print(id)
            confidence = "  {0}%".format(round(confidence))
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            id = "unknown"
            confidence = "  {0}%".format(round(confidence-30))

        cv2.putText(
            img,
            str(id),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255)
        )
        cv2.putText(
            img,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0)
        )

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 297:
        break

cam.release()
cv2.destroyAllWindows()