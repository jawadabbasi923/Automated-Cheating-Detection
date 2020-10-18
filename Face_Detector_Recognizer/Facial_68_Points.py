# Drawing 68 points on face

import cv2
import numpy as np
import dlib
import imutils

cap = cv2.VideoCapture(0)

#cv2.fac e.LBPHFaceRecognizer_create()
print(dir(cv2.face))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Data/shape_predictor_68_face_landmarks.dat")

while True:
    _ , frame = cap.read()

    frame = imutils.resize(frame, width= 900, height= 600)

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)
        print (landmarks.part(1).x)
        print (landmarks.part(1).y)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
