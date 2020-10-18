# Face detector in a video by using your webcame.
# If you want to save images to some directory, uncomment "ImageSaver" 

import cv2, time
import os

def ImageSaver ( image_path, img, no ):

    os.chdir(image_path)

    filename = "Image" + str(no) + ".jpg"

    cv2.imwrite(filename, img)

    print('Successfully saved')

video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")

a = 1

while True:
    a = a + 1

    check, frame = video.read()

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Capturing", frame)

#    ImageSaver("/Users/abc/Documents", frame, a)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

print (a)

video.release()

cv2.destroyAllWindows()
