import cv2
import dlib
import sqlite3
import numpy as np
import requests
import json

#face_recognizer = dlib.get_frontal_face_detector()
face_recognizer = cv2.CascadeClassifier("/Users/jawadabbasi/Documents/FinalProject/venv/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()

cam = cv2.VideoCapture(0)

recognizer.read("/Users/jawadabbasi/Documents/FinalProject/venv/VideoModel_For_Project/trained.yml")

def getUser(Id):

    conn = sqlite3.connect("/Users/jawadabbasi/Documents/FinalProject/ProjectDataBase.db")

    sqlQuery = "select * from UserData where Id = " + str(Id)

    result = conn.execute(sqlQuery)

    user = None

    for row in result:
        user = row

    conn.commit()

    conn.close()

    return user

def getStudentFromSQL(Arid_No1):
    response = requests.get("http://10.211.55.3/WebApi/api/Detection/getStudentFromSQL?Arid_No="+Arid_No1)

    student = []
    if response.ok:
        jData = json.loads(response.content)

        student = [jData["Arid_No"], jData["Student_Name"], jData["Student_Gender"]]

    return student


# student = getStudentFromSQL("2186")
# print (student)

id = ""

font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    ret, frame = cam.read()

    frame = cv2.flip(frame, 1)

    gray_Scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = face_recognizer.detectMultiScale(gray_Scale, 1.5, 6)

    # for rect in rects:
    #     x1 = rect.left()
    #     y1 = rect.top()
    #     x2 = rect.right()
    #     y2 = rect.bottom()

    for (x,y,w,h) in rects:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv2.rectangle(frame, (x, y), (x + w, y + h + 30), (255, 0, 0), 3)

        id, conf = recognizer.predict(gray_Scale[y: y + h, x: x + w])

        student = getStudentFromSQL(str(id))

        if student != None:
            cv2.putText(frame, str(student[1]), (x, y + h), font, 0.90, (0,255,0), 2)
            #cv2.putText(frame, str(student[2]), (x , y + h + 30), font, 0.90, (0, 255, 0), 2)

    cv2.imshow("Frames", frame)

    if cv2.waitKey(100) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

