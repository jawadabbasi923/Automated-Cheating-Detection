import cv2
import dlib
import sqlite3
import requests

face_recognizer = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#If you are using SQLite :

# def InsertOrUpdateToSqlite(u_Id, UserName1, Gender1):

#     conn = sqlite3.connect("ProjectDataBase.db")

#     sqlQuery = "select * from 'UserData' where Id = " + str(u_Id)

#     result = conn.execute(sqlQuery)

#     isRecordExist = 0

#     for row in result:
#         isRecordExist = 1

#     if isRecordExist == 1:
#         sqlQuery = "update UserData set UName = " + str(UserName1) + "and Gender = " + str(Gender1) + "where Id = "+ u_Id

#     else:
#        sqlQuery = ""

#     conn.execute("INSERT INTO UserData VALUES (?, ?, ?)", (u_Id, UserName1, Gender1))

#     conn.commit()

#     conn.close()

# For SQL Server

def insertToSQLUsingAPI(a_Arid_No, a_Stu_Name, a_Stu_Gender):
    print ("Api")
    dataToPost = {"Arid_No": a_Arid_No, "Student_Name": a_Stu_Name, "Student_Gender": a_Stu_Gender}
    response = requests.post("http://IP_Adress/API_Project_Name/api/Detection/AddStudent", data=dataToPost)
    print (response.status_code, response.content)

Id = input("Enter Id: ")

UName = input("Enter Name: ")

Gender = input("Enter Gender: ")

insertToSQLUsingAPI(Id, UName, Gender)

#InsertOrUpdateToSqlite(Id,UName,Gender)

Id = Id.split('-')[2]

print (Id)

cam = cv2.VideoCapture(0)

a = 0

while True:

    ret, frame = cam.read()

    frame = cv2.flip(frame, 1)

    gray_Scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = face_recognizer.detectMultiScale(gray_Scale, 1.3, 5)

    # for rect in rects:
    #     x1 = rect.left()
    #     y1 = rect.top()
    #     x2 = rect.right()
    #     y2 = rect.bottom()
    for (x,y,w,h) in rects:

        a += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv2.imwrite("DataSet/UserImage." + str(Id) + "." + str(a) + ".jpg", gray_Scale)

    cv2.imshow("Frames", frame)

    key = cv2.waitKey(100)

    if key == ord('q'):
        cam.release()
        break

print (str(a) + " Pictures are detected for training model")

cv2.destroyAllWindows()
