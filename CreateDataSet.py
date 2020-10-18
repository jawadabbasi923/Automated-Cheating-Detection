import cv2
import dlib
import sqlite3
import requests

face_recognizer = cv2.CascadeClassifier("/Users/jawadabbasi/Documents/FinalProject/venv/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")

def InsertOrUpdateToSqlite(u_Id, UserName1, Gender1):

    conn = sqlite3.connect("/Users/jawadabbasi/Documents/FinalProject/ProjectDataBase.db")

    sqlQuery = "select * from 'UserData' where Id = " + str(u_Id)

    result = conn.execute(sqlQuery)

    isRecordExist = 0

    for row in result:
        isRecordExist = 1

    if isRecordExist == 1:
        sqlQuery = "update UserData set UName = " + str(UserName1) + "and Gender = " + str(Gender1) + "where Id = "+ u_Id

    else:
       sqlQuery = ""

    conn.execute("INSERT INTO UserData VALUES (?, ?, ?)", (u_Id, UserName1, Gender1))

    conn.commit()

    conn.close()

def insertToSQLUsingAPI(a_Arid_No, a_Stu_Name, a_Stu_Gender):
    print ("Api")
    dataToPost = {"Arid_No": a_Arid_No, "Student_Name": a_Stu_Name, "Student_Gender": a_Stu_Gender}
    response = requests.post("http://10.211.55.3/WebApi/api/Detection/AddStudent", data=dataToPost)
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

        cv2.imwrite("/Users/jawadabbasi/Documents/FinalProject/venv/VideoModel_For_Project/DataForProject/UserImage." + str(Id) + "." + str(a) + ".jpg", gray_Scale)

    cv2.imshow("Frames", frame)

    key = cv2.waitKey(100)

    if key == ord('q'):
        cam.release()
        break

print (str(a) + " Pictures are detected for training model")

cv2.destroyAllWindows()



#
# import cv2
# import numpy as np
# from PIL import Image
# import time
# import dlib
# import imutils
# import requests
# import sqlite3
# from datetime import datetime
# import json
#
# # Model For Face Detectiom
# detector = dlib.get_frontal_face_detector()
#
# # Model For Detection Of 68 Points
# predictor = dlib.shape_predictor("/Users/jawadabbasi/Documents/FinalProject/venv/Data/shape_predictor_68_face_landmarks.dat")
#
# # Model For Face Recognizer
# recognizer = cv2.face.LBPHFaceRecognizer_create()
#
# # Self Trained Model For Recognition Of Person
# recognizer.read("/Users/jawadabbasi/Documents/FinalProject/venv/VideoModel_For_Project/trained.yml")
#
# # Function to get User from Database through API
# def getUser(Id):
#     data = requests.get("http://10.211.55.3/WebApi/api/Detection/getCandidate?Stu_Id1="+str(Id))
#     if data.ok:
#         jsonCand = json.loads(data.content)
#         return jsonCand
#
# # Function to draw Box on Face
# def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
#     """Draw a 3D box as annotation of pose"""
#     point_3d = []
#     dist_coeffs = np.zeros((4, 1))
#     rear_size = 1
#     rear_depth = 0
#     point_3d.append((-rear_size, -rear_size, rear_depth))
#     point_3d.append((-rear_size, rear_size, rear_depth))
#     point_3d.append((rear_size, rear_size, rear_depth))
#     point_3d.append((rear_size, -rear_size, rear_depth))
#     point_3d.append((-rear_size, -rear_size, rear_depth))
#
#     front_size = img.shape[1]
#     front_depth = front_size * 2
#     point_3d.append((-front_size, -front_size, front_depth))
#     point_3d.append((-front_size, front_size, front_depth))
#     point_3d.append((front_size, front_size, front_depth))
#     point_3d.append((front_size, -front_size, front_depth))
#     point_3d.append((-front_size, -front_size, front_depth))
#     point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
#
#     # Map to 2d img points
#     (point_2d, _) = cv2.projectPoints(point_3d,
#                                       rotation_vector,
#                                       translation_vector,
#                                       camera_matrix,
#                                       dist_coeffs)
#     point_2d = np.int32(point_2d.reshape(-1, 2))
#
#     # # Draw all the lines
#     # cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
#     k = (point_2d[5] + point_2d[8]) // 2
#     # cv2.line(img, tuple(point_2d[1]), tuple(
#     #    point_2d[6]), color, line_width, cv2.LINE_AA)
#     # cv2.line(img, tuple(point_2d[2]), tuple(
#     #     point_2d[7]), color, line_width, cv2.LINE_AA)
#     # cv2.line(img, tuple(point_2d[3]), tuple(
#     #     point_2d[8]), color, line_width, cv2.LINE_AA)
#
#     return (point_2d[2], k)
#
# # Read Image
#
# # im = cv2.imread("/Users/jawadabbasi/Documents/FinalProject/venv/Data/headPose.jpg");
#
# # face_cascade = cv2.CascadeClassifier("/Users/jawadabbasi/Documents/FinalProject/venv/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")
# #
# # gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# #
# # faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)
# #
# # for x, y, w, h in faces:
# #
# #     im = im[y:y + h, x:x + w]
# #     im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# # #    cv2.imshow("cropped", im)
#
# #Funtion to check User
# def checkUser(gray, idp, facePoints):
#     id, conf = recognizer.predict(
#         gray[facePoints[1]: facePoints[1] + facePoints[3], facePoints[0]: facePoints[0] + facePoints[2]])
#
#     print ("Model User Id " + str(id))
#     print ("Image User Id " + str(idp))
#
#     if id == idp:
#         return True
#     else:
#         return False
#
# # 3D model points.
# model_points = np.array([
#
#     (0.0, 0.0, 0.0),  # Nose tip
#
#     (0.0, -330.0, -65.0),  # Chin
#
#     (-225.0, 170.0, -135.0),  # Left eye left corner
#
#     (225.0, 170.0, -135.0),  # Right eye right corne
#
#     (-150.0, -150.0, -125.0),  # Left Mouth corner
#
#     (150.0, -150.0, -125.0)  # Right mouth corner
#
# ])
#
# #Function to detect Cheater and save Offender to databse.
# def Take_Paper(student_ID, endH, endM):
#     cap = cv2.VideoCapture(0)
#     timeer1 = 0
#     while True:
#
#         now = datetime.now()
#
#         current_time = now.strftime("%H:%M:%S")
#         cu_hour = int(current_time.split(':')[0])
#         cu_mint = int(current_time.split(':')[1])
#
#         if cu_hour == endH and cu_mint == endM:
#             break
#
#         _, im = cap.read()
#
#         # im = imutils.resize(im, width=900, height=600)
#
#         im = cv2.flip(im, 1)
#
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#
#         size = im.shape
#
#         faces = detector(gray)
#
#         if len(faces) != 0:
#             print ("Face Found.")
#             for face in faces:
#                 x = face.left()
#                 y = face.top()
#                 w = face.right() - x
#                 h = face.bottom() - y
#                 # cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
#
#                 facePoints = np.array([x, y, w, h])
#
#                 if checkUser(gray, student_ID, facePoints):
#                     print ("User is known.")
#
#                     landmarks = predictor(gray, face)
#                     # print (landmarks.part(1).x)
#                     # print (landmarks.part(1).y)
#
#                     # for n in range(0, 68):
#                     #     x = landmarks.part(n).x
#                     #     y = landmarks.part(n).y
#                     #     cv2.circle(im, (x, y), 4, (255, 0, 0), -1)
#
#                     # 2D image points. If you change the image, you need to change vector
#                     image_points = np.array([
#                         (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
#
#                         (landmarks.part(8).x, landmarks.part(8).y),  # Chin
#
#                         (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
#
#                         (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corne
#
#                         (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
#
#                         (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
#
#                     ], dtype="double")
#
#                     # Camera internals
#
#                     focal_length = size[1]
#
#                     center = (size[1] / 2, size[0] / 2)
#
#                     camera_matrix = np.array(
#                         [[focal_length, 0, center[0]],
#
#                          [0, focal_length, center[1]],
#
#                          [0, 0, 1]], dtype="double"
#                     )
#
#                     # print ("Camera Matrix :\n {0}".format(camera_matrix))
#
#                     dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
#
#                     (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
#                                                                                   camera_matrix,
#                                                                                   dist_coeffs,
#                                                                                   flags=cv2.SOLVEPNP_ITERATIVE)
#
#                     # print ("Rotation Vector:\n {0}".format(rotation_vector))
#
#                     # print ("Translation Vector:\n {0}".format(translation_vector))
#
#                     # Project a 3D point (0, 0, 1000.0) onto the image plane.
#
#                     # We use this to draw a line sticking out of the nose
#
#                     (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
#                                                                      translation_vector,
#                                                                      camera_matrix, dist_coeffs)
#
#                     for p in image_points:
#                         cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
#
#                     # print (image_points[0][0])
#                     # print (image_points[0][1])
#                     p1 = (int(image_points[0][0]), int(image_points[0][1]))
#                     p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
#                     x1, x2 = draw_annotation_box(im, rotation_vector, translation_vector, camera_matrix)
#
#                     cv2.line(im, p1, p2, (255, 0, 0), 2)
#                     # cv2.line(im, tuple(x1), tuple(x2), (255, 255, 0), 2)
#
#                     m = p2[0] - p1[0]
#
#                     if m > 350 or m < -350:
#                         timeer1 += 1
#                         if timeer1 < 3:
#                             cv2.putText(im, "Warning", (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 255, 0), 2)
#                             print ("Warning")
#                         if timeer1 > 3:
#                             dataToPost = {"Stu_Id": student_ID,"Reason":"Looking left/right"}
#                             response = requests.post("http://10.211.55.3/WebApi/api/Detection/AddCheater",data=dataToPost)
#                             print (response.status_code, response.content)
#                     else:
#                         timeer1 = 0
#
#                     # Display image
#                     cv2.imshow("Output", im)
#                 else:
#                     print ("User Unknown")
#                     dataToPost = {"Stu_Id": student_ID,"Reason":"Student with specified Id not found"}
#                     response = requests.post("http://10.211.55.3/WebApi/api/Detection/AddCheater", data=dataToPost)
#                     print (response.status_code, response.content)
#         else:
#             dataToPost = {"Stu_Id": student_ID, "Reason": "No Student/Face Found"}
#             response = requests.post("http://10.211.55.3/WebApi/api/Detection/AddCheater", data=dataToPost)
#             print (response.status_code, response.content)
#             print ("No Student Found")
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#         time.sleep(1.0)
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# #TakePaper(101, 22, 01)
#
# # Function to check Timetable in database after every minute and call funtion Take_Paper
# def Check_TimeTable_Start_Paper():
#     while True:
#         now = datetime.now()
#
#         current_time = now.strftime("%H:%M:%S")
#
#         cu_secs = int(current_time.split(':')[2])
#         cu_hour = int(current_time.split(':')[0])
#         cu_mint = int(current_time.split(':')[1])
#
#         print (str(cu_secs))
#
#         if cu_secs == 0:
#             # Search all record in database from table 'TimeTable', if startTime of paper in now at current hour and minute
#             data = requests.get("http://10.211.55.3/WebApi/api/Detection/GetDataOfTimeTable")
#             if data.ok:
#                 jsonData = json.loads(data.content)
#                 if len(jsonData) > 0:
#                     for jdata in jsonData:
#                         print (jdata)
#                         s_ID = int(jdata["Stu_Id"])
#                         d_Hour = int(jdata["Start_Time"].split(':')[0])
#                         d_Mint = int(jdata["Start_Time"].split(':')[1])
#                         d_EndHour = int(jdata["End_Time"].split(':')[0])
#                         d_EndMint = int(jdata["End_Time"].split(':')[1])
#
#                         print (current_time)
#                         print (cu_hour, d_Hour)
#                         print (cu_mint, d_Mint)
#                         if cu_hour == d_Hour and cu_mint == d_Mint:
#                             # Get Id of student from database
#                             # Pass that ID and Endtime of paper to doDetectionCheating Function
#                             print ("Start Paper")
#                             Take_Paper(s_ID, d_EndHour, d_EndMint)
#                             print ("Finish Paper")
#
#                             cv2.destroyAllWindows()
#
#         time.sleep(1.0)
#
#         # print("Current Time =", current_time)
#
# Check_TimeTable_Start_Paper()

