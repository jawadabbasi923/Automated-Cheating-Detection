import cv2

img = cv2.imread("/Users/jawadabbasi/Documents/FinalProject/venv/Data/JawadPic.jpeg",1)

#print ( type(img))

#print (img.shape)

#print (img)

# cv2.imshow("Dash", img)
#
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# QT platform
# pip3 install --upgrade opencv-python==4.1.2.30
# https://www.youtube.com/watch?v=-ZrDjwXZGxI

face_cascade = cv2.CascadeClassifier("/Users/jawadabbasi/Documents/FinalProject/venv/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)

# print (type(faces))
#
# print (faces)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

cv2.imshow("Gray", img)

cv2.waitKey(0)

cv2.destroyAllWindows()