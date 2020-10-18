import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path_for_dataset = "DataSet"

def getImagesForTraining(path):

    imagesPaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faces = []
    userIds = []

    for imagePath in imagesPaths:
        faceImage = Image.open(imagePath).convert('L')

        faceNP = np.array(faceImage, 'uint8')
        #cv2.imshow("hda", faceImage)
        #graysa = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)

        id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNP)
        userIds.append(id)

        cv2.imshow("Training "+str(id)+" ", faceNP)

        cv2.waitKey(100)
    return np.array(userIds), faces

user_ids, faces = getImagesForTraining(path_for_dataset)

recognizer.train(faces , user_ids)

recognizer.save("Model/trained.yml")

cv2.destroyAllWindows()
