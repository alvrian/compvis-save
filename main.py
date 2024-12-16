import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

face_list = []
class_list = []

### Train ###
train_path = './dataset/'
person_name = os.listdir(train_path)

for idx, name in enumerate(person_name):
    folder_path = train_path + name
    
    for img_name in os.listdir(folder_path):
        img_path = train_path + name + '/' + img_name
        img_gray = cv2.imread(img_path, 0)
        
        # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
        # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
        detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

        if len(detected_face) < 1:
            continue
        
        for face_rect in detected_face:
            x, y, w, h = face_rect
            face_img = img_gray[y:y+h, x:x+w]
            face_list.append(face_img)
            class_list.append(idx)

# pip install opencv-contrib-python
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))


### Test ###
test_path = './dataset/test/'

for img_name in os.listdir(test_path):
    img_path = test_path +  img_name
    img_gray = cv2.imread(img_path, 0)
    img_bgr = cv2.imread(img_path)
    
    # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
    # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
    detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

    if len(detected_face) < 1:
        continue
    
    for face_rect in detected_face:
        x, y, w, h = face_rect
        face_img = img_gray[y:y+h, x:x+w]
        
        res, confidence = face_recognizer.predict(face_img)
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 1)
        text = person_name[res] + ": " + str(confidence)
        cv2.putText(img_bgr, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('Result', img_bgr)
        cv2.waitKey(0)
        



# ### haarcascade_frontalface_default
# img_path = train_path + 'Gus Fring' + '/' + '1.jpg'
# img = cv2.imread(img_path)

# # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
# # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
# # factors = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
# # for n in factors:
# detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)

# for face_rect in detected_face:
#     x, y, w, h = face_rect
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
#     cv2.imshow('Figure', img)
#     cv2.waitKey(0)

