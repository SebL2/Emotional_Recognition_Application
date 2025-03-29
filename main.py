import torch
import os
from deepface import DeepFace
import cv2
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml" #can try different models and different benchmarks for this 
)
video_capture = cv2.VideoCapture(0)
file_path = os.path.join(os.getcwd()+"\\temp", "img.jpg")
def detect_face(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40)) #array of coordinates where the face is
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4) 
    return faces
#can i use pytorch to do differnet emotions?



def detect_emotion(face):
    #code here for emotion detection
    
    cv2.imwrite(file_path, face)
    result = DeepFace.analyze(img_path = file_path, actions = ['emotion'],enforce_detection = False, silent = True, detector_backend="opencv")   
    os.remove(file_path)
    return result
while True:
    
    result, video_frame = video_capture.read()  
    if not result:
        break  
    face = detect_face(video_frame) 
    for (x, y, w, h) in face:
        face_image = video_frame[y:y+h, x:x+w]
        
        emotions = detect_emotion(face_image)
        cv2.putText(video_frame, emotions[0]['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('',video_frame) #the application window

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()