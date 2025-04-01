import torch
import os
from deepface import DeepFace
from PIL import Image
import torchvision.transforms as transforms
from network import EmotionNetwork
import cv2
model = EmotionNetwork()
model.load_state_dict(torch.load("./model/trained.pth"));
video_capture = cv2.VideoCapture(0)
file_path = os.path.join(os.getcwd()+"\\temp", "img.jpg")
def detect_face(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40)) #array of coordinates where the face is
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4) 
    return faces
def detect_emotion_deepFace(face):
    cv2.imwrite(file_path, face)    
    result = DeepFace.analyze(img_path = file_path, actions = ['emotion'],enforce_detection = False, silent = True, detector_backend="opencv")   
    os.remove(file_path)
    return result

def detect_emotion_NW(model,face,file_path):
    #code here for emotion detection
    emotion_chart = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    cv2.imwrite(file_path, face)    
    image = Image.open(file_path).convert("L")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((48,48))
    ])
    tensor = transform(image)
    tensor = torch.unsqueeze(tensor,0)
    result = model(tensor)
    print(result)
    predicted_emotion =  emotion_chart[torch.argmax(result,1)]
    os.remove(file_path)
    return predicted_emotion

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml" #can try different models and different benchmarks for this 
)

while True:
    result, video_frame = video_capture.read()  
    if not result:
        break  
    face = detect_face(video_frame) 
    for (x, y, w, h) in face:
        face_image = video_frame[y:y+h, x:x+w]
        
        emotion = detect_emotion_NW(model,face_image,file_path)
        # emotion = detect_emotion_deepFace(face_image)
        # cv2.putText(video_frame, emotion[0]['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(video_frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('',video_frame) #the application window

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()