from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import csv
class Emotion:
    def __init__(self, face_detection_path, emotion_detection_path,video_path):
        self.face_detection_path = face_detection_path
        self.emotion_detection_path = emotion_detection_path
        self.video_path = video_path
        self.dict = {"angry":0,"disgust":0,"fear":0,"happy":0,"sad":0,"surprise":0,"neutral":0}
    def load_face_detection(self):
        detection_model = cv2.CascadeClassifier(self.face_detection_path)
        return detection_model
    def load_emotion_detection(self):
        emotion_model = load_model(self.emotion_detection_path,compile=False)
        return emotion_model
    def preprocess_input(self,x, v2=True):
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x
    def get_labels(self):
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',4: 'sad', 5: 'surprise', 6: 'neutral'}
    def detect_faces(self,detection_model, gray_image_array):
        return detection_model.detectMultiScale(gray_image_array, 1.3, 5)
    def apply_offsets(self,face_coordinates, offsets):
        x, y, width, height = face_coordinates
        x_off, y_off = offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)
    def addEmotionCount(self,s,dict):
        if(s is "angry"):
            dict["angry"] += 1
        elif(s is "disgust"):
            dict["disgust"] += 1
        elif(s is "fear"):
            dict["fear"] += 1
        elif(s is "happy"):
            dict["happy"] += 1
        elif(s is "sad"):
            dict["sad"] += 1
        elif(s is "surprise"):
            dict["surprise"] += 1
        elif(s is "neutral"):
            dict["neutral"] += 1
    def start(self,face_detection,emotion_classifier,emotion_labels):
        # hyper-parameters for bounding boxes shape
        emotion_offsets = (20, 40)
        # getting input model shapes for inference
        emotion_target_size = emotion_classifier.input_shape[1:3]
        # Playing video from file:
        cap = cv2.VideoCapture(self.video_path)
        currentFrame = 0
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        s=""
        while(length>0):
            if((currentFrame % 30 is 0)):
            # Capture frame-by-frame
                ret, frame = cap.read()
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detect_faces(face_detection, gray_image)
                for face_coordinates in faces:
                    x1, x2, y1, y2 = self.apply_offsets(face_coordinates, emotion_offsets)
                    #print(x1,x2,y1,y2)
                    gray_face = gray_image[y1:y2, x1:x2]
                    try:
                        gray_face = cv2.resize(gray_face, (emotion_target_size))
                    except:
                        continue
                    gray_face = self.preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)
                    emotion_prediction = emotion_classifier.predict(gray_face)
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = emotion_labels[emotion_label_arg]
                    s = emotion_text
                    self.addEmotionCount(s,self.dict)
            currentFrame = currentFrame+1
            length = length-1
    def emotion_count(self):
        return self.dict
