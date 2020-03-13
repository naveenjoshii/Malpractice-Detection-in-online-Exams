from flask import Flask,render_template,request
import numpy as np
import os
import cv2
from Emotion import Emotion
from Phone import Phone
from flask import send_from_directory
import spacy
import speech_recognition as sr
from moviepy.editor import *
app = Flask(__name__)

VIDEO_FOLDER = "D:\\Main Project\\Main Web-App\\uploads\\videos"
QUESTION_PAPER_FOLDER = "D:\\Main Project\\Main Web-App\\uploads\\question_paper"
RESULT_FOLDER = "D:\\Main Project\\Main Web-App\\results\\"
TMP_PATH="D:\\Main Project\\Main Web-App\\temp\\"
MULTIPLE_FACE_PATH = "D:\\Main Project\\Main Web-App\\Multiple_faces"
AUDIO_PATH = "D:\\Main Project\\Main Web-App\\Audio\\"
detection_config_path = "D:\\Main Project\\Main Web-App\\Models\\detection_config.json"
detection_model_path = "D:\\Main Project\\Main Web-App\\Models\\detection_model-ex-010--loss-0004.931.h5"
emotion_model_path= "D:\\Main Project\\Main Web-App\\Models\\fer2013_mini_XCEPTION.102-0.66.hdf5"
frontal_face_path = "D:\\Main Project\\Main Web-App\\Models\\haarcascade_frontalface_default.xml"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def upload_file():
    res=""
    if request.method == 'GET':
        return render_template('index.html')
    else:
        video = request.files['video']
        question_paper = request.files['text']
        video_full_path = os.path.join(VIDEO_FOLDER, video.filename)
        question_paper_full_path = os.path.join(QUESTION_PAPER_FOLDER, question_paper.filename)
        video.save(video_full_path)
        question_paper.save(question_paper_full_path)
        m,p = multiple_faces_and_phone(video_full_path)
        emotions = emotion(video_full_path)
        path_of_audio = extract_audio_from_video(video_full_path)
        extracted_text = convert_audio_to_text(path_of_audio)
        similarity = get_similarity(question_paper_full_path,extracted_text)
        if m.count(1)>10 and p.count(1)>5 and emotions['fear'] > 2 and similarity > 40:
            res = "True"
        else:
            res = "False"
    return render_template('output.html',result = res)
@app.route('/show')
def show():
    mult=os.listdir(MULTIPLE_FACE_PATH)
    phon=os.listdir(RESULT_FOLDER)
    m=len(mult)
    p=len(phon)
    return render_template('results.html',mult = mult,m=m,phon=phon,p=p)
def multiple_faces_and_phone(video_file):
    detector =  cv2.CascadeClassifier(frontal_face_path)
    cap = cv2.VideoCapture(video_file)
    currentFrame = 0
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(fps)
    print(length)
    mul_faces=[]
    phone_count=[]
    while(length>0):
        mul_count=0
        if((currentFrame % fps == 0)):
            _, img = cap.read()
            tmp_path=TMP_PATH+str(currentFrame)+".png"
            cv2.imwrite(tmp_path,img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=3,minSize=(50, 50))
            if(len(faces)==1):
                pass
            elif len(faces)>1:
                mul_count=1
                for face_coordinates in faces:
                    draw_bounding_box(face_coordinates, img, (0,255,0))
                f = str(currentFrame)+"_"+str(len(faces))+".png"
                cv2.imwrite(MULTIPLE_FACE_PATH+f,img)
            if mul_count == 1:
                mul_faces.append(1)
            else:
                mul_faces.append(0)
            detected = phone(tmp_path)
            if detected:
                phone_count.append(1)
            else:
                phone_count.append(0)
            os.remove(tmp_path)
        currentFrame = currentFrame+1
        length = length-1
    print(mul_faces)
    print(phone_count)
    return mul_faces,phone_count
def emotion(video_path):
    emotion = Emotion(frontal_face_path,emotion_model_path,video_path)
    emotion_labels = emotion.get_labels()
    face_detection = emotion.load_face_detection()
    emotion_classifier = emotion.load_emotion_detection()
    emotion.start(face_detection,emotion_classifier,emotion_labels)
    emotions = emotion.emotion_count()
    return emotions

def phone(img):
    phone = Phone(detection_model_path,detection_config_path,RESULT_FOLDER)
    phone.detect_phone(img)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def extract_audio_from_video(video_full_path):
    video = VideoFileClip(video_full_path)
    audio = video.audio
    path = AUDIO_PATH + video_full_path[video_full_path.rfind("\\")+1:] +"_copy.wav"
    audio = audio.write_audiofile(path)
    return path

def convert_audio_to_text(path):
    query=""
    r = sr.Recognizer()
    fa = sr.AudioFile(path)
    with fa as source:
        at = r.record(source)
    try:
        query = r.recognize_google(at, language='en-in')
    except:
        pass
    print(query)
    return query

def get_similarity(question_paper_full_path,extracted_text):
    nlp = spacy.load('en')
    with open(question_paper_full_path, 'r') as content_file:
        content = content_file.read()
    print(content)
    d1 = nlp(u''+content)
    d2 = nlp(u''+extracted_text)
    similarity = d1.similarity(d2)
    return similarity

@app.route('/Multiple_faces/<filename>')
def send_file(filename):
    return send_from_directory(MULTIPLE_FACE_PATH, filename)
@app.route('/results/<filename>')
def send_file_phone(filename):
    return send_from_directory(RESULT_FOLDER, filename)
if __name__ == '__main__':
    app.run()
    app.debug = True
    app.run(debug=True)
    app.debug = True
