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
import glob
from csv import writer
app = Flask(__name__)

VIDEO_FOLDER = "D:\\Main Project\\Main Web-App\\uploads\\videos"
QUESTION_PAPER_FOLDER = "D:\\Main Project\\Main Web-App\\uploads\\question_paper"
RESULT_FOLDER = "D:\\Main Project\\Main Web-App\\results\\"
TMP_PATH="D:\\Main Project\\Main Web-App\\temp\\"
MULTIPLE_FACE_PATH = "D:\\Main Project\\Main Web-App\\Multiple_faces\\"
AUDIO_PATH = "D:\\Main Project\\Main Web-App\\Audio\\"
detection_config_path = "D:\\Main Project\\Main Web-App\\Models\\detection_config.json"
detection_model_path = "D:\\Main Project\\Main Web-App\\Models\\detection_model-ex-010--loss-0004.931.h5"
emotion_model_path= "D:\\Main Project\\Main Web-App\\Models\\fer2013_mini_XCEPTION.102-0.66.hdf5"
frontal_face_path = "D:\\Main Project\\Main Web-App\\Models\\haarcascade_frontalface_default.xml"
dataset_path = 'D:\\Main Project\\Main Web-App\\dataset\\dataset.csv'
@app.route('/')
def index():
    folders=[TMP_PATH,MULTIPLE_FACE_PATH,RESULT_FOLDER,VIDEO_FOLDER,QUESTION_PAPER_FOLDER]
    for i in folders:
        files = glob.glob(i+"\\*")
        for f in files:
            os.remove(f)
    return render_template('index.html')

def calculateProbability(listname,weightage):
    return (float(listname.count(1))/len(listname))*weightage
def create_dataset(d):
    with open(dataset_path, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(d)
    print("successfully added to the dataset thanks for your contribution")

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
        extracted_text = convert_audio_to_text("D:\\Main Project\\Main Web-App\\Audio\\audio.wav")
        similarity = get_similarity(question_paper_full_path,extracted_text)
        totalweigh = 0
        data={}
        emotion_freq = emotions.values()
        data['Multiple_faces'] = calculateProbability(m,0.2)
        data['phone']  = calculateProbability(p,0.5)
        data['fear'] = ((float(emotions['fear']))/sum(emotion_freq))*0.15
        data['similarity']  =(similarity/100)*0.15
        data_list=data.values()
        totalweigh = data['Multiple_faces']+data['phone'] +data['fear']+data['similarity']
    return render_template('output.html',result = totalweigh,emotions = emotion_freq,data=data_list)
@app.route('/show/<emotions>/<data_without_op>')
def show(emotions,data_without_op):
    print(emotions)
    #################
    print(data_without_op)
    emotions = list(map(int,emotions[1:len(emotions)-1].split(",")))
    data_without_op = list(data_without_op[1:len(data_without_op)-1].split(","))
    mult=os.listdir(MULTIPLE_FACE_PATH)
    phon=os.listdir(RESULT_FOLDER)
    m=len(mult)
    p=len(phon)
    return render_template('results.html',mult = mult,m=m,phon=phon,p=p,emotions=emotions,data_without_op=data_without_op)
@app.route('/Multiple_faces/<filename>')
def send_file(filename):
    return send_from_directory(MULTIPLE_FACE_PATH, filename)

@app.route('/results/<filename>')
def send_file_phone(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/add_output_data/<data_op>/<data_without_op>')
def add_output_data(data_op,data_without_op):
    data = list(data_without_op[1:len(data_without_op)-1].split(","))
    data.append(data_op)
    create_dataset(data)
    print("Added to data set thanks for contributing")
    return render_template('index.html')


@app.route('/showallmultiple')
def showallmultiple():
    mult=os.listdir(MULTIPLE_FACE_PATH)
    m=len(mult)
    return render_template("thumnails.html",mult=mult,m=m)
@app.route('/showallphones')
def showallphones():
    phon=os.listdir(RESULT_FOLDER)
    p=len(phon)
    return render_template("thumnails.html",phon=phon,p=p)
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
        if((currentFrame % 10 == 0)):
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
            print("Phone detected ",detected)
            phone_count.append(detected)
            os.remove(tmp_path)
            print(mul_faces)
            print(phone_count)
        currentFrame = currentFrame+1
        length = length-1
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
    detected = phone.detect_phone(img)
    return detected

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



if __name__ == '__main__':
    app.run()
    app.debug = True
    app.run(debug=True)
    app.debug = True
