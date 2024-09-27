import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential # type: ignore
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D # type: ignore
import datetime
from threading import Thread
import pandas as pd
from tensorflow.keras.models import load_model  # Make sure this line is in the code


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor = 0.6

# Modify your existing model to match the architecture in the saved weights file

# Load the pre-trained weights
emotion_model = load_model('emotion_detection_model.keras')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
music_dist = {0: "songs/angry.csv", 1: "songs/disgusted.csv", 2: "songs/fearful.csv",3: "songs/happy.csv", 4: "songs/neutral.csv", 5: "songs/sad.csv", 6: "songs/surprised.csv"}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]

class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._numFrames / self.elapsed()

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

class VideoCamera(object):
    def __init__(self):
        self.captured_frame = None
        self.video_stream = WebcamVideoStream(src=0).start()  # Start the video stream

    def get_frame(self):
        global cap1
        global df1
        image = self.video_stream.read()  # Read frame from the video stream
        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            roi_gray_frame = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            show_text[0] = maxindex
            cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            df1 = music_rec()

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), df1  # Return the image and the song recommendations


def music_rec():
    try:
        df = pd.read_csv(music_dist[show_text[0]], encoding='utf-8')  # Read song CSV based on detected mood
        df = df[['Name', 'Album', 'Artist']]  # Extract relevant columns
        df = df.head(15)  # Limit the results to the top 15 songs
        return df
    except Exception as e:
        print(f"Error in music_rec: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


