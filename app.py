import camera
from flask import Flask, render_template, Response, jsonify
from camera import *
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

os.environ['PYTHONIOENCODING'] = 'UTF-8'

# Load the pre-trained emotion detection model
emotion_model = load_model('emotion_detection_model.keras')  # Replace with the actual path

app = Flask(__name__)

# Table for music recommendations
headings = ("Name", "Album", "Artist")
df1 = music_rec()  # Assuming this function is defined
df1 = df1.head(15)

# Function to preprocess the image for emotion detection
def preprocess_image(frame):
    cropped_img = cv2.resize(frame, (48, 48))  # Resize the frame to 48x48
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    cropped_img = np.expand_dims(cropped_img, axis=-1)  # Add channel dimension
    cropped_img = np.expand_dims(cropped_img, axis=0)  # Add batch dimension
    return cropped_img

@app.route('/')
def index():
    print(df1.to_json(orient='records'))
    return render_template('index.html', headings=headings, data=df1)

import numpy as np
import cv2

def gen(camera):
    while True:
        global df1
        frame, df1 = camera.get_frame()  # Capture the frame from the camera

        # Check if the frame is a byte stream (e.g., from JPEG encoding)
        if isinstance(frame, bytes):
            # Decode the byte stream into a numpy array
            nparr = np.frombuffer(frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocess the frame for emotion detection
        if frame is not None:
            cropped_img = preprocess_image(frame)
            
            # Make emotion prediction
            prediction = emotion_model.predict(cropped_img)
            print(f"Prediction: {prediction}")
        
        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Convert the numpy array to bytes
        if ret:
            frame = jpeg.tobytes()

        # Return the frame as a response
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    try:
        return Response(gen(VideoCamera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/t')
def gen_table():
    try:
        return jsonify(df1.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/captured_image')
def captured_image():
    try:
        ret, jpeg = cv2.imencode('.jpg', camera.captured_frame)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        return Response(f"Error: {e}", status=500)

if __name__ == '__main__':
    app.debug = True
    app.run()
