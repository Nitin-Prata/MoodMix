# Moodify: AI-Powered Emotion Detection with Spotify Integration

## Overview

**Moodify** is an AI-powered web application that detects human emotions through live video feed and recommends a personalized Spotify playlist based on the detected mood. The project leverages a pre-trained emotion detection model for real-time emotion recognition and the Spotify API to fetch appropriate music playlists corresponding to the identified emotions.

---

## Features

1. **Real-Time Emotion Detection**: Uses a camera feed to capture facial expressions and predicts the user's emotion using a machine learning model.
2. **Dynamic Music Playlists**: Based on the detected mood, the app fetches personalized Spotify playlists corresponding to emotions such as Happy, Sad, Angry, and more.
3. **Integration with Spotify**: The app connects with the Spotify API to dynamically retrieve playlists and recommend them to the user.
4. **CSV Output**: Emotion-based playlists are saved as CSV files containing song details like name, album, and artist.
5. **Web Interface**: A simple and user-friendly interface built using HTML, CSS, and JavaScript to display real-time camera feed and control music selection.

---

## Project Structure

moodify │ ├── .cache # Spotify cache file ├── app.py # Main application file (Flask-based routing) ├── camera.py # Handles camera input and emotion detection ├── emotion_detection_model.h5 # Pre-trained emotion detection model ├── haarcascade_frontalface_default.xml # Face detection model using Haar cascades ├── songs/ # Directory to store CSV files for playlists ├── Spotipy.py # Handles Spotify API integration ├── static/ # Static assets (CSS, JS) ├── templates/ # HTML templates for the web interface ├── train.py # Script to train the emotion detection model ├── utils.py # Utility functions used throughout the project └── pycache/ # Compiled Python files


---

## Setup and Installation

### Prerequisites

- Python 3.x
- Pip (Python Package Manager)
- A Spotify Developer Account (to generate API keys)
- A webcam or external camera for emotion detection

  Set Up Spotify API Credentials:

Go to the Spotify Developer Dashboard and create a new application.
Retrieve the Client ID and Client Secret.
Set the SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET as environment variables.
On Windows, you can set them like this:

set SPOTIPY_CLIENT_ID=your_client_id
set SPOTIPY_CLIENT_SECRET=your_client_secret

On Linux/MacOS:

export SPOTIPY_CLIENT_ID=your_client_id
export SPOTIPY_CLIENT_SECRET=your_client_secret

Run the Application: Start the Flask server:
python app.py
Open your browser and go to http://127.0.0.1:5000/ to see the web interface.

How It Works
Emotion Detection
The camera.py script captures a live video feed from the user's camera and uses a pre-trained Convolutional Neural Network (CNN) model (emotion_detection_model.h5) to detect emotions.
It first detects the user's face using OpenCV's Haar Cascade (haarcascade_frontalface_default.xml) and then analyzes the facial expression to classify the mood into one of the following:
Angry
Disgusted
Fearful
Happy
Neutral
Sad
Surprised
Spotify Integration
The Spotipy.py script integrates with the Spotify API using the spotipy library.
Based on the detected emotion, it fetches a relevant playlist from Spotify by calling the getTrackIDs function, which extracts track IDs from the specified playlist.
The getTrackFeatures function retrieves additional details (song name, album, artist) for each track, and the data is saved into CSV files (stored in the songs/ directory).
CSV Output
For each mood, a corresponding CSV file (e.g., angry.csv, happy.csv) is generated with the following details:
Name: The name of the song.
Album: The album the song belongs to.
Artist: The primary artist who performed the song.
Key Python Files
app.py:

This file is the main entry point for the web application. It sets up the Flask server, handles routing, and links the camera feed with the emotion detection system.
camera.py:

This script manages the camera feed and processes the live video stream. It detects faces using the Haar Cascade model and then uses the trained emotion detection model (emotion_detection_model.h5) to predict the user's emotion in real-time.
Spotipy.py:

Handles Spotify API authentication and playlist retrieval based on detected moods. It fetches song details from the Spotify API and stores them in CSV files for easy reference and display.
train.py:

A script to train the emotion detection model from a dataset of facial expressions. This script includes functions for preprocessing images, training the model, and saving it for later use.
utils.py:

Contains helper functions that are used throughout the project for repetitive tasks like logging, data processing, or model utilities.
Future Improvements
Better Emotion Model: Enhance the emotion detection model to include more sophisticated deep learning architectures or fine-tuning for higher accuracy.
Playlist Customization: Allow users to link their own Spotify account and fetch personalized playlists based on their mood.
UI Enhancements: Improve the user interface for smoother interaction and better design.
Mobile Support: Add support for mobile devices by enabling access to the device camera and mobile-friendly UI.
Acknowledgments
Keras and TensorFlow: For the deep learning libraries used to build the emotion detection model.
OpenCV: For the real-time face detection features.
Spotipy: For Spotify API integration.
Flask: For building the web application framework.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any questions or feedback, feel free to contact me at:igotkarthik@gmail.com
