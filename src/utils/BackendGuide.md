
# Python Backend Setup Guide

This document explains how to set up the Python backend for the Face Speech Nexus application.

## Requirements

Make sure you have the following Python packages installed:

```bash
pip install flask flask-cors opencv-python numpy tensorflow scikit-learn joblib keras-facenet ultralytics whisper sounddevice scipy
```

## Backend Implementation

Create a file named `app.py` with the following content:

```python
from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib
from keras_facenet import FaceNet
from ultralytics import YOLO
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import os
import threading
import base64
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize models
facenet = FaceNet()
model = YOLO("yolov8n-face.pt")
svm_model = joblib.load('svm_face_recognition_model1.pkl')
whisper_model = whisper.load_model("base")

# Load face embeddings and labels
faces_embeddings = np.load("faces-embeddings_done_2classes.npz")
Y = faces_embeddings['arr_0']
labels = faces_embeddings['arr_1']

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Current state
current_status = {
    "isDetecting": False,
    "isRecording": False,
    "isProcessing": False,
    "lastUpdated": time.time() * 1000  # Convert to milliseconds for JavaScript
}
detected_faces = []
transcriptions = []

# Cap for video capture
cap = None
processing_thread = None
is_running = False

def calculate_similarity(embedding1, embeddings2):
    similarity = np.dot(embedding1, embeddings2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embeddings2, axis=1))
    return similarity

def record_audio(filename="speech.wav", duration=5, samplerate=16000):
    """Record audio from the microphone."""
    print("Recording...")
    current_status["isRecording"] = True
    current_status["lastUpdated"] = time.time() * 1000
    
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    wav.write(filename, samplerate, audio_data)
    
    current_status["isRecording"] = False
    current_status["isProcessing"] = True
    current_status["lastUpdated"] = time.time() * 1000
    print("Recording finished.")

def transcribe_audio(filename="speech.wav", person_name="Unknown"):
    """Transcribe recorded audio using Whisper."""
    result = whisper_model.transcribe(filename)
    text = result["text"]
    
    # Add to transcriptions
    transcriptions.append({
        "text": text,
        "timestamp": time.time() * 1000,
        "personName": person_name
    })
    
    current_status["isProcessing"] = False
    current_status["lastUpdated"] = time.time() * 1000
    
    return text

def process_video():
    global detected_faces, is_running
    
    while is_running:
        ret, frame = cap.read()
        if not ret:
            continue
            
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = model.predict(source=frame, show=False, save=False)
        
        # Clear previous faces
        detected_faces = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                face_img = rgb_img[y1:y2, x1:x2]
                if face_img.size == 0:  # Skip if face extraction failed
                    continue
                    
                face_img = cv.resize(face_img, (160, 160))
                face_img = np.expand_dims(face_img, axis=0)
                
                ypred = facenet.embeddings(face_img)
                similarities = calculate_similarity(ypred, Y).flatten()
                
                best_match_idx = np.argmax(similarities)
                max_similarity = similarities[best_match_idx]
                
                if max_similarity > 0.6:
                    face_idx = svm_model.predict(ypred)[0]
                    predicted_name = encoder.inverse_transform([face_idx])[0] if face_idx in encoded_labels else "Visitor"
                else:
                    predicted_name = "Visitor"
                
                # Add to detected faces
                detected_faces.append({
                    "id": str(len(detected_faces)),
                    "name": predicted_name,
                    "confidence": float(max_similarity),
                    "x": float(x1) / frame.shape[1] * 100,  # Convert to percentage
                    "y": float(y1) / frame.shape[0] * 100,
                    "width": float(x2 - x1) / frame.shape[1] * 100,
                    "height": float(y2 - y1) / frame.shape[0] * 100
                })
                
                # Only record and transcribe for non-visitors
                if predicted_name != "Visitor" and not current_status["isRecording"] and not current_status["isProcessing"]:
                    # Start a new thread for audio recording to avoid blocking
                    audio_thread = threading.Thread(target=lambda: (record_audio(), transcribe_audio(person_name=predicted_name)))
                    audio_thread.start()
        
        time.sleep(0.1)  # Avoid consuming too many resources

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(current_status)

@app.route('/api/faces', methods=['GET'])
def get_faces():
    return jsonify(detected_faces)

@app.route('/api/transcriptions', methods=['GET'])
def get_transcriptions():
    return jsonify(transcriptions)

@app.route('/api/start', methods=['GET'])
def start_system():
    global cap, processing_thread, is_running
    
    if is_running:
        return jsonify({"status": "already_running"})
    
    try:
        cap = cv.VideoCapture(0)
        is_running = True
        current_status["isDetecting"] = True
        current_status["lastUpdated"] = time.time() * 1000
        
        processing_thread = threading.Thread(target=process_video)
        processing_thread.start()
        
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/stop', methods=['GET'])
def stop_system():
    global is_running, cap
    
    if not is_running:
        return jsonify({"status": "already_stopped"})
    
    is_running = False
    current_status["isDetecting"] = False
    current_status["isRecording"] = False
    current_status["isProcessing"] = False
    current_status["lastUpdated"] = time.time() * 1000
    
    if cap:
        cap.release()
    
    return jsonify({"status": "stopped"})

@app.route('/api/data', methods=['GET'])
def get_all_data():
    return jsonify({
        "faces": detected_faces,
        "transcriptions": transcriptions,
        "status": current_status
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## Running the Backend

1. Save the file as `app.py`
2. Make sure all your model files are in the same directory:
   - `yolov8n-face.pt`
   - `svm_face_recognition_model1.pkl`
   - `faces-embeddings_done_2classes.npz`
3. Run the backend server:
   ```bash
   python app.py
   ```
4. The server will start on http://localhost:5000

## Connecting the Frontend

Once the backend is running:
1. Open the React application in your browser
2. Enter the backend URL (e.g., http://localhost:5000) in the connection field
3. Click "Connect"

The frontend will then connect to your Python backend and display real-time face recognition and speech transcription.
¸