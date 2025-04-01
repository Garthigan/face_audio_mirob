from flask import Flask, jsonify, request, Response
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
import queue

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

# Audio processing queue
audio_queue = queue.Queue()
processing_queue = queue.Queue()

# Cap for video capture
cap = None
processing_thread = None
audio_thread = None
transcription_thread = None
is_running = False

def calculate_similarity(embedding1, embeddings2):
    similarity = np.dot(embedding1, embeddings2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embeddings2, axis=1))
    return similarity

def audio_recording_worker():
    """Continuously record audio in chunks and put them in the queue."""
    samplerate = 16000
    chunk_size = 3  # seconds
    channels = 1
    dtype = np.int16
    silence_threshold = 500  # Adjust this based on your microphone
    
    while is_running:
        try:
            current_status["isRecording"] = True
            current_status["lastUpdated"] = time.time() * 1000
            
            # Record audio chunk
            audio_chunk = sd.rec(int(chunk_size * samplerate), 
                                samplerate=samplerate, 
                                channels=channels, 
                                dtype=dtype)
            sd.wait()  # Wait until recording is finished
            
            # Check if the audio chunk contains significant sound
            if np.max(np.abs(audio_chunk)) > silence_threshold:
                # Get current timestamp for the chunk
                timestamp = time.time() * 1000
                
                # Put the chunk in the queue with timestamp
                audio_queue.put((audio_chunk, timestamp))
                print("Audio chunk recorded and queued")
            
            current_status["isRecording"] = False
            current_status["lastUpdated"] = time.time() * 1000
            
        except Exception as e:
            print(f"Error in audio recording: {e}")
            time.sleep(1)

def audio_processing_worker():
    """ Process audio chunks from the queue and transcribe them. """
    samplerate = 16000
    
    while is_running:
        try:
            if not audio_queue.empty():
                current_status["isProcessing"] = True
                current_status["lastUpdated"] = time.time() * 1000
                
                # Get the oldest audio chunk from the queue
                audio_chunk, timestamp = audio_queue.get()
                
                # Normalize audio
                audio_chunk = audio_chunk.astype(np.float32) / np.iinfo(np.int16).max
                
                # Save temporarily to file (Whisper works with files)
                temp_filename = f"temp_audio_{timestamp}.wav"
                wav.write(temp_filename, samplerate, audio_chunk)
                
                try:
                    # Transcribe using Whisper with specific settings
                    result = whisper_model.transcribe(
                        temp_filename,
                        language="en",
                        task="transcribe",
                        fp16=False,  # Use float32 for better accuracy
                        temperature=0.0,  # Lower temperature for more deterministic results
                        word_timestamps=True  # Get word-level timestamps
                    )
                    text = result["text"].strip()
                    
                    if text:  # Only process if we got some text
                        # Find the most recent face detection to associate with this transcription
                        person_name = "Unknown"
                        if detected_faces:
                            # Get the face that was detected closest to when this audio was recorded
                            person_name = detected_faces[-1]["name"] if detected_faces[-1]["name"] != "Visitor" else "Unknown"
                        
                        # Add to transcriptions
                        transcriptions.append({
                            "text": text,
                            "timestamp": timestamp,
                            "personName": person_name
                        })
                        print(f"Transcribed text: {text}")
                except Exception as e:
                    print(f"Error in transcription: {e}")
                finally:
                    # Clean up temporary file
                    try:
                        os.remove(temp_filename)
                    except Exception as e:
                        print(f"Error removing temp file: {e}")
                
                current_status["isProcessing"] = False
                current_status["lastUpdated"] = time.time() * 1000
                
        except Exception as e:
            print(f"Error in audio processing: {e}")
            time.sleep(1)

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
    global cap, processing_thread, audio_thread, transcription_thread, is_running
    
    if is_running:
        return jsonify({"status": "already_running"})
    
    try:
        # Try different camera indices if 0 doesn't work
        for camera_index in range(2):  # Try first two camera indices
            cap = cv.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Successfully opened camera at index {camera_index}")
                # Set camera properties for better performance
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv.CAP_PROP_FPS, 30)
                break
            else:
                print(f"Failed to open camera at index {camera_index}")
                cap.release()
        
        if not cap or not cap.isOpened():
            return jsonify({"status": "error", "message": "Failed to open camera"})
            
        is_running = True
        current_status["isDetecting"] = True
        current_status["lastUpdated"] = time.time() * 1000
        
        # Start video processing thread
        processing_thread = threading.Thread(target=process_video)
        processing_thread.start()
        
        # Start audio recording thread
        audio_thread = threading.Thread(target=audio_recording_worker)
        audio_thread.start()
        
        # Start transcription thread
        transcription_thread = threading.Thread(target=audio_processing_worker)
        transcription_thread.start()
        
        return jsonify({"status": "started"})
    except Exception as e:
        print(f"Error starting system: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/stop', methods=['GET'])
def stop_system():
    global is_running, cap, processing_thread, audio_thread, transcription_thread
    
    if not is_running:
        return jsonify({"status": "already_stopped"})
    
    is_running = False
    current_status["isDetecting"] = False
    current_status["isRecording"] = False
    current_status["isProcessing"] = False
    current_status["lastUpdated"] = time.time() * 1000
    
    # Wait for threads to finish
    if processing_thread:
        processing_thread.join()
    if audio_thread:
        audio_thread.join()
    if transcription_thread:
        transcription_thread.join()
    
    if cap:
        cap.release()
    
    # Clear queues
    with audio_queue.mutex:
        audio_queue.queue.clear()
    with processing_queue.mutex:
        processing_queue.queue.clear()
    
    return jsonify({"status": "stopped"})

@app.route('/api/data', methods=['GET'])
def get_all_data():
    return jsonify({
        "faces": detected_faces,
        "transcriptions": transcriptions,
        "status": current_status
    })

@app.route('/video_feed')
def video_feed():
    try:
        if cap is None:
            print("Error: Camera not initialized in video_feed endpoint")
            return "Camera not initialized", 500
            
        def generate_frames():
            while is_running:
                try:
                    if cap is None:
                        print("Error: Camera not initialized")
                        break
                        
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to read frame from camera")
                        break
                    else:
                        # Mirror the frame horizontally
                        frame = cv.flip(frame, 1)
                        # Encode frame as JPEG with quality 80
                        ret, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 80])
                        if not ret:
                            print("Error: Failed to encode frame")
                            continue
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        # Add a small delay to control frame rate
                        time.sleep(0.03)  # ~30 FPS
                except Exception as e:
                    print(f"Error in generate_frames: {str(e)}")
                    break

        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame',
                        headers={
                            'Access-Control-Allow-Origin': '*',
                            'Access-Control-Allow-Methods': 'GET',
                            'Cache-Control': 'no-cache, no-store, must-revalidate',
                            'Pragma': 'no-cache',
                            'Expires': '0',
                            'Connection': 'keep-alive'
                        })
    except Exception as e:
        print(f"Error in video_feed endpoint: {str(e)}")
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)