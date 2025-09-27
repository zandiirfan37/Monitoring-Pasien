import cv2
import sounddevice as sd
import numpy as np
import requests
import time
import os
import torch
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- Constants ---
IMAGE_DIR = "images"
AUDIO_DIR = "audios"
SPECTROGRAM_DIR = "spectrograms"

# --- Model Loading ---
# Load your pre-trained PyTorch model
# Make sure to place your model in the 'model' directory
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best (3).pt')
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit()

# Load the model
# model = torch.load(model_path)
# model.eval() # Set the model to evaluation mode

# --- Preprocessing (NEEDS TO BE IMPLEMENTED) ---
def preprocess_image(image):
    # This is a placeholder. You need to implement the actual preprocessing required by your model.
    # For example, resizing, normalization, etc.
    return image

def preprocess_audio(audio_path):
    # This is a placeholder. You need to implement the actual preprocessing required by your model.
    # This function now receives the audio file path
    y, sr = librosa.load(audio_path)
    spect = librosa.feature.melspectrogram(y=y, sr=sr)
    spect_db = librosa.power_to_db(spect, ref=np.max)
    return spect_db

# --- Main Capture Loop ---
def capture_and_analyze():
    # Create directories if they don't exist
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

    while True:
        # --- Capture Image ---
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()

        if ret:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_filename = f"{timestamp}.jpg"
            image_path = os.path.join(IMAGE_DIR, image_filename)
            cv2.imwrite(image_path, frame)
            print(f"Image captured: {image_path}")

            # --- Capture Audio ---
            fs = 44100  # Sample rate
            seconds = 5  # Duration of recording
            audio_filename = f"{timestamp}.wav"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)

            print("Recording audio...")
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            sd.wait()  # Wait until recording is finished
            write(audio_path, fs, myrecording) # Save as WAV file
            print(f"Audio recorded: {audio_path}")

            # --- Convert to Spectrogram ---
            spectrogram_filename = f"{timestamp}.png"
            spectrogram_path = os.path.join(SPECTROGRAM_DIR, spectrogram_filename)
            spect = preprocess_audio(audio_path)
            
            fig, ax = plt.subplots()
            librosa.display.specshow(spect, sr=fs, x_axis='time', y_axis='mel', ax=ax)
            plt.savefig(spectrogram_path)
            plt.close(fig)
            print(f"Spectrogram saved: {spectrogram_path}")


            # --- Analyze with Model (Placeholder) ---
            # Preprocess the data
            # processed_image = preprocess_image(frame)
            # processed_spectrogram = preprocess_image(spect) # Assuming same preprocessing as image

            # Run inference
            # with torch.no_grad():
            #     # The following is a placeholder. You need to adapt this to your model's input format
            #     # image_prediction = model(processed_image)
            #     # audio_prediction = model(processed_spectrogram)
            #     image_prediction = "Placeholder Image Prediction"
            #     audio_prediction = "Placeholder Audio Prediction"

            image_prediction = "Normal"
            audio_prediction = "Normal"


            # --- Send to Backend ---
            try:
                response = requests.post("http://127.0.0.1:5001/data", json={
                    'image_path': image_path,
                    'audio_path': audio_path,
                    'prediction': f'Image: {image_prediction}, Audio: {audio_prediction}'
                })
                if response.status_code == 201:
                    print("Data sent to backend successfully")
                else:
                    print(f"Failed to send data to backend. Status code: {response.status_code}")
            except requests.exceptions.ConnectionError as e:
                print(f"Could not connect to the backend: {e}")

        else:
            print("Failed to capture image")

        time.sleep(5)

if __name__ == '__main__':
    capture_and_analyze()