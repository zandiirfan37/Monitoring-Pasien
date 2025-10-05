import cv2
import sounddevice as sd
import numpy as np
import requests
import time
import os
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Constants ---
IMAGE_DIR = "images"
AUDIO_DIR = "audios"
SPECTROGRAM_DIR = "spectrograms"

# --- Model Loading ---
# 1. Load kedua model
face_model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best (3).pt')
if not os.path.exists(face_model_path):
    print(f"Model not found at {face_model_path}")
    exit()
face_model = YOLO(face_model_path)

# PLEASE REPLACE WITH THE ACTUAL PATH TO YOUR AUDIO MODEL
audio_model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best (3).pt')
if not os.path.exists(audio_model_path):
    print(f"Audio model not found at {audio_model_path}")
    print("Please replace 'path/to/your/audio/model.pt' with the actual path to your audio model.")
    # exit() # You might want to exit if the model is not found
    audio_model = face_model # Using face model as a placeholder
else:
    audio_model = YOLO(audio_model_path)


# 3. Unified classes & mapping
unified_classes = ['angry','disgust','fear','happy','neutral','sad','surprize']

face_to_unified = {
    'Angry':    'angry',
    'Disgust':  'disgust',
    'Fear':     'fear',
    'Happy':    'happy',
    'Neutral':  'neutral',
    'Sad':      'sad',
    'Surprize': 'surprize',
}

audio_to_unified = {
    'Angry':        'angry',
    'Disgust':      'disgust',
    'Fear':      'fear',
    'Happy':        'happy',
    'Neutral': 'neutral',
    'Sad':          'sad',
    'Surprize':     'surprize',
}

# 4. Fungsi inferensi → np.ndarray probabilitas
def infer_probs(model, img_path, img_size=224):
    res     = model.predict(source=img_path, imgsz=img_size, conf=0.0, verbose=False)[0]
    raw     = res.cpu().probs.data
    return raw.cpu().numpy()

# --- Main Capture Loop ---
def capture_and_analyze(alpha=0.6, img_size=224):
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
            y, sr = librosa.load(audio_path)
            spect = librosa.feature.melspectrogram(y=y, sr=sr)
            spect_db = librosa.power_to_db(spect, ref=np.max)
            
            fig, ax = plt.subplots()
            librosa.display.specshow(spect_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            plt.savefig(spectrogram_path)
            plt.close(fig)
            print(f"Spectrogram saved: {spectrogram_path}")


            # --- Analyze with Fusion Model ---
            pf = infer_probs(face_model,  image_path, img_size)
            pa = infer_probs(audio_model, spectrogram_path, img_size)

            # akumulasi probabilitas ke unified index
            vf = np.zeros(len(unified_classes))
            va = np.zeros(len(unified_classes))
            for idx,p in enumerate(pf):
                uni_i = unified_classes.index(face_to_unified[face_model.names[idx]])
                vf[uni_i] += p
            for idx,p in enumerate(pa):
                uni_i = unified_classes.index(audio_to_unified[audio_model.names[idx]])
                va[uni_i] += p

            vfus = alpha * vf + (1 - alpha) * va
            prediction = unified_classes[int(np.argmax(vfus))]


            # --- Send to Backend ---
            try:
                response = requests.post("http://127.0.0.1:5001/data", json={
                    'image_path': image_path,
                    'audio_path': audio_path,
                    'prediction': prediction
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
