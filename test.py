import cv2
import pickle
import numpy as np
import os
import csv
import time
import winsound
import face_recognition
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(str1)

if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# Load Face Data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

if FACES.shape[1] != 128:
    print("ERROR: Your saved face data uses the old raw-pixel format (30,000 values).")
    print("Please delete data/faces_data.pkl and data/names.pkl, then re-register everyone.")
    exit(1)

if len(FACES) != len(LABELS):
    print(f"Mismatch found: {len(FACES)} faces vs {len(LABELS)} labels. Syncing...")
    min_val = min(len(FACES), len(LABELS))
    FACES = FACES[:min_val]
    LABELS = LABELS[:min_val]

KNOWN_ENCODINGS = list(FACES)
KNOWN_NAMES = list(LABELS)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
imgBackground = cv2.imread("background.png")

COL_NAMES = ['NAME', 'TIME']
recorded_attendance = []

current_name = None
stable_count = 0
STABLE_THRESHOLD = 15
# Euclidean distance in 128-dim face embedding space.
# Same person: typically < 0.45. Different people: typically > 0.55.
CONFIDENCE_THRESHOLD = 0.45

print("System Ready. Scanning for faces...")

while True:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(haar_faces) == 0:
        current_name = None
        stable_count = 0

    if len(haar_faces) > 0:
        # Convert Haar (x,y,w,h) → face_recognition (top,right,bottom,left) format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in haar_faces]
        face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
    else:
        face_locations = []
        face_encodings = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        x, y, w, h = left, top, right - left, bottom - top

        if KNOWN_ENCODINGS:
            distances = face_recognition.face_distance(KNOWN_ENCODINGS, face_encoding)
            best_idx = int(np.argmin(distances))
            best_dist = distances[best_idx]
            name = KNOWN_NAMES[best_idx] if best_dist < CONFIDENCE_THRESHOLD else "Unknown"
        else:
            name = "Unknown"

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        color = (0, 0, 255)
        status_text = "SCANNING..."

        if name == current_name:
            stable_count += 1
        else:
            current_name = name
            stable_count = 1

        if name != "Unknown" and name not in recorded_attendance:
            status_text = f"HOLD STILL... {stable_count}/{STABLE_THRESHOLD}"
            color = (0, 165, 255)

        if stable_count >= STABLE_THRESHOLD and name not in recorded_attendance and name != "Unknown":
            file_path = f"Attendance/Attendance_{date}.csv"
            file_exists = os.path.isfile(file_path)

            with open(file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow([name, timestamp])

            recorded_attendance.append(name)
            winsound.Beep(1000, 300)
            print(f"Logged: {name} at {timestamp}")
            speak(f"Attendance taken for {name}")
            stable_count = 0
            current_name = None

        if name in recorded_attendance:
            color = (0, 255, 0)
            status_text = "VERIFIED"

        if name == "Unknown":
            color = (0, 0, 255)
            status_text = "UNKNOWN"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Attendance System", imgBackground)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
