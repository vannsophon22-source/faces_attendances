import cv2
import pickle
import numpy as np
import os
import csv
import time
import winsound
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch

# --- 1. SETUP & FEEDBACK FUNCTIONS ---
def speak(str1):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(str1)

# Ensure the Attendance directory exists
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# Load Face Data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# FIX: Sync data if samples are inconsistent (Prevents ValueError)
if len(FACES) != len(LABELS):
    print(f"Mismatch found: {len(FACES)} faces vs {len(LABELS)} labels. Syncing...")
    min_val = min(len(FACES), len(LABELS))
    FACES = FACES[:min_val]
    LABELS = LABELS[:min_val]

# Initialize KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load Camera and Assets
video = cv2.VideoCapture(0)

# --- DETECT FROM AFAR ---
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
imgBackground = cv2.imread("background.png")

COL_NAMES = ['NAME', 'TIME']
recorded_attendance = []  # prevents logging the same person multiple times per session

# --- STABILITY TRACKING ---
current_name = None
stable_count = 0
STABLE_THRESHOLD = 15 # number of consecutive frames required for stable recognition (adjust as needed)
CONFIDENCE_THRESHOLD = 16000 # distance threshold for KNN (lower = more strict, better for far faces, adjust as needed)

print("System Ready. Scanning for faces...")

# --- 2. MAIN LOOP ---
while True:
    ret, frame = video.read()

    # --- UPSCALE FRAME for better far detection ---
    # scale_factor controls how much the image size is increased. 1.5 means 150% of original size, which helps detect far faces but may reduce performance. Adjust as needed.
    scale_factor = 1.5
    enlarged = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    # scaleFactor=1.05 → thorough scan at many sizes (shrink)
    # minNeighbors=12   → detect faces multiple times and only return if detected at least 12 times (higher = more strict, better for far faces)
    # minSize=(60,60)  → detect faces as small as 60x60px
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=12, minSize=(60, 60))

    faces_original = []
    for (x, y, w, h) in faces:
        x = int(x / scale_factor)
        y = int(y / scale_factor)
        w = int(w / scale_factor)
        h = int(h / scale_factor)
        # Store original coordinates for processing and UI because detection was done on enlarged image
        # divide by 1.5 specifically because we scaled up to 1.5
        faces_original.append((x, y, w, h))

    # If no face detected, reset stability
    if len(faces_original) == 0:
        current_name = None
        stable_count = 0

    for (x, y, w, h) in faces_original:
        # Process face for prediction
        crop_img = frame[y:y+h, x:x+w, :]

        # --- LARGER IMAGE SIZE for better far recognition ---
        # resize to 100x100
        resized_img = cv2.resize(crop_img, (100, 100)).flatten().reshape(1, -1)

        # Check confidence
        distances, indices = knn.kneighbors(resized_img)
        avg_distance = np.mean(distances[0])

        if avg_distance > CONFIDENCE_THRESHOLD:
            name = "Unknown"
        else:
            output = knn.predict(resized_img)
            name = str(output[0])

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        # Default UI
        color = (0, 0, 255)
        status_text = "SCANNING..."

        # Stability check
        # same name increase stable_count
        if name == current_name:
            stable_count += 1
        else:
            current_name = name # reset
            stable_count = 1

        # Show progress
        if name != "Unknown" and name not in recorded_attendance:
            status_text = f"HOLD STILL... {stable_count}/{STABLE_THRESHOLD}"
            color = (0, 165, 255)

        # Log attendance when stable enough, name not in record and not unknown
        if stable_count >= STABLE_THRESHOLD and name not in recorded_attendance and name != "Unknown":
            now = datetime.now()
            if now.hour == 14 and 0 <= now.minute < 15:
                file_path = f"Attendance/Attendance_{date}.csv" # build the file path using today dates, saved in attendance
                file_exists = os.path.isfile(file_path)

                with open(file_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(COL_NAMES)
                    writer.writerow([name, timestamp])

                recorded_attendance.append(name)
                winsound.Beep(1000, 300) # Beep sound for feedback
                print(f"Logged: {name} at {timestamp}")
                speak(f"Attendance taken for {name}")

                stable_count = 0
                current_name = None
            else:
                status_text = "LATE - NOT RECORDED"
                color = (0, 165, 255)

        # If already recorded, show GREEN
        if name in recorded_attendance:
            color = (0, 255, 0)
            status_text = "VERIFIED"

        # If unknown, show RED
        if name == "Unknown":
            color = (0, 0, 255)
            status_text = "UNKNOWN"

        # Draw UI
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2) # box around face
        cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1) # filled rectangle for name label
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2) # name label
        cv2.putText(frame, status_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2) # status text below box

    # Display result
    # Place the processed frame onto the background template for a nicer UI
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Attendance System", imgBackground)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
