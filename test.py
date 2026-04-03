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
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
imgBackground = cv2.imread("background.png")

COL_NAMES = ['NAME', 'TIME']
recorded_attendance = [] # To prevent logging the same person multiple times per session

print("System Ready. Scanning for faces...")

# --- 2. MAIN LOOP ---
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Process face for prediction
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        name = str(output[0])
        
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        
        # Default UI: Red for scanning
        color = (0, 0, 255) 
        status_text = "SCANNING..."

        # --- AUTOMATIC LOGGING LOGIC ---
        now = datetime.now()
        
        if name not in recorded_attendance:
            # Check if time is between 2:00 PM (14) and 2:15 PM
            if now.hour == 14 and 0 <= now.minute <= 15:
                # Log to CSV
                file_path = f"Attendance/Attendance_{date}.csv"
                file_exists = os.path.isfile(file_path)
                
                with open(file_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(COL_NAMES)
                    writer.writerow([name, timestamp])
                
                # Success Feedback
                recorded_attendance.append(name)
                winsound.Beep(1000, 300) # Quick beep
                print(f"Logged: {name} at {timestamp}")
                speak(f"Attendance taken for {name}")
            
            else:
                # Handle Late Students (Optional: log them as LATE if you wish)
                color = (0, 165, 255) # Orange box for late
                status_text = "LATE - NOT RECORDED"

        # If already recorded, change UI to GREEN
        if name in recorded_attendance:
            color = (0, 255, 0)
            status_text = "VERIFIED"

        # Draw UI
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display result
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Attendance System", imgBackground)
    
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()