import cv2
import pickle
import numpy as np
import os
import face_recognition

DATA_DIR = 'data'
CAPTURE_TARGET = 30  # number of face encodings to capture

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml'))

faces_data = []
i = 0

name = input("Enter Your Name: ")
print(f"Look at the camera. Capturing {CAPTURE_TARGET} samples — move your head slightly for variety.")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.putText(frame, f"Captured: {len(faces_data)}/{CAPTURE_TARGET}",
                (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
    cv2.imshow("Add Face - Press Q to quit", frame)

    # Every 3rd frame, encode using the Haar-detected location (skips slow HOG detection)
    if len(faces) > 0 and len(faces_data) < CAPTURE_TARGET and i % 3 == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, y, w, h = faces[0]  # use the first detected face
        # face_recognition location format is (top, right, bottom, left)
        face_location = [(y, x + w, y + h, x)]
        encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_location)
        if encodings:
            faces_data.append(encodings[0])

    i += 1
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= CAPTURE_TARGET:
        break

video.release()
cv2.destroyAllWindows()

if not faces_data:
    print("No face encodings captured. Make sure your face is clearly visible and well-lit.")
    exit(1)

faces_array = np.array(faces_data)  # shape (N, 128)
names_list = [name] * len(faces_data)

names_path = os.path.join(DATA_DIR, 'names.pkl')
faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')

if os.path.exists(names_path):
    with open(names_path, 'rb') as f:
        existing_names = pickle.load(f)
    existing_names += names_list
else:
    existing_names = names_list

with open(names_path, 'wb') as f:
    pickle.dump(existing_names, f)

if os.path.exists(faces_path):
    with open(faces_path, 'rb') as f:
        existing_faces = pickle.load(f)
    if existing_faces.shape[1] != 128:
        print("ERROR: Existing data uses the old raw-pixel format.")
        print("Delete data/faces_data.pkl and data/names.pkl, then re-register everyone.")
        exit(1)
    existing_faces = np.append(existing_faces, faces_array, axis=0)
else:
    existing_faces = faces_array

with open(faces_path, 'wb') as f:
    pickle.dump(existing_faces, f)

print(f"Done! Registered {len(faces_data)} face encodings for '{name}'.")
