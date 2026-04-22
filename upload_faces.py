import cv2
import pickle
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# configuration
SAMPLES_PER_IMAGE = 10   # How many augmented samples to generate per uploaded photo
IMG_SIZE = (100, 100)      # Must match what Add_faces.py and test.py use
DATA_DIR = "data"

# HELPER FUNCTIONS

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_haar():
    xml_path = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"Cannot find {xml_path}.\n"
            "Make sure haarcascade_frontalface_default.xml is inside your 'data/' folder."
        )
    return cv2.CascadeClassifier(xml_path)

# teaching the system to recognize new faces from uploaded photos
def augment_face(face_img):
    """Generate multiple augmented versions of a single face crop."""
    samples = []
    h, w = face_img.shape[:2]

    # 1. Original, resized 100x100
    samples.append(cv2.resize(face_img, IMG_SIZE))

    # 2. Horizontal flip
    samples.append(cv2.resize(cv2.flip(face_img, 1), IMG_SIZE))

    # 3-4. Brightness variations
    for gamma in [0.7, 1.3]:
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)], dtype=np.uint8)
        adjusted = cv2.LUT(face_img, table)
        samples.append(cv2.resize(adjusted, IMG_SIZE))

    # 5-6. Small rotations
    center = (w // 2, h // 2)
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_img, M, (w, h))
        samples.append(cv2.resize(rotated, IMG_SIZE))

    # 7. Slight zoom-in crop
    margin = int(min(h, w) * 0.1)
    if margin > 0:
        cropped = face_img[margin:h-margin, margin:w-margin]
        samples.append(cv2.resize(cropped, IMG_SIZE))

    # 8. Gaussian blur (simulates slight out-of-focus)
    blurred = cv2.GaussianBlur(face_img, (3, 3), 0)
    samples.append(cv2.resize(blurred, IMG_SIZE))

    # 9. Grayscale-converted back to BGR (lighting robustness)
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    samples.append(cv2.resize(gray_bgr, IMG_SIZE))

    # 10. Small translation
    M2 = np.float32([[1, 0, 3], [0, 1, 3]])
    translated = cv2.warpAffine(face_img, M2, (w, h))
    samples.append(cv2.resize(translated, IMG_SIZE))

    return samples[:SAMPLES_PER_IMAGE]

def extract_faces_from_image(image_path, facedetect):
    """Load an image, detect faces, return list of face crops."""
    img = cv2.imread(image_path)
    if img is None:
        return [], f"Could not read image: {os.path.basename(image_path)}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #scalefactor controls how much the image size is reduced at each image scale. 1.1 means reduce by 10% each time
    #minNeighbors scan the img multiple times and only return a face if it detects it at least this many times. Higher = more strict, better for far faces
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    crops = []
    # crop out the detected faces and store them in a list
    for (x, y, w, h) in faces:
        crop = img[y:y+h, x:x+w]
        crops.append(crop)

    if len(crops) == 0:
        return [], f"No face detected in: {os.path.basename(image_path)}"

    return crops, None

def save_face_data(name, all_samples):
    """Append new face samples and labels to the pkl files."""
    ensure_data_dir()

    faces_array = np.array(all_samples)                        # (N, 100, 100, 3)
    faces_flat  = faces_array.reshape(len(all_samples), -1)    # (N, 30000)

    names_list = [name] * len(all_samples)

    # --- names.pkl ---
    names_path = os.path.join(DATA_DIR, "names.pkl")
    if os.path.exists(names_path):
        with open(names_path, "rb") as f:
            existing_names = pickle.load(f)
        existing_names += names_list
    else:
        existing_names = names_list

    with open(names_path, "wb") as f:
        pickle.dump(existing_names, f)

    # --- faces_data.pkl ---
    faces_path = os.path.join(DATA_DIR, "faces_data.pkl")
    if os.path.exists(faces_path):
        with open(faces_path, "rb") as f:
            existing_faces = pickle.load(f)
        existing_faces = np.append(existing_faces, faces_flat, axis=0)
    else:
        existing_faces = faces_flat

    with open(faces_path, "wb") as f:
        pickle.dump(existing_faces, f)

    return len(all_samples)

# GUI Application

class UploadFacesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Upload Faces - Attendance System")
        self.root.geometry("520x480")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        self.selected_files = []

        # Title
        tk.Label(root, text="Face Upload Registration",
                 font=("Helvetica", 18, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(pady=(24, 4))

        tk.Label(root, text="Upload photos of yourself from multiple angles.",
                 font=("Helvetica", 10),
                 bg="#1e1e2e", fg="#a6adc8").pack()

        # Name entry
        name_frame = tk.Frame(root, bg="#1e1e2e")
        name_frame.pack(pady=20)

        tk.Label(name_frame, text="Your Name:", font=("Helvetica", 11),
                 bg="#1e1e2e", fg="#cdd6f4").grid(row=0, column=0, padx=8)

        self.name_var = tk.StringVar()
        name_entry = tk.Entry(name_frame, textvariable=self.name_var,
                              font=("Helvetica", 11), width=22,
                              bg="#313244", fg="#cdd6f4",
                              insertbackground="#cdd6f4", relief="flat",
                              bd=6)
        name_entry.grid(row=0, column=1, padx=8)

        # Buttons
        btn_frame = tk.Frame(root, bg="#1e1e2e")
        btn_frame.pack(pady=4)

        tk.Button(btn_frame, text="📁  Select Photos",
                  command=self.select_files,
                  font=("Helvetica", 11), bg="#89b4fa", fg="#1e1e2e",
                  activebackground="#74c7ec", relief="flat",
                  padx=14, pady=8, cursor="hand2").grid(row=0, column=0, padx=8)

        # register
        tk.Button(btn_frame, text="Register",
                  command=self.register,
                  font=("Helvetica", 11, "bold"), bg="#a6e3a1", fg="#1e1e2e",
                  activebackground="#94e2d5", relief="flat",
                  padx=14, pady=8, cursor="hand2").grid(row=0, column=1, padx=8)

        # File list label
        self.files_label = tk.Label(root, text="No photos selected.",
                                    font=("Helvetica", 10),
                                    bg="#1e1e2e", fg="#a6adc8")
        self.files_label.pack(pady=6)

        # File listbox
        list_frame = tk.Frame(root, bg="#1e1e2e")
        list_frame.pack(padx=30, fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(list_frame, font=("Helvetica", 9),
                                  bg="#313244", fg="#cdd6f4",
                                  selectbackground="#89b4fa",
                                  relief="flat", bd=0,
                                  yscrollcommand=scrollbar.set,
                                  height=8)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        # Progress bar
        self.progress = ttk.Progressbar(root, length=460, mode="determinate")
        self.progress.pack(pady=12)

        # Status label
        self.status_label = tk.Label(root, text="",
                                     font=("Helvetica", 10),
                                     bg="#1e1e2e", fg="#a6e3a1",
                                     wraplength=460)
        self.status_label.pack(pady=(0, 16))

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select face photos",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if files:
            self.selected_files = list(files)
            self.listbox.delete(0, tk.END)
            for f in self.selected_files:
                self.listbox.insert(tk.END, "  " + os.path.basename(f))
            self.files_label.config(
                text=f"{len(self.selected_files)} photo(s) selected."
            )
            self.status_label.config(text="")

    def register(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing Name", "Please enter your name first.")
            return
        if not self.selected_files:
            messagebox.showwarning("No Photos", "Please select at least one photo.")
            return

        try:
            facedetect = load_haar()
        except FileNotFoundError as e:
            messagebox.showerror("Missing File", str(e))
            return

        all_samples = []
        errors = []

        # total number of photo to process (e.g. 5 photos = maximum 5)
        self.progress["maximum"] = len(self.selected_files)
        self.progress["value"] = 0

        # process each selected photo, extract faces, augment them, and collect samples
        for i, path in enumerate(self.selected_files):
            crops, err = extract_faces_from_image(path, facedetect)
            if err:
                errors.append(err)
            else:
                for crop in crops:
                    samples = augment_face(crop)
                    all_samples.extend(samples)

            self.progress["value"] = i + 1
            self.root.update_idletasks()

        if not all_samples:
            msg = "No faces could be detected in any of the selected photos.\n\nTips:\n- Use clear, well-lit photos\n- Make sure your face is visible and not too small"
            if errors:
                msg += "\n\nDetails:\n" + "\n".join(errors)
            messagebox.showerror("No Faces Found", msg)
            return

        total_saved = save_face_data(name, all_samples)

        status = f"Success! Registered '{name}' with {total_saved} face samples from {len(self.selected_files) - len(errors)} photo(s)."
        if errors:
            status += f"\n {len(errors)} photo(s) had no detectable face and were skipped."

        self.status_label.config(text=status)
        messagebox.showinfo("Registration Complete",
                            f"'{name}' has been registered!\n\n"
                            f"Samples saved: {total_saved}\n"
                            f"You can now run test.py to take attendance.")

        # Reset
        self.selected_files = []
        self.listbox.delete(0, tk.END)
        self.files_label.config(text="No photos selected.")
        self.progress["value"] = 0


# entry point
if __name__ == "__main__":
    ensure_data_dir()
    root = tk.Tk()
    app = UploadFacesApp(root)
    root.mainloop()