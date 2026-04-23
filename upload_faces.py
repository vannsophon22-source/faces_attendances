import cv2
import pickle
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import face_recognition
from PIL import Image

DATA_DIR = "data"

# HELPER FUNCTIONS

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

MAX_WIDTH = 800  # resize large photos before processing to speed up HOG detection

def extract_encodings_from_image(image_path):
    """Load an image, detect faces with face_recognition, return list of 128-dim encodings."""
    try:
        pil_img = Image.open(image_path).convert('RGB')
        # Downscale large images — HOG is much faster on smaller images
        w, h = pil_img.size
        if w > MAX_WIDTH:
            pil_img = pil_img.resize((MAX_WIDTH, int(h * MAX_WIDTH / w)), Image.LANCZOS)
        rgb_img = np.array(pil_img)
    except Exception as e:
        return [], f"Could not read image: {os.path.basename(image_path)} ({e})"

    locations = face_recognition.face_locations(rgb_img, model="hog")
    encodings = face_recognition.face_encodings(rgb_img, locations)

    if not encodings:
        return [], f"No face detected in: {os.path.basename(image_path)}"

    return encodings, None

def save_face_data(name, encodings):
    """Append new face encodings and labels to the pkl files."""
    ensure_data_dir()

    faces_array = np.array(encodings)  # (N, 128)
    names_list = [name] * len(encodings)

    names_path = os.path.join(DATA_DIR, "names.pkl")
    if os.path.exists(names_path):
        with open(names_path, "rb") as f:
            existing_names = pickle.load(f)
        existing_names += names_list
    else:
        existing_names = names_list

    with open(names_path, "wb") as f:
        pickle.dump(existing_names, f)

    faces_path = os.path.join(DATA_DIR, "faces_data.pkl")
    if os.path.exists(faces_path):
        with open(faces_path, "rb") as f:
            existing_faces = pickle.load(f)
        if existing_faces.shape[1] != 128:
            raise ValueError("Existing data uses the old raw-pixel format. Delete data/faces_data.pkl and data/names.pkl, then re-register.")
        existing_faces = np.append(existing_faces, faces_array, axis=0)
    else:
        existing_faces = faces_array

    with open(faces_path, "wb") as f:
        pickle.dump(existing_faces, f)

    return len(encodings)

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

        all_encodings = []
        errors = []

        self.progress["maximum"] = len(self.selected_files)
        self.progress["value"] = 0

        for i, path in enumerate(self.selected_files):
            encodings, err = extract_encodings_from_image(path)
            if err:
                errors.append(err)
            else:
                all_encodings.extend(encodings)

            self.progress["value"] = i + 1
            self.root.update_idletasks()

        if not all_encodings:
            msg = "No faces could be detected in any of the selected photos.\n\nTips:\n- Use clear, well-lit photos\n- Make sure your face is visible and not too small"
            if errors:
                msg += "\n\nDetails:\n" + "\n".join(errors)
            messagebox.showerror("No Faces Found", msg)
            return

        try:
            total_saved = save_face_data(name, all_encodings)
        except ValueError as e:
            messagebox.showerror("Data Format Error", str(e))
            return

        status = f"Success! Registered '{name}' with {total_saved} face encoding(s) from {len(self.selected_files) - len(errors)} photo(s)."
        if errors:
            status += f"\n {len(errors)} photo(s) had no detectable face and were skipped."

        self.status_label.config(text=status)
        messagebox.showinfo("Registration Complete",
                            f"'{name}' has been registered!\n\n"
                            f"Face encodings saved: {total_saved}\n"
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