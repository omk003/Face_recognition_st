import face_recognition
import os
import pickle

# Folder where known face images are stored
known_faces_dir = "known_faces"
known_encodings = []
known_names = []

# Iterate through each person’s folder in the dataset
for name in os.listdir(known_faces_dir):
    person_folder = os.path.join(known_faces_dir, name)
    if os.path.isdir(person_folder):
        for filename in os.listdir(person_folder):
            file_path = os.path.join(person_folder, filename)
            # Load image and get the encoding
            image = face_recognition.load_image_file(file_path)
            encoding = face_recognition.face_encodings(image)[0]
            # Append encoding and the person's name
            known_encodings.append(encoding)
            known_names.append(name)

# Save the encodings and names using pickle
with open("known_faces.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)
