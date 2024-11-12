# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# import pickle

# # Set up Streamlit interface
# st.title("Live Face Identification System")

# # Start the webcam feed
# run = st.checkbox('Start Camera')
# FRAME_WINDOW = st.image([])

# # Load known face encodings and names from the pickle file
# with open("known_faces.pkl", "rb") as f:
#     known_encodings, known_names = pickle.load(f)

# # Capture video if 'Start Camera' is checked
# if run:
#     # OpenCV video capture
#     video_capture = cv2.VideoCapture(0)

#     while run:
#         # Capture frame-by-frame
#         ret, frame = video_capture.read()

#         # Check if the frame was captured properly
#         if not ret:
#             st.error("Failed to capture image from camera.")
#             break
        
#         # Resize frame for faster processing (optional)
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
#         # Convert the frame to RGB
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#         # Detect face locations
#         face_locations = face_recognition.face_locations(rgb_small_frame)

#         # Debug output to check face locations
#         # st.write("Detected face locations:", face_locations)

#         # Get face encodings only if faces are detected
#         face_encodings = []
#         if face_locations:
#             try:
#                 face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
#             except Exception as e:
#                 st.error(f"Error during face encoding: {e}")
#                 st.stop()
        
#         # Process each face found in the frame
#         face_names = []
#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_encodings, face_encoding)
#             name = "Unknown"
            
#             # If match found, find the name of the matched face
#             if True in matches:
#                 first_match_index = matches.index(True)
#                 name = known_names[first_match_index]
            
#             face_names.append(name)

#         # Draw labels and bounding boxes on the original frame
#         for (top, right, bottom, left), name in zip(face_locations, face_names):
#             # Scale back up face locations since the frame we detected on was scaled to 1/4 size
#             top *= 4
#             right *= 4
#             bottom *= 4
#             left *= 4

#             # Draw a box around the face
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

#             # Draw a label with a name below the face
#             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
#             font = cv2.FONT_HERSHEY_DUPLEX
#             cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#         # Display the resulting frame in Streamlit
#         FRAME_WINDOW.image(frame, channels="BGR")

#     # Release the capture when done
#     video_capture.release()
# else:
#     st.write("Camera is off")

import streamlit as st
import cv2
import face_recognition
import numpy as np
import pickle
from PIL import Image

# Set up Streamlit interface
st.title("Live Face Identification System")

# Start the webcam feed
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

# Load known face encodings and names from the pickle file
with open("known_faces.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Capture video if 'Start Camera' is checked
if run:
    # OpenCV video capture
    video_capture = cv2.VideoCapture(0)

    while run:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Check if the frame was captured properly
        if not ret:
            st.error("Failed to capture image from camera.")
            break

        # Resize frame for faster processing (optional)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the frame to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Get face encodings only if faces are detected
        face_encodings = []
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Process each face found in the frame
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            
            # If match found, find the name of the matched face
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            
            face_names.append(name)

        # Draw labels and bounding boxes on the original frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected on was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting frame in Streamlit
        FRAME_WINDOW.image(frame, channels="BGR")

    # Release the capture when done
    video_capture.release()
else:
    st.write("Camera is off")

# Add a face to the dataset using uploaded images
st.sidebar.subheader("Add a New Face")
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
new_name = st.sidebar.text_input("Enter the name for the new face")

if uploaded_image and new_name:
    # Load the uploaded image
    image = face_recognition.load_image_file(uploaded_image)
    
    # Detect faces in the uploaded image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    if face_encodings:
        # Use the first detected face encoding
        known_encodings.append(face_encodings[0])
        known_names.append(new_name)
        
        # Save the updated known faces and names
        with open("known_faces.pkl", "wb") as f:
            pickle.dump((known_encodings, known_names), f)
        
        st.sidebar.success(f"Successfully added {new_name} to the database!")
    else:
        st.sidebar.error("No face detected in the uploaded image. Please try another image.")

