import streamlit as st
import cv2
import pickle
import numpy as np
from deepface import DeepFace
import os

st.title("Smart Attendance System")

# Load embeddings
if not os.path.exists("embeddings/embeddings.pkl"):
    st.warning("No embeddings found. Please train the model first.")
else:
    with open("embeddings/embeddings.pkl", "rb") as f:
        known_embeddings = pickle.load(f)

    def find_match(embedding):
        for name, embeds in known_embeddings.items():
            for e in embeds:
                dist = np.linalg.norm(np.array(e) - np.array(embedding))
                if dist < 10:
                    return name
        return "Unknown"

    img_file = st.camera_input("Take a photo")

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        name = find_match(embedding)

        st.success(f"Detected: {name}")
