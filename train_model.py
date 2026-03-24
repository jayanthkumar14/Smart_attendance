from deepface import DeepFace
import os
import pickle

dataset_path = "dataset"
embeddings = {}

for student in os.listdir(dataset_path):
    student_path = os.path.join(dataset_path, student)
    embeddings[student] = []

    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)

        try:
            embedding = DeepFace.represent(img_path=img_path, model_name="Facenet")[0]["embedding"]
            embeddings[student].append(embedding)
        except:
            pass

os.makedirs("embeddings", exist_ok=True)

with open("embeddings/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Training complete!")
