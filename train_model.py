#!/usr/bin/env python3
"""
train_model.py
==============
Custom face-recognition model trainer.

Pipeline
--------
1. Load enrolled face images from the database / enrolled_faces/ folder.
2. Augment images (flip, brightness, rotation).
3. Extract Facenet512 embeddings via DeepFace.
4. Train a lightweight softmax classifier (sklearn SVM / MLP) on top
   of the frozen embeddings.
5. Serialise the classifier to models/face_classifier.pkl.

Run
---
    python train_model.py

Requirements
------------
deepface, scikit-learn, opencv-python, numpy, Pillow, tqdm
"""

import os, sys, pickle, json, logging
import numpy as np
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# Paths
ROOT          = os.path.dirname(__file__)
ENROLLED_DIR  = os.path.join(ROOT, 'enrolled_faces')
MODELS_DIR    = os.path.join(ROOT, 'models')
CLASSIFIER_PT = os.path.join(MODELS_DIR, 'face_classifier.pkl')
LABEL_MAP_PT  = os.path.join(MODELS_DIR, 'label_map.json')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── DeepFace / sklearn imports ────────────────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False
    log.warning("DeepFace not installed – using random embeddings (demo mode).")

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME       = "Facenet512"
DETECTOR_BACKEND = "opencv"
AUGMENT_FACTOR   = 5        # copies per image


# ─────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────
def augment(img: np.ndarray) -> list[np.ndarray]:
    variants = [img]

    # Horizontal flip
    variants.append(cv2.flip(img, 1))

    # Brightness jitter
    for delta in [-30, +30]:
        bright = np.clip(img.astype(int) + delta, 0, 255).astype(np.uint8)
        variants.append(bright)

    # Small rotation (±10°)
    h, w = img.shape[:2]
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h))
        variants.append(rot)

    return variants[:AUGMENT_FACTOR]


# ─────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────
def get_embedding(img: np.ndarray) -> np.ndarray | None:
    if not DEEPFACE_OK:
        return np.random.rand(512).astype(np.float32)
    try:
        result = DeepFace.represent(
            img_path          = img,
            model_name        = MODEL_NAME,
            detector_backend  = DETECTOR_BACKEND,
            enforce_detection = True,
        )
        return np.array(result[0]['embedding'], dtype=np.float32)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────
def train():
    log.info("Scanning enrolled_faces/ …")

    if not os.path.isdir(ENROLLED_DIR) or not os.listdir(ENROLLED_DIR):
        log.error("No images found in enrolled_faces/. Enrol students first.")
        sys.exit(1)

    X, y = [], []

    for fname in tqdm(os.listdir(ENROLLED_DIR), desc="Extracting embeddings"):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Filename convention: <student_id>_<name>.jpg
        parts     = os.path.splitext(fname)[0].split('_', 1)
        label     = parts[0]                   # student_id as label
        img_path  = os.path.join(ENROLLED_DIR, fname)
        img       = cv2.imread(img_path)

        if img is None:
            log.warning(f"Cannot read {fname}, skipping.")
            continue

        for aug_img in augment(img):
            emb = get_embedding(aug_img)
            if emb is not None:
                X.append(emb)
                y.append(label)

    if len(set(y)) < 2:
        log.error("Need ≥ 2 enrolled students to train a classifier.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)
    log.info(f"Dataset: {len(X)} samples, {len(set(y))} classes.")

    # Encode labels
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)

    # Save label map (index → student_id)
    label_map = {int(i): lbl for i, lbl in enumerate(le.classes_)}
    with open(LABEL_MAP_PT, 'w') as f:
        json.dump(label_map, f, indent=2)
    log.info(f"Label map saved → {LABEL_MAP_PT}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # SVM classifier with L2 normalisation
    clf = Pipeline([
        ('norm', Normalizer(norm='l2')),
        ('svm',  SVC(kernel='rbf', C=10, gamma='scale',
                     probability=True, class_weight='balanced'))
    ])
    clf.fit(X_train, y_train)
    log.info("Classifier trained.")

    # Evaluation
    y_pred = clf.predict(X_test)
    log.info("\n" + classification_report(y_test, y_pred,
             target_names=le.classes_))

    # Persist
    with open(CLASSIFIER_PT, 'wb') as f:
        pickle.dump({'classifier': clf, 'label_encoder': le}, f)
    log.info(f"Model saved → {CLASSIFIER_PT}")


# ─────────────────────────────────────────────────────────────────────
# Inference helper (used by the API if you want SVM-based recognition)
# ─────────────────────────────────────────────────────────────────────
class TrainedClassifier:
    def __init__(self, model_path: str = CLASSIFIER_PT):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.clf = data['classifier']
        self.le  = data['label_encoder']

    def predict(self, embedding: np.ndarray):
        """Returns (student_id_str, confidence_float)."""
        emb  = embedding.reshape(1, -1)
        pred = self.clf.predict(emb)[0]
        prob = self.clf.predict_proba(emb)[0].max()
        return self.le.inverse_transform([pred])[0], float(prob)


if __name__ == '__main__':
    train()
