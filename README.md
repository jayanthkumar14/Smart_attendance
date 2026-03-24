# 🎓 FaceAttend — Smart Attendance System

A full-stack deep learning attendance system using **face recognition** (DeepFace + Facenet512).  
Students are enrolled once; attendance is marked automatically by scanning their faces.

---

## 🏗 Architecture

```
smart_attendance/
├── app.py                        ← Flask app factory & entry point
├── train_model.py                ← Standalone model training script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml            ← Web + PostgreSQL
├── .env.example
│
├── backend/
│   ├── database/
│   │   └── models.py             ← SQLAlchemy ORM (Student, Attendance, Subject)
│   ├── routes/
│   │   ├── students.py           ← Enrol / CRUD API  (/api/students/*)
│   │   └── attendance.py         ← Mark / query API  (/api/attendance/*)
│   └── utils/
│       └── face_recognition.py   ← FaceRecognitionEngine (DeepFace)
│
├── frontend/
│   └── templates/
│       └── index.html            ← Single-page UI (camera, enrol, records, reports)
│
├── enrolled_faces/               ← Saved face crops (auto-created)
└── models/                       ← Trained SVM classifier (auto-created)
```

---

## ⚡ Quick Start (Local)

### 1 — Clone & install
```bash
git clone <repo-url>
cd smart_attendance

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2 — Configure
```bash
cp .env.example .env
# Edit .env — at minimum set SECRET_KEY
```

### 3 — Run
```bash
python app.py
```
Open **http://localhost:5000**

---

## 🐳 Docker (recommended for production)

```bash
docker-compose up --build
```
Opens on **http://localhost:5000**. PostgreSQL is wired automatically.

---

## 🤖 Deep Learning Pipeline

### Face Recognition Engine (`backend/utils/face_recognition.py`)

| Step | Detail |
|------|--------|
| **Detection** | OpenCV Haar Cascade (fast) or MTCNN (accurate) |
| **Embedding** | Facenet512 via DeepFace → 512-d float vector |
| **Matching** | Cosine distance < **0.40** → recognised |
| **Storage** | Embeddings serialised as JSON in SQLite/PostgreSQL |

### Custom SVM Classifier (`train_model.py`)

Run after enrolling several students for better accuracy:

```bash
python train_model.py
```

Pipeline:
1. Load all enrolled face images
2. **Data augmentation** (flip, brightness, rotation) — 5× per image
3. Extract Facenet512 embeddings for each augmented image
4. Train **SVM (RBF kernel, C=10)** with L2 normalisation
5. Evaluate with classification report
6. Save model → `models/face_classifier.pkl`

---

## 📡 REST API Reference

### Students

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/students/` | List all students |
| `POST` | `/api/students/enroll` | Enrol new student |
| `GET`  | `/api/students/<id>` | Get single student |
| `PUT`  | `/api/students/<id>` | Update / re-enrol |
| `DELETE` | `/api/students/<id>` | Deactivate |

**Enrol payload:**
```json
{
  "student_id": "2021CS001",
  "name": "Alice Smith",
  "email": "alice@example.com",
  "department": "Computer Science",
  "year": "2nd Year",
  "image_base64": "<base64-jpeg>"
}
```

### Attendance

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/attendance/mark` | Mark from image |
| `GET`  | `/api/attendance/` | All records (filterable) |
| `GET`  | `/api/attendance/today` | Today's summary |
| `GET`  | `/api/attendance/report/<roll>` | Per-student report |
| `POST` | `/api/attendance/manual` | Manual override |
| `GET`  | `/api/attendance/stats` | Dashboard stats |

**Mark payload:**
```json
{
  "image_base64": "<base64-jpeg>",
  "subject": "Mathematics"
}
```

---

## 🗄 Database

Default: **SQLite** (`attendance.db` in project root).

Switch to **PostgreSQL** for production by setting:
```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

Also works with **MySQL** via `mysql+pymysql://...`.

Tables:
- `students` — profile + 512-d face embedding
- `attendance` — date, subject, confidence, status
- `subjects` — subject catalogue (seeded automatically)

---

## 🖥 UI Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Stats + quick camera attendance |
| **Enrol Student** | Webcam / upload + form |
| **Students** | Grid of all enrolled faces |
| **Take Attendance** | Full camera scan with annotations |
| **Records** | Filterable log with CSV export |
| **Reports** | Per-student attendance % with history |

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `change-me` | Flask session secret |
| `DATABASE_URL` | `sqlite:///attendance.db` | DB connection string |
| `MODEL_NAME` | `Facenet512` | DeepFace model |
| `RECOGNITION_THRESHOLD` | `0.40` | Cosine distance threshold |
| `ENROLLED_FACES_DIR` | `enrolled_faces` | Face image storage |

---

## 📦 Key Dependencies

- **Flask** — web framework
- **DeepFace** — face recognition (Facenet512)
- **TensorFlow** — DeepFace backend
- **OpenCV** — image processing & face detection
- **SQLAlchemy** — ORM (SQLite / PostgreSQL / MySQL)
- **scikit-learn** — SVM classifier & evaluation
- **MTCNN** — optional accurate face detector

---

## 📄 License

MIT
