import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# ---------------------------
# Setup
# ---------------------------
DB_FILE = "faces.pkl"

# Load or create face database
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)
    print(f"[INFO] Loaded {len(face_db)} faces from {DB_FILE}")
else:
    face_db = {}
    print("[INFO] No database found, starting fresh.")

# Initialize ArcFace
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)

# Enrollment mode toggle
enroll_mode = False
current_name = None
embeddings = []
NUM_SAMPLES = 20  # number of frames to capture during enrollment

def get_embedding(frame):
    faces = app.get(frame)
    if len(faces) > 0:
        return faces[0].normed_embedding, faces[0].bbox
    return None, None

# ---------------------------
# Main Loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Enrollment mode
    if enroll_mode and current_name is not None:
        emb, bbox = get_embedding(frame)
        if emb is not None:
            embeddings.append(emb)
            cv2.putText(frame, f"Capturing {len(embeddings)}/{NUM_SAMPLES}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if len(embeddings) >= NUM_SAMPLES:
                # Average embeddings
                final_emb = np.mean(embeddings, axis=0)
                face_db[current_name] = final_emb

                # Save to pickle
                with open(DB_FILE, "wb") as f:
                    pickle.dump(face_db, f)

                print(f"[INFO] Enrollment complete for {current_name} and saved to {DB_FILE}")
                enroll_mode = False
                embeddings = []

    else:
        # Recognition mode
        emb, bbox = get_embedding(frame)
        if emb is not None:
            best_match = None
            best_score = 0.0

            for name, db_emb in face_db.items():
                score = cosine_similarity([emb], [db_emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = name

            if best_score > 0.5:  # Threshold
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{best_match} ({best_score:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
            else:
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("ArcFace Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # Quit
        break
    elif key == ord("e"):  # Enroll
        current_name = input("Enter name for enrollment: ")
        enroll_mode = True
        embeddings = []
        print(f"[INFO] Enrollment started for {current_name}. Please turn your head slowly...")

cap.release()
cv2.destroyAllWindows()
