import os
import re
import cv2
import logging
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from geopy.distance import geodesic
from firebase_setup import db

# === Flask setup ===
app = Flask(__name__)
CORS(app)

# --- Office geofence ---
OFFICE_LAT = 26.9236
OFFICE_LONG = 75.8068
ALLOWED_RADIUS = 0.04  # km

# --- Folders ---
REGISTERED_IMG_DIR = "registered_faces"   # keep user ref image (optional)
REGISTERED_VEC_DIR = "registered_vectors" # store embeddings
TEMP_DIR = "temp"
os.makedirs(REGISTERED_IMG_DIR, exist_ok=True)
os.makedirs(REGISTERED_VEC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
)
# also show werkzeug request logs
logging.getLogger("werkzeug").setLevel(logging.INFO)
logging.info("🚀 Starting Caffeina Attendance Server (InsightFace)…")

# === InsightFace: detector+recognizer ===
# ArcFace (r100) embeddings + RetinaFace detector, CPU via onnxruntime
from insightface.app import FaceAnalysis
try:
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    logging.info("✅ InsightFace model (buffalo_l) loaded.")
except Exception as e:
    logging.exception(f"Failed to load InsightFace models: {e}")
    raise

# --- Config ---
# ArcFace outputs L2-normalized embeddings. Cosine similarity in [ -1, 1 ].
# Good same-person similarity is typically >= 0.35~0.4 depending on image quality.
SIMILARITY_THRESHOLD = 0.40


# === Helpers ===
def safe_filename(email: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.@-]", "_", email)

def read_rgb(path: str):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def largest_face(faces):
    if not faces:
        return None
    # pick face with largest box area
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

def extract_embedding(image_path: str):
    """
    Returns (embedding: np.ndarray of shape (512,), bbox) or (None, None)
    """
    img = read_rgb(image_path)
    if img is None:
        logging.warning(f"Image read failed: {image_path}")
        return None, None

    faces = face_app.get(img)
    if len(faces) == 0:
        return None, None

    face = largest_face(faces)
    emb = face.normed_embedding  # L2-normalized 512-dim
    if emb is None or emb.size == 0:
        return None, None
    return emb.astype(np.float32), face.bbox

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    # embeddings from ArcFace are already normalized, dot -> cosine sim
    return float(np.dot(u, v))


# === ROUTES ===

# Register a user's reference face
@app.route("/register_face", methods=["POST"])
def register_face():
    try:
        email = request.form.get("email")
        image_file = request.files.get("image")

        if not email or not image_file:
            logging.warning("Missing email or image in /register_face")
            return jsonify({"status": "error", "message": "Missing email or image."}), 400

        safe_email = safe_filename(email)
        img_path = os.path.join(REGISTERED_IMG_DIR, f"{safe_email}.jpg")
        vec_path = os.path.join(REGISTERED_VEC_DIR, f"{safe_email}.npy")

        image_file.save(img_path)
        emb, bbox = extract_embedding(img_path)
        if emb is None:
            os.remove(img_path)
            logging.info(f"[{email}] no face detected.")
            return jsonify({"status": "error", "message": "No face detected. Please try again."}), 400

        np.save(vec_path, emb)
        logging.info(f"✅ Registered face for {email} (saved vector: {vec_path})")
        return jsonify({"status": "success", "message": f"Face registered for {email}"}), 200

    except Exception as e:
        logging.exception("Error in /register_face")
        return jsonify({"status": "error", "message": str(e)}), 500


# Verify and mark attendance
@app.route("/verify", methods=["POST"])
def verify():
    try:
        email = request.form.get("email")
        lat = request.form.get("latitude")
        lon = request.form.get("longitude")
        image_file = request.files.get("image")

        if not email or not lat or not lon or not image_file:
            logging.warning("Missing required fields in /verify")
            return jsonify({"status": "error", "message": "Missing required fields."}), 400

        lat, lon = float(lat), float(lon)
        safe_email = safe_filename(email)

        # Save temporary probe image
        temp_path = os.path.join(TEMP_DIR, f"{safe_email}_probe.jpg")
        image_file.save(temp_path)

        # Geofence
        dist_km = geodesic((OFFICE_LAT, OFFICE_LONG), (lat, lon)).km
        logging.info(f"[{email}] Distance from office: {dist_km:.3f} km")
        if dist_km > ALLOWED_RADIUS:
            os.remove(temp_path)
            logging.warning(f"[{email}] outside radius ({dist_km:.3f} km)")
            return jsonify({"status": "failed", "message": "❌ You are outside the office area."}), 403

        # Check registration
        vec_path = os.path.join(REGISTERED_VEC_DIR, f"{safe_email}.npy")
        if not os.path.exists(vec_path):
            os.remove(temp_path)
            logging.warning(f"[{email}] no registered vector found.")
            return jsonify({"status": "failed", "message": "No registered face found. Please contact admin."}), 404

        # Extract embeddings
        probe_emb, _ = extract_embedding(temp_path)
        os.remove(temp_path)
        if probe_emb is None:
            logging.info(f"[{email}] probe face not detected.")
            return jsonify({"status": "failed", "message": "Face not detected properly."}), 400

        ref_emb = np.load(vec_path)
        sim = cosine_similarity(ref_emb, probe_emb)
        logging.info(f"[{email}] cosine similarity = {sim:.3f}")

        if sim < SIMILARITY_THRESHOLD:
            logging.warning(f"[{email}] face mismatch (sim={sim:.3f} < {SIMILARITY_THRESHOLD})")
            return jsonify({"status": "failed", "message": "❌ Face does not match."}), 401

        # Mark attendance
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        attendance_ref = db.collection("attendance").document(email)
        attendance_ref.set({
            today: {
                "time": current_time,
                "status": "Present",
                "location": {"lat": lat, "lon": lon},
                "similarity": round(sim, 3)
            }
        }, merge=True)

        logging.info(f"✅ Attendance marked for {email} at {current_time} (sim={sim:.3f})")
        return jsonify({"status": "success", "message": "✅ Attendance marked successfully!"}), 200

    except Exception as e:
        logging.exception("Error in /verify")
        return jsonify({"status": "error", "message": str(e)}), 500


# Health check (optional)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    logging.info("🌐 Flask server at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
