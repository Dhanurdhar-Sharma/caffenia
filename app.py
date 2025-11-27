import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from insightface.app import FaceAnalysis
from firebase_admin import credentials, firestore, initialize_app
import logging
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename

# ---------------------------------------------------------
# ✅ CONFIGURATION
# ---------------------------------------------------------
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Firebase Setup
cred = credentials.Certificate("firebase_config.json")
initialize_app(cred)
db = firestore.client()
logging.info("✅ Connected to Firestore")

# ---------------------------------------------------------
# 🧠 FACE MODEL LOADING (Stable for Render)
# ---------------------------------------------------------
face_app = None

def get_face_app():
    """Safely load and cache InsightFace for CPU."""
    global face_app
    if face_app is not None:
        return face_app

    try:
        logging.info("⚙️ Loading InsightFace (antelopev2, CPU)...")
        model_dir = "/opt/render/project/.models"
        os.makedirs(model_dir, exist_ok=True)

        face_app = FaceAnalysis(
            name="antelopev2",
            root=model_dir,
            allowed_modules=['detection', 'recognition'],
            providers=["CPUExecutionProvider"]
        )
        face_app.prepare(ctx_id=0)
        logging.info("✅ InsightFace loaded successfully!")
    except Exception as e:
        logging.error(f"❌ Failed to load InsightFace model: {e}")
        traceback.print_exc()
        raise e

    return face_app


# ---------------------------------------------------------
# 🧩 HELPER: Extract Face Embedding
# ---------------------------------------------------------
def extract_embedding(img_path):
    try:
        app_model = get_face_app()
        img = cv2.imread(img_path)
        faces = app_model.get(img)

        if not faces:
            raise ValueError("No face detected")

        emb = faces[0].embedding
        return emb, faces[0]
    except Exception as e:
        logging.error(f"Error extracting embedding: {e}")
        traceback.print_exc()
        raise e


# ---------------------------------------------------------
# ✅ HEALTH CHECK
# ---------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ---------------------------------------------------------
# 🧠 MODEL STATUS
# ---------------------------------------------------------
@app.route("/model_status", methods=["GET"])
def model_status():
    try:
        app_model = get_face_app()
        return jsonify({"status": "ok", "models": list(app_model.models.keys())}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------
# 🧍 REGISTER FACE
# ---------------------------------------------------------
@app.route("/register_face", methods=["POST"])
def register_face():
    try:
        email = request.form.get("email")
        if not email:
            return jsonify({"status": "error", "message": "Email required"}), 400

        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No image uploaded"}), 400

        img_file = request.files["image"]
        filename = secure_filename(img_file.filename)
        os.makedirs("uploads", exist_ok=True)
        img_path = os.path.join("uploads", filename)
        img_file.save(img_path)

        emb, _ = extract_embedding(img_path)
        emb_list = emb.tolist()

        # Store embedding in Firestore
        db.collection("faces").document(email).set({"embedding": emb_list})
        logging.info(f"✅ Face registered for {email}")

        return jsonify({"status": "success", "message": f"Face registered for {email}"}), 200

    except Exception as e:
        logging.error("Error in /register_face")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------
# 📸 VERIFY FACE & MARK ATTENDANCE
# ---------------------------------------------------------
@app.route("/verify", methods=["POST"])
def verify():
    try:
        email = request.form.get("email")
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")

        if not email:
            return jsonify({"status": "error", "message": "Email required"}), 400

        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No image provided"}), 400

        img_file = request.files["image"]
        filename = secure_filename(img_file.filename)
        os.makedirs("temp", exist_ok=True)
        img_path = os.path.join("temp", filename)
        img_file.save(img_path)

        emb, _ = extract_embedding(img_path)

        # Retrieve stored embedding
        doc = db.collection("faces").document(email).get()
        if not doc.exists:
            return jsonify({"status": "error", "message": "No registered face found"}), 404

        stored_emb = np.array(doc.to_dict()["embedding"])
        similarity = np.dot(emb, stored_emb) / (np.linalg.norm(emb) * np.linalg.norm(stored_emb))

        # Threshold for similarity
        threshold = 0.6
        if similarity < threshold:
            return jsonify({
                "status": "error",
                "message": f"Face mismatch ({similarity:.2f})",
                "similarity": float(similarity)
            }), 401

        # Mark attendance
        today = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M:%S")

        attendance_ref = db.collection("attendance").document(email)
        data = attendance_ref.get().to_dict() or {}
        data[today] = {
            "status": "Present",
            "time": time_now,
            "location": {"lat": latitude, "lon": longitude},
            "similarity": float(similarity)
        }
        attendance_ref.set(data)

        logging.info(f"✅ Attendance marked for {email} at {time_now}")
        return jsonify({
            "status": "success",
            "message": f"Attendance marked successfully ({similarity:.2f})",
            "similarity": float(similarity)
        }), 200

    except Exception as e:
        logging.error("Error in /verify")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------
# 🏁 RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    logging.info(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
