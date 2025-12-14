import os
import cv2
import json
import numpy as np
from flask import Flask, request, jsonify
from insightface.app import FaceAnalysis
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin
import logging
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename
from geopy.distance import geodesic

# ---------------------------------------------------------
# ✅ CONFIGURATION
# ---------------------------------------------------------
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------
# 🔥 FIREBASE INITIALIZATION (Local + Render)
# ---------------------------------------------------------
try:
    firebase_json = os.getenv("FIREBASE_CRED_JSON")

    if firebase_json:
        logging.info("🌍 Using Firebase credentials from environment variable")
        cred_info = json.loads(firebase_json)
    else:
        logging.info("💻 Using local firebase_config.json file")
        with open("firebase_config.json") as f:
            cred_info = json.load(f)

    cred = credentials.Certificate(cred_info)
    if not firebase_admin._apps:
        initialize_app(cred)

    db = firestore.client()
    logging.info("✅ Connected to Firestore successfully!")

except Exception as e:
    logging.error(f"❌ Failed to initialize Firebase: {e}")
    traceback.print_exc()
    raise e


# ---------------------------------------------------------
# 🧠 LIGHTWEIGHT FACE MODEL (for Render/Low Storage)
# ---------------------------------------------------------
face_app = None

def get_face_app():
    """Load and cache lightweight InsightFace model."""
    global face_app
    if face_app is not None:
        return face_app

    try:
        model_dir = "/opt/render/project/.models" if os.getenv("RENDER") else "models"
        os.makedirs(model_dir, exist_ok=True)

        # 💡 Try antelopev2 (lighter); fallback to buffalo_l if needed
        try:
            logging.info("⚙️ Loading InsightFace model: antelopev2")
            face_app = FaceAnalysis(
                name="antelopev2",
                root=model_dir,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"]
            )
            face_app.prepare(ctx_id=0)
            logging.info("✅ InsightFace (antelopev2) loaded successfully!")
        except Exception as err:
            logging.warning(f"⚠️ antelopev2 failed ({err}), using buffalo_l instead")
            face_app = FaceAnalysis(
                name="buffalo_l",
                root=model_dir,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"]
            )
            face_app.prepare(ctx_id=0)
            logging.info("✅ InsightFace (buffalo_l) loaded successfully!")

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
        if img is None:
            raise ValueError("Invalid image path or format")

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

        db.collection("faces").document(email).set({"embedding": emb_list})
        logging.info(f"✅ Face registered for {email}")

        os.remove(img_path)

        return jsonify({
            "status": "success",
            "message": f"Face registered for {email}"
        }), 200

    except Exception as e:
        logging.error("Error in /register_face")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------
# 📸 VERIFY FACE & MARK ATTENDANCE (with Location Check)
# ---------------------------------------------------------
@app.route("/verify", methods=["POST"])
def verify():
    try:
        email = request.form.get("email")
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")

        if not email or not latitude or not longitude:
            return jsonify({"status": "error", "message": "Email and location required"}), 400

        # 🌍 Office coordinates
        OFFICE_LAT, OFFICE_LON = 26.92362149151839, 75.80682636101716  # Jaipur office example
        MAX_DISTANCE_KM = 0.02  # 20 meters

        user_location = (float(latitude), float(longitude))
        office_location = (OFFICE_LAT, OFFICE_LON)
        distance_km = geodesic(user_location, office_location).km

        if distance_km > MAX_DISTANCE_KM:
            return jsonify({
                "status": "error",
                "message": f"You are outside office range ({distance_km:.2f} km)",
                "distance_km": round(distance_km, 2)
            }), 403

        # ✅ Face verification
        img_file = request.files["image"]
        filename = secure_filename(img_file.filename)
        os.makedirs("temp", exist_ok=True)
        img_path = os.path.join("temp", filename)
        img_file.save(img_path)

        emb, _ = extract_embedding(img_path)

        doc = db.collection("faces").document(email).get()
        if not doc.exists:
            return jsonify({"status": "error", "message": "No registered face found"}), 404

        stored_emb = np.array(doc.to_dict()["embedding"])
        similarity = np.dot(emb, stored_emb) / (np.linalg.norm(emb) * np.linalg.norm(stored_emb))

        threshold = 0.55
        if similarity < threshold:
            logging.warning(f"❌ Face mismatch for {email} ({similarity:.2f})")
            return jsonify({
                "status": "error",
                "message": f"Face mismatch ({similarity:.2f})",
                "similarity": float(similarity),
                "distance_km": round(distance_km, 2)
            }), 401

        today = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M:%S")

        attendance_ref = db.collection("attendance").document(email)
        data = attendance_ref.get().to_dict() or {}
        data[today] = {
            "status": "Present",
            "time": time_now,
            "location": {"lat": latitude, "lon": longitude},
            "distance_km": round(distance_km, 2),
            "similarity": float(similarity)
        }
        attendance_ref.set(data)

        os.remove(img_path)
        logging.info(f"✅ Attendance marked for {email} at {time_now}")

        return jsonify({
            "status": "success",
            "message": f"Attendance marked successfully ({similarity:.2f})",
            "similarity": float(similarity),
            "distance_km": round(distance_km, 2),
            "time": time_now
        }), 200

    except Exception as e:
        logging.error("Error in /verify")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------
# 🌐 ROOT ROUTE
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "☕ Caffeina Staff API is live!",
        "endpoints": ["/register_face", "/verify", "/health"]
    })


# ---------------------------------------------------------
# 🏁 RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    logging.info(f"🚀 Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port)
