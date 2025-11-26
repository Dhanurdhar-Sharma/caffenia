import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Load from Render environment variable if present
firebase_json = os.getenv("FIREBASE_CRED_JSON")

if firebase_json:
    cred_dict = json.loads(firebase_json)
    cred = credentials.Certificate(cred_dict)
else:
    # Local fallback (for testing)
    cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred)
db = firestore.client()
