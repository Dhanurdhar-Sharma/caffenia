import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Load service account credentials from environment variable
firebase_json = os.getenv("FIREBASE_CRED_JSON")

if not firebase_json:
    raise Exception("❌ FIREBASE_CRED_JSON environment variable not set!")

cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

print("✅ Connected to Firestore:", db)
