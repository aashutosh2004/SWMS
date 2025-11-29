import os
import io
import json
import base64
import datetime
import uuid
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf

# ----- Fix MKL crash on Windows -----
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ----- PATHS -----
APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))

MODEL_PATH = os.path.join(ROOT_DIR, "models", "waste_mobilenetv2.h5")
LABELS_PATH = os.path.join(ROOT_DIR, "data", "labels.json")
LOGS_PATH = os.path.join(ROOT_DIR, "data", "logs.json")
UPLOADS_DIR = os.path.join(ROOT_DIR, "data", "uploads")

IMG_SIZE = (224, 224)

os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Create empty log file if missing
if not os.path.exists(LOGS_PATH):
    with open(LOGS_PATH, "w") as f:
        json.dump([], f)

# ----- BIN MAPPING -----
BIN_MAP = {
    "cardboard": "paper/cardboard",
    "paper": "paper/cardboard",
    "plastic": "plastic",
    "glass": "glass",
    "metal": "metal",
    "trash": "landfill/other"
}

# ----- INIT FLASK APP -----
app = Flask(__name__, static_folder="static", template_folder="templates")

# -------- LOAD MODEL & LABELS --------
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    CLASS_NAMES = json.load(f)


# ---------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)[None, ...]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr


# ---------------------------------------
# RUN MODEL
# ---------------------------------------
def predict_image(img: Image.Image):
    inp = preprocess_image(img)
    pred = model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(pred))

    return {
        "class": CLASS_NAMES[idx],
        "prob": float(pred[idx]),
        "probs": {CLASS_NAMES[i]: float(p) for i, p in enumerate(pred)}
    }


# ---------------------------------------
# LOG STORAGE
# ---------------------------------------
def log_prediction(entry: dict):
    try:
        with open(LOGS_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except:
        logs = []

    logs.insert(0, entry)  # newest first
    logs = logs[:1000]  # keep last 1000 entries

    with open(LOGS_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


# ======================================
# ROUTES
# ======================================

# HOME PAGE
@app.route("/")
def index():
    return render_template("index.html", classes=CLASS_NAMES)


# ---------------------------------------
# PREDICT ROUTE  (UPLOAD + CAMERA)
# ---------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    img = None
    filename = None
    source = None

    # ---- Upload File ----
    if "image" in request.files:
        file = request.files["image"]
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        path = os.path.join(UPLOADS_DIR, filename)
        file.save(path)
        img = Image.open(path)
        source = "upload"

    # ---- Camera Base64 ----
    else:
        data = request.get_json(silent=True)
        if not data or "imageBase64" not in data:
            return jsonify({"error": "No image received"}), 400

        b64 = data["imageBase64"].split(",")[-1]
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        filename = f"{uuid.uuid4().hex}.jpg"
        img.save(os.path.join(UPLOADS_DIR, filename))
        source = "camera"

    # ---- MODEL PREDICTION ----
    result = predict_image(img)
    pred_class = result["class"]
    prob = result["prob"]
    bin_name = BIN_MAP.get(pred_class, "unknown")

    # ---- LOG ENTRY ----
    entry = {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "pred_class": pred_class,
        "prob": prob,
        "bin": bin_name,
        "source": source,
        "filename": filename
    }

    log_prediction(entry)

    # ---- RESPONSE ----
    return jsonify({
        "class": pred_class,
        "prob": prob,
        "probs": result["probs"],
        "bin": bin_name
    })


# ---------------------------------------
# DASHBOARD PAGE
# ---------------------------------------
@app.route("/dashboard")
def dashboard():

    # load logs
    try:
        with open(LOGS_PATH, "r") as f:
            logs = json.load(f)
    except:
        logs = []

    # compute bin counts
    bin_counts = {}
    for log in logs:
        b = log["bin"]
        bin_counts[b] = bin_counts.get(b, 0) + 1

    # compute class counts
    class_counts = {c: 0 for c in CLASS_NAMES}
    for log in logs:
        class_counts[log["pred_class"]] += 1

    return render_template(
        "dashboard.html",
        recent=logs[:50],
        bin_counts=bin_counts,
        class_counts=class_counts,
        classes=CLASS_NAMES
    )


# Serve saved uploaded images
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)


# ---------------------------------------
# RUN APP
# ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
