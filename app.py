import os
import base64
from io import BytesIO
from pathlib import Path
from collections import deque, Counter
from difflib import get_close_matches

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import mediapipe as mp

try:
    from google import genai
except Exception:
    genai = None

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "gru_original_fine_tuned.keras"
CLASSES_PATH = ROOT / "class_names.npy"
DATASET_PATH = ROOT / "SSL400_Data"

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model produced by train_once.py. Training is deliberately
# kept out of Space startup so every restart does not retrain the network.
if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model file is missing. Run train_once.py once, then upload "
        "gru_original_fine_tuned.keras and class_names.npy to this Space."
    )
if not CLASSES_PATH.exists():
    raise RuntimeError("class_names.npy is missing. Upload it beside app.py.")

print("Loading trained GRU model...")
model = load_model(str(MODEL_PATH))
valid_actions = np.load(str(CLASSES_PATH), allow_pickle=True).tolist()
print(f"Model ready: {len(valid_actions)} classes; input shape={model.input_shape}")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = None
if GEMINI_API_KEY and genai is not None:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini client ready")
    except Exception as exc:
        print("Gemini initialization warning:", exc)
else:
    print("Gemini key not configured; sentence generation will use detected words.")

# Match the original app's two-hand MediaPipe landmark extraction.
try:
    mp_hands = mp.solutions.hands
except AttributeError:
    import mediapipe.python.solutions.hands as mp_hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

sequence_frames = []
prediction_history = deque(maxlen=10)
last_valid_left = np.zeros(63, dtype=np.float32)
last_valid_right = np.zeros(63, dtype=np.float32)
waiting_for_next_word = False
last_prediction = None

VIDEO_DATABASE = {}
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
if DATASET_PATH.exists():
    for root, _dirs, files in os.walk(DATASET_PATH):
        for filename in files:
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                word = Path(root).name.upper()
                category = Path(root).parent.name
                VIDEO_DATABASE[word] = {
                    "category": category,
                    "path": str(Path(root) / filename),
                }
print("Videos indexed:", len(VIDEO_DATABASE))

WORD_MAP = {
    "سلام": "HELLO", "خداحافظ": "GOOD", "ممنون": "THANK YOU", "متشکرم": "THANK YOU",
    "کتاب": "BOOK", "آب": "DRINK", "غذا": "FOOD", "خانه": "HOUSE", "برو": "GO", "بیا": "COME"
}


def extract_keypoints(results):
    global last_valid_left, last_valid_right
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    has_hand = False
    if results.multi_hand_landmarks and results.multi_handedness:
        has_hand = True
        for idx, info in enumerate(results.multi_handedness):
            label = info.classification[0].label
            landmarks = results.multi_hand_landmarks[idx]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32).flatten()
            if label == "Left":
                left, last_valid_left = coords, coords
            else:
                right, last_valid_right = coords, coords
    else:
        left, right = last_valid_left, last_valid_right
    return np.concatenate([left, right]), has_hand


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "model_loaded": model is not None, "classes": len(valid_actions), "videos": len(VIDEO_DATABASE)})


@app.route("/predict", methods=["POST"])
def predict():
    global waiting_for_next_word, last_prediction
    if waiting_for_next_word:
        return jsonify(last_prediction)
    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get("image")
        if not image_data:
            return jsonify({"success": False, "error": "No image"}), 400
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    keypoints, has_hand = extract_keypoints(results)
    if not has_hand and len(sequence_frames) == 0:
        return jsonify({"success": True, "completed": False, "prediction": "No Hand", "frames": 30})

    sequence_frames.append(keypoints)
    if len(sequence_frames) < 30:
        return jsonify({"success": True, "completed": False, "prediction": f"Collecting {len(sequence_frames)}/30", "frames": 30})

    sequence = np.asarray(sequence_frames[-30:], dtype=np.float32)
    prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    top_indexes = prediction.argsort()[-5:][::-1]
    top5 = []
    for index in top_indexes:
        confidence = float(prediction[index])
        word = valid_actions[index] if index < len(valid_actions) else f"class_{index}"
        top5.append({"class": int(index), "word": str(word), "confidence": round(confidence * 100, 2)})

    best = top5[0]
    word = best["word"]
    confidence = best["confidence"]
    if confidence >= 18:
        prediction_history.append(word)
    final_word = word
    if prediction_history:
        voted = Counter(prediction_history).most_common(1)[0]
        if voted[1] >= 2:
            final_word = voted[0]

    video_info = VIDEO_DATABASE.get(str(final_word).upper())
    response = {
        "success": True, "completed": True, "prediction": word,
        "matched_word": final_word, "last_word": final_word,
        "confidence": confidence, "top5": top5,
        "category": video_info["category"] if video_info else "GENERAL",
        "video_found": str(final_word).upper() in VIDEO_DATABASE,
        "locked": True,
    }
    last_prediction = response
    waiting_for_next_word = True
    return jsonify(response)


@app.route("/reset_frames", methods=["POST"])
def reset_frames():
    global sequence_frames, waiting_for_next_word, last_prediction
    sequence_frames = []
    prediction_history.clear()
    waiting_for_next_word = False
    last_prediction = None
    return jsonify({"success": True})


@app.route("/search_word")
def search_word():
    word = request.args.get("word", "").strip().upper()
    word = WORD_MAP.get(word, word)
    if word in VIDEO_DATABASE:
        return jsonify({"success": True, "word": word})
    matches = get_close_matches(word, VIDEO_DATABASE.keys(), n=1, cutoff=0.4)
    if matches:
        return jsonify({"success": True, "word": matches[0]})
    # If the requested class exists in the model but the video library is not uploaded,
    # return a useful failure instead of inventing a video path.
    return jsonify({"success": False, "error": "Not found"})


@app.route("/video")
def video():
    word = request.args.get("word", "").strip().upper()
    word = WORD_MAP.get(word, word)
    if word not in VIDEO_DATABASE:
        matches = get_close_matches(word, VIDEO_DATABASE.keys(), n=1, cutoff=0.3)
        if matches:
            word = matches[0]
        else:
            return "Video not found", 404
    path = Path(VIDEO_DATABASE[word]["path"])
    return send_from_directory(str(path.parent), path.name, conditional=True)


@app.route("/gemini", methods=["POST"])
def gemini_translate():
    data = request.get_json(silent=True) or {}
    words = data.get("words", [])
    sentence = " ".join(words) if words else ""
    try:
        if client and words:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Translate these sign-language gloss words into a natural Persian sentence: {words}",
            )
            if response.text:
                sentence = response.text.strip()
        return jsonify({"success": True, "sentence": sentence})
    except Exception as exc:
        app.logger.warning("Gemini request failed: %s", exc)
        return jsonify({"success": False, "sentence": sentence})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
