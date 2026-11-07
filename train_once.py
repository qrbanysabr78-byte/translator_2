"""Run once outside the Hugging Face Space to create the model artifacts.

Required environment:
  KAGGLE_API_TOKEN=<your Kaggle API token>

Outputs:
  gru_original_fine_tuned.keras
  class_names.npy
  SSL400_Data/ (source videos used by the web app's sample-video feature)
"""
import os
import glob
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, GRU, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "SSL400_Data"
TASK_PATH = ROOT / "hand_landmarker.task"
MODEL_PATH = ROOT / "gru_original_fine_tuned.keras"
CLASSES_PATH = ROOT / "class_names.npy"


def download_dataset():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Set KAGGLE_API_TOKEN in your environment before running this script.")
    if not DATASET_PATH.exists() or not any(DATASET_PATH.iterdir()):
        DATASET_PATH.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download", "-d",
            "yohanabhishek/ssl400-dynamic-sri-lankan-sign-language-dataset",
            "-p", str(ROOT)
        ], check=True)
        archives = sorted(ROOT.glob("*.zip"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not archives:
            raise RuntimeError("Kaggle download finished but no dataset ZIP was found.")
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(DATASET_PATH)


def extract_keypoints_from_video(video_path, landmarker, sequence_length=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
    sequence_data, current_frame, target_ptr = [], 0, 0
    last_valid = [0.0] * 126
    success = True
    while success and target_ptr < len(frame_indices):
        success, frame = cap.read()
        if not success:
            break
        if current_frame == frame_indices[target_ptr]:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect(mp_image)
            pts = []
            if results.hand_landmarks:
                for hand in results.hand_landmarks:
                    for lm in hand:
                        pts.extend([lm.x, lm.y, lm.z])
            while len(pts) < 126:
                pts.extend([0.0, 0.0, 0.0])
            last_valid = pts[:126]
            sequence_data.append(last_valid)
            target_ptr += 1
        current_frame += 1
    cap.release()
    if len(sequence_data) == sequence_length:
        return np.asarray(sequence_data, dtype=np.float32)
    return None


def main():
    download_dataset()
    if not TASK_PATH.exists():
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, TASK_PATH)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    landmarker = HandLandmarker.create_from_options(options)
    all_videos = glob.glob(str(DATASET_PATH / "**" / "*.mp4"), recursive=True)
    if not all_videos:
        raise RuntimeError(f"No MP4 videos found under {DATASET_PATH}")

    actions = sorted({Path(path).parent.name.strip().upper() for path in all_videos})
    label_map = {label: idx for idx, label in enumerate(actions)}
    sequences, labels = [], []
    print(f"Extracting 30-frame, 126-feature samples from {len(all_videos)} videos...")
    for video_path in tqdm(all_videos):
        label = Path(video_path).parent.name.strip().upper()
        try:
            sequence = extract_keypoints_from_video(video_path, landmarker, 30)
            if sequence is not None:
                sequences.append(sequence)
                labels.append(label_map[label])
        except Exception as exc:
            print("Skipped", video_path, exc)

    landmarker.close()
    if not sequences:
        raise RuntimeError("No valid video sequences were extracted.")

    unique_classes, counts = np.unique(labels, return_counts=True)
    valid_classes = unique_classes[counts >= 2]
    kept = [(seq, lab) for seq, lab in zip(sequences, labels) if lab in valid_classes]
    old_to_new = {old: new for new, old in enumerate(sorted(valid_classes.tolist()))}
    X = np.asarray([item[0] for item in kept], dtype=np.float32)
    y_raw = np.asarray([old_to_new[item[1]] for item in kept], dtype=np.int64)
    valid_actions = [actions[i] for i in sorted(valid_classes.tolist())]
    np.save(ROOT / "X_data.npy", X)
    np.save(ROOT / "y_data.npy", y_raw)
    np.save(CLASSES_PATH, np.asarray(valid_actions, dtype=object))
    with open(ROOT / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({name: idx for idx, name in enumerate(valid_actions)}, f, ensure_ascii=False)

    y = to_categorical(y_raw, num_classes=len(valid_actions))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    train_labels = np.argmax(y_train, axis=1)
    class_weights_array = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = dict(enumerate(class_weights_array))

    model = Sequential([
        Bidirectional(GRU(128, return_sequences=True, activation="tanh"), input_shape=(X.shape[1], X.shape[2])),
        BatchNormalization(), Dropout(0.2),
        Bidirectional(GRU(64, return_sequences=False, activation="tanh")),
        BatchNormalization(), Dropout(0.2),
        Dense(256, activation="relu"), BatchNormalization(), Dropout(0.15),
        Dense(128, activation="relu"), Dropout(0.15),
        Dense(len(valid_actions), activation="softmax"),
    ])
    model.compile(
        optimizer=AdamW(learning_rate=5e-4, weight_decay=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"]
    )
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_accuracy", mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]
    model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=120,
        batch_size=16, class_weight=class_weights, shuffle=True, callbacks=callbacks,
    )
    print("Training complete. Artifacts:", MODEL_PATH, CLASSES_PATH)
    print("Keep SSL400_Data too if the app must serve the original sample videos.")


if __name__ == "__main__":
    main()
