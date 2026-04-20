import os
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Settings
# -----------------------------
DATASET_DIR = Path("data/train")
MODEL_PATH = "asr_model.pkl"
SAMPLE_RATE = 16000

AVIATION_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliett", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu"
]

DIGIT_WORDS = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
]

ALL_CLASSES = AVIATION_WORDS + DIGIT_WORDS


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(file_path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract fixed-length feature vector from one audio file:
    MFCC mean/std + delta mean/std + zero crossing + spectral centroid.
    """
    y, sr = librosa.load(file_path, sr=sr)

    # Trim leading and trailing silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # If the signal is too short, pad it
    if len(y) < int(0.2 * sr):
        y = np.pad(y, (0, int(0.2 * sr) - len(y)))

    # Normalize amplitude
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Additional spectral features
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1),
        np.std(mfcc_delta2, axis=1),
        np.mean(zcr, axis=1),
        np.std(zcr, axis=1),
        np.mean(centroid, axis=1),
        np.std(centroid, axis=1),
        np.mean(bandwidth, axis=1),
        np.std(bandwidth, axis=1),
        np.mean(rolloff, axis=1),
        np.std(rolloff, axis=1),
    ])

    return features.astype(np.float32)


def load_dataset(dataset_dir: Path):
    X = []
    y = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for class_name in ALL_CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"[WARNING] Folder not found: {class_dir}")
            continue

        wav_files = list(class_dir.glob("*.wav"))
        if not wav_files:
            print(f"[WARNING] No WAV files in: {class_dir}")
            continue

        for wav_path in wav_files:
            try:
                features = extract_features(wav_path)
                X.append(features)
                y.append(class_name)
            except Exception as e:
                print(f"[ERROR] Failed to process {wav_path}: {e}")

    if not X:
        raise ValueError("No data loaded. Check your dataset folders and WAV files.")

    return np.array(X), np.array(y)


# -----------------------------
# Training
# -----------------------------
def main():
    print("Loading dataset...")
    X, y = load_dataset(DATASET_DIR)

    print(f"Total samples: {len(X)}")
    print(f"Feature vector size: {X.shape[1]}")
    print(f"Classes found: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=16,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42,
        early_stopping=False
    )

    print("Training model...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("\nTest accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "classes": ALL_CLASSES
    }, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()