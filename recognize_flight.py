from pathlib import Path

import joblib
import librosa
import numpy as np


MODEL_PATH = "asr_model.pkl"
DEFAULT_AUDIO_DIR = Path("data/test_sequence")
SAMPLE_RATE = 16000

AVIATION_TO_LETTER = {
    "alpha": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E",
    "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I", "juliett": "J",
    "kilo": "K", "lima": "L", "mike": "M", "november": "N", "oscar": "O",
    "papa": "P", "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T",
    "uniform": "U", "victor": "V", "whiskey": "W", "xray": "X", "yankee": "Y",
    "zulu": "Z"
}

DIGIT_TO_SYMBOL = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}


def extract_features_from_signal(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.effects.trim(y, top_db=25)

    min_len = int(0.25 * sr)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))

    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    feature_vector = np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),          # 26
        np.mean(delta, axis=1), np.std(delta, axis=1),        # 26
        np.mean(delta2, axis=1), np.std(delta2, axis=1),      # 26
        np.mean(zcr, axis=1), np.std(zcr, axis=1),            # 2
        np.mean(centroid, axis=1), np.std(centroid, axis=1),  # 2
        np.mean(bandwidth, axis=1), np.std(bandwidth, axis=1),# 2
        np.mean(rolloff, axis=1), np.std(rolloff, axis=1),    # 2
    ])

    return feature_vector.astype(np.float32)
    return feature_vector.astype(np.float32)


def segment_audio(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    energy_threshold_ratio: float = 0.18,
    min_segment_duration: float = 0.18,
    min_silence_duration: float = 0.16
):
    """
    Split a long recording into separate speech segments based on RMS energy.
    """
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    threshold = energy_threshold_ratio * np.max(rms)
    speech_frames = rms > threshold

    segments = []
    start_frame = None

    for i, is_speech in enumerate(speech_frames):
        if is_speech and start_frame is None:
            start_frame = i
        elif not is_speech and start_frame is not None:
            end_frame = i

            start_sample = max(0, start_frame * hop_length)
            end_sample = min(len(y), end_frame * hop_length)
            duration = (end_sample - start_sample) / sr

            if duration >= min_segment_duration:
                segments.append((start_sample, end_sample))

            start_frame = None

    if start_frame is not None:
        start_sample = max(0, start_frame * hop_length)
        end_sample = len(y)
        duration = (end_sample - start_sample) / sr

        if duration >= min_segment_duration:
            segments.append((start_sample, end_sample))

    # merge segments if the pause between them is too short
    merged_segments = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            prev_start, prev_end = merged_segments[-1]
            curr_start, curr_end = seg

            silence = (curr_start - prev_end) / sr
            if silence < min_silence_duration:
                merged_segments[-1] = (prev_start, curr_end)
            else:
                merged_segments.append(seg)

    return merged_segments


def decode_token(token: str) -> str:
    if token in AVIATION_TO_LETTER:
        return AVIATION_TO_LETTER[token]
    if token in DIGIT_TO_SYMBOL:
        return DIGIT_TO_SYMBOL[token]
    return "?"


def predict_sequence(audio_path: Path):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=25)

    segments = segment_audio(y, sr)

    if not segments:
        print("Speech segments were not found.")
        return

    print(f"Segments found: {len(segments)}\n")

    predicted_tokens = []
    decoded_symbols = []

    for idx, (start, end) in enumerate(segments, start=1):
        segment = y[start:end]

        features = extract_features_from_signal(segment, sr)
        features_scaled = scaler.transform([features])

        probs = model.predict_proba(features_scaled)[0]
        best_idx = np.argmax(probs)
        token = model.classes_[best_idx]
        confidence = probs[best_idx]

        predicted_tokens.append(token)
        decoded_symbols.append(decode_token(token))

        print(f"Segment {idx}: {token:<10} | confidence = {confidence:.3f}")

    final_string = "".join(decoded_symbols)

    print("\nRecognized words:")
    print(" ".join(predicted_tokens))

    print("\nDecoded flight / board number:")
    print(final_string)


def main():
    if not DEFAULT_AUDIO_DIR.exists():
        print(f"Folder not found: {DEFAULT_AUDIO_DIR}")
        return

    wav_files = sorted(DEFAULT_AUDIO_DIR.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in: {DEFAULT_AUDIO_DIR}")
        return

    print("Available test files:")
    for i, wav_file in enumerate(wav_files, start=1):
        print(f"{i}. {wav_file.name}")

    try:
        choice = int(input("\nChoose file number: ").strip())
        audio_path = wav_files[choice - 1]
    except Exception:
        print("Invalid input.")
        return

    print(f"\nProcessing file: {audio_path.name}\n")
    predict_sequence(audio_path)


if __name__ == "__main__":
    main()