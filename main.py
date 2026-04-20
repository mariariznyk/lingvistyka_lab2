from pathlib import Path
import random
import tempfile
import time

import numpy as np
import pyttsx3
import soundfile as sf
from scipy.signal import resample


OUTPUT_DIR = Path("aviation_asr_dataset/test_sequence")
LABELS_FILE = OUTPUT_DIR / "labels.txt"
SAMPLE_RATE = 16000
NUM_SEQUENCES = 10

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

ALL_WORDS = AVIATION_WORDS + DIGIT_WORDS


def get_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    try:
        engine.stop()
    except Exception:
        pass
    return voices


def normalize_audio(y: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y.astype(np.float32)


def ensure_mono(y: np.ndarray) -> np.ndarray:
    if len(y.shape) == 1:
        return y.astype(np.float32)
    return np.mean(y, axis=1).astype(np.float32)


def trim_silence_simple(y: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    idx = np.where(np.abs(y) > threshold)[0]
    if len(idx) == 0:
        return y
    return y[idx[0]: idx[-1] + 1]


def add_silence(y: np.ndarray, sr: int, min_ms: int = 80, max_ms: int = 220) -> np.ndarray:
    left_ms = random.randint(min_ms, max_ms)
    right_ms = random.randint(min_ms, max_ms)

    left_pad = np.zeros(int(sr * left_ms / 1000), dtype=np.float32)
    right_pad = np.zeros(int(sr * right_ms / 1000), dtype=np.float32)

    return np.concatenate([left_pad, y, right_pad])


def add_pause_between_words(sr: int, min_ms: int = 180, max_ms: int = 420) -> np.ndarray:
    pause_ms = random.randint(min_ms, max_ms)
    return np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)


def change_volume(y: np.ndarray, min_gain: float = 0.85, max_gain: float = 1.10) -> np.ndarray:
    gain = random.uniform(min_gain, max_gain)
    y = y * gain
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def change_speed(y: np.ndarray, speed_factor: float) -> np.ndarray:
    if abs(speed_factor - 1.0) < 1e-3:
        return y.astype(np.float32)
    new_length = max(1, int(len(y) / speed_factor))
    return resample(y, new_length).astype(np.float32)


def add_noise(y: np.ndarray, noise_level: float = 0.0025) -> np.ndarray:
    noise = np.random.normal(0, noise_level, len(y)).astype(np.float32)
    y = y + noise
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def load_wav(path: Path):
    y, sr = sf.read(str(path))
    y = ensure_mono(y)

    if sr != SAMPLE_RATE:
        new_length = int(len(y) * SAMPLE_RATE / sr)
        y = resample(y, new_length).astype(np.float32)
        sr = SAMPLE_RATE

    return y.astype(np.float32), sr


def augment_word_audio(y: np.ndarray, sr: int) -> np.ndarray:
    y = trim_silence_simple(y, threshold=0.01)
    y = normalize_audio(y)

    y = change_speed(y, random.uniform(0.93, 1.07))
    y = change_volume(y, 0.85, 1.10)

    if random.random() < 0.8:
        y = add_noise(y, random.uniform(0.001, 0.004))

    y = normalize_audio(y)
    return y


def synthesize_one_file(text: str, out_path: Path, voice_id: str, rate: int) -> bool:
    engine = None
    try:
        engine = pyttsx3.init()
        engine.setProperty("voice", voice_id)
        engine.setProperty("rate", rate)
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        time.sleep(0.15)
        return out_path.exists()
    except Exception as e:
        print(f"[TTS ERROR] {text} -> {e}")
        return False
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        time.sleep(0.1)


def generate_random_sequence():
    """
    Примеры:
    ['alpha', 'bravo', 'one', 'nine', 'zero']
    ['delta', 'mike', 'seven', 'two']
    ['xray', 'three', 'eight']
    """
    letters_count = random.randint(1, 3)
    digits_count = random.randint(1, 4)

    letters = random.sample(AVIATION_WORDS, letters_count)
    digits = [random.choice(DIGIT_WORDS) for _ in range(digits_count)]

    sequence = letters + digits

    if random.random() < 0.35:
        extra_count = random.randint(0, 2)
        for _ in range(extra_count):
            sequence.insert(random.randint(0, len(sequence)), random.choice(ALL_WORDS))

    return sequence


def build_sequence_audio(sequence_words, voices, tmpdir: Path):
    parts = []

    # Один голос и близкий темп на всю последовательность,
    # чтобы она звучала естественнее и была похожа на один "рейс"
    voice = random.choice(voices)
    base_rate = random.randint(140, 185)

    for idx, word in enumerate(sequence_words):
        raw_path = tmpdir / f"part_{idx}_{word}.wav"

        ok = synthesize_one_file(word, raw_path, voice.id, base_rate)
        if not ok:
            raise RuntimeError(f"Failed to synthesize word: {word}")

        wait_count = 0
        while (not raw_path.exists() or raw_path.stat().st_size == 0) and wait_count < 20:
            time.sleep(0.1)
            wait_count += 1

        if not raw_path.exists() or raw_path.stat().st_size == 0:
            raise RuntimeError(f"Empty synthesized file for word: {word}")

        y, sr = load_wav(raw_path)
        y = augment_word_audio(y, sr)
        parts.append(y)

        if idx < len(sequence_words) - 1:
            parts.append(add_pause_between_words(sr))

        try:
            raw_path.unlink()
        except OSError:
            pass

    if not parts:
        raise RuntimeError("No audio parts created.")

    full_audio = np.concatenate(parts)
    full_audio = add_silence(full_audio, SAMPLE_RATE, min_ms=100, max_ms=250)

    if random.random() < 0.7:
        full_audio = add_noise(full_audio, random.uniform(0.0008, 0.0025))

    full_audio = normalize_audio(full_audio)
    return full_audio, voice.name, base_rate


def generate_sequences():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voices = get_voices()

    if not voices:
        raise RuntimeError("No voices found in pyttsx3.")

    print("Available voices:")
    for i, v in enumerate(voices, start=1):
        print(f"{i}. {v.name} | {v.id}")

    label_lines = []

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        for i in range(1, NUM_SEQUENCES + 1):
            sequence_words = generate_random_sequence()
            out_path = OUTPUT_DIR / f"seq{i}.wav"

            try:
                audio, voice_name, rate = build_sequence_audio(sequence_words, voices, tmpdir)
                sf.write(out_path, audio, SAMPLE_RATE)

                text_line = " ".join(sequence_words)
                label_lines.append(f"{out_path.name}: {text_line}")

                print(f"OK -> {out_path.name}")
                print(f"     text  = {text_line}")
                print(f"     voice = {voice_name}")
                print(f"     rate  = {rate}")
            except Exception as e:
                print(f"[ERROR] seq{i}: {e}")

    LABELS_FILE.write_text("\n".join(label_lines), encoding="utf-8")
    print(f"\nSaved labels to: {LABELS_FILE}")


if __name__ == "__main__":
    generate_sequences()