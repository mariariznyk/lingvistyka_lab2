from pathlib import Path
import random
import tempfile
import time

import numpy as np
import pyttsx3
import soundfile as sf
from scipy.signal import resample


OUTPUT_DIR = Path("aviation_asr_dataset/train")
SAMPLE_RATE = 16000
SAMPLES_PER_WORD = 12

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


def change_volume(y: np.ndarray, min_gain: float = 0.85, max_gain: float = 1.1) -> np.ndarray:
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


def augment_audio(y: np.ndarray, sr: int) -> np.ndarray:
    y = trim_silence_simple(y, threshold=0.01)
    y = normalize_audio(y)

    y = change_speed(y, random.uniform(0.93, 1.07))
    y = change_volume(y, 0.85, 1.1)

    if random.random() < 0.8:
        y = add_noise(y, random.uniform(0.001, 0.004))

    y = add_silence(y, sr)
    y = normalize_audio(y)
    return y


def synthesize_one_file(text: str, out_path: Path, voice_id: str, rate: int) -> bool:
    """
    Самый важный фикс:
    создаём отдельный engine на КАЖДЫЙ файл.
    """
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


def generate_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voices = get_voices()

    if not voices:
        raise RuntimeError("No voices found in pyttsx3.")

    print("Available voices:")
    for i, v in enumerate(voices, start=1):
        print(f"{i}. {v.name} | {v.id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for word in ALL_WORDS:
            class_dir = OUTPUT_DIR / word
            class_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nGenerating: {word}")

            for i in range(SAMPLES_PER_WORD):
                voice = random.choice(voices)
                rate = random.randint(140, 185)

                raw_path = tmpdir / f"{word}_{i}_raw.wav"
                final_path = class_dir / f"{word}_{i+1:02d}.wav"

                ok = synthesize_one_file(word, raw_path, voice.id, rate)
                if not ok:
                    print(f"[SKIP] {word}_{i+1:02d}")
                    continue

                # небольшое ожидание, чтобы файл точно дописался
                wait_count = 0
                while (not raw_path.exists() or raw_path.stat().st_size == 0) and wait_count < 20:
                    time.sleep(0.1)
                    wait_count += 1

                if not raw_path.exists() or raw_path.stat().st_size == 0:
                    print(f"[SKIP] empty file: {raw_path.name}")
                    continue

                try:
                    y, sr = load_wav(raw_path)
                    y = augment_audio(y, sr)
                    sf.write(final_path, y, sr)
                    print(f"  OK -> {final_path.name} | {voice.name} | rate={rate}")
                except Exception as e:
                    print(f"[AUDIO ERROR] {raw_path.name}: {e}")

                try:
                    raw_path.unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    generate_dataset()