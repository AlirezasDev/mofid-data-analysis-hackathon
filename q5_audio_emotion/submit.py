"""
Question 5: Audio Emotion Classification for ShEMO-like dataset

This module implements an audio prediction pipeline for emotion probabilities.
It extracts simple waveform features from WAV files, trains a multiclass
classifier, and writes a probability submission file for required emotion labels.
"""
# Implement by: Alireza Sepehri

import csv
import math
import wave
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline


TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")
SUBMISSION_FILE = Path("submission.csv")
EMOTIONS = ["A", "H", "N", "S", "W"]
LABEL_MAP = {label: label for label in EMOTIONS}


def parse_label_from_filename(filename: str) -> str:
    """Parse emotion label from filename according to dataset format."""
    stem = Path(filename).stem
    if len(stem) >= 2 and stem[1] in LABEL_MAP:
        return LABEL_MAP[stem[1]]
    raise ValueError(f"Invalid audio filename format: {filename}")


def parse_file_id_from_filename(filename: str) -> int:
    """Parse integer file_id from the numeric prefix of the filename."""
    stem = Path(filename).stem
    numeric_prefix = "".join(ch for ch in stem if ch.isdigit())
    if numeric_prefix:
        return int(numeric_prefix)
    raise ValueError(f"Invalid audio filename identifier: {filename}")


def load_waveform_features(path: Path) -> np.ndarray:
    """Extract basic waveform-based audio features from a WAV file."""
    with wave.open(str(path), "rb") as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        sample_rate = wav.getframerate()
        frames = wav.readframes(n_frames)

    if n_frames == 0 or sample_width not in (1, 2, 4):
        return np.zeros(10, dtype=float)

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sample_width]
    waveform = np.frombuffer(frames, dtype=dtype)

    if n_channels > 1:
        waveform = waveform.reshape(-1, n_channels).mean(axis=1)

    if sample_width == 1:
        waveform = waveform.astype(np.float32) - 128.0
    waveform = waveform.astype(np.float32)
    if np.max(np.abs(waveform)) > 0:
        waveform /= np.max(np.abs(waveform))

    duration = n_frames / sample_rate
    energy = np.mean(waveform ** 2)
    peak = np.max(np.abs(waveform))
    zcr = float(np.mean(np.abs(np.diff(np.sign(waveform)))) / 2.0)
    mean = float(np.mean(waveform))
    std = float(np.std(waveform))
    minimum = float(np.min(waveform))
    maximum = float(np.max(waveform))

    spec = np.abs(np.fft.rfft(waveform))
    spec_sum = np.sum(spec) + 1e-8
    spec_centroid = float(np.sum(np.arange(len(spec)) * spec) / spec_sum)
    spec_bandwidth = float(np.sqrt(np.sum(((np.arange(len(spec)) - spec_centroid) ** 2) * spec) / spec_sum))

    return np.array([
        duration,
        energy,
        peak,
        zcr,
        mean,
        std,
        minimum,
        maximum,
        spec_centroid,
        spec_bandwidth,
    ], dtype=float)


def load_audio_dataset(data_dir: Path, labeled: bool = True) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Load audio features and labels (or ids) from a directory of WAV files."""
    features = []
    labels = []
    file_ids = []

    audio_files = sorted(data_dir.glob("*.wav"), key=lambda p: parse_file_id_from_filename(p.name))
    for path in audio_files:
        features.append(load_waveform_features(path))
        file_ids.append(parse_file_id_from_filename(path.name))
        if labeled:
            labels.append(parse_label_from_filename(path.name))

    if labeled:
        return np.vstack(features), np.array(labels, dtype=str), file_ids
    return np.vstack(features), np.array(file_ids, dtype=int), file_ids


def build_model() -> Pipeline:
    """Build a standard scaler + logistic regression pipeline for multiclass probabilities."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    multi_class="multinomial",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate the model using stratified cross-validation and log loss."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = build_model()
    cv_scores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X, y_encoded), start=1):
        model.fit(X[train_idx], y_encoded[train_idx])
        proba = model.predict_proba(X[valid_idx])
        loss = -np.mean(np.log(np.clip(proba[np.arange(len(valid_idx)), y_encoded[valid_idx]], 1e-15, 1.0)))
        cv_scores.append(loss)
        print(f"Fold {fold} log loss: {loss:.5f}")

    mean_loss = float(np.mean(cv_scores))
    print(f"Average log loss: {mean_loss:.5f}")
    return mean_loss


def write_submission(file_ids: List[int], probabilities: np.ndarray, path: Path) -> None:
    """Write a submission CSV with probabilities for each emotion label."""
    output = pd.DataFrame(
        {
            "file_id": file_ids,
            "A": probabilities[:, EMOTIONS.index("A")],
            "H": probabilities[:, EMOTIONS.index("H")],
            "N": probabilities[:, EMOTIONS.index("N")],
            "S": probabilities[:, EMOTIONS.index("S")],
            "W": probabilities[:, EMOTIONS.index("W")],
        }
    )
    output.to_csv(path, index=False)


def main() -> None:
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            "Expected directories data/train and data/test with WAV files."
        )

    print("Loading training audio data...")
    X_train, y_train, _ = load_audio_dataset(TRAIN_DIR, labeled=True)
    print(f"Loaded {X_train.shape[0]} training samples.")

    print("Evaluating model performance with cross-validation...")
    evaluate_model(X_train, y_train)

    print("Training final model...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    model = build_model()
    model.fit(X_train, y_encoded)

    print("Loading test audio data...")
    X_test, _, file_ids = load_audio_dataset(TEST_DIR, labeled=False)
    print(f"Loaded {X_test.shape[0]} test samples.")

    print("Predicting probabilities for submission...")
    probabilities = model.predict_proba(X_test)

    print("Writing submission file...")
    write_submission(file_ids, probabilities, SUBMISSION_FILE)
    print(f"Created submission file: {SUBMISSION_FILE.resolve()}")


if __name__ == "__main__":
    main()
