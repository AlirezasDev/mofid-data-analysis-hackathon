# Question 5: Audio Emotion Classification

This project implements an emotion prediction pipeline for audio recordings.
It extracts basic waveform features from WAV files, trains a multiclass classifier,
and generates `submission.csv` with probability estimates for each emotion.

## Data expectations

Place your audio files under:

- `q5_audio_emotion/data/train/`
- `q5_audio_emotion/data/test/`

Training filenames should use the format:

```
<id><gender><emotion>.wav
```

Examples:
- `1200FS.wav`  -> female sadness
- `0812AM.wav`  -> male anger

Emotion labels:
- `A` = anger
- `H` = happiness
- `N` = neutral
- `S` = sadness
- `W` = surprise

## How to run

```bash
cd q5_audio_emotion
python submit.py
```

The script will create `submission.csv` in the project directory.
