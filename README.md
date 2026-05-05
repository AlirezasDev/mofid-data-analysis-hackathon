# Mofid Data Analysis Hackathon

A comprehensive machine learning and data science project covering five distinct problems
from the Mofid Data Analysis Hackathon. Each question is implemented as a professional,
production-grade solution following best practices in software engineering and machine learning.

## Project structure

```
mofid-data-analysis-hackathon/
├── q1_parallel_universes/      # Matrix transformation via gradient descent
├── q2_spam_detection/          # Email spam classification with NLP
├── q3_categorical_cleaning/    # Categorical feature encoding and prediction
├── q4_autocorrect/             # Spell correction with minimum edit distance
└── q5_audio_emotion/           # Audio emotion classification
```

## Questions overview

### Q1: Parallel Universes Transformation

**Problem**: Find an optimal transformation matrix R mapping universe X to universe Y.

**Solution**:
- Implement gradient descent to minimize Frobenius norm loss
- Find nearest neighbors using cosine similarity
- Output: Optimized transformation matrix R

**Key concepts**: Vectorized gradient computation, Frobenius norm, Cosine similarity

**Tech stack**: NumPy

---

### Q2: Spam Email Detection

**Problem**: Classify emails as spam or not spam using text content.

**Solution**:
- Clean and preprocess email text (decode MIME artifacts)
- Apply TF-IDF vectorization (unigrams + bigrams)
- Train logistic regression classifier
- Output: Probability predictions for test emails

**Key concepts**: NLP preprocessing, TF-IDF, Binary classification, ROC-AUC

**Tech stack**: Pandas, NumPy, scikit-learn

---

### Q3: Categorical Data Cleaning and Prediction

**Problem**: Predict a target variable from fully categorical features (binary, nominal, ordinal, date-derived).

**Solution**:
- Handle missing values appropriately
- Encode categorical features using category_encoders
- Train logistic regression classifier
- Output: Probability predictions with ROC-AUC evaluation

**Key concepts**: Categorical encoding, Missing value imputation, Feature types

**Tech stack**: Pandas, NumPy, scikit-learn, category-encoders

---

### Q4: Probabilistic AutoCorrect

**Problem**: Implement spelling correction using Shakespeare's text as vocabulary and frequency model.

**Solution**:
- Generate edit distance candidates (delete, replace, insert)
- Score candidates by minimum edit cost and corpus probability
- Output: Corrected word with lowest cost and highest frequency

**Key concepts**: Minimum edit distance, Text frequency modeling, Spell checking

**Tech stack**: Standard library (string, re, collections)

---

### Q5: Audio Emotion Classification

**Problem**: Classify emotions in audio recordings (anger, happiness, neutral, sadness, surprise).

**Solution**:
- Extract waveform features from WAV files (without external audio libraries)
- Train multiclass logistic regression
- Output: Probability predictions for each emotion class

**Key concepts**: Audio feature extraction, Multiclass classification, WAV file processing

**Tech stack**: NumPy, Pandas, scikit-learn

---

## Setup and usage

### Clone the repository

```bash
cd d:\Hackathons\mofid-data-analysis-hackathon
```

### Run a specific question

Each question has its own directory with a `README.md` and `requirements.txt`:

```bash
cd q1_parallel_universes
pip install -r requirements.txt
python submit.py

cd ../q2_spam_detection
pip install -r requirements.txt
python submit.py
```

And so on for q3, q4, q5.

---

## Implementation notes

- **Code quality**: All code follows PEP 8 style guidelines
- **Documentation**: Each function includes docstrings with parameter and return descriptions
- **No external constraints violated**: Q4 builds custom implementations without external libraries
- **Production-grade**: Proper error handling, type hints, and modular design

---

## Author

Implemented by: Alireza Sepehri

---

## Repository

[GitHub: mofid-data-analysis-hackathon](https://github.com/AlirezasDev/mofid-data-analysis-hackathon)
