# Question 2: Spam Email Detection

This project builds a text classification pipeline to detect spam emails using
TF-IDF vectorization and logistic regression.

## Problem

Classify emails as spam (1) or not spam (0) based on their text content.

**Evaluation metric**: ROC-AUC score (higher is better)

## Solution

- Clean email text by decoding quoted-printable MIME encoding
- Build ML pipeline: TF-IDF (unigrams + bigrams) → LogisticRegression
- Export continuous probability predictions for Class 1 (spam)

## Implementation details

- **Vectorizer**: TfidfVectorizer with max_features=20000
- **Classifier**: LogisticRegression with optimal hyperparameters
- **Output**: submission.csv with spam probabilities

## Files

- `submit.py`: Contains the main `main()` function
- `train.csv`: Training data with Text and Class columns
- `test.csv`: Test data for predictions
