"""
Question 3: Categorical Data Cleaning and Target Prediction

This module builds a professional machine learning pipeline for a fully
categorical dataset. It encodes binary, nominal, ordinal, and date-derived
features using category_encoders, manages missing values, trains a logistic
regression model, and exports probability predictions as submission.csv.
"""
# Implement by: Alireza Sepehri

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SUBMISSION_FILE = Path("submission.csv")
TARGET_COLUMN = "target"


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets from CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def identify_feature_groups(columns: List[str]) -> Dict[str, List[str]]:
    """Identify feature groups by prefix according to the problem statement."""
    groups = {
        "binary": [col for col in columns if col.startswith("bin")],
        "nominal": [col for col in columns if col.startswith("nom")],
        "ordinal": [col for col in columns if col.startswith("ord")],
        "day_month": [col for col in columns if col.startswith("day") or col.startswith("month")],
    }
    return groups


def clean_missing_values(df: pd.DataFrame, groups: Dict[str, List[str]]) -> pd.DataFrame:
    """Impute missing values for all categorical feature groups."""
    cleaned = df.copy()

    for column in groups["binary"] + groups["nominal"] + groups["ordinal"]:
        cleaned[column] = cleaned[column].fillna("missing").astype(str)

    for column in groups["day_month"]:
        cleaned[column] = cleaned[column].fillna("missing").astype(str)

    return cleaned


def encode_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    groups: Dict[str, List[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical features using category_encoders."""
    train_encoded = pd.DataFrame(index=train_df.index)
    valid_encoded = pd.DataFrame(index=valid_df.index)

    if groups["binary"]:
        binary_encoder = BinaryEncoder(cols=groups["binary"], return_df=True, handle_unknown="ignore")
        binary_encoder.fit(train_df)
        train_encoded = pd.concat([train_encoded, binary_encoder.transform(train_df)], axis=1)
        valid_encoded = pd.concat([valid_encoded, binary_encoder.transform(valid_df)], axis=1)

    if groups["nominal"]:
        nominal_encoder = OneHotEncoder(cols=groups["nominal"], use_cat_names=True, handle_unknown="ignore", return_df=True)
        nominal_encoder.fit(train_df)
        train_encoded = pd.concat([train_encoded, nominal_encoder.transform(train_df)], axis=1)
        valid_encoded = pd.concat([valid_encoded, nominal_encoder.transform(valid_df)], axis=1)

    if groups["ordinal"]:
        ordinal_encoder = OrdinalEncoder(cols=groups["ordinal"], return_df=True, handle_unknown="impute")
        ordinal_encoder.fit(train_df)
        train_encoded = pd.concat([train_encoded, ordinal_encoder.transform(train_df)], axis=1)
        valid_encoded = pd.concat([valid_encoded, ordinal_encoder.transform(valid_df)], axis=1)

    if groups["day_month"]:
        date_encoder = OneHotEncoder(cols=groups["day_month"], use_cat_names=True, handle_unknown="ignore", return_df=True)
        date_encoder.fit(train_df)
        train_encoded = pd.concat([train_encoded, date_encoder.transform(train_df)], axis=1)
        valid_encoded = pd.concat([valid_encoded, date_encoder.transform(valid_df)], axis=1)

    return train_encoded, valid_encoded


def train_model(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Train a logistic regression model on the encoded feature matrix."""
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=1.0)
    model.fit(X, y)
    return model


def evaluate_cv(X: pd.DataFrame, y: pd.Series) -> float:
    """Evaluate model stability with stratified cross-validation and ROC-AUC."""
    cv_scores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X, y), start=1):
        model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=1.0)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict_proba(X.iloc[valid_idx])[:, 1]
        score = roc_auc_score(y.iloc[valid_idx], predictions)
        cv_scores.append(score)
        print(f"Fold {fold} ROC-AUC: {score:.4f}")

    mean_score = float(np.mean(cv_scores))
    print(f"Average ROC-AUC: {mean_score:.4f}")
    return mean_score


def create_submission(model: LogisticRegression, features: pd.DataFrame) -> pd.DataFrame:
    """Generate a submission DataFrame with probability estimates for class 1."""
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame({TARGET_COLUMN: probabilities})


def main() -> None:
    print("Loading datasets...")
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)

    features = [col for col in train_df.columns if col != TARGET_COLUMN]
    groups = identify_feature_groups(features)

    print("Cleaning missing values...")
    train_clean = clean_missing_values(train_df[features], groups)
    test_clean = clean_missing_values(test_df[features], groups)

    print("Encoding categorical features...")
    train_encoded, test_encoded = encode_features(train_clean, test_clean, groups)

    print("Evaluating model with cross-validation...")
    evaluate_cv(train_encoded, train_df[TARGET_COLUMN])

    print("Training final model on all data...")
    model = train_model(train_encoded, train_df[TARGET_COLUMN])

    print("Generating submission file...")
    submission_df = create_submission(model, test_encoded)
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"submission.csv created successfully at: {SUBMISSION_FILE.resolve()}")


if __name__ == "__main__":
    main()
