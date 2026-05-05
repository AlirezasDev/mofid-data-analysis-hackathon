# Question 3: Categorical Data Cleaning and Target Prediction

This project implements a professional data preparation and machine learning
pipeline for a fully categorical dataset.

## Project overview

- Clean missing values for all categorical groups
- Encode binary, nominal, ordinal, and date-derived features
- Train a logistic regression model
- Generate `submission.csv` with probability predictions for class 1

## Expected dataset structure

Place the input data files in `q3_categorical_cleaning/data/`:

- `train.csv`
- `test.csv`

The training set must contain a `target` column.

## How to run

```bash
cd q3_categorical_cleaning
python submit.py
```

The script will produce `submission.csv` in the project root.
