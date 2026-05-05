"""
Question 2: Spam Email Detection
Implementation of an NLP pipeline using TF-IDF and Logistic Regression
to predict spam probabilities, optimized for ROC-AUC scoring.
"""
# Implement by: Alireza Sepehri

import re
import quopri
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def clean_email_text(text: str) -> str:
    """
    Cleans raw email text by decoding quoted-printable characters and removing extra spaces.
    
    Args:
        text (str): Raw email text which might contain MIME encoding.
        
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # Decode quoted-printable encoding (e.g., '=2E' -> '.', '=2C' -> ',')
    try:
        text = quopri.decodestring(text.encode('utf-8')).decode('utf-8', errors='ignore')
    except Exception:
        pass  # Fallback to original text if decoding fails
    
    # Remove excessive whitespaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    """
    Main function to load data, train the NLP model, and generate submission.csv.
    """
    print("Loading datasets...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    print("Cleaning text data...")
    train_df['Text'] = train_df['Text'].apply(clean_email_text)
    test_df['Text'] = test_df['Text'].apply(clean_email_text)

    # Define the Machine Learning Pipeline
    # TF-IDF converts text to numerical vectors, Logistic Regression predicts probabilities
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 2))),
        ('clf', LogisticRegression(C=2.0, max_iter=500, random_state=42, solver='liblinear'))
    ])

    print("Training the model...")
    # Train the pipeline on the training data
    pipeline.fit(train_df['Text'], train_df['Class'])

    print("Predicting probabilities for the test set...")
    # predict_proba returns an array of shape (n_samples, 2)
    # We take the 2nd column (index 1) which represents the probability of Class 1 (Spam)
    spam_probabilities = pipeline.predict_proba(test_df['Text'])[:, 1]

    print("Saving submission file...")
    # The output requires a single column with a header. 
    submission = pd.DataFrame({'spam_probability': spam_probabilities})
    submission.to_csv('submission.csv', index=False)
    print("submission.csv successfully generated!")

if __name__ == "__main__":
    main()
