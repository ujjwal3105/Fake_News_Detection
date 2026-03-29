import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import load_and_preprocess

def train():
    # ================= LOAD DATA =================
    df = load_and_preprocess('../data/fake_news_dataset.csv')

    print("✅ Dataset Loaded")
    print("Columns:", df.columns)
    print("Dataset size:", len(df))

    # ================= SAFETY CHECK =================
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    # Remove empty rows
    df = df.dropna()
    df = df[df['text'].str.strip() != ""]

    print("After cleaning, size:", len(df))

    # ================= FEATURES =================
    X = df['text']
    y = df['label']

    # Better vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),   # unigrams + bigrams
        stop_words='english'
    )

    X_vec = vectorizer.fit_transform(X)

    print("Vector shape:", X_vec.shape)

    # ================= TRAIN TEST SPLIT =================
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # ================= MODEL =================
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Accuracy check
    accuracy = model.score(X_test, y_test)
    print("✅ Model Accuracy:", accuracy)

    # ================= SAVE MODEL =================
    os.makedirs('../models', exist_ok=True)

    joblib.dump(model, '../models/fake_news_model.pkl')
    joblib.dump(vectorizer, '../models/vectorizer.pkl')

    print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    train()
