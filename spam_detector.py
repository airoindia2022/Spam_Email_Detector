"""
Spam Email / SMS Detector using Machine Learning (Naive Bayes)
---------------------------------------------------------------
Trains a text classifier on a sample dataset and lets you test
your own messages interactively.

Run:  python spam_detector.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def load_data(path="messages.csv"):
    df = pd.read_csv(path)
    return df["message"], df["label"]


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True)),
        ("classifier", MultinomialNB()),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Model trained successfully!")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.1f}%\n")
    print(classification_report(y_test, predictions))
    return model


def main():
    print("=" * 50)
    print("   SPAM MESSAGE DETECTOR - AI Project")
    print("=" * 50)

    X, y = load_data()
    model = train_model(X, y)

    print("\nType a message to check if it's SPAM or NOT SPAM.")
    print("Type 'quit' to exit.\n")

    while True:
        message = input("Enter message: ").strip()
        if message.lower() == "quit":
            print("Goodbye!")
            break
        if not message:
            continue
        prediction = model.predict([message])[0]
        proba = model.predict_proba([message]).max() * 100
        label = "SPAM 🚨" if prediction == "spam" else "NOT SPAM ✅"
        print(f"  -> {label}  (confidence: {proba:.1f}%)\n")


if __name__ == "__main__":
    main()
