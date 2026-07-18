"""
train_model.py
----------------
Trains a classical ML fake news detector using TF-IDF + Logistic Regression
(with Passive Aggressive Classifier as a comparison baseline).

USAGE:
    python train_model.py

DATA:
    By default this looks for a real dataset at data/train.csv with columns:
        title, text, label      (label values: REAL / FAKE, or 0/1)
    If that file is not found, it falls back to the small demo dataset at
    data/sample_news.csv so the pipeline can be tested end-to-end immediately.

    For a real project, download the "Fake and Real News Dataset" from Kaggle
    (search: Fake and Real News Dataset by Clement Bisaillon) and place the
    combined/labeled CSV at data/train.csv.

OUTPUT:
    models/vectorizer.joblib   -> fitted TF-IDF vectorizer
    models/model.joblib        -> best trained classifier
    models/metrics.txt          -> evaluation report
"""

import os
import re
import string
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = "data"
MODELS_DIR = "models"
REAL_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_news.csv")


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip URLs/punctuation/numbers/extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data() -> pd.DataFrame:
    if os.path.exists(REAL_DATA_PATH):
        print(f"Loading real dataset from {REAL_DATA_PATH} ...")
        df = pd.read_csv(REAL_DATA_PATH)
    else:
        print(f"No data/train.csv found. Falling back to demo dataset: {SAMPLE_DATA_PATH}")
        print("NOTE: This demo dataset is tiny and only meant to prove the pipeline works.")
        print("For real accuracy, download a proper dataset (e.g. Kaggle 'Fake and Real News Dataset')")
        print("and place it at data/train.csv with columns: title, text, label")
        df = pd.read_csv(SAMPLE_DATA_PATH)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Combine title + text if both exist
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    elif "title" in df.columns:
        df["content"] = df["title"].fillna("")
    else:
        raise ValueError("Dataset must contain a 'text' and/or 'title' column.")

    # Normalize labels to 0/1 (0 = REAL, 1 = FAKE)
    label_map = {"REAL": 0, "FAKE": 1, "TRUE": 0, "FALSE": 1, "0": 0, "1": 1}
    df["label"] = (
        df["label"].astype(str).str.strip().str.upper().map(label_map)
    )
    df = df.dropna(subset=["label", "content"])
    df["label"] = df["label"].astype(int)

    df["content"] = df["content"].apply(clean_text)
    return df[["content", "label"]]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=1, max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Passive Aggressive (SGD)": SGDClassifier(
            loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0, max_iter=1000, random_state=42
        ),
    }

    report_lines = []
    best_model = None
    best_name = None
    best_acc = -1

    for name, model in candidates.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        report_lines.append(f"\n=== {name} ===")
        report_lines.append(f"Accuracy: {acc:.4f}")
        report_lines.append("Classification report:\n" + classification_report(y_test, preds, target_names=["REAL", "FAKE"]))
        report_lines.append("Confusion matrix (rows=actual, cols=predicted, order=[REAL, FAKE]):\n" + str(confusion_matrix(y_test, preds)))
        print(f"{name}: accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "model.joblib"))

    with open(os.path.join(MODELS_DIR, "metrics.txt"), "w") as f:
        f.write(f"Best model: {best_name} (accuracy={best_acc:.4f})\n")
        f.write("\n".join(report_lines))

    print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")
    print(f"Saved vectorizer -> {MODELS_DIR}/vectorizer.joblib")
    print(f"Saved model      -> {MODELS_DIR}/model.joblib")
    print(f"Saved metrics    -> {MODELS_DIR}/metrics.txt")


if __name__ == "__main__":
    main()
