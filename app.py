"""
app.py
-------
Streamlit web app for the Fake News Detector.

USAGE:
    streamlit run app.py

Requires models/vectorizer.joblib and models/model.joblib to exist.
Run `python train_model.py` first if they don't.
"""

import os
import re
import string
import joblib
import streamlit as st

MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.caption("TF-IDF + classical ML (Logistic Regression / Passive Aggressive)")

if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH)):
    st.error(
        "No trained model found. Run `python train_model.py` first to train and save the model, "
        "then restart this app."
    )
    st.stop()

vectorizer, model = load_artifacts()

with st.sidebar:
    st.header("About")
    st.write(
        "This demo classifies news article text as **REAL** or **FAKE** using a "
        "TF-IDF vectorizer and a classical ML model trained in `train_model.py`."
    )
    st.write(
        "⚠️ This is trained on a small demo dataset for pipeline testing. "
        "For real-world accuracy, retrain on a large labeled dataset "
        "(e.g. the Kaggle 'Fake and Real News Dataset') by placing it at "
        "`data/train.csv` and re-running `python train_model.py`."
    )

title = st.text_input("Headline (optional)", placeholder="e.g. Local council approves new budget")
text = st.text_area(
    "Article text",
    height=220,
    placeholder="Paste the news article text here...",
)

if st.button("Check", type="primary"):
    combined = f"{title} {text}".strip()
    if not combined:
        st.warning("Please enter a headline or article text.")
    else:
        cleaned = clean_text(combined)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        label = "FAKE ⚠️" if pred == 1 else "REAL ✅"

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            confidence = proba[pred]
            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: **{confidence * 100:.1f}%**")
            st.progress(float(confidence))
        else:
            # Some models (e.g. SGDClassifier with hinge loss) don't support predict_proba
            st.subheader(f"Prediction: {label}")
            st.caption("Confidence score not available for this model type.")

        st.divider()
        st.caption(
            "Note: This tool flags patterns learned from training data — it is not a fact-checker "
            "and should not be treated as a definitive verdict on truthfulness."
        )
