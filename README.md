# Fake News Detection using Machine Learning

A classical ML pipeline that classifies news article text as **REAL** or **FAKE**,
using **TF-IDF** features and a **Logistic Regression** (with Passive Aggressive / SGD
as a comparison baseline), plus a simple **Streamlit** web app to try it interactively.

## Project structure

```
fake_news_detector/
├── data/
│   └── sample_news.csv     # small synthetic demo dataset (39 rows) — lets the pipeline run out of the box
├── models/                 # created after training: vectorizer.joblib, model.joblib, metrics.txt
├── train_model.py          # cleans data, trains models, saves the best one
├── app.py                  # Streamlit app for interactive predictions
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a real dataset (recommended before submitting/using this seriously)

The included `data/sample_news.csv` is a **tiny synthetic dataset** (39 rows) meant
only to prove the pipeline works end-to-end — it is NOT big enough for a real
project or accurate predictions.

For real results, download a proper labeled dataset, for example:

- **Kaggle: "Fake and Real News Dataset"** by Clément Bisaillon
  (search "Fake and Real News Dataset Kaggle" — contains `Fake.csv` and `True.csv`,
  ~44,000 articles total)
- **LIAR dataset** (short statements, multi-class truthfulness labels)

Combine/label the data into one CSV with columns `title`, `text`, `label`
(label values `REAL`/`FAKE`, or `0`/`1`), and save it as `data/train.csv`.
`train_model.py` automatically prefers `data/train.csv` over the demo file if present.

## 3. Train the model

```bash
python train_model.py
```

This will:
1. Load `data/train.csv` if present, otherwise fall back to the demo dataset
2. Clean the text (lowercase, strip URLs/HTML/punctuation/numbers)
3. Vectorize with TF-IDF (`max_features=5000`, English stop words removed)
4. Train Logistic Regression and a Passive Aggressive-style SGD classifier
5. Print accuracy, classification report, and confusion matrix for both
6. Save the best-performing model + vectorizer to `models/`

## 4. Run the web app

```bash
streamlit run app.py
```

Paste in a headline and/or article text and click **Check** to get a
REAL/FAKE prediction with a confidence score.

## How it works (for your report/presentation)

1. **Text cleaning** — lowercase, remove URLs, HTML tags, punctuation, and digits.
2. **Feature extraction (TF-IDF)** — converts cleaned text into numeric vectors
   where each word's weight reflects how important/distinctive it is to a
   document relative to the whole corpus.
3. **Classification** — Logistic Regression learns a decision boundary between
   REAL and FAKE based on those TF-IDF features. A Passive Aggressive/SGD model
   is trained alongside it for comparison; the better-performing one is saved.
4. **Evaluation** — accuracy, precision/recall/F1, and a confusion matrix are
   computed on a held-out 20% test split.
5. **Inference (the app)** — new text goes through the same cleaning +
   TF-IDF transform, then the saved model predicts REAL/FAKE with a
   confidence score.

## Limitations & next steps

- This is a **bag-of-words style** approach — it looks at word patterns, not
  actual fact-checking. It will pick up on *writing style* cues common in
  sensational/fake content, not verify claims against real facts.
- Accuracy is only as good as the training data. The bundled demo dataset is
  far too small/simple for production use — swap in a real dataset (see step 2).
- Possible extensions for a stronger project/report:
  - Add more classical models (Naive Bayes, SVM) and compare with cross-validation
  - Add n-grams (`ngram_range=(1,2)`) to the vectorizer
  - Try deep learning (LSTM) or transformer models (BERT/DistilBERT) for higher accuracy
  - Add a "confidence too low → uncertain" threshold instead of forcing REAL/FAKE
  - Deploy the Streamlit app publicly (Streamlit Community Cloud) for a live demo link
