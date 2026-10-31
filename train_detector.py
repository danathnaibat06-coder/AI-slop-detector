"""Train a simple human-vs-AI text classifier (TF-IDF + logistic regression).

Usage:
  pip install scikit-learn joblib
  python train_detector.py                      # train + show accuracy
  python train_detector.py --predict "some text here"   # test a text

Expects human.jsonl and ai_groq.jsonl in the same folder.
"""
import argparse, json, random, sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

MODEL_FILE = "detector.joblib"


def pick_text(row):
    """Find the text field: prefer 'text', else the longest string value."""
    if isinstance(row.get("text"), str):
        return row["text"]
    strings = [v for v in row.values() if isinstance(v, str)]
    return max(strings, key=len) if strings else ""


def load(path):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = pick_text(json.loads(line)).strip()
            if len(t) > 200:
                texts.append(t)
    return texts


def train():
    human = load("human.jsonl")
    ai = load("ai_groq.jsonl")
    print(f"human texts: {len(human)} | ai texts: {len(ai)}")

    n = min(len(human), len(ai))
    random.seed(42)
    random.shuffle(human)
    random.shuffle(ai)
    human, ai = human[:n], ai[:n]
    print(f"using {n} of each (balanced)")

    X = human + ai
    y = [0] * n + [1] * n  # 0 = human, 1 = AI
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    print(classification_report(y_te, pred, target_names=["human", "ai"]))
    joblib.dump(model, MODEL_FILE)
    print(f"saved model to {MODEL_FILE}")


def predict(text):
    model = joblib.load(MODEL_FILE)
    p_ai = model.predict_proba([text])[0][1]
    label = "AI" if p_ai >= 0.5 else "HUMAN"
    print(f"{label}  (AI probability: {p_ai:.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", help="classify this text using the saved model")
    args = ap.parse_args()
    if args.predict:
        predict(args.predict)
    else:
        train()
