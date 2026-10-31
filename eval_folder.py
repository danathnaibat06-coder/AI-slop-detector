"""Score every .txt file in tests/human and tests/ai with detector.joblib.

Usage: python eval_folder.py
Put human-written texts in tests/human and AI-written texts in tests/ai.
"""
import glob, joblib

obj = joblib.load("detector.joblib")

if isinstance(obj, dict):
    vec = next((v for v in obj.values() if hasattr(v, "transform") and not hasattr(v, "predict_proba")), None)
    clf = next((v for v in obj.values() if hasattr(v, "predict_proba")), None)
    if clf is None:
        raise SystemExit("Couldn't find the model inside detector.joblib. Send me the error and train_detector.py.")
    predict_proba = (lambda t: clf.predict_proba(vec.transform(t))) if vec is not None else clf.predict_proba
    classes = list(clf.classes_)
else:
    predict_proba, classes = obj.predict_proba, list(obj.classes_)

def is_ai(c):
    return c == 1 or str(c).lower() in ("ai", "1", "true")
ai_idx = next(i for i, c in enumerate(classes) if is_ai(c))

total = correct = 0
for truth in ("human", "ai"):
    ok = n = 0
    for path in sorted(glob.glob(f"tests/{truth}/*.txt")):
        text = open(path, encoding="utf-8", errors="ignore").read()
        p_ai = predict_proba([text])[0][ai_idx]
        pred = "ai" if p_ai >= 0.5 else "human"
        n += 1; ok += pred == truth
        print(f"{'OK   ' if pred == truth else 'WRONG'} {path:25s} AI probability: {p_ai * 100:.0f}%")
    print(f"== {truth}: {ok}/{n} correct\n")
    total += n; correct += ok

if total:
    print(f"Overall: {correct}/{total} = {correct / total * 100:.0f}%")
else:
    print("No .txt files found in tests/human or tests/ai")
