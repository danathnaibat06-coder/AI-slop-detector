"""Clean, de-confound, and split into train/val/test.

Usage:
  python 03_build_dataset.py --human human.jsonl \
      --ai ai_haiku.jsonl ai_sonnet.jsonl ai_other.jsonl \
      --holdout-generator ai_other --out-dir dataset/

Splits are by ARTICLE TITLE (no topic leakage). One generator is held out
entirely and written to test_heldout_gen.jsonl to measure generalization.
"""
import argparse, hashlib, json, os, re

PREAMBLE = re.compile(r"^(certainly|sure|here(?:'s| is)|of course|absolutely)[^\n]*\n+", re.I)
PUNCT = str.maketrans({"\u2014": ", ", "\u2013": "-", "\u2018": "'", "\u2019": "'",
                       "\u201c": '"', "\u201d": '"', "\u2026": "..."})

def clean_ai(t):
    t = PREAMBLE.sub("", t.strip())
    t = re.sub(r"^#+\s.*$", "", t, flags=re.M)             # headings
    t = re.sub(r"(\*\*|__|\*|`)", "", t)                    # emphasis
    t = re.sub(r"^\s*([-*\u2022]|\d+\.)\s+", "", t, flags=re.M)  # bullets
    return re.sub(r"\n{3,}", "\n\n", t).strip()

def split_of(title):
    h = int(hashlib.md5(title.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--ai", nargs="+", required=True)
    ap.add_argument("--holdout-generator", required=True, help="substring of the ai filename to hold out")
    ap.add_argument("--normalize-punct", action=argparse.BooleanOptionalAction, default=True,
                    help="map em dashes/curly quotes to ASCII so the model can't key on them")
    ap.add_argument("--len-tol", type=float, default=0.3)
    ap.add_argument("--out-dir", default="dataset")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    norm = (lambda t: t.translate(PUNCT)) if a.normalize_punct else (lambda t: t)

    human = {}
    for l in open(a.human):
        r = json.loads(l); r["text"] = norm(r["text"]); human[r["id"]] = r

    out = {k: [] for k in ["train", "val", "test", "test_heldout_gen"]}
    seen = set(); dropped = 0
    for path in a.ai:
        held = a.holdout_generator in path
        for l in open(path):
            r = json.loads(l); h = human.get(r["human_id"])
            if not h: continue
            t = norm(clean_ai(r["text"])); n = len(t.split())
            key = hashlib.md5(t.lower().encode()).hexdigest()
            if key in seen or abs(n - h["n_words"]) > a.len_tol * h["n_words"] or n < 100:
                dropped += 1; continue                      # dup or length mismatch
            seen.add(key)
            ai = {"id": f"ai_{key[:10]}", "title": h["title"], "text": t, "label": 1,
                  "generator": r["generator"], "style": r["style"]}
            if held:
                out["test_heldout_gen"].append(ai)          # human side comes from test split
            else:
                out[split_of(h["title"])].append(ai)

    for h in human.values():
        row = {k: h[k] for k in ["id", "title", "text", "label"]}
        s = split_of(h["title"])
        out[s].append(row)
        if s == "test":
            out["test_heldout_gen"].append(row)

    for name, rows in out.items():
        with open(f"{a.out_dir}/{name}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        pos = sum(r["label"] for r in rows)
        print(f"{name}: {len(rows)} rows ({pos} AI / {len(rows)-pos} human)")
    print("dropped:", dropped)

if __name__ == "__main__":
    main()
