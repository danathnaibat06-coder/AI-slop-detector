"""Sample Wikipedia articles and cut them into 150-400 word human chunks.

Usage: python 01_fetch_wikipedia.py --n-articles 20000 --out human.jsonl

Note: HF's wikimedia/wikipedia only ships 20231101.en. Wikipedia contamination
by AI text is small, but for a strictly pre-2022 corpus use a dump from
dumps.wikimedia.org (e.g. enwiki-20211201) and swap the loader below.
"""
import argparse, json, re
from datasets import load_dataset

CUT = re.compile(r"\n(References|External links|See also|Further reading|Notes|Bibliography)\s*\n.*", re.S)

def clean(text):
    text = CUT.sub("", text)
    text = re.sub(r"\[\d+\]", "", text)          # citation remnants
    return text

def chunk(text, lo, hi):
    chunks, cur, n = [], [], 0
    for p in (p.strip() for p in text.split("\n")):
        w = len(p.split())
        if w < 20:                                # headings, stubs, list items
            continue
        cur.append(p); n += w
        if n >= lo:
            if n <= hi:
                chunks.append("\n\n".join(cur))
            cur, n = [], 0
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-articles", type=int, default=20000)
    ap.add_argument("--chunks-per-article", type=int, default=2)
    ap.add_argument("--min-words", type=int, default=150)
    ap.add_argument("--max-words", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="human.jsonl")
    a = ap.parse_args()

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    ds = ds.shuffle(seed=a.seed, buffer_size=20000)

    kept = 0
    with open(a.out, "w") as f:
        for row in ds:
            chunks = chunk(clean(row["text"]), a.min_words, a.max_words)[: a.chunks_per_article]
            for i, c in enumerate(chunks):
                f.write(json.dumps({
                    "id": f"{row['id']}_{i}", "title": row["title"], "text": c,
                    "n_words": len(c.split()), "label": 0, "source": "wikipedia",
                }) + "\n")
            if chunks:
                kept += 1
            if kept >= a.n_articles:
                break
    print(f"wrote chunks from {kept} articles to {a.out}")

if __name__ == "__main__":
    main()
