"""Fetch human-written news (CNN/DailyMail dataset) as a 2nd human source.

Usage: python 01b_fetch_news.py --n-articles 300 --out human_news.jsonl
The story's 'highlights' (a short summary) are saved in the "title" field,
so the AI script can write a matching AI article about the same story.
"""
import argparse, json, re
from datasets import load_dataset

def clean(text):
    head = text[:150]
    if "(CNN)" in head:
        text = re.sub(r"^.*?\(CNN\)\s*(--|-)?\s*", "", text, count=1)
    return text.strip()

def first_chunk(text, lo, hi):
    sents = re.split(r"(?<=[.!?])\s+", text)
    cur, n = [], 0
    for s in sents:
        cur.append(s); n += len(s.split())
        if n >= lo:
            return " ".join(cur) if n <= hi else None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-articles", type=int, default=300)
    ap.add_argument("--min-words", type=int, default=150)
    ap.add_argument("--max-words", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="human_news.jsonl")
    a = ap.parse_args()

    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    ds = ds.shuffle(seed=a.seed, buffer_size=1000)

    kept = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            art = row["article"]
            if "PUBLISHED:" in art[:400] or "UPDATED:" in art[:400] or art.startswith("By ."):
                continue                      # messy Daily Mail headers
            chunk = first_chunk(clean(art), a.min_words, a.max_words)
            if not chunk:
                continue
            f.write(json.dumps({
                "id": f"news_{i}", "title": row["highlights"].replace("\n", " "),
                "text": chunk, "n_words": len(chunk.split()), "label": 0, "source": "cnn",
            }) + "\n")
            kept += 1
            if kept % 50 == 0:
                print(kept, "articles so far")
            if kept >= a.n_articles:
                break
    print(f"wrote {kept} news articles to {a.out}")

if __name__ == "__main__":
    main()
