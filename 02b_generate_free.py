"""Generate AI text with Groq's FREE API (one request per human chunk).

Usage:
  python 02b_generate_free.py --limit 100 --out ai_groq.jsonl

Safe to stop and re-run: it skips chunks it already finished and appends.
If you hit the free daily limit it stops cleanly; run it again tomorrow.
Needs the environment variable GROQ_API_KEY.
"""
import argparse, json, os, random, sys, time
import requests

URL = "https://api.groq.com/openai/v1/chat/completions"

STYLES = {
    "wiki_article": "Write the opening section of a Wikipedia article titled \"{title}\".",
    "wiki_intro":   "Write an encyclopedic introduction to the topic: {title}.",
    "explain":      "Explain the topic \"{title}\" to an educated general reader.",
    "overview":     "Write an informative overview of {title}, covering its history and significance.",
    "textbook":     "Write a textbook-style passage about {title}.",
    "essay":        "Write a short essay about {title}.",
}
SUFFIX = (" Aim for about {n} words. Write plain prose paragraphs only: no headings, "
          "no bullet points, no markdown, and no preamble or closing remarks. Output only the text.")

class Stop(Exception):
    pass

def generate(key, model, prompt, temp, max_tokens):
    for attempt in range(6):
        try:
            r = requests.post(URL, timeout=90,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "temperature": temp, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]})
        except requests.RequestException as e:
            print("  network error, retrying:", e); time.sleep(5); continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 20))
            if wait > 120:
                raise Stop("Daily free limit reached. Run the same command again later.")
            print(f"  rate limited, waiting {wait:.0f}s"); time.sleep(wait + 1); continue
        if r.status_code in (401, 403):
            raise Stop("Groq rejected your key. Check GROQ_API_KEY.")
        if r.status_code in (400, 404):
            raise Stop(f"Groq error {r.status_code}: {r.text[:300]}\n"
                       "If it mentions the model, pick a current one from console.groq.com/docs/models "
                       "and pass it with --model.")
        print(f"  error {r.status_code}, retrying"); time.sleep(5)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="human.jsonl")
    ap.add_argument("--out", default="ai_groq.jsonl")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--model-name", default="groq")
    ap.add_argument("--limit", type=int, default=100, help="max generations this run")
    ap.add_argument("--delay", type=float, default=2.5, help="seconds between requests")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit('GROQ_API_KEY is not set. Run:  $env:GROQ_API_KEY="gsk_your_key"')

    rows = [json.loads(l) for l in open(a.human, encoding="utf-8")]
    rng = random.Random(a.seed)
    plan = [(r, rng.choice(list(STYLES)), round(rng.uniform(0.7, 1.0), 2)) for r in rows]

    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["human_id"] for l in open(a.out, encoding="utf-8")}
    print(f"{len(done)} already done, {len(rows) - len(done)} remaining, doing up to {a.limit} now")

    n = 0
    try:
        with open(a.out, "a", encoding="utf-8") as f:
            for r, style, temp in plan:
                if r["id"] in done:
                    continue
                if n >= a.limit:
                    break
                prompt = STYLES[style].format(title=r["title"]) + SUFFIX.format(n=r["n_words"])
                text = generate(key, a.model, prompt, temp, int(r["n_words"] * 2) + 2000)
                if not text:
                    print("  skipping", r["title"]); continue
                f.write(json.dumps({"human_id": r["id"], "text": text, "label": 1,
                                    "generator": a.model_name, "style": style,
                                    "temperature": temp}) + "\n")
                f.flush(); n += 1
                print(f"[{n}/{a.limit}] {r['title']}")
                time.sleep(a.delay)
    except Stop as e:
        print("\nStopped:", e)
    print(f"\nFinished this run: {n} new generations saved to {a.out}")

if __name__ == "__main__":
    main()
