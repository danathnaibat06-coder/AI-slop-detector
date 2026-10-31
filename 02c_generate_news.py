"""Generate AI news articles matching human_news.jsonl (free Groq API).

Usage: python 02c_generate_news.py --limit 100 --out ai_news.jsonl --model openai/gpt-oss-120b --model-name gpt-oss-120b
Resumable. Needs the environment variable GROQ_API_KEY.
"""
import argparse, json, os, random, sys, time
import requests

URL = "https://api.groq.com/openai/v1/chat/completions"
PROMPTS = [
    "Write a news article based on these highlights:\n{h}",
    "You are a journalist. Write a news report covering this story:\n{h}",
    "Write a news story that includes these key points:\n{h}",
]
SUFFIX = ("\n\nAim for about {n} words. Plain prose paragraphs only: no headline, no bullet "
          "points, no markdown, no preamble. Output only the article text.")

class Stop(Exception):
    pass

def generate(key, model, prompt, temp, max_tokens):
    for _ in range(6):
        try:
            r = requests.post(URL, timeout=120, headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "temperature": temp, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]})
        except requests.RequestException as e:
            print("  network error, retrying:", e); time.sleep(5); continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 20))
            if wait > 120:
                raise Stop("Daily free limit reached for this model. Try again later or use another model.")
            print(f"  rate limited, waiting {wait:.0f}s"); time.sleep(wait + 1); continue
        if r.status_code in (401, 403):
            raise Stop("Groq rejected your key. Check GROQ_API_KEY.")
        if r.status_code in (400, 404):
            raise Stop(f"Groq error {r.status_code}: {r.text[:300]}")
        print(f"  error {r.status_code}, retrying"); time.sleep(5)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="human_news.jsonl")
    ap.add_argument("--out", default="ai_news.jsonl")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--model-name", default="gpt-oss-120b")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit('GROQ_API_KEY is not set. Run:  $env:GROQ_API_KEY="gsk_your_key"')

    rows = [json.loads(l) for l in open(a.human, encoding="utf-8")]
    rng = random.Random(a.seed)
    plan = [(r, rng.randrange(len(PROMPTS)), round(rng.uniform(0.7, 1.0), 2)) for r in rows]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["human_id"] for l in open(a.out, encoding="utf-8")}
    print(f"{len(done)} already done, doing up to {a.limit} now")

    n = 0
    try:
        with open(a.out, "a", encoding="utf-8") as f:
            for r, pi, temp in plan:
                if r["id"] in done: continue
                if n >= a.limit: break
                prompt = PROMPTS[pi].format(h=r["title"]) + SUFFIX.format(n=r["n_words"])
                text = generate(key, a.model, prompt, temp, int(r["n_words"] * 2) + 2000)
                if not text:
                    print("  skipping", r["id"]); continue
                f.write(json.dumps({"human_id": r["id"], "text": text, "label": 1,
                                    "generator": a.model_name, "style": f"news{pi}",
                                    "temperature": temp}) + "\n")
                f.flush(); n += 1
                print(f"[{n}/{a.limit}] {r['id']}")
                time.sleep(a.delay)
    except Stop as e:
        print("\nStopped:", e)
    print(f"\nFinished this run: {n} new generations saved to {a.out}")

if __name__ == "__main__":
    main()
