"""Generate AI text matched to each human chunk (same title, same target length).

Two steps, because batch jobs are async:
  python 02_generate_ai.py submit  --human human.jsonl --model claude-haiku-5-5
  python 02_generate_ai.py collect --batch-id msgbatch_xxx --out ai_haiku.jsonl

Run it once per generator model. Keep at least one model OUT of training and
use it only as a held-out test set (see 03_build_dataset.py).
To add other generators (local vLLM, OpenRouter, etc.), write a function that
returns {id: text} and call it from `collect`'s output format below.
"""
import argparse, json, random, time
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

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

def submit(a):
    rng = random.Random(a.seed)
    client = anthropic.Anthropic()
    rows = [json.loads(l) for l in open(a.human)]
    reqs = []
    for r in rows:
        style = rng.choice(list(STYLES))
        temp = round(rng.uniform(0.7, 1.0), 2)
        prompt = STYLES[style].format(title=r["title"]) + SUFFIX.format(n=r["n_words"])
        reqs.append(Request(
            custom_id=f"{r['id']}|{style}|{temp}",
            params=MessageCreateParamsNonStreaming(
                model=a.model, max_tokens=int(r["n_words"] * 2.5) + 100,
                temperature=temp, messages=[{"role": "user", "content": prompt}]),
        ))
    batch = client.messages.batches.create(requests=reqs)
    print("batch id:", batch.id, "| requests:", len(reqs))

def collect(a):
    client = anthropic.Anthropic()
    while True:
        b = client.messages.batches.retrieve(a.batch_id)
        if b.processing_status == "ended":
            break
        print("status:", b.processing_status, b.request_counts)
        time.sleep(60)
    n = 0
    with open(a.out, "w") as f:
        for res in client.messages.batches.results(a.batch_id):
            if res.result.type != "succeeded":
                continue
            hid, style, temp = res.custom_id.split("|")
            text = "".join(bl.text for bl in res.result.message.content if bl.type == "text")
            f.write(json.dumps({
                "human_id": hid, "text": text, "label": 1, "generator": a.model_name or "unknown",
                "style": style, "temperature": float(temp),
            }) + "\n"); n += 1
    print(f"wrote {n} generations to {a.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("submit"); s.add_argument("--human", default="human.jsonl")
    s.add_argument("--model", required=True); s.add_argument("--seed", type=int, default=0)
    c = sp.add_parser("collect"); c.add_argument("--batch-id", required=True)
    c.add_argument("--out", required=True); c.add_argument("--model-name", default=None)
    a = ap.parse_args()
    submit(a) if a.cmd == "submit" else collect(a)
