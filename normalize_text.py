import json, sys

MAP = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": ", ",
                     "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                     "\u2026": "...", "\u00a0": " ", "\u202f": " "})

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:
        r["text"] = r["text"].translate(MAP)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(path, len(rows), "rows cleaned")
