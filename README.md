# AI slop detector

I wanted to see if a simple model could tell human writing from AI writing. It sort of can, but not reliably. This is a learning project, so don't use it to judge whether a real person wrote something.

## What I did

I collected human text from Wikipedia and old CNN news. Then I asked a few AI models (through Groq's free API) to write about the same topics, so the two piles match. I cleaned the text, trained a TF-IDF + logistic regression classifier, and tested it on texts it had never seen.

## How it went

On a test set from the same sources as the training data, it got 93%. That number looks good but doesn't mean much, because the test text looks just like the training text.

On a small set of outside texts (6 human, 3 AI), it got all 6 human ones right but only 1 of the 3 AI ones. Nine texts is tiny, so treat that as a rough hint. What it shows is that the model is decent at saying "this looks human" and weak at catching AI text from models or styles it hasn't seen.

## Limits

- The human text is only Wikipedia and news from 2007-2015, so other kinds of writing could get flagged wrongly.
- The AI text comes from just a couple of models.
- The outside test is very small.
- Some of the AI news articles mix up facts from different years.
- No AI detector is fully reliable. Treat the output as a weak hint, not proof.

## Running it

Set your Groq key as an environment variable. Don't put it in a file.

    $env:GROQ_API_KEY="your-key"

Then:

    pip install -r requirements.txt
    python 01_fetch_wikipedia.py --n-articles 800 --out human.jsonl
    python 01b_fetch_news.py --n-articles 300 --out human_news.jsonl
    python 02b_generate_free.py --limit 200 --out ai_groq.jsonl --model openai/gpt-oss-120b
    python 02c_generate_news.py --limit 100 --out ai_news.jsonl
    python normalize_text.py human.jsonl ai_groq.jsonl
    python train_detector.py
    python train_detector.py --predict "some text here"

The data and the trained model aren't in this repo, so you'll need to run those steps to build them.