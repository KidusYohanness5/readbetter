# readbetter (MVP)

PDF text extraction, optional OpenAI refine + TTS, and a browser reader with transcript.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY
python app.py
```

Open `http://127.0.0.1:8765` (or the port shown in the terminal).

## Env (optional)

- `OPENAI_API_KEY` — extraction refine, vision OCR fallback, and `/api/tts`
- `READBETTER_USE_LLM_REFINE=0` — disable per-page LLM refine
- `PORT` — default `8765`
- `READBETTER_MAX_UPLOAD_BYTES` — override max upload size (bytes)

## Deploy on Vercel

1. Import this repo; leave the root directory as the project root (same folder as `app.py`).
2. **Project → Settings → Environment variables:** add `OPENAI_API_KEY` (and any other vars you use).
3. **Hobby limits:** Vercel rejects request bodies around **4.5 MB**; on Vercel we default uploads to **4 MB** so `/api/extract` is less likely to fail with an opaque **500**. Use a smaller PDF or upgrade / set `READBETTER_MAX_UPLOAD_BYTES` if your plan allows larger bodies.
4. **`vercel.json`** sets Python function **`maxDuration` 60s** (PDF + LLM can be slow). On Hobby, if deploy fails, lower it in the dashboard to what your plan allows.
5. Static UI lives in **`public/`** (`index.html`, `favicon.svg`) so the CDN serves the page and `app.py` handles `/api/*`.
6. Browser console messages about **zustand** / **Radix Dialog** come from **Vercel’s preview toolbar**, not this app—disable the toolbar in Vercel settings or ignore them.
