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
