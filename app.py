"""
read.better MVP: PDF extract + optional OCR, TTS, PDF.js UI.

Run: .venv/bin/python readbetter-mvp/app.py
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

_here = Path(__file__).resolve().parent
for _env in (
    _here / ".env",
    _here.parent / "project1" / ".env",
    _here.parent / ".env",
):
    if _env.is_file():
        load_dotenv(_env)

_DIR = Path(__file__).resolve().parent
_PUBLIC = _DIR / "public"
# Root copies are required for Vercel: `public/` is CDN-only and not in the Python bundle.
# Prefer root (next to app.py), fall back to public/ for local setups.
_INDEX_HTML = _DIR / "index.html" if (_DIR / "index.html").is_file() else _PUBLIC / "index.html"
_FAVICON_SVG = _DIR / "favicon.svg" if (_DIR / "favicon.svg").is_file() else _PUBLIC / "favicon.svg"

# Vercel serverless request bodies are ~4.5MB on Hobby; larger uploads often fail with 500 before our code runs.
_default_upload_cap = 4 * 1024 * 1024 if os.environ.get("VERCEL") else 12 * 1024 * 1024
MAX_UPLOAD_BYTES = int(os.environ.get("READBETTER_MAX_UPLOAD_BYTES", str(_default_upload_cap)))
MAX_PAGES = 80
# Page text shorter than this (after strip) triggers OCR attempt
MIN_TEXT_CHARS = 48
# Cap OCR calls per upload (OpenAI vision / tesseract passes)
MAX_OCR_PAGES = int(os.environ.get("READBETTER_MAX_OCR_PAGES", "40"))
# When OPENAI_API_KEY is set, refine each page for TTS + sentence boundaries (set "0" to disable)
READBETTER_USE_LLM_REFINE = os.environ.get("READBETTER_USE_LLM_REFINE", "1")
# Max characters per page sent to the refiner (avoid huge prompts)
MAX_CHARS_PER_PAGE_REFINE = int(os.environ.get("READBETTER_MAX_CHARS_PER_PAGE_REFINE", "14000"))

REFINE_READALOUD_SYSTEM = """You segment one page of extracted PDF text for text-to-speech read-along (one page of a longer document). Your job is to preserve what is on the page, not to curate or summarize the document.

Return ONLY valid JSON (no markdown fences): {"sentences": ["...", ...]}

DEFAULT — READ THE PAGE: Treat everything in "Page N text" as spoken content unless it matches the tiny STRIP rules below. Include in reading order: titles, section and subsection headers, author and affiliation lines, abstract, every paragraph, parentheticals, lists, figure/table titles and captions, footnote and endnote lines if present, reference-list entries, journal metadata, copyright/boilerplate if it appears — all of it. Do not delete lines because they "look like" a running header, margin note, or navigation; false drops are worse than reading an extra repeated line.

STRIP ONLY (machine junk, not prose): Remove bare URL fragments (http, https, www., domain-only tails) from inside strings. Drop standalone lines that are only an email address or only a DOI/URL slug with no other words. Everything else stays.

SEGMENTATION: Use several strings with breaks at natural sentence boundaries. A period is not always sentence-end: abbreviations (Dr., U.S., U.K., e.g., i.e., Fig., Figs., p., pp., et al., vs., St., Jr., Sr., No., Vol., ed., eds., trans., ch., sec.), decimals (3.14), version strings (v1.0), and "City, ST" stay inside the same spoken unit. Do not add or invent words.

PAGE BREAKS: Layout page breaks are not sentence boundaries. If "Page N text" ends mid-clause, the last array item must not get a fake . ! ? REFERENCE ONLY blocks (before or after Page N text) are hints only — never copy them into JSON.

SOURCE OF TRUTH: Output strings must come only from "Page N text" after STRIP rules.

LISTS: Start a new string when a bullet or numbered item begins a new entry; do not attach a new section's first bullet to the prior paragraph.

EMPTY: Whitespace-only page → {"sentences": []}."""

# Short tail only: long snippets encourage the model to echo prior-page text into page N output.
_CONTINUATION_TAIL_CHARS = int(os.environ.get("READBETTER_CONTINUATION_TAIL_CHARS", "200"))
_CONTINUATION_HEAD_CHARS = int(os.environ.get("READBETTER_CONTINUATION_HEAD_CHARS", "220"))


def _strip_obvious_layout_noise(text: str) -> str:
    """
    Non-LLM path / LLM-fallback only. Same philosophy as the refiner: drop almost nothing.
    Removes only standalone pagination ticks and bare machine identifiers (no words to speak).
    """
    if not (text or "").strip():
        return text
    out_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            out_lines.append(line)
            continue
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if re.match(r"(?i)^page\s+\d+", s):
            continue
        if re.match(r"(?i)^page\s+\d+\s+of\s+\d+", s):
            continue
        if re.fullmatch(r"\d{1,4}\s*[/|]\s*\d{1,4}", s):
            continue
        if re.match(r"(?i)^doi:\s*\S+\s*$", s):
            continue
        if re.match(r"(?i)^https?://\S+\s*$", s):
            continue
        if re.fullmatch(r"\S+@\S+\.\S+", s):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _parse_json_sentences_from_llm(content: str) -> list[str]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    data = json.loads(text)
    arr = data.get("sentences")
    if not isinstance(arr, list):
        return []
    out: list[str] = []
    for x in arr:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _assign_norm_ranges(sentences: list[dict]) -> str:
    """Fill normStart/normEnd on each dict; return joined norm string."""
    norm = ""
    for s in sentences:
        t = s["text"].strip()
        if not t:
            continue
        if norm:
            norm += " "
        start = len(norm)
        norm += t
        s["normStart"] = start
        s["normEnd"] = len(norm)
    return norm


def _build_page_ranges_refined(sentences: list[dict], page_count: int) -> list[dict]:
    ranges: list[dict] = []
    for p in range(1, page_count + 1):
        rel = [s for s in sentences if s.get("startPage", 0) <= p <= s.get("endPage", 0)]
        if not rel:
            ranges.append({"pageNum": p, "normStart": 0, "normEnd": 0, "source": "text"})
            continue
        ranges.append(
            {
                "pageNum": p,
                "normStart": min(s["normStart"] for s in rel),
                "normEnd": max(s["normEnd"] for s in rel),
                "source": "text",
            }
        )
    return ranges


def _raw_page_tail_for_context(raw_page_text: str, max_chars: int = _CONTINUATION_TAIL_CHARS) -> str:
    """Last characters of a page (normalized whitespace) for the next page's LLM prompt."""
    chunk = re.sub(r"\s+", " ", (raw_page_text or "").strip())
    if len(chunk) <= max_chars:
        return chunk
    return chunk[-max_chars:]


def _raw_page_head_for_context(raw_page_text: str, max_chars: int = _CONTINUATION_HEAD_CHARS) -> str:
    """First characters of the next page (normalized) so the refiner can avoid a false full stop at page bottom."""
    chunk = re.sub(r"\s+", " ", (raw_page_text or "").strip())
    if len(chunk) <= max_chars:
        return chunk
    return chunk[:max_chars]


def _raw_page_end_lacks_sentence_terminus(raw_page_text: str) -> bool:
    """
    True if extracted page text does not end with . ! ? (PDF often cuts mid-sentence).
    Used to undo an LLM-added period when merging with the next page.
    """
    t = (raw_page_text or "").rstrip()
    if not t:
        return False
    while t and t[-1] in "\"'”’)]}>":
        t = t[:-1].rstrip()
    if not t:
        return False
    if t.endswith(("...", "…")):
        return False
    return t[-1] not in ".!?"


def _refine_page_for_readaloud(
    page_text: str,
    page_num: int,
    client: OpenAI,
    previous_page_tail: str | None = None,
    next_page_head: str | None = None,
) -> list[str]:
    chunk = page_text.strip()
    if not chunk:
        return []
    if len(chunk) > MAX_CHARS_PER_PAGE_REFINE:
        chunk = chunk[:MAX_CHARS_PER_PAGE_REFINE]

    user_parts: list[str] = []
    if previous_page_tail and page_num > 1:
        user_parts.append(
            f"REFERENCE ONLY — do not output any of this text in your JSON; it is not part of "
            f"page {page_num}. It shows how page {page_num - 1} ended so you can avoid a false "
            f"sentence break at the top of page {page_num} when the clause continues.\n\n"
            f"…{previous_page_tail}\n"
        )
    user_parts.append(f"Page {page_num} text (all sentences in your JSON must come from below):\n\n{chunk}")
    if next_page_head:
        user_parts.append(
            f"REFERENCE ONLY — start of page {page_num + 1} (do not output this text in JSON). "
            "If the Page N text above ends mid-clause and this snippet clearly continues the same "
            "sentence (often lowercase or a number continuing a list), your LAST array item for "
            "this page must NOT end with . ! ? — do not add a full stop at the page break.\n\n"
            f"…{next_page_head}\n"
        )
    user_content = "\n\n".join(user_parts)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.15,
        messages=[
            {"role": "system", "content": REFINE_READALOUD_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        max_tokens=8192,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_json_sentences_from_llm(raw)


def _looks_like_sentence_terminus(text: str) -> bool:
    """True if trailing text likely ends a complete sentence (do not merge with next page)."""
    t = text.rstrip()
    if not t:
        return False
    while t and t[-1] in "\"'”’)]}>":
        t = t[:-1].rstrip()
    if not t:
        return False
    if t.endswith(("...", "…")):
        return True
    return t[-1] in ".!?"


def _merge_cross_page_sentence_fragments(
    sentences: list[dict],
    page_texts: list[str] | None = None,
) -> list[dict]:
    """
    Join segments that were split only because of a page boundary (LLM or regex added a false stop).
    """
    if len(sentences) < 2:
        return sentences
    out: list[dict] = []
    acc = dict(sentences[0])
    for nxt in sentences[1:]:
        merge = False
        strip_trailing_period = False
        if acc.get("endPage", 0) and nxt.get("startPage", 0) == acc["endPage"] + 1:
            a = acc["text"].rstrip()
            b = nxt["text"].lstrip()
            ep = acc["endPage"]
            raw_prev = (
                page_texts[ep - 1]
                if page_texts and 1 <= ep <= len(page_texts)
                else ""
            )
            if a.endswith("-") and b and b[0].isalpha():
                merge = True
            elif (
                a
                and b
                and not _looks_like_sentence_terminus(a)
                and (b[0].islower() or b[0].isdigit())
            ):
                merge = True
            elif a and b and a[-1] in ",:;" and b and b[0].islower():
                merge = True
            elif (
                raw_prev
                and a.endswith(".")
                and not a.endswith("...")
                and b
                and (b[0].islower() or b[0].isdigit())
                and _raw_page_end_lacks_sentence_terminus(raw_prev)
            ):
                merge = True
                strip_trailing_period = True
        if merge:
            a = acc["text"].rstrip()
            b = nxt["text"].lstrip()
            if a.endswith("-") and b and b[0].isalpha():
                merged_text = a[:-1].rstrip() + b
            elif strip_trailing_period and a.endswith("."):
                merged_text = a[:-1].rstrip() + (" " if b else "") + b
            else:
                merged_text = a + (" " if a and b else "") + b
            acc = {
                "text": merged_text,
                "startPage": acc["startPage"],
                "endPage": nxt["endPage"],
            }
        else:
            out.append(acc)
            acc = dict(nxt)
    out.append(acc)
    return out


def _refine_all_pages_readaloud(page_texts: list[str], client: OpenAI) -> list[dict]:
    """One LLM call per page; sentence units with page numbers; global norm ranges."""
    sentences_out: list[dict] = []
    prev_tail = ""
    n_pages = len(page_texts)
    for page_num, pt in enumerate(page_texts, start=1):
        if not (pt or "").strip():
            prev_tail = ""
            continue
        next_head: str | None = None
        if page_num < n_pages:
            nxt = page_texts[page_num]
            if (nxt or "").strip():
                next_head = _raw_page_head_for_context(nxt)
        try:
            parts = _refine_page_for_readaloud(
                pt,
                page_num,
                client,
                previous_page_tail=prev_tail or None,
                next_page_head=next_head,
            )
        except Exception:
            parts = []
        if not parts:
            norm_page = re.sub(r"\s+", " ", _strip_obvious_layout_noise(pt)).strip()
            parts = _split_sentences(norm_page) if norm_page else []
        parts = _expand_overlong_sentences(parts)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            sentences_out.append(
                {
                    "text": part,
                    "startPage": page_num,
                    "endPage": page_num,
                }
            )
        prev_tail = _raw_page_tail_for_context(pt)
    return _merge_cross_page_sentence_fragments(sentences_out, page_texts)


def _norm_offset(raw_full: str, prefix_len: int) -> int:
    chunk = raw_full[:prefix_len]
    return len(re.sub(r"\s+", " ", chunk).strip())


def _split_sentences(norm: str) -> list[str]:
    text = re.sub(r"\s+", " ", norm).strip()
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for c in chunks:
        c = c.strip()
        if len(c) >= 2:
            out.append(c)
    if not out and text:
        return [text]
    return out


def _expand_overlong_sentences(parts: list[str], max_chars: int = 340) -> list[str]:
    """
    Split run-on segments (often page 1 from the LLM as one giant "sentence").
    Keeps short clauses intact; only expands when a segment exceeds max_chars.
    """
    out: list[str] = []
    for raw in parts:
        s = (raw or "").strip()
        if not s:
            continue
        if len(s) <= max_chars:
            out.append(s)
            continue
        sub = _split_sentences(s)
        if len(sub) > 1:
            out.extend(_expand_overlong_sentences(sub, max_chars))
            continue
        sub2 = re.split(r"(?<=[;])\s+|\n+", s)
        sub2 = [x.strip() for x in sub2 if len(x.strip()) >= 2]
        if len(sub2) > 1:
            out.extend(_expand_overlong_sentences(sub2, max_chars))
        else:
            out.append(s)
    return out


def _ocr_tesseract(pix: fitz.Pixmap) -> str:
    import io

    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img) or ""


def _ocr_openai(pix: fitz.Pixmap, client: OpenAI) -> str:
    b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe all visible text from this document page. "
                            "Preserve reading order (columns top-to-bottom). "
                            "Output plain text only, no commentary."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip()


def _page_text_with_optional_ocr(
    page: fitz.Page,
    page_index: int,
    ocr_budget: list[int],
    client: OpenAI | None,
) -> tuple[str, str]:
    """
    Returns (text, source) where source is 'text', 'ocr_tesseract', or 'ocr_openai'.
    """
    direct = page.get_text("text") or ""
    if len(direct.strip()) >= MIN_TEXT_CHARS:
        return direct, "text"

    if ocr_budget[0] <= 0:
        return direct, "text"

    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    ocr_budget[0] -= 1

    try:
        t = _ocr_tesseract(pix).strip()
        if len(t) >= MIN_TEXT_CHARS // 2:
            return t, "ocr_tesseract"
    except Exception:
        pass

    if client and os.environ.get("OPENAI_API_KEY"):
        try:
            t = _ocr_openai(pix, client).strip()
            if t:
                return t, "ocr_openai"
        except Exception:
            pass

    return direct, "text"


def extract_document(data: bytes) -> dict:
    doc = fitz.open(stream=data, filetype="pdf")
    ocr_budget = [MAX_OCR_PAGES]
    client: OpenAI | None = None
    if os.environ.get("OPENAI_API_KEY"):
        try:
            client = OpenAI()
        except Exception:
            client = None

    page_texts: list[str] = []
    page_sources: list[str] = []
    ocr_used = False

    try:
        n = min(len(doc), MAX_PAGES)
        for i in range(n):
            page = doc[i]
            txt, src = _page_text_with_optional_ocr(page, i, ocr_budget, client)
            page_texts.append(txt)
            page_sources.append(src)
            if src.startswith("ocr"):
                ocr_used = True
    finally:
        doc.close()

    raw_full = "\n".join(page_texts)
    page_count = len(page_texts)

    # Per-page norm ranges in *extracted* raw space (for fallback / debugging)
    page_norm_ranges_raw: list[dict] = []
    pos = 0
    for i, pt in enumerate(page_texts):
        start_raw = pos
        pos += len(pt)
        if i < len(page_texts) - 1:
            pos += 1  # newline between pages
        ns = _norm_offset(raw_full, start_raw)
        ne = _norm_offset(raw_full, pos)
        page_norm_ranges_raw.append(
            {
                "pageNum": i + 1,
                "normStart": ns,
                "normEnd": ne,
                "source": page_sources[i],
            }
        )

    llm_refined = False
    sentences: list[dict] = []
    norm: str = ""

    use_refine = (
        client is not None
        and os.environ.get("OPENAI_API_KEY")
        and READBETTER_USE_LLM_REFINE != "0"
    )

    if use_refine:
        try:
            cand = _refine_all_pages_readaloud(page_texts, client)
            cand = [s for s in cand if (s.get("text") or "").strip()]
            if cand:
                sentences = cand
                norm = _assign_norm_ranges(sentences)
                llm_refined = True
        except Exception:
            sentences = []

    if not sentences:
        norm = re.sub(r"\s+", " ", _strip_obvious_layout_noise(raw_full)).strip()

        def page_for_norm_offset(off: int) -> int:
            for pr in page_norm_ranges_raw:
                if pr["normStart"] <= off < pr["normEnd"]:
                    return pr["pageNum"]
            if page_norm_ranges_raw:
                return page_norm_ranges_raw[-1]["pageNum"]
            return 1

        sentence_texts = _split_sentences(norm)
        cursor = 0
        for sent in sentence_texts:
            start = norm.find(sent, cursor)
            if start < 0:
                continue
            end = start + len(sent)
            sp = page_for_norm_offset(start)
            ep = page_for_norm_offset(max(start, end - 1))
            sentences.append(
                {
                    "text": sent,
                    "normStart": start,
                    "normEnd": end,
                    "startPage": sp,
                    "endPage": ep,
                }
            )
            cursor = end
        page_norm_ranges = page_norm_ranges_raw
    else:
        page_norm_ranges = _build_page_ranges_refined(sentences, page_count)
        for i, pr in enumerate(page_norm_ranges):
            if i < len(page_sources):
                pr["source"] = page_sources[i]

    return {
        "norm": norm,
        "raw": raw_full,
        "pages": [
            {
                "pageNum": pr["pageNum"],
                "normStart": pr["normStart"],
                "normEnd": pr["normEnd"],
                "source": pr["source"],
            }
            for pr in page_norm_ranges
        ],
        "sentences": sentences,
        "sentence_count": len(sentences),
        "ocr_used": ocr_used,
        "page_count": page_count,
        "llm_refined": llm_refined,
    }


async def homepage(_: Request) -> HTMLResponse:
    html = _INDEX_HTML.read_text(encoding="utf-8")
    return HTMLResponse(html)


async def favicon(_: Request) -> FileResponse:
    return FileResponse(_FAVICON_SVG, media_type="image/svg+xml")


async def extract_pdf(request: Request) -> JSONResponse:
    try:
        form = await request.form()
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {
                "error": (
                    "Upload could not be read. On Vercel, keep PDFs under ~4 MB on Hobby, "
                    f"or set READBETTER_MAX_UPLOAD_BYTES. Details: {e!s}"
                ),
            },
            status_code=413,
        )
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse({"error": "Missing file field `file`"}, status_code=400)
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "File too large (max 12MB)"}, status_code=413)
    if len(data) < 16:
        return JSONResponse({"error": "Empty or invalid PDF"}, status_code=400)
    try:
        payload = extract_document(data)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Could not read PDF: {e!s}"}, status_code=422)
    if not payload["sentences"]:
        return JSONResponse(
            {
                "error": "No readable text after extraction/OCR.",
                "sentences": [],
            },
            status_code=422,
        )
    return JSONResponse(payload)


MAX_TTS_CHARS = 4096


async def tts_speech(request: Request) -> Response:
    if not os.environ.get("OPENAI_API_KEY"):
        return JSONResponse({"error": "OPENAI_API_KEY not configured"}, status_code=503)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    # Verbatim transcript string — no rewriting so audio matches on-screen text (TTS engine may still interpret pronunciation).
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)
    if len(text) > MAX_TTS_CHARS:
        return JSONResponse({"error": f"Text too long (max {MAX_TTS_CHARS} chars)"}, status_code=413)
    voice = body.get("voice") or "alloy"
    if voice not in {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}:
        voice = "alloy"
    try:
        speed = float(body.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    speed = max(0.25, min(4.0, speed))

    client = OpenAI()
    try:
        speech = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=speed,
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=502)

    return Response(content=speech.content, media_type="audio/mpeg")


routes = [
    Route("/", homepage, methods=["GET"]),
    Route("/favicon.svg", favicon, methods=["GET"]),
    Route("/api/extract", extract_pdf, methods=["POST"]),
    Route("/api/tts", tts_speech, methods=["POST"]),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
