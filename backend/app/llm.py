"""Gemini helpers — strictly optional accelerators around CLIP, never a
dependency of it.

Two jobs, both aimed at the text-query path (image↔image CLIP already separates
real matches at ~0.73 from unrelated ones at ~0.55 and needs no help):

  translate_to_english() — CLIP's text tower is English-only, so a Thai query
      scores in the noise band. The built-in dictionary in embeddings.py fixes
      the words we predicted; this fixes the ones we didn't.
  rerank() — CLIP text↔image recalls well but ranks badly (it put a tablet top
      for "black wallet"). The model looks at the actual crops and rejects them.

Every entry point returns None on any failure, and callers must treat that as
"carry on without me" — a dead API key or dead booth wifi degrades search back
to plain CLIP rather than breaking it.
"""
from __future__ import annotations

import base64
import json
import logging

import httpx

from . import config

log = logging.getLogger("uvicorn.error")

_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_TRANSLATE_PROMPT = """You rewrite Thai lost-and-found descriptions into a short English phrase for a CLIP image-search query.
Keep only what is visible on the object: its type, colour, material, brand, pattern.
Drop places, dates, people's names, times, and anything about how it was lost.
Reply as JSON: {"english": "<short phrase>"}

Description: """

_RERANK_PROMPT = """You are checking which CCTV crops show the item someone lost. Each image below is preceded by its ID.
Score every ID from 0 to 100 for how likely it is the SAME KIND of object the person describes, judging type first, then colour and other visible details.
An image of a different kind of object must score under 20 even if the colour matches.
These crops come from CCTV and many are tiny, dark or blurry. If you cannot actually make out what the object is, score it under 30 — a dark rectangle is not evidence of a wallet. Never infer the object type from silhouette or colour alone; only score high when the image genuinely shows the item.
Reply as JSON: {"scores": [{"id": "<id>", "score": <0-100>}]}

The person lost: """


def enabled() -> bool:
    return bool(config.GEMINI_API_KEY)


def _generate(parts: list[dict]) -> str | None:
    """One JSON-mode call. Returns raw text, or None if anything at all went wrong."""
    if not enabled():
        return None
    try:
        r = httpx.post(
            _URL.format(model=config.GEMINI_MODEL),
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            timeout=config.GEMINI_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:  # noqa: BLE001 — any failure means "fall back", never raise
        log.warning("gemini %s failed: %r", config.GEMINI_MODEL, e)
        return None


def translate_to_english(text: str) -> str | None:
    if not text.strip() or not config.LLM_TRANSLATE:
        return None
    raw = _generate([{"text": _TRANSLATE_PROMPT + text}])
    if not raw:
        return None
    try:
        out = (json.loads(raw).get("english") or "").strip()
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None
    return out or None


def rerank(description: str, candidates: list[tuple[str, bytes]]) -> dict[str, int] | None:
    """Score crops against a description. `candidates` is [(id, jpeg_bytes)].

    Returns {id: 0-100} covering only the ids the model answered for; callers
    keep their CLIP order for anything missing rather than dropping it.
    """
    if not candidates or not config.LLM_RERANK:
        return None
    parts: list[dict] = [{"text": _RERANK_PROMPT + description}]
    for cid, blob in candidates:
        parts.append({"text": f"ID: {cid}"})
        parts.append({"inline_data": {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(blob).decode("ascii"),
        }})
    raw = _generate(parts)
    if not raw:
        return None
    try:
        rows = json.loads(raw).get("scores") or []
        return {str(r["id"]): int(r["score"]) for r in rows if "id" in r and "score" in r}
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError, KeyError):
        log.warning("gemini rerank: unparsable reply")
        return None
