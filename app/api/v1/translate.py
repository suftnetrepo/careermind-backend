"""
Tranquis translation endpoints.

All endpoints require a valid JWT (get_current_user).
Text translation counts against the per-user daily quota (free: 10/day).
Voice (audio transcription) and camera (image OCR) are Pro-only and
currently always auth-checked but not quota-gated — the quota check
on /text is the primary gate for free users.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

import uuid as _uuid

from app.core.deps import get_current_user
from app.core.quota import check_and_increment_quota, get_quota_status
from app.db.engine import get_db
from app.db.models import User, TranslationHistory, PhrasebookEntry

router = APIRouter(prefix="/translate", tags=["translate"])


# ── Whisper hallucination filter ──────────────────────────────────────────────
WHISPER_HALLUCINATIONS = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by",
    "transcribed by",
    "www.",
    "http",
    "",
}


def is_hallucination(text: str) -> bool:
    t = text.strip().lower()
    if not t or len(t) < 3:
        return True
    for phrase in WHISPER_HALLUCINATIONS:
        if phrase in t:
            return True
    return False


# ── Text translation (quota-gated) ────────────────────────────────────────────

@router.post("/text")
async def translate_text(
    payload: dict,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Translate text via GPT-4o. Costs 1 daily quota credit for free users."""
    import openai

    text: str = (payload.get("text") or "").strip()
    source_lang: str = payload.get("source_lang", "auto")
    target_lang: str = payload.get("target_lang", "English")

    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    if len(text) > 5000:
        raise HTTPException(status_code=422, detail="text must be 5000 characters or fewer")

    # Quota check (increments count; raises 429 if exceeded)
    quota = await check_and_increment_quota(db, user)

    source_clause = (
        f"from {source_lang} " if source_lang and source_lang.lower() != "auto"
        else ""
    )
    prompt = (
        f"Translate the following text {source_clause}to {target_lang}. "
        "Return ONLY the translated text with no additional commentary, "
        "explanation, or punctuation outside the translation itself.\n\n"
        f"{text}"
    )

    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.3,
    )
    translated = response.choices[0].message.content or ""

    translated_text = translated.strip()

    # Save to history
    entry = TranslationHistory(
        id=_uuid.uuid4(),
        user_id=user.id,
        source_text=text,
        translated_text=translated_text,
        source_lang=source_lang,
        target_lang=target_lang,
        mode="text",
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)

    return {
        "id": str(entry.id),
        "translated_text": translated_text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "quota": quota,
    }


# ── Audio transcription (Pro feature, auth required) ─────────────────────────

@router.post("/transcribe")
async def transcribe_audio(
    payload: dict,
    user: User = Depends(get_current_user),
):
    """Transcribe audio via Whisper and filter hallucinations."""
    import openai, base64, tempfile, os

    audio_b64: str = payload.get("audio_base64", "")
    if not audio_b64:
        raise HTTPException(status_code=422, detail="audio_base64 is required")

    audio_bytes = base64.b64decode(audio_b64)
    suffix = ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        client = openai.AsyncOpenAI()
        with open(tmp_path, "rb") as f:
            result = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        text: str = result.text or ""
    finally:
        os.unlink(tmp_path)

    if is_hallucination(text):
        raise HTTPException(status_code=422, detail="No speech detected")

    return {"text": text}


# ── Camera OCR translation (Pro feature, auth required) ──────────────────────

@router.post("/image")
async def translate_image(
    payload: dict,
    user: User = Depends(get_current_user),
):
    """Extract and translate text from an image (GPT-4o Vision)."""
    import openai, base64

    image_b64: str = payload.get("image_base64", "")
    target_lang: str = payload.get("target_lang", "English")

    if not image_b64:
        raise HTTPException(status_code=422, detail="image_base64 is required")

    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Extract all text from this image and translate it to {target_lang}. "
                            "Return a JSON object with two fields: "
                            '"extracted_text" (the original text found in the image) and '
                            '"translated_text" (the translation). '
                            "If no text is found, set both to empty strings."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
                    },
                ],
            }
        ],
        max_tokens=1024,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    import json
    try:
        data = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        data = {"extracted_text": "", "translated_text": ""}

    return {
        "extracted_text": data.get("extracted_text", ""),
        "translated_text": data.get("translated_text", ""),
        "target_lang": target_lang,
    }


# ── TTS (auth required, no quota) ────────────────────────────────────────────

@router.post("/tts")
async def text_to_speech(
    payload: dict,
    user: User = Depends(get_current_user),
):
    """Convert translated text to speech via OpenAI TTS and return base64 MP3."""
    import openai, base64

    text: str = (payload.get("text") or "").strip()
    voice: str = payload.get("voice", "nova")  # alloy|echo|fable|onyx|nova|shimmer

    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    if len(text) > 4096:
        raise HTTPException(status_code=422, detail="text must be 4096 characters or fewer")

    client = openai.AsyncOpenAI()
    response = await client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
    )
    audio_bytes = response.read()
    return {"audio_base64": base64.b64encode(audio_bytes).decode()}


# ── Quota status (free endpoint, no increment) ────────────────────────────────

@router.get("/quota")
async def translation_quota(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's daily translation quota status."""
    return await get_quota_status(db, user)
