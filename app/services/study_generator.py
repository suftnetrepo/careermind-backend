from openai import AsyncOpenAI
from app.config import get_settings
import asyncio
import json
import logging
import random

logger = logging.getLogger(__name__)
settings = get_settings()

QUIZ_SIZE = 25
QUIZ_MIN = 20
QUIZ_BATCH_SIZE = 5
# 5 easy, 15 medium, 5 hard — one batch per entry, generated in parallel.
# A single 25-question request is slow (~30s) and unreliable on count (it
# returned 8 and 24 in testing); batches of 5 come back complete in ~10s.
QUIZ_BATCHES = ["easy", "medium", "medium", "medium", "hard"]
OPTION_IDS = ["a", "b", "c", "d"]


def _schema(name: str, items_key: str, item_properties: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    items_key: {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": item_properties,
                            "required": list(item_properties),
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [items_key],
                "additionalProperties": False,
            },
        },
    }


# The model writes answers as text and we assign a–d after shuffling. When it
# lettered the options itself, the right answer was "b" 21 times out of 25 and
# explanations referred to letters, which made reordering unsafe.
QUIZ_SCHEMA = _schema("quiz", "questions", {
    "question":       {"type": "string"},
    "topic":          {"type": "string"},
    "difficulty":     {"type": "string", "enum": ["easy", "medium", "hard"]},
    "correct_answer": {"type": "string"},
    "wrong_answers":  {"type": "array", "items": {"type": "string"}},
    "explanation":    {"type": "string"},
})

FLASHCARD_SCHEMA = _schema("flashcards", "flashcards", {
    "front": {"type": "string"},
    "back":  {"type": "string"},
    "topic": {"type": "string"},
    "tip":   {"type": "string"},
})


def _to_quiz_item(raw: dict) -> dict | None:
    """Shuffle the answers into lettered options; None if the item is malformed."""
    answers = [raw["correct_answer"], *raw["wrong_answers"]]
    if len(raw["wrong_answers"]) != 3 or len({a.strip().lower() for a in answers}) != 4:
        return None
    random.shuffle(answers)
    options = [{"id": OPTION_IDS[i], "text": a} for i, a in enumerate(answers)]
    return {
        "question":    raw["question"],
        "topic":       raw["topic"],
        "difficulty":  raw["difficulty"],
        "options":     options,
        "correct":     OPTION_IDS[answers.index(raw["correct_answer"])],
        "explanation": raw["explanation"],
    }


async def _quiz_batch(
    client: AsyncOpenAI,
    role: str,
    level: str,
    difficulty: str,
    topics: list[str],
    questions_text: str,
) -> list[dict]:
    prompt = f"""You are an expert in {role} interviews at {level} level.

The candidate just completed an interview covering these topics:
{questions_text}

Generate exactly {QUIZ_BATCH_SIZE} multiple choice quiz questions, all at "{difficulty}" difficulty, to test the candidate's knowledge of these topics deeply.
Focus this set on: {', '.join(topics)}.

Rules:
- Questions must test real understanding, not just memorisation
- Give one correct answer and exactly 3 plausible wrong answers
- Only one answer may be correct
- Make all four answers similar in length and level of detail, so the correct one doesn't stand out
- Be specific to {role} at {level} level
- The explanation says why the correct answer is right and why the wrong answers are wrong,
  referring to them by what they say (answers are shuffled before display)"""

    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000,
                response_format=QUIZ_SCHEMA,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError(f"no content (refusal: {response.choices[0].message.refusal!r})")
            items = [q for q in map(_to_quiz_item, json.loads(content)["questions"]) if q]
            if items:
                return items[:QUIZ_BATCH_SIZE]
            raise ValueError("no valid questions in batch")
        except Exception:
            logger.exception("Quiz batch (%s) failed, attempt %d", difficulty, attempt + 1)
    return []


async def generate_quiz(
    role: str,
    level: str,
    questions: list[dict],
    transcript: list[dict] | None = None,
) -> list[dict]:
    """
    Generate 25 multiple choice quiz questions
    based on the interview questions and
    ideal answers.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    questions_text = "\n".join([
        f"- {q['question']} (topic: {q.get('topic', '')})"
        for q in questions
    ])
    # Give each batch its own share of the topics so batches don't repeat each other
    topics = list(dict.fromkeys(q.get("topic", "") for q in questions if q.get("topic"))) or [role]
    batch_topics = [
        [t for j, t in enumerate(topics) if j % len(QUIZ_BATCHES) == i] or topics
        for i in range(len(QUIZ_BATCHES))
    ]

    batches = await asyncio.gather(*[
        _quiz_batch(client, role, level, difficulty, batch_topics[i], questions_text)
        for i, difficulty in enumerate(QUIZ_BATCHES)
    ])
    questions_list = [q for batch in batches for q in batch]

    if len(questions_list) < QUIZ_MIN:
        raise ValueError(f"Only got {len(questions_list)} quiz questions")

    return questions_list[:QUIZ_SIZE]


async def generate_flashcards(
    role: str,
    level: str,
    questions: list[dict],
    cv_text: str | None = None,
) -> list[dict]:
    """
    Generate flashcards from interview
    questions with ideal answers on the back.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    questions_text = "\n".join([
        f"Q: {q['question']}\n"
        f"Topic: {q.get('topic', '')}\n"
        f"Keywords: {', '.join(q.get('ideal_keywords', []))}"
        for q in questions
    ])

    prompt = f"""You are an expert interview coach for {role} at {level} level.

Create flashcards for these interview questions — exactly one flashcard per question, in the same order.
Each flashcard front is the interview question.
Each flashcard back is the ideal answer — clear, structured, specific, what a strong candidate would say: 3-5 sentences, with a concrete example where relevant.
Each flashcard has a topic area and one tip: the key thing to remember when answering.

Some questions ask about the candidate's own past work. For those, never invent their projects, employers or results.
{"Use only facts from their CV below; where the CV gives no detail, show the structure with placeholders like [your result]." if cv_text else "Show a strong answer structure with placeholders like [your project] and [your result] instead of made-up specifics."}

Interview questions:
{questions_text}
{f"""
CANDIDATE CV:
{cv_text[:4000]}""" if cv_text else ""}"""

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=4000,
        response_format=FLASHCARD_SCHEMA,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"no flashcard content (refusal: {response.choices[0].message.refusal!r})")
    flashcards = json.loads(content)["flashcards"]
    if not flashcards:
        raise ValueError("no flashcards returned")
    return flashcards
