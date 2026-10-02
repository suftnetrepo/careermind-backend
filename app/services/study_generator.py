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
SHORT_ANSWER_WORDS = 15
MAX_WEAK_AREAS = 5


def find_weak_areas(transcript: list[dict] | None) -> list[str]:
    """Short answers suggest the candidate struggled. Each one is paired with the
    question Alex asked, so the quiz knows which topic to target. The reply to
    Alex's opening greeting ("Yes, I'm ready") is skipped — it's always short."""
    weak_areas = []
    last_alex = None
    alex_turns = 0
    for line in transcript or []:
        text = (line.get("text") or "").strip()
        if line.get("role") == "alex":
            last_alex = text
            alex_turns += 1
        elif line.get("role") == "user" and text and alex_turns > 1:
            if len(text.split()) < SHORT_ANSWER_WORDS:
                weak_areas.append(f"Asked: '{last_alex}' — gave a very short answer: '{text}'")
    return weak_areas[:MAX_WEAK_AREAS]


def _weak_areas_text(weak_areas: list[str], difficulty: str) -> str:
    if not weak_areas:
        return ""
    # Each batch has one difficulty, so the targeting instruction depends on it
    instruction = {
        "hard":   "At least 3 of these 5 hard questions must target these weak areas.",
        "medium": "Include questions on these weak areas in this set.",
    }.get(difficulty, "")
    return f"""
CANDIDATE WEAK AREAS (from transcript):
The candidate gave short or weak answers here:
{chr(10).join(f'- {w}' for w in weak_areas)}
{instruction}
"""


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
    weak_areas: list[str],
) -> list[dict]:
    prompt = f"""You are an expert in {role} interviews at {level} level.

The candidate just completed an interview covering these topics:
{questions_text}

Generate exactly {QUIZ_BATCH_SIZE} multiple choice quiz questions, all at "{difficulty}" difficulty, to test the candidate's knowledge of these topics deeply.
Focus this set on: {', '.join(topics)}{" — but the weak areas below take priority" if weak_areas and difficulty == "hard" else ""}.
{_weak_areas_text(weak_areas, difficulty)}
Rules:
- Questions must test real understanding, not just memorisation
- If the candidate struggled on a topic (gave short answers), include more questions on that topic — especially at medium and hard difficulty
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

    weak_areas = find_weak_areas(transcript)

    batches = await asyncio.gather(*[
        _quiz_batch(client, role, level, difficulty, batch_topics[i], questions_text, weak_areas)
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


FEEDBACK_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "interview_feedback",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "overall_score":       {"type": "integer"},
                "technical_score":     {"type": "integer"},
                "communication_score": {"type": "integer"},
                "examples_score":      {"type": "integer"},
                "structure_score":     {"type": "integer"},
                "strengths":           {"type": "array", "items": {"type": "string"}},
                "improvements":        {"type": "array", "items": {"type": "string"}},
                "recommended_focus":   {"type": "string"},
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic":       {"type": "string"},
                            # Quoting first makes coverage decisions consistent run to run
                            "evidence":    {"type": "string"},
                            "covered":     {"type": "boolean"},
                            "score":       {"type": "integer"},
                            "feedback":    {"type": "string"},
                            "improvement": {"type": "string"},
                        },
                        "required": ["topic", "evidence", "covered", "score", "feedback", "improvement"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "overall_score", "technical_score", "communication_score", "examples_score",
                "structure_score", "strengths", "improvements", "recommended_focus", "questions",
            ],
            "additionalProperties": False,
        },
    },
}


def no_answers_feedback(questions: list[dict]) -> dict:
    """Feedback for an interview where the candidate never answered — no model call needed."""
    return {
        "overall_score": 0, "technical_score": 0, "communication_score": 0,
        "examples_score": 0, "structure_score": 0,
        "strengths": [],
        "improvements": ["No answers were recorded — check your microphone and try another interview."],
        "recommended_focus": "Complete an interview with spoken answers to get a score.",
        "questions": [
            {"topic": q.get("topic", ""), "evidence": "", "covered": False, "score": 0,
             "feedback": "Not reached in this interview.", "improvement": ""}
            for q in questions
        ],
    }


async def generate_feedback(
    role: str,
    level: str,
    questions: list[dict],
    transcript: list[dict],
) -> dict:
    """
    Score the candidate's interview answers
    using GPT-4o. Returns structured feedback
    with per-question scores and overall
    dimension scores.
    """
    if not any(t.get("role") == "user" and (t.get("text") or "").strip() for t in transcript):
        return no_answers_feedback(questions)

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Alex rephrases, reorders and adds follow-ups, so answers can't be matched to
    # planned questions by position — give the model the whole conversation
    plan_text = "\n\n".join(
        f"Question {i + 1}: {q.get('question', '')}\n"
        f"Topic: {q.get('topic', '')}\n"
        f"Type: {q.get('type', '')}\n"
        f"Expected keywords: {', '.join(q.get('ideal_keywords', []))}"
        for i, q in enumerate(questions)
    )
    conversation_text = "\n".join(
        f"{'Interviewer' if t.get('role') == 'alex' else 'Candidate'}: {t.get('text', '')}"
        for t in transcript
    )

    prompt = f"""You are an expert interviewer scoring a {level}-level {role} interview.

Score each answer honestly and specifically. Be fair but rigorous — this feedback will help the candidate improve.

PLANNED QUESTIONS (the interviewer worked through these, rephrasing them in conversation):
{plan_text}

FULL INTERVIEW TRANSCRIPT, IN ORDER:
{conversation_text}

Scoring guide:
90-100: Exceptional — clear, specific, with strong examples
75-89:  Strong — good understanding, minor gaps
60-74:  Adequate — basic understanding, lacks depth or examples
40-59:  Weak — vague or incomplete
0-39:   Poor — wrong, missing or very short answer

Rules:
- Score based on what they ACTUALLY said, not what they could have said
- The questions array must have exactly {len(questions)} items, one per planned question, in the same order
- Match answers to planned questions by meaning, including follow-up answers on the same topic
- For each question, first fill "evidence": quote the candidate's words that answer it (combine several turns if needed), or "" if they said nothing on it. Check every candidate turn against every question
- "covered": true if the interviewer asked that question (in any wording) OR the candidate gave an answer on that topic; false only if neither happened
- Score a volunteered answer on its merits, even if the interviewer didn't formally ask it
- A question that was asked but not answered, or answered with "I don't know", "not sure" or almost nothing, scores 0-20
- For a question that was not covered, set score to 0, feedback to "Not reached in this interview." and improvement to ""
- Do not penalise the dimension or overall scores for questions that were not covered
- overall_score should roughly match the average score of the covered questions
- Strengths and improvements must be SPECIFIC to this interview, never generic — reference actual things the candidate said
- Give 2-3 strengths and 2-3 improvements; if there is genuinely nothing strong, give fewer strengths
- feedback: 1-2 sentences specific to what they said; improvement: one concrete actionable tip"""

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4000,
        response_format=FEEDBACK_SCHEMA,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"no feedback content (refusal: {response.choices[0].message.refusal!r})")
    feedback = json.loads(content)

    # Keep the headline score consistent with the per-question scores the user sees
    covered = [q["score"] for q in feedback["questions"] if q["covered"]]
    if covered:
        feedback["overall_score"] = round(sum(covered) / len(covered))
    for key in ("overall_score", "technical_score", "communication_score", "examples_score", "structure_score"):
        feedback[key] = max(0, min(100, int(feedback[key])))
    return feedback
