from openai import AsyncOpenAI
from app.config import get_settings
import asyncio
import json
import logging
import random

logger = logging.getLogger(__name__)
settings = get_settings()

# Models offered for quiz and flashcard generation — "Standard" and "Premium"
STUDY_MODELS = ("gpt-4o", "gpt-4o-mini")
DEFAULT_STUDY_MODEL = "gpt-4o"

QUIZ_SIZE = 25
QUIZ_MIN = 20
QUIZ_BATCHES = 5
# Each parallel batch writes 5 questions on its own share of the role's topics:
# 1 easy, 3 medium, 1 hard — 5/15/5 across the quiz. One 25-question request was
# slow (up to 30s) and unreliable on count (it returned 8 and 24 in testing).
BATCH_MIX = {"easy": 1, "medium": 3, "hard": 1}
WEAK_AREA_QUESTIONS = 3   # hard questions aimed at weak answers, one per batch
WEAK_AREA_ANGLES = [
    "the underlying concept",
    "how to implement it in practice",
    "debugging it or choosing between trade-offs",
]
DUPLICATE_OVERLAP = 0.7   # share of key words two questions must share to count as the same
DUPLICATE_MIN_WORDS = 3
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "is", "are", "be", "by",
    "with", "what", "which", "how", "why", "when", "does", "do", "you", "your", "would",
    "can", "could", "should", "following", "best", "describes", "true", "statement",
    "correct", "way", "use", "using", "used", "primary", "main", "purpose", "it", "its",
    "that", "this", "from", "most", "into", "an", "app", "application",
}

ROLE_TOPICS = {
    "React Native Developer": [
        "Core components (View, Text, ScrollView, FlatList, SectionList)",
        "Navigation (React Navigation, stack, tab, drawer)",
        "State management (useState, useReducer, Context, Redux, Zustand)",
        "Hooks (useEffect, useMemo, useCallback, useRef, custom hooks)",
        "Styling (StyleSheet, Flexbox, responsive design, platform-specific)",
        "Native modules and bridging",
        "Performance optimisation (memo, lazy loading, Hermes)",
        "Animations (Animated API, Reanimated, LayoutAnimation)",
        "Networking (fetch, Axios, REST, GraphQL)",
        "Storage (AsyncStorage, MMKV, SQLite)",
        "Push notifications (FCM, APNs, Expo)",
        "Testing (Jest, React Native Testing Library, Detox)",
        "Deployment (App Store, Play Store, CodePush, EAS)",
    ],
    "AI Engineer": [
        "RAG pipeline design and evaluation",
        "LLM fine-tuning vs prompt engineering",
        "Vector databases and embeddings",
        "MLOps and model deployment",
        "OpenAI API and tool use",
        "LangChain and LlamaIndex",
        "Evaluation metrics (RAGAS, BLEU)",
        "Context window management",
        "Agent architectures",
        "Safety and hallucination mitigation",
    ],
    "Python Developer": [
        "Core Python (decorators, generators, context managers, metaclasses)",
        "Async programming (asyncio, await, event loop)",
        "Testing (pytest, mocking, fixtures)",
        "FastAPI and REST API design",
        "Database (SQLAlchemy, Alembic, raw SQL)",
        "Performance and profiling",
        "Data structures and algorithms",
        "Type hints and mypy",
        "Packaging and virtual environments",
        "Concurrency and multiprocessing",
    ],
    "Full Stack Developer": [
        "React fundamentals and hooks",
        "Next.js (SSR, SSG, App Router)",
        "TypeScript",
        "REST API design",
        "Database design and SQL",
        "Authentication and security",
        "CSS and responsive design",
        "Testing (unit, integration, e2e)",
        "CI/CD and deployment",
        "Performance optimisation",
    ],
    "Frontend Developer": [
        "React hooks and lifecycle",
        "State management",
        "TypeScript",
        "CSS and responsive design",
        "Browser APIs and performance",
        "Testing",
        "Accessibility",
        "Build tools and bundlers",
        "Security (XSS, CSRF)",
        "Web vitals and optimisation",
    ],
    "Backend Developer": [
        "API design (REST, GraphQL)",
        "Database design and optimisation",
        "Authentication and authorisation",
        "Caching strategies",
        "Message queues",
        "Microservices architecture",
        "Security best practices",
        "Testing strategies",
        "Scalability and performance",
        "Containerisation and deployment",
    ],
    "Data Scientist": [
        "Statistics and probability",
        "Machine learning algorithms",
        "Feature engineering",
        "Model evaluation and validation",
        "pandas and numpy",
        "Scikit-learn",
        "Deep learning basics",
        "Data visualisation",
        "SQL for data analysis",
        "A/B testing",
    ],
    "DevOps Engineer": [
        "Docker and containerisation",
        "Kubernetes orchestration",
        "CI/CD pipelines",
        "Infrastructure as Code (Terraform)",
        "Cloud platforms (AWS/GCP/Azure)",
        "Monitoring and observability",
        "Security and secrets management",
        "Networking fundamentals",
        "Linux administration",
        "Incident response",
    ],
}

DEFAULT_TOPICS = [
    "Core concepts and fundamentals",
    "Best practices and patterns",
    "Problem solving and debugging",
    "Performance and scalability",
    "Testing and quality",
    "Security considerations",
    "Tools and ecosystem",
    "Architecture and design",
    "Communication and process",
    "Real-world application",
]
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
def _quiz_schema(topic_labels: list[str]) -> dict:
    # Topic is restricted to the role's list so the results screen can group by it
    return _schema("quiz", "questions", {
        "question":       {"type": "string"},
        "topic":          {"type": "string", "enum": topic_labels},
        "difficulty":     {"type": "string", "enum": ["easy", "medium", "hard"]},
        "correct_answer": {"type": "string"},
        "wrong_answers":  {"type": "array", "items": {"type": "string"}},
        "explanation":    {"type": "string"},
    })


def _topic_label(topic: str) -> str:
    """'Navigation (React Navigation, stack, tab, drawer)' -> 'Navigation'"""
    return topic.split("(")[0].strip()


def _key_words(question: str, role: str) -> set[str]:
    """Content words, singular, without filler or the role name — so 'schedules
    coroutines' and 'schedule coroutines' compare equal."""
    ignore = STOPWORDS | {w.lower() for w in role.split()}
    words = set()
    for w in question.lower().split():
        w = w.strip(".,?!:;'\"()`")
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w and w not in ignore:
            words.add(w)
    return words


def _is_duplicate(question: str, role: str, seen: list[set[str]]) -> bool:
    words = _key_words(question, role)
    for other in seen:
        smaller = min(len(words), len(other))
        if smaller >= DUPLICATE_MIN_WORDS and len(words & other) / smaller >= DUPLICATE_OVERLAP:
            return True
    seen.append(words)
    return False


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
    model: str,
    role: str,
    level: str,
    batch_topics: list[str],
    all_topics: list[str],
    mix: dict[str, int],
    weak_area: tuple[str, str] | None = None,
    avoid: list[str] | None = None,
) -> list[dict]:
    labels = [_topic_label(t) for t in all_topics]
    topics_text = "\n".join(f"- {_topic_label(t)}: {t}" for t in batch_topics)
    weak_text = f"""
WEAK AREA FROM THIS INTERVIEW — the candidate gave a short or weak answer here:
- {weak_area[0]}
Make one hard question target this weak area, testing {weak_area[1]}. Use the closest topic from this full list: {', '.join(labels)}.
""" if weak_area else ""
    avoid_text = (
        "\nDO NOT repeat or rephrase any of these existing questions:\n"
        + "\n".join(f"- {q}" for q in avoid) + "\n"
    ) if avoid else ""
    mix_text = ", ".join(f"{n} {d}" for d, n in mix.items() if n)

    prompt = f"""You are an expert technical interviewer specialising in {role} at {level} level.

Write {sum(mix.values())} multiple choice questions for a {level}-level {role} candidate: {mix_text}.

TOPICS FOR THIS SET (cover each of them):
{topics_text}
{weak_text}{avoid_text}
STRICT RULES:
1. Every question must be SPECIFIC to {role} — name the real APIs, libraries, tools and behaviours a {role} works with. No generic software engineering questions unless directly relevant to {role}
2. No duplicates — each question tests something different
3. "topic" must be the topic's short name from the list above
4. Give one unambiguously correct answer and exactly 3 plausible wrong answers that someone who knows {role} might pick
5. Make all four answers similar in length and level of detail, so the correct one doesn't stand out
6. The explanation says why the correct answer is right and why the wrong answers are wrong, referring to them by what they say (answers are shuffled before display)"""

    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=3000,
                response_format=_quiz_schema(labels),
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError(f"no content (refusal: {response.choices[0].message.refusal!r})")
            items = [q for q in map(_to_quiz_item, json.loads(content)["questions"]) if q]
            if items:
                return items
            raise ValueError("no valid questions in batch")
        except Exception:
            logger.exception("Quiz batch failed (%s), attempt %d", ", ".join(map(_topic_label, batch_topics)), attempt + 1)
    return []


async def generate_quiz(
    role: str,
    level: str,
    questions: list[dict],
    transcript: list[dict] | None = None,
    model: str = DEFAULT_STUDY_MODEL,
) -> list[dict]:
    """
    Generate 25 role-specific multiple choice questions covering every topic in
    ROLE_TOPICS for the role, with hard questions aimed at weak answers from the
    interview transcript.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    topics = ROLE_TOPICS.get(role, DEFAULT_TOPICS)
    weak_areas = find_weak_areas(transcript)

    # Split the topics across the batches so every topic is covered and
    # parallel batches never write about the same thing
    batch_topics = [topics[i::QUIZ_BATCHES] for i in range(QUIZ_BATCHES)]
    # Exactly WEAK_AREA_QUESTIONS batches aim their hard question at a weak area,
    # each from a different angle — giving every batch the same weak area
    # produced near-identical questions
    assignments = [
        (weak_areas[i % len(weak_areas)], WEAK_AREA_ANGLES[i]) if weak_areas and i < WEAK_AREA_QUESTIONS else None
        for i in range(QUIZ_BATCHES)
    ]
    batches = await asyncio.gather(*[
        _quiz_batch(client, model, role, level, bt, topics, BATCH_MIX, assignments[i])
        for i, bt in enumerate(batch_topics) if bt
    ])

    # Deduplicate by question text
    seen: list[set[str]] = []
    unique = [q for batch in batches for q in batch if not _is_duplicate(q["question"], role, seen)]

    # Top up anything lost to duplicates or short batches — the missing
    # difficulties, on the least-covered topics, avoiding existing questions
    missing = QUIZ_SIZE - len(unique)
    if missing > 0:
        have = {d: sum(q["difficulty"] == d for q in unique) for d in BATCH_MIX}
        want = {d: n * QUIZ_BATCHES for d, n in BATCH_MIX.items()}
        mix = {d: max(0, want[d] - have[d]) for d in BATCH_MIX}
        while sum(mix.values()) > missing:
            mix["medium" if mix["medium"] else max(mix, key=mix.get)] -= 1
        while sum(mix.values()) < missing:
            mix["medium"] += 1
        counts = {_topic_label(t): sum(q["topic"] == _topic_label(t) for q in unique) for t in topics}
        least = sorted(topics, key=lambda t: counts[_topic_label(t)])[:max(2, missing)]
        extra = await _quiz_batch(
            client, model, role, level, least, topics, mix, avoid=[q["question"] for q in unique],
        )
        unique += [q for q in extra if not _is_duplicate(q["question"], role, seen)]

    if len(unique) < QUIZ_MIN:
        raise ValueError(f"Only got {len(unique)} unique questions")

    return unique[:QUIZ_SIZE]


async def generate_flashcards(
    role: str,
    level: str,
    questions: list[dict],
    cv_text: str | None = None,
    model: str = DEFAULT_STUDY_MODEL,
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
        model=model,
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
- Strengths and improvements must be SPECIFIC to this interview, never generic — describe what the candidate actually did or said, without quoting it
- Give 2-3 strengths and 2-3 improvements; if there is genuinely nothing strong, give fewer strengths
- feedback: 1-2 sentences specific to what they said; improvement: one concrete actionable tip
- NEVER mention the candidate's employer names, company names, or specific organisations from their CV or answers
- NEVER quote the candidate's exact words in feedback, improvement, strengths or improvements ("evidence" is the only field that quotes them)
- Refer to experience generically: "your previous role", "a past project", "your experience" — never by company name
- Strengths and improvements must describe skills and behaviours, not specific employers or organisations"""

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
