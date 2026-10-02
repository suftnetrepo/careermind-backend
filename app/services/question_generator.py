from fastapi import HTTPException
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings
import json
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

ROLES = {
    "AI Engineer":           "LLMs, RAG pipelines, vector search, MLOps, Python, OpenAI, LangChain",
    "Python Developer":      "Python, FastAPI, Django, REST APIs, async, testing, databases",
    "Full Stack Developer":  "React, Node.js, Next.js, TypeScript, PostgreSQL, REST APIs",
    "Data Scientist":        "Python, pandas, scikit-learn, ML models, statistics, visualisation",
    "Product Manager":       "roadmaps, prioritisation, metrics, stakeholder management, Agile",
    "Business Analyst":      "requirements gathering, process mapping, SQL, data analysis, documentation",
    "DevOps Engineer":       "Docker, Kubernetes, CI/CD, AWS/GCP, Terraform, monitoring",
    "UX Designer":           "user research, Figma, wireframes, usability testing, design systems",
    "JavaScript Developer":  "JavaScript, TypeScript, React, Node.js, browser APIs, testing",
    "Project Manager":       "delivery, risk management, stakeholder communication, Agile, budgets",
}


# Structured Outputs guarantees this shape. Plain json_object mode with a prompt
# asking for an array failed ~30% of the time: a single question object instead
# of a list, or an outright refusal with no content.
QUESTIONS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "interview_questions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question":       {"type": "string"},
                            "type":           {"type": "string", "enum": ["technical", "behavioural", "system_design"]},
                            "difficulty":     {"type": "string", "enum": ["easy", "medium", "hard"]},
                            "topic":          {"type": "string"},
                            "follow_up":      {"type": "string"},
                            "ideal_keywords": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["question", "type", "difficulty", "topic", "follow_up", "ideal_keywords"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["questions"],
            "additionalProperties": False,
        },
    },
}


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    reraise=False,
)
async def _call_openai(client, model, messages):
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
        response_format=QUESTIONS_SCHEMA,
    )
    # A refusal arrives with no content — treat it as a failure so it gets retried
    message = response.choices[0].message
    if not message.content:
        raise ValueError(f"OpenAI returned no content (refusal: {message.refusal!r})")
    return response


async def generate_questions(
    role: str,
    level: str,
    focus: str,
    job_description: str | None = None,
    cv_text: str | None = None,
    custom_prompt: str | None = None,
    preset_prompts: list[str] | None = None,
) -> list[dict]:
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    role_context = ROLES.get(role, f"{role} role")

    # Role basics always go in; candidate-supplied context is layered on top
    context_parts = [f"Role: {role}\nLevel: {level}\nKey skills: {role_context}"]

    if job_description:
        context_parts.append(f"""
JOB DESCRIPTION (tailor questions to this):
{job_description[:3000]}
""")

    if cv_text:
        context_parts.append(f"""
CANDIDATE CV (personalise questions to their experience — reference specific roles, technologies and projects they mention):
{cv_text[:4000]}
""")

    if preset_prompts:
        context_parts.append(f"""
INTERVIEW PREFERENCES (from the candidate):
{chr(10).join(f'- {p}' for p in preset_prompts)}
""")

    if custom_prompt:
        context_parts.append(f"""
ADDITIONAL CANDIDATE NOTE:
{custom_prompt}
""")

    context = "\n".join(context_parts)

    focus_instruction = {
        "technical":   "Focus 70% on technical questions, 30% behavioural.",
        "behavioural": "Focus 70% on behavioural/situational questions, 30% technical.",
        "mixed":       "Mix technical and behavioural questions equally.",
    }.get(focus, "Mix technical and behavioural questions equally.")

    prompt = f"""You are an expert interviewer. Generate 8 interview questions for a {level}-level {role} candidate.

{context}

{focus_instruction}

Return 8 questions in the "questions" array. Each question must have:
- "question": the question text (string)
- "type": one of "technical", "behavioural", "system_design"  
- "difficulty": one of "easy", "medium", "hard"
- "topic": a short topic label (e.g. "RAG pipelines", "Conflict resolution")
- "follow_up": one natural follow-up question (string)
- "ideal_keywords": array of 3-5 key concepts a strong answer should include

Make the questions realistic, specific to the level, and varied in difficulty.
If the candidate's CV mentions specific projects, companies or technologies, reference them directly in your questions. Make the interview feel personal to them."""

    try:
        response = await _call_openai(
            client, settings.openai_model, [{"role": "user", "content": prompt}]
        )
        parsed = json.loads(response.choices[0].message.content)
    except Exception:
        logger.exception("Question generation failed after retries")
        raise HTTPException(
            status_code=502,
            detail="Question generation failed. Please try again.",
        )

    # Handle if GPT wraps in an object
    if isinstance(parsed, dict):
        for key in ("questions", "items", "data"):
            if key in parsed:
                parsed = parsed[key]
                break

    if not isinstance(parsed, list) or len(parsed) == 0:
        logger.error("Question generation returned no questions: %.300s", json.dumps(parsed))
        raise HTTPException(
            status_code=502,
            detail="Question generation returned empty. Please try again.",
        )

    return parsed
