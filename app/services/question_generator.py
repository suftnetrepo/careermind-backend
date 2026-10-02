from fastapi import HTTPException
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings
import json

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
        response_format={"type": "json_object"},
    )
    # OpenAI occasionally returns a completion with no content — treat it as a
    # failure so it gets retried rather than crashing in json.loads
    if not response.choices[0].message.content:
        raise ValueError("OpenAI returned empty content")
    return response


async def generate_questions(
    role: str,
    level: str,
    focus: str,
    job_description: str | None = None,
) -> list[dict]:
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    role_context = ROLES.get(role, f"{role} role")

    if job_description:
        context = f"""
The candidate has applied for this specific role. Use the job description to tailor every question:

JOB DESCRIPTION:
{job_description[:3000]}

Role type: {role}
Level: {level}
"""
    else:
        context = f"""
Role: {role}
Level: {level}
Key skills and technologies: {role_context}
"""

    focus_instruction = {
        "technical":   "Focus 70% on technical questions, 30% behavioural.",
        "behavioural": "Focus 70% on behavioural/situational questions, 30% technical.",
        "mixed":       "Mix technical and behavioural questions equally.",
    }.get(focus, "Mix technical and behavioural questions equally.")

    prompt = f"""You are an expert interviewer. Generate 8 interview questions for a {level}-level {role} candidate.

{context}

{focus_instruction}

Return ONLY a valid JSON array. Each question must have:
- "question": the question text (string)
- "type": one of "technical", "behavioural", "system_design"  
- "difficulty": one of "easy", "medium", "hard"
- "topic": a short topic label (e.g. "RAG pipelines", "Conflict resolution")
- "follow_up": one natural follow-up question (string)
- "ideal_keywords": array of 3-5 key concepts a strong answer should include

Make the questions realistic, specific to the level, and varied in difficulty.
Return only the JSON array, no other text."""

    try:
        response = await _call_openai(
            client, settings.openai_model, [{"role": "user", "content": prompt}]
        )
        parsed = json.loads(response.choices[0].message.content)
    except Exception:
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
        raise HTTPException(
            status_code=502,
            detail="Question generation returned empty. Please try again.",
        )

    return parsed
