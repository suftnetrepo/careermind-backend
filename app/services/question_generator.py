from openai import AsyncOpenAI
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

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    # Handle if GPT wraps in an object
    if isinstance(parsed, dict):
        for key in ("questions", "items", "data"):
            if key in parsed:
                parsed = parsed[key]
                break

    return parsed if isinstance(parsed, list) else []
