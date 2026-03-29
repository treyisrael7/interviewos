"""Role Intelligence: infer role profile from job description text via LLM."""

import json
import logging
import re
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# Default when inference fails
DEFAULT_PROFILE: dict[str, Any] = {
    "domain": "general_business",
    "seniority": "entry",
    "roleTitleGuess": "",
    "focusAreas": ["communication", "problem solving"],
    "questionMix": {"behavioral": 40, "roleSpecific": 30, "scenario": 30},
}

# Valid domain values for schema validation
VALID_DOMAINS = frozenset(
    {
        "technical",
        "finance",
        "healthcare_social_work",
        "sales_marketing",
        "operations",
        "education",
        "general_business",
    }
)

VALID_SENIORITIES = frozenset({"entry", "mid", "senior"})


def infer_role_profile(jd_text: str) -> dict[str, Any]:
    """
    Infer role profile from job description text using an LLM.

    Returns JSON with domain, seniority, roleTitleGuess, focusAreas, questionMix.
    On failure, returns DEFAULT_PROFILE.
    """
    if not jd_text or len(jd_text.strip()) < 50:
        logger.info("Role intelligence: text too short, using default")
        return DEFAULT_PROFILE.copy()

    if not settings.openai_api_key:
        logger.warning("Role intelligence: OpenAI not configured, using default")
        return DEFAULT_PROFILE.copy()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)

        system_prompt = """You analyze job descriptions and return a structured JSON profile.

Output ONLY valid JSON, no markdown or other text, matching this schema:
{
  "domain": one of: "technical", "finance", "healthcare_social_work", "sales_marketing", "operations", "education", "general_business",
  "seniority": one of: "entry", "mid", "senior",
  "roleTitleGuess": a short guessed job title (e.g. "Software Engineer", "Social Worker", "Marketing Manager"),
  "focusAreas": array of 4–8 short competency strings extracted from the job description (e.g. "data analysis", "client relationships", "project management"),
  "questionMix": { "behavioral": N, "roleSpecific": N, "scenario": N } where the three numbers MUST sum to 100
}

Rules:
- domain: pick the single best match for the role's primary industry/function.
- seniority: infer from years of experience, title, responsibilities.
- focusAreas: key skills, competencies, or knowledge areas mentioned in the job description.
- questionMix: behavioral=STAR/experience questions, roleSpecific=domain knowledge/skills questions, scenario=hypothetical situation questions. Sum must equal 100."""

        # Truncate to avoid token limits (approx 12k chars ~3k tokens for input)
        text_for_llm = jd_text[:12000] if len(jd_text) > 12000 else jd_text

        response = client.chat.completions.create(
            model=settings.model_fast,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this job description:\n\n{text_for_llm}"},
            ],
            max_tokens=500,
        )

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return DEFAULT_PROFILE.copy()

        # Extract JSON (handle markdown code blocks)
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            raw = json_match.group(0)

        data = json.loads(raw)

        # Validate and normalize
        domain = str(data.get("domain", "")).strip().lower()
        if domain not in VALID_DOMAINS:
            domain = "general_business"

        seniority = str(data.get("seniority", "")).strip().lower()
        if seniority not in VALID_SENIORITIES:
            seniority = "entry"

        role_title = str(data.get("roleTitleGuess", "")).strip()

        focus_raw = data.get("focusAreas")
        if isinstance(focus_raw, list):
            focus_areas = [str(x).strip() for x in focus_raw if x][:8]
        else:
            focus_areas = DEFAULT_PROFILE["focusAreas"].copy()
        if len(focus_areas) < 4:
            focus_areas = focus_areas + ["adaptability", "collaboration"][: 4 - len(focus_areas)]

        qm = data.get("questionMix") or {}
        if isinstance(qm, dict):
            b = int(qm.get("behavioral", 40))
            r = int(qm.get("roleSpecific", 30))
            s = int(qm.get("scenario", 30))
            total = b + r + s
            if total != 100:
                # Normalize to sum to 100
                if total > 0:
                    b = round(100 * b / total)
                    r = round(100 * r / total)
                    s = 100 - b - r
                else:
                    b, r, s = 40, 30, 30
            question_mix = {"behavioral": b, "roleSpecific": r, "scenario": s}
        else:
            question_mix = DEFAULT_PROFILE["questionMix"].copy()

        return {
            "domain": domain,
            "seniority": seniority,
            "roleTitleGuess": role_title,
            "focusAreas": focus_areas,
            "questionMix": question_mix,
        }
    except Exception as e:
        logger.warning("Role intelligence inference failed: %s", e, exc_info=True)
        return DEFAULT_PROFILE.copy()
