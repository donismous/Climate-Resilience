"""LLM integration for the climate risk dashboard, using Google Gemini's
free tier (Google AI Studio, not Vertex AI -- no billing required).

Provides:
- summarize_dashboard: narrate the currently filtered view
- explain_drivers: turn PCA + SHAP output into a plain-language insight
- draft_recommendations: persona-specific recommendations
- ChatSession: a topic-scoped chat, gated behind a persona selection

All functions send small, pre-computed summaries to the model rather than
raw dataframes -- the model's job is narration/interpretation, not analysis.
"""

import os

import google.generativeai as genai
import google.api_core.exceptions as google_exceptions


genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-flash-latest"

TOPIC_SCOPE_INSTRUCTIONS = """You are an assistant embedded in a climate
risk dashboard. You only discuss topics related to climate risk, the
ND-GAIN vulnerability/readiness indicators, the composite risk score
methodology, the forecasts shown in this dashboard, and practical
implications for the user's stated persona. If asked about anything
unrelated, politely decline and redirect to what you can help with."""

def _model(system_instruction: str) -> genai.GenerativeModel:
    return genai.GenerativeModel(MODEL_NAME, system_instruction=system_instruction)


def _generate(model: genai.GenerativeModel, prompt: str, max_output_tokens: int) -> str:
    """Wraps generate_content with a friendly message on quota errors."""
    try:
        response = model.generate_content(prompt, generation_config={"max_output_tokens": max_output_tokens})
        return response.text
    except google_exceptions.ResourceExhausted:
        return (
            "The AI summary service has hit its free-tier daily request limit. "
            "Please try again later."
        )


def summarize_dashboard(filtered_stats: dict) -> str:
    """Narrate the currently filtered dashboard view.

    Args:
        filtered_stats: small dict of precomputed numbers, e.g.
            {"countries_shown": [...], "year_range": [1995, 2040],
             "mean_risk_score": 0.52, "highest_risk_country": "SOM",
             "lowest_risk_country": "EST", "trend": "declining"}
    """
    prompt = f"""Summarize this climate risk dashboard view in 3-4 sentences
for a general audience. Data: {filtered_stats}"""

    response = _model(TOPIC_SCOPE_INSTRUCTIONS).generate_content(prompt)
    return response.text


def explain_drivers(pca_summary: dict, shap_summary: dict) -> str:
    """Turn PCA + SHAP output into a plain-language driver explanation."""
    prompt = f"""Based on this PCA and SHAP analysis of what drives the
composite climate risk score, explain in plain language (no jargon like
'eigenvalue' or 'Shapley') which underlying factors matter most and why.

PCA summary: {pca_summary}
SHAP feature importances: {shap_summary}"""

    response = _model(TOPIC_SCOPE_INSTRUCTIONS).generate_content(prompt)
    return response.text


def draft_recommendations(
    persona: str, industry: str | None, dashboard_summary: str, driver_summary: str
) -> str:
    """Draft persona-specific recommendations focused on contributing to
    climate risk reduction, not personal financial self-protection.

    Args:
        persona: "individual", "business", or "government"
        industry: only used when persona == "business"
        dashboard_summary: output of summarize_dashboard
        driver_summary: output of explain_drivers
    """
    audience = persona if persona != "business" else f"a business in the {industry} industry"

    prompt = f"""Based on this climate risk analysis, draft recommendations
for {audience} focused specifically on how they can contribute to REDUCING
climate risk globally or for the countries/regions involved -- not on how
they can protect their own assets, investments, or personal financial
interests from climate risk.

Do NOT recommend things like: reallocating investments away from
high-risk regions, personal financial hedging, building personal resilience
infrastructure for self-protection, or diversifying income away from
vulnerable areas. These protect the individual/business from risk but do
nothing to reduce the risk itself.

Instead, focus on concrete actions that address the underlying drivers of
risk identified below -- for example: supporting or investing in adaptation
and readiness-building initiatives, advocating for or funding policy and
infrastructure improvements, engaging in practices that reduce vulnerability
or improve governance/economic stability in affected regions, or otherwise
contributing resources, expertise, or influence toward genuinely lowering
climate risk.

Respond ONLY in concise bullet points (3-5 bullets, one short actionable
sentence each). Be specific to this audience and grounded in the data
below. No preamble, no closing remarks.

Dashboard summary: {dashboard_summary}
Driving factors: {driver_summary}"""

    response = _model(TOPIC_SCOPE_INSTRUCTIONS).generate_content(prompt)
    return response.text


class ChatSession:
    """A topic-scoped chat session, gated behind a persona selection."""

    def __init__(self, persona: str, industry: str | None, context_summary: str):
        """
        Args:
            persona: "individual", "business", or "government"
            industry: only used when persona == "business"
            context_summary: combined dashboard + driver summary, given to
                the model once as grounding context, not repeated per turn
        """
        audience = persona if persona != "business" else f"a business in the {industry} industry"

        system_instruction = f"""{TOPIC_SCOPE_INSTRUCTIONS}

The user you are speaking with is: {audience}. Tailor your answers to what
matters for this audience specifically.

Dashboard context for this conversation:
{context_summary}"""

        self._model = _model(system_instruction)
        self._chat = self._model.start_chat(history=[])

    def ask(self, question: str) -> str:
        response = self._chat.send_message(question)
        return response.text

    @property
    def history(self) -> list[dict]:
        """Expose history in a simple role/content shape for the frontend."""
        return [
            {
                "role": "user" if msg.role == "user" else "assistant",
                "content": msg.parts[0].text,
            }
            for msg in self._chat.history
        ]

def summarize_world_map(movers: dict) -> str:
    """Narrate the world map view: top movers + global trend."""
    prompt = f"""Summarize this global climate risk picture. Respond ONLY in
concise bullet points (max 6 bullets, one short sentence each). Short preamble.

Countries that improved the most long-term: {movers['improved']}
Countries that worsened the most long-term: {movers['worsened']}
Global average trend: {movers['global_direction']} (slope: {movers['global_mean_slope']:.5f}/year)"""

    response = _model(TOPIC_SCOPE_INSTRUCTIONS).generate_content(
        prompt, generation_config={"max_output_tokens": 1000}
    )
    return response.text


def summarize_country_detail(detail: dict) -> str:
    """Narrate a single country's performance, trend, and indicator breakdown."""
    prompt = f"""Summarize this country's climate risk profile. Respond ONLY
in concise bullet points (max 6 bullets). Short preamble.

Country: {detail['country_name']} ({detail['country']})
Trend: {detail['trend']['direction']} (slope: {detail['trend']['slope_per_year']:.5f}/year, statistically significant: {detail['trend']['significant']})
Strongest indicators (most favorable): {detail['strongest_indicators']}
Weakest indicators (least favorable): {detail['weakest_indicators']}

For each weakest indicator, briefly explain in plain language why it likely
drags down this country's risk profile."""

    response = _model(TOPIC_SCOPE_INSTRUCTIONS).generate_content(
        prompt, generation_config={"max_output_tokens": 1000}
    )
    return response.text
