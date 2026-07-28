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

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# gemini-2.0-flash is fast and free-tier friendly. Swap to
# gemini-1.5-flash if 2.0 isn't available on your account/region.
MODEL_NAME = "gemini-2.0-flash"

TOPIC_SCOPE_INSTRUCTIONS = """You are an assistant embedded in a climate
risk dashboard. You only discuss topics related to climate risk, the
ND-GAIN vulnerability/readiness indicators, the composite risk score
methodology, the forecasts shown in this dashboard, and practical
implications for the user's stated persona. If asked about anything
unrelated, politely decline and redirect to what you can help with."""


def _model(system_instruction: str) -> genai.GenerativeModel:
    return genai.GenerativeModel(MODEL_NAME, system_instruction=system_instruction)


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
    """Draft persona-specific recommendations.

    Args:
        persona: "individual", "business", or "government"
        industry: only used when persona == "business"
        dashboard_summary: output of summarize_dashboard
        driver_summary: output of explain_drivers
    """
    audience = persona if persona != "business" else f"a business in the {industry} industry"

    prompt = f"""Based on this climate risk analysis, draft 3-5 concrete,
actionable recommendations for {audience}. Be specific and practical, not
generic. Ground the recommendations in the data below.

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
