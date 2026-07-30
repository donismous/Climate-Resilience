"""LLM integration for the climate risk dashboard, using the Mistral AI API
(La Plateforme free "Experiment" tier).

Provides:
- summarize_dashboard: narrate the currently filtered view
- explain_drivers: turn PCA + SHAP output into a plain-language insight
- draft_recommendations: persona-specific recommendations
- ChatSession: a topic-scoped chat, gated behind a persona selection

All functions send small, pre-computed summaries to the model rather than
raw dataframes -- the model's job is narration/interpretation, not analysis.
"""

import os

from mistralai.client import Mistral
from mistralai.client.errors import SDKError

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

MODEL_NAME = "mistral-small-latest"

# Max output tokens per function. Kept small
MAX_TOKENS_DASHBOARD_SUMMARY = 200
MAX_TOKENS_WORLD_SUMMARY = 300
MAX_TOKENS_COUNTRY_SUMMARY = 350
MAX_TOKENS_DRIVERS = 250
MAX_TOKENS_RECOMMENDATIONS = 300
MAX_TOKENS_CHAT = 500

TOPIC_SCOPE_INSTRUCTIONS = """You are an assistant embedded in a climate
risk dashboard. You discuss topics related to climate risk and its drivers:
the ND-GAIN vulnerability/readiness indicators, the composite risk score
methodology, the forecasts shown in this dashboard, climate adaptation and
mitigation policy, energy and economic policy as it relates to climate
resilience, and practical implications and actions relevant to whoever you
are speaking with. If asked about something with no connection to climate
risk or its drivers, politely decline and redirect to what you can help
with.

When asked about contested policy or empirical questions within scope
(e.g. energy policy choices, adaptation strategy tradeoffs), give a fair,
balanced overview of the main perspectives and evidence rather than a
single confident opinion, and note where genuine expert disagreement
exists."""


def _generate(system_instruction: str, prompt: str, max_tokens: int) -> str:
    """Single-turn completion with a friendly message on API errors."""
    try:
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except SDKError as error:
        if "429" in str(error) or "rate limit" in str(error).lower():
            return "The AI service has hit its free-tier request limit. Please try again later."
        return "The AI service is temporarily unavailable. Please try again shortly."
    except Exception:
        return "Something went wrong generating this response. Please try again."


def summarize_dashboard(filtered_stats: dict) -> str:
    """Narrate the currently filtered dashboard view.

    Args:
        filtered_stats: small dict of precomputed numbers, e.g.
            {"countries_shown": [...], "year_range": [2020, 2028],
             "mean_risk_score": 0.52, "highest_risk_country": "SOM",
             "lowest_risk_country": "EST", "trend": "declining"}
    """
    prompt = f"""Summarize this climate risk dashboard view for a general
audience. Start with one short introductory sentence, then respond in
concise bullet points (3-5 bullets max, each one short sentence). No
closing remarks.

Data: {filtered_stats}"""
    return _generate(TOPIC_SCOPE_INSTRUCTIONS, prompt, max_tokens=MAX_TOKENS_DASHBOARD_SUMMARY)


def explain_drivers(pca_summary: dict, shap_summary: dict) -> str:
    """Turn PCA + SHAP output into a plain-language driver explanation."""
    prompt = f"""Based on this PCA and SHAP analysis of what drives the
composite climate risk score, explain which underlying factors matter most
and why. Start with one short introductory sentence, then respond in
concise bullet points (max 5 bullets, one short sentence each). No jargon
like 'eigenvalue' or 'Shapley'. No closing remarks.

PCA summary: {pca_summary}
SHAP feature importances: {shap_summary}"""
    return _generate(TOPIC_SCOPE_INSTRUCTIONS, prompt, max_tokens=MAX_TOKENS_DRIVERS)


def draft_recommendations(
    persona: str, industry: str | None, dashboard_summary: str, driver_summary: str
) -> str:
    """Draft persona-specific recommendations focused on contributing to
    climate risk reduction, not personal financial self-protection.
    """
    audience = persona if persona != "business" else f"a business in the {industry} industry"

    scale_guidance = {
        "individual": """This is an average private citizen, not a policymaker,
investor, or NGO director. Recommend things a normal person can realistically
do with their own time, money, and voice -- e.g. reducing personal carbon
footprint and consumption, making informed everyday choices (energy use,
transport, food), volunteering with local resilience/disaster-preparedness
efforts, joining or supporting community climate initiatives, voting with
climate resilience in mind and contacting local representatives, spreading
accurate awareness among friends/family/community, or making small,
affordable personal contributions (e.g. modest recurring donations to
reputable, vetted climate organizations). Avoid suggesting they personally
"fund," "mobilize capital toward," or "advocate for" large-scale international
programs -- that is not realistic for an individual citizen.""",
        "business": f"""This is a business in the {industry} industry.
Recommend actions realistic for a company of unspecified size -- e.g.
adapting supply chains, investing in resilient operations, adopting
sustainable practices relevant to their industry, engaging in industry
coalitions or standards efforts, or responsible investment/procurement
choices tied to climate readiness.""",
        "government/institution": """This is a government or institutional
actor. Recommend policy, regulatory, investment, and governance actions
realistic for a public institution -- e.g. adaptation and readiness
investment, international cooperation, regulatory frameworks, and
institutional capacity-building.""",
    }.get(persona, "")

    prompt = f"""Based on this climate risk analysis, draft recommendations
for {audience} focused specifically on how they can contribute to REDUCING
climate risk globally or for the countries/regions involved -- not on how
they can protect their own assets, investments, or personal financial
interests from climate risk.

{scale_guidance}

Start with one short introductory sentence, then respond in concise bullet
points (3-5 bullets, one short actionable sentence each). Be specific to
this audience and grounded in the data below. No closing remarks.

Dashboard summary: {dashboard_summary}
Driving factors: {driver_summary}"""
    return _generate(TOPIC_SCOPE_INSTRUCTIONS, prompt, max_tokens=MAX_TOKENS_RECOMMENDATIONS)


def summarize_world_map(movers: dict) -> str:
    """Narrate the world map view: current best/worst, long-term movers, and global trend."""
    prompt = f"""Summarize this global climate risk picture. Start with one
short introductory sentence, then respond in concise bullet points (max 8
bullets, one short sentence each). No closing remarks.

Countries that perform best (status quo, lowest current risk scores): {movers['best_current']}
Countries that perform worst (status quo, highest current risk scores): {movers['worst_current']}
Countries that improved the most long-term (trend): {movers['improved']}
Countries that worsened the most long-term (trend): {movers['worsened']}
Global average trend: {movers['global_direction']} (slope: {movers['global_mean_slope']:.5f}/year)"""
    return _generate(TOPIC_SCOPE_INSTRUCTIONS, prompt, max_tokens=MAX_TOKENS_WORLD_SUMMARY)


def summarize_country_detail(detail: dict) -> str:
    """Narrate a single country's performance, trend, and indicator breakdown."""
    prompt = f"""Summarize this country's climate risk profile. Start with
one short introductory sentence, then respond in concise bullet points (max
6 bullets).

Country: {detail['country_name']} ({detail['country']})
Trend: {detail['trend']['direction']} (slope: {detail['trend']['slope_per_year']:.5f}/year, statistically significant: {detail['trend']['significant']})
Strongest indicators (most favorable): {detail['strongest_indicators']}
Weakest indicators (least favorable): {detail['weakest_indicators']}

For each weakest indicator, briefly explain in plain language why it likely
drags down this country's risk profile."""
    return _generate(TOPIC_SCOPE_INSTRUCTIONS, prompt, max_tokens=MAX_TOKENS_COUNTRY_SUMMARY)


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

        self.system_instruction = f"""{TOPIC_SCOPE_INSTRUCTIONS}

The user you are speaking with is: {audience}. Tailor your answers to what
matters for this audience specifically.

Dashboard context for this conversation:
{context_summary}

Keep answers concise. Prefer short bullet points when listing multiple
things; use brief plain sentences for simple/direct questions. Avoid long
paragraphs."""

        self.history: list[dict] = [{"role": "system", "content": self.system_instruction}]

    def ask(self, question: str) -> str:
        self.history.append({"role": "user", "content": question})
        try:
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=self.history,
                max_tokens=MAX_TOKENS_CHAT,
            )
            answer = response.choices[0].message.content
        except SDKError as error:
            if "429" in str(error) or "rate limit" in str(error).lower():
                answer = "The AI chat service has hit its free-tier request limit. Please try again later."
            else:
                answer = "The AI chat service is temporarily unavailable. Please try again shortly."
        except Exception:
            answer = "Something went wrong. Please try again."

        self.history.append({"role": "assistant", "content": answer})
        return answer
