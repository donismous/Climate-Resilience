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

MODEL_NAME = "mistral-small-2506"

# Max output tokens per function. Kept small
MAX_TOKENS_DASHBOARD_SUMMARY = 200
MAX_TOKENS_WORLD_SUMMARY = 400
MAX_TOKENS_COUNTRY_SUMMARY = 400
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

def _format_shap_trend(trend: list[dict]) -> str:
    """Format only the significant SHAP importance changes as plain text.
    If none are significant, says so explicitly rather than listing
    unchanged values."""
    significant = [t for t in trend if t["significant"]]
    if not significant:
        return "No indicator's importance is forecasted to change significantly through 2040 -- the key drivers remain largely the same."
    lines = [
        f"- {t['name']}: importance {t['direction']} ({t['change']:+.1f} percentage points by 2040)"
        for t in significant
    ]
    return "\n".join(lines)

def _format_indicators(indicators: list[dict]) -> str:
    """Format indicators as a favorability percentage + qualitative bucket
    (higher = better, for every indicator) plus its effect on risk score."""
    lines = []
    for ind in indicators:
        favorable_pct = (1 - ind['latest_value']) * 100
        if favorable_pct >= 66.7:
            bucket = "high"
        elif favorable_pct >= 33.3:
            bucket = "moderate"
        else:
            bucket = "low"

        shap = ind.get("shap_contribution")
        effect = ""
        if shap is not None:
            effect = f", {'reduces' if shap < 0 else 'increases'} this country's risk score"
        lines.append(f"- {ind['name']}: {favorable_pct:.0f}% favorable ({bucket}){effect}")
    return "\n".join(lines)

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
reputable, vetted climate organizations). Consider the specific countries income level
and top indicators. Avoid suggesting they personally
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
    """Narrate the world map view: current best/worst, forecast trend, and
    global indicator strengths/weaknesses.
    """
    best_ind_text = "\n".join(f"- {i['name']}" for i in movers['best_indicators_global'])
    worst_ind_text = "\n".join(f"- {i['name']}" for i in movers['worst_indicators_global'])
    shap_trend_text = _format_shap_trend(movers.get('global_shap_trend', []))

    prompt = f"""Summarize this global climate risk picture. Start with one
short introductory sentence, then respond in concise bullet points (max 8
bullets, one short sentence each). Be aware that the trend figures are
forward-looking forecasts (through {movers['forecast_target_year']}), not
historical patterns. When referring to indicators, use just their plain
name (e.g. "Governance", "Water") -- do not append category labels like
"(readiness)" or "(vulnerability)" after the indicator name.

Countries that perform best today: {movers['best_current']}
Countries that perform worst today: {movers['worst_current']}
Countries with the best forecasted trend: {movers['improved']}
Countries with the worst forecasted trend: {movers['worsened']}
Indicators that most reduce risk globally: {movers['best_indicators_global']}
Indicators that most increase risk globally: {movers['worst_indicators_global']}
How much each global driver's importance is forecasted to change by 2040:
{shap_trend_text}
Global average forecasted trend: {movers['global_direction']} (slope: {movers['global_mean_slope']:.5f}/year)"""
    return _generate(TOPIC_SCOPE_INSTRUCTIONS, prompt, max_tokens=MAX_TOKENS_WORLD_SUMMARY)


def summarize_country_detail(detail: dict) -> str:
    """Narrate a single country's performance, forecast trend, and the
    indicators that most drive its risk score (by SHAP contribution).
    """
    strongest_text = _format_indicators(detail['strongest_indicators'])
    weakest_text = _format_indicators(detail['weakest_indicators'])
    shap_trend_text = _format_shap_trend(detail.get('shap_importance_trend', []))

    prompt = f"""Summarize this country's climate risk profile. Start with
one short introductory sentence, then respond in concise bullet points (max
8 bullets). Be aware that the trend is a forward-looking forecast, not a
historical pattern. When referring to indicators, use just their plain
name (e.g. "Governance", "Capacity") -- do not append category labels like
"(readiness)" or "(vulnerability)" after the indicator name. Use ONLY the
value given for each indicator -- do not invert it, convert it, or compute
any second number yourself.

You MUST organize your response into exactly two clearly labeled sections:
one titled "Risk-reducing indicators" covering ONLY the three indicators
listed under "REDUCES RISK" below, and one titled "Risk-increasing
indicators" covering ONLY the three listed under "INCREASES RISK" below.
Never place an indicator from one list under the other section's heading.

Country: {detail['country_name']} ({detail['country']})
Latest actual risk score: {detail['trend']['latest_actual_value']:.3f} (as of {detail['trend']['latest_actual_year']})
Forecasted trend: {detail['trend']['direction']} (projected slope: {detail['trend']['forecast_slope_per_year']:.5f}/year, forecasted risk score by 2040: {detail['trend'].get('forecast_2040_value'):.3f})

REDUCES RISK:
{strongest_text}

INCREASES RISK:
{weakest_text}

For each indicator under REDUCES RISK, briefly note the mechanism by which
it helps this country. For each indicator under INCREASES RISK, briefly
explain the mechanism by which it likely drags down this country's risk
profile. If an indicator has a favorable value but still appears under
INCREASES RISK, its actual effect on risk is small -- say so plainly (e.g.
"has only a minor negative effect despite an otherwise good score") rather
than inventing a specific causal story to explain why a good score would
increase risk. Do not invent specific numeric targets, dates, or
statistics beyond what is given above.

How much each indicator's importance for this country is forecasted to
change by 2040:
{shap_trend_text}"""
    return _generate(TOPIC_SCOPE_INSTRUCTIONS, prompt, max_tokens=MAX_TOKENS_COUNTRY_SUMMARY)

def explain_global_drivers(global_importance: list, feature_dependence: dict) -> str:
    """Narrate what drives global climate risk, and how consistently each
    driver behaves (based on correlation between its raw value and its
    SHAP contribution across all countries/years).
    """
    prompt = f"""Explain what drives global climate risk. Start with one
short introductory sentence, then respond in concise bullet points (max 7,
one short sentence each).

Globally most impactful indicators (on risk score): {global_importance}

Consistency of each indicator's effect (correlation between the indicator's
raw value and how much it pushes risk up or down; "consistent" means the
indicator reliably affects risk in the same direction, "inconsistent"
means its effect on risk varies): {feature_dependence}

For indicators marked "inconsistent", note only that their effect on risk
varies rather than describing a specific pattern -- do not invent a shape
or threshold you cannot verify from this data."""
    return _generate(TOPIC_SCOPE_INSTRUCTIONS, prompt, max_tokens=350)


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

def summarize_alert_tracker(alerts: list[dict]) -> str:
    currently = [a for a in alerts if a["status"] == "currently_above_threshold"]
    entering = [a for a in alerts if a["status"] == "forecasted_to_cross_threshold"]
    recovering = [a for a in alerts if a["status"] == "forecasted_to_recover"]

    def _format_alert(a: dict, verb: str) -> str:
        drivers = ", ".join(d["name"] for d in a.get("top_risk_drivers", []))
        return f"- {a['country_name']}: forecasted to {verb} in {a['crossing_year']}, driven mainly by {drivers}"

    entering_sorted = sorted(entering, key=lambda a: a["crossing_year"])
    recovering_sorted = sorted(recovering, key=lambda a: a["crossing_year"])

    entering_text = "\n".join(_format_alert(a, "cross the threshold") for a in entering_sorted[:10]) or "None currently forecasted."
    recovering_text = "\n".join(_format_alert(a, "drop back below the threshold") for a in recovering_sorted[:10]) or "None currently forecasted."

    prompt = f"""Explain this climate risk alert tracker in one short
introductory sentence describing what the alert threshold means and how
many countries are currently flagged with no forecasted recovery
({len(currently)} total, already shown in the accompanying table). Then
respond in two clearly labeled sections (max 5 bullets each):

1. "Newly entering the alert list" -- countries not currently flagged but
forecasted to cross the threshold soon.
2. "Forecasted to leave the alert list" -- countries currently flagged but
forecasted to recover below the threshold soon.

Always state the specific year given for each country, and its top
driver(s). Do not invent specific numeric targets or dates beyond what is
given.

Countries newly entering the alert list, sorted by soonest:
{entering_text}

Countries forecasted to leave the alert list, sorted by soonest:
{recovering_text}"""
    return _generate(TOPIC_SCOPE_INSTRUCTIONS, prompt, max_tokens=450)

def explain_trend_comparison(countries: list[dict]) -> str:
    """Compare 2-5 selected countries' climate risk trajectories: current
    position, forecasted direction, and top driver behind each.
    """
    def _format_country(c: dict) -> str:
        top_reducer = c["top_reducer"]["name"] if c.get("top_reducer") else "unknown"
        top_increaser = c["top_increaser"]["name"] if c.get("top_increaser") else "unknown"
        return (
            f"- {c['country_name']}: risk score {c['latest_actual_value']:.3f} "
            f"(as of {c['latest_actual_year']}), forecasted {c['direction']} "
            f"to {c['forecast_2040_value']:.3f} by 2040. "
            f"Top risk-reducing factor: {top_reducer}. Top risk-increasing factor: {top_increaser}."
        )

    countries_text = "\n".join(_format_country(c) for c in countries)

    prompt = f"""Compare these countries' climate risk trajectories. Start
with one short introductory sentence, then respond in bullet points (max 2
per country): one bullet on its current position relative to the others,
one bullet on why its forecast moves in the direction it does, based on
its top driver given below. Do not invent specific numeric targets, dates,
or statistics beyond what is given.

{countries_text}"""
    return _generate(TOPIC_SCOPE_INSTRUCTIONS, prompt, max_tokens=450)
