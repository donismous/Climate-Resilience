'''
Initializes a FastAPI instance for the application with defined endpoints.
'''
from dotenv import load_dotenv
load_dotenv()

import pycountry
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from package_folder.climate import (
    prediction_function,
    all_predictions,
    get_global_movers,
    get_country_detail,
    get_cached_recommendation,
    get_cached_summary,
    get_country_tiers,
    get_feature_importance,
    get_global_drivers,
    get_alerts,
    get_trend_comparison_data,
    flag_url,
    get_indicators,
    get_country_feature_attribution
)

from package_folder.llm_integration import (
    explain_drivers,
    draft_recommendations,
    ChatSession,
    summarize_world_map,
    summarize_country_detail,
    explain_global_drivers,
    summarize_alert_tracker,
    explain_trend_comparison
)

# FastAPI instance
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for chat sessions.
_chat_sessions: dict[str, ChatSession] = {}

CACHEABLE_PERSONAS = {"individual", "government/institution"}


# Root endpoint
@app.get("/")
def root():
    return {'greeting': "hello"}


# Prediction endpoint
@app.get("/predict")
def predict(
    country: str = Query(
        ...,
        min_length=3,
        max_length=3,
        pattern="^[A-Za-z]{3}$",
        description="ISO3 country code, e.g. FRA",
    ),
    year: int = Query(..., description="Calendar year"),
):
    """Return every ND-GAIN indicator's value for a country and year.

    Shaped identically to /predict_all: {"count": N, "data": [...]}, where
    each record has the same fields as a /predict_all row.

    Args:
        country: ISO3 country code, e.g. "FRA". Must be a real ISO 3166-1
            alpha-3 code, checked before hitting the data lookup.
        year: Calendar year (actual or forecast, up to the horizon in the
            precomputed data).
    """
    country = country.upper()
    if pycountry.countries.get(alpha_3=country) is None:
        raise HTTPException(
            status_code=422,
            detail=f"{country!r} is not a valid ISO3 country code.",
        )

    try:
        records = prediction_function(country, year)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    return {"count": len(records), "data": records}


# All-countries endpoint
@app.get("/predict_all")
def predict_all(year: int | None = None):
    """Return every country's risk score, optionally filtered to one year.

    Args:
        year: Optional calendar year to narrow results to (still all countries).
    """
    records = all_predictions(year)
    return {"count": len(records), "data": records}


# Vulnerability x Readiness tiers endpoint
@app.get("/tiers")
def tiers(year: int | None = None):
    """Return every country's Vulnerability x Readiness tier assignment.

    Args:
        year: Optional calendar year to narrow results to (still all countries).
    """
    records = get_country_tiers(year)
    return {"count": len(records), "data": records}


@app.get("/feature-importance")
def feature_importance(
    country: str = Query("global", description="'global' or an ISO3 country code"),
    year: int | None = Query(None, description="Optional year filter (country scope only)"),
):
    """Return each ND-GAIN sub-indicator's contribution to the composite
    risk score, ranked by SHAP importance (highest first).

    Args:
        country: "global" for the overall model-wide contribution, or an
            ISO3 country code for that country's own SHAP breakdown.
        year: Optional calendar year to narrow a country's results to
            (ignored for "global", which has no year dimension).
    """
    if country.upper() == "GLOBAL":
        result = get_feature_importance()
        return {"count": len(result["data"]), "data": result["data"], "caveat": result["caveat"]}

    country = country.upper()
    if pycountry.countries.get(alpha_3=country) is None:
        raise HTTPException(status_code=422, detail=f"{country!r} is not a valid ISO3 country code.")
    result = get_country_feature_attribution(country, year)
    return {"count": len(result["data"]), "data": result["data"]}


@app.get("/indicators")
def indicators():
    """Return each ND-GAIN sub-indicator's global contribution to the
    composite risk score, ranked by SHAP importance (highest first)."""
    result = get_indicators()
    return {"count": len(result["data"]), "data": result["data"], "caveat": result["caveat"]}



# ------- LLM-powered endpoints -------

class DriverInput(BaseModel):
    pca_summary: dict
    shap_summary: dict


class RecommendationRequest(BaseModel):
    scope: str  # "world" or an ISO3 code, used for cache lookup
    persona: str  # "individual", "business", or "government/institution"
    industry: str | None = None
    dashboard_summary: str
    driver_summary: str


class ChatStartRequest(BaseModel):
    session_id: str
    persona: str
    industry: str | None = None
    context_summary: str


class ChatMessageRequest(BaseModel):
    session_id: str
    message: str


@app.post("/explain-drivers")
def drivers(input: DriverInput):
    return {"explanation": explain_drivers(input.pca_summary, input.shap_summary)}


@app.post("/recommendations")
def recommendations(request: RecommendationRequest):
    if request.persona == "business" and not request.industry:
        raise HTTPException(status_code=422, detail="industry is required when persona is 'business'.")

    if request.persona in CACHEABLE_PERSONAS:
        cached = get_cached_recommendation(request.scope, request.persona)
        if cached is not None:
            return {"recommendations": cached}

    text = draft_recommendations(
        persona=request.persona,
        industry=request.industry,
        dashboard_summary=request.dashboard_summary,
        driver_summary=request.driver_summary,
    )
    return {"recommendations": text}


@app.post("/chat/start")
def chat_start(request: ChatStartRequest):
    if request.persona == "business" and not request.industry:
        raise HTTPException(status_code=422, detail="industry is required when persona is 'business'.")
    _chat_sessions[request.session_id] = ChatSession(
        persona=request.persona,
        industry=request.industry,
        context_summary=request.context_summary,
    )
    return {"status": "started", "session_id": request.session_id}


@app.post("/chat/message")
def chat_message(request: ChatMessageRequest):
    session = _chat_sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No chat session found for this session_id. Call /chat/start first.")
    answer = session.ask(request.message)
    return {"answer": answer}


@app.get("/world-summary")
def world_summary(top_n: int = 5):
    movers = get_global_movers(top_n=top_n)
    cached = get_cached_summary("world")
    summary = cached if cached is not None else summarize_world_map(movers)
    return {"movers": movers, "summary": summary}


@app.get("/country-detail/{country}")
def country_detail(country: str):
    country = country.upper()
    if pycountry.countries.get(alpha_3=country) is None:
        raise HTTPException(status_code=422, detail=f"{country!r} is not a valid ISO3 country code.")
    try:
        detail = get_country_detail(country)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))

    cached = get_cached_summary(country)
    summary = cached if cached is not None else summarize_country_detail(detail)
    return {"detail": detail, "summary": summary, "flag": flag_url(country)}

@app.get("/global-drivers")
def global_drivers():
    drivers = get_global_drivers()
    cached = get_cached_summary("global_drivers")
    explanation = cached if cached is not None else explain_global_drivers(
        drivers["global_importance"],
        drivers["feature_dependence"],
    )
    return {"drivers": drivers, "explanation": explanation}

@app.get("/alerts")
def alerts():
    alert_list = get_alerts()
    cached = get_cached_summary("alerts")
    explanation = cached if cached is not None else summarize_alert_tracker(alert_list)
    return {"alerts": alert_list, "explanation": explanation}


class TrendComparisonRequest(BaseModel):
    countries: list[str]

@app.post("/trend-comparison")
def trend_comparison(request: TrendComparisonRequest):
    if len(request.countries) < 2:
        raise HTTPException(status_code=422, detail="Select at least 2 countries to compare.")
    if len(request.countries) > 5:
        raise HTTPException(status_code=422, detail="Select at most 5 countries to compare.")

    comparison_data = get_trend_comparison_data(request.countries)
    if len(comparison_data) < 2:
        raise HTTPException(status_code=404, detail="Not enough valid countries found for comparison.")

    explanation = explain_trend_comparison(comparison_data)
    return {"data": comparison_data, "explanation": explanation}
