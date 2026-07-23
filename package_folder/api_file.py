'''
Initializes a FastAPI instance for the application with defined endpoints.
'''

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from package_folder.climate import prediction_function, all_predictions

# FastAPI instance
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Return the composite climate risk score for a country and year.

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
        result = prediction_function(country, year)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    return {"country": country.upper(), "year": year, **result}

# All-countries endpoint
@app.get("/predict_all")
def predict_all(year: int | None = None):
    """Return every country's risk score, optionally filtered to one year.

    Args:
        year: Optional calendar year to narrow results to (still all countries).
    """
    records = all_predictions(year)
    return {"count": len(records), "data": records}
