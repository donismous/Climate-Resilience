'''
Initializes a FastAPI instance for the application with defined endpoints.
'''

from fastapi import FastAPI, HTTPException

from package_folder.climate import prediction_function

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {'greeting': "hello"}

# Prediction endpoint
@app.get("/predict")
def predict(country: str, year: int):
    """Return the composite climate risk score for a country and year.

    Args:
        country: ISO3 country code, e.g. "FRA".
        year: Calendar year (actual or forecast, up to the horizon in the
            precomputed data).
    """
    try:
        result = prediction_function(country, year)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    return {"country": country.upper(), "year": year, **result}
