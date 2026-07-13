'''
Initializes a FastAPI instance for the application with defined endpoints.
'''

from fastapi import FastAPI
import pickle

from package_folder.climate import prediction_function

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {'greeting':"hello"}

# DUMMY Prediction endpoint
@app.get("/predict")
def predict(a: float, b: float, c: float, d: float):
    # Use the function in our package to run the prediction
    prediction = prediction_function(a, b, c, d)

    # Return prediction
    return {"prediction": int(prediction)}
