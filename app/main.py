from fastapi import FastAPI, HTTPException

from app.schemas import Customer
from app.predictor import predict
from app.transformers import FeatureEngineering

app = FastAPI(
    title="Bank Customer Churn API",
    version="1.0"
)


@app.get("/")
def home():

    return {
        "message":"Bank Customer Churn API"
    }


@app.post("/predict/logistic")
def predict_logistic(customer: Customer):

    prediction, probability = predict(
        customer.model_dump(),
        "logistic"
    )

    return {

        "model":"Logistic Regression",

        "prediction":prediction,

        "probability":round(probability,4)

    }


@app.post("/predict/gradient")
def predict_gradient(customer: Customer):

    prediction, probability = predict(
        customer.model_dump(),
        "gradient"
    )

    return {

        "model":"Gradient Boosting",

        "prediction":prediction,

        "probability":round(probability,4)

    }