import joblib
import pandas as pd
from app.transformers import FeatureEngineering


logistic_model = joblib.load(
    "models/logistic_pipeline.pkl"
)

gradient_model = joblib.load(
    "models/gradient_pipeline.pkl"
)


MODELS = {
    "logistic": logistic_model,
    "gradient": gradient_model
}


def predict(customer, model_name):

    df = pd.DataFrame([customer])

    model = MODELS[model_name]

    prediction = int(
        model.predict(df)[0]
    )

    probability = float(
        model.predict_proba(df)[0][1]
    )

    return prediction, probability