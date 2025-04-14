import joblib
import pandas as pd

def make_prediction(data, model_path="model/user_trained_model.pkl", columns_path="model/feature_columns.pkl"):
    # Load model and expected feature columns
    model = joblib.load(model_path)
    expected_columns = joblib.load(columns_path)

    # Align incoming data with the expected feature order
    data = data[expected_columns]

    # Predict
    return model.predict(data)
