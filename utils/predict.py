import joblib
import pandas as pd

def make_prediction(data, model_path="model/user_trained_model.pkl", columns_path="model/feature_columns.pkl"):
    # Load model and expected column order
    model = joblib.load(model_path)
    expected_columns = joblib.load(columns_path)

    # Reorder and select only the columns used during training
    aligned_data = data[expected_columns]

    # Make prediction
    return model.predict(aligned_data)
