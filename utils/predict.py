import joblib

def make_prediction(data, model_path="model.pkl"):
    model = joblib.load(model_path)
    return model.predict(data)
