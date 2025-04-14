import joblib

def make_prediction(data, model_path="model/user_trained_model.pkl"):
    model = joblib.load(model_path)
    return model.predict(data)
