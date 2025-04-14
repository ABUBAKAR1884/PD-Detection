import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_model(data, target_column, model_type="Random Forest"):
    """
    Trains a machine learning model based on user-provided data and model type.
    
    Args:
        data (DataFrame): The dataset containing features and target column.
        target_column (str): The name of the target column to predict.
        model_type (str): The type of model to train. Options: "Random Forest", 
                          "Logistic Regression", "Neural Network".
    
    Returns:
        model (sklearn model): The trained model object.
        accuracy (float): Accuracy score on the validation set.
        f1 (float): F1 score on the validation set.
    """
    # Split data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    if model_type == "Random Forest":
        model = RandomForestClassifier()
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Neural Network":
        model = MLPClassifier(max_iter=1000)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Evaluate model
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')

    # Save trained model
    joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")


    return model, accuracy, f1
